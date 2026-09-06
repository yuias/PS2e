//! EE-side view of the GS: privileged registers, CSR/IMR interrupt logic,
//! and the command stream into the renderer.
//!
//! The renderer ([`Gs`]) either lives inline or, with the `threads` feature,
//! on a worker thread that consumes batches of register writes. Everything
//! the EE can observe stays on this side and is decided at enqueue time
//! (SIGNAL/FINISH flags, CSR, display registers), so emulated behaviour does
//! not depend on how far the worker has got; the worker only owns VRAM and
//! the rasterizer. Reads that need VRAM (screenshots, dumps, statistics)
//! drain the queue first.

use super::Gs;
use tracing::{debug, trace};

/// One composited display frame: width, height, RGBA8.
pub type Frame = (u32, u32, Vec<u8>);

/// The composited frame as the front end hands it out: the pixels sit
/// behind an `Arc` so a reader takes a pointer, not a megabyte.
pub type SharedFrame = (u32, u32, std::sync::Arc<Vec<u8>>);

/// Renderer statistics, mirrored for the run summary.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub prims_drawn: u64,
    pub prims_textured: u64,
    pub pixels_shaded: u64,
    pub prims_split: u64,
    pub tex_psm_hist: [u64; 64],
    pub frame: [u64; 2],
    pub zbuf: [u64; 2],
    pub test: [u64; 2],
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            prims_drawn: 0,
            prims_textured: 0,
            pixels_shaded: 0,
            prims_split: 0,
            tex_psm_hist: [0; 64],
            frame: [0; 2],
            zbuf: [0; 2],
            test: [0; 2],
        }
    }
}

impl Stats {
    fn of(gs: &Gs) -> Self {
        Self {
            prims_drawn: gs.prims_drawn,
            prims_textured: gs.prims_textured,
            pixels_shaded: gs.pixels_shaded,
            prims_split: gs.prims_split,
            tex_psm_hist: gs.tex_psm_hist,
            frame: [gs.ctx[0].frame, gs.ctx[1].frame],
            zbuf: [gs.ctx[0].zbuf, gs.ctx[1].zbuf],
            test: [gs.ctx[0].test, gs.ctx[1].test],
        }
    }
}

/// Renderer command.
enum Cmd {
    /// General register write (GIF/VIF/VU1 paths).
    Reg(u8, u64),
    /// A run of HWREG words (IMAGE transfer payload), kept as one command
    /// so the renderer decodes the transfer setup once per run instead of
    /// once per 64 bits.
    Image(Vec<u64>),
    /// Privileged display register (PMODE, DISPFB, ...), kept in order
    /// with the drawing that precedes it.
    Priv(u32, u64),
    /// Vertical blank with the field now displayed and the deinterlace
    /// mode: composite the display into the shared frame slot.
    Vblank(bool, super::Deinterlace),
    /// Reply with the current display (after draining).
    Frame(std::sync::mpsc::SyncSender<Frame>),
    /// Reply with a copy of VRAM.
    Vram(std::sync::mpsc::SyncSender<Box<[u8]>>),
    /// Reply with the statistics.
    Stats(std::sync::mpsc::SyncSender<Stats>),
    /// Turn the internal-2x overlay on or off.
    Internal2x(bool),
    /// Reply with the renderer's serialized state (after draining).
    Snapshot(std::sync::mpsc::SyncSender<Vec<u8>>),
    /// Replace the renderer with a state from [`Cmd::Snapshot`].
    Restore(Vec<u8>),
    /// Start over on a blank renderer (power cycle).
    Reset,
}

/// A save state's view of the display side: the EE-visible privileged
/// register copies plus the renderer's own state, which may have to make a
/// round trip through the worker thread to be read or written.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GsState {
    pmode: u64,
    smode1: u64,
    smode2: u64,
    dispfb1: u64,
    display1: u64,
    dispfb2: u64,
    display2: u64,
    bgcolor: u64,
    csr: u64,
    imr: u64,
    priv_shadow: [u64; 32],
    intc_pending: bool,
    internal_2x: bool,
    /// The renderer, serialized where it lives.
    pub(crate) renderer: Vec<u8>,
}

/// Flush a batch to the worker once it holds this many commands.
const BATCH_MAX: usize = 4096;
/// HWREG words per Image command at most (512 KiB), so a long upload
/// streams to the worker instead of landing all at once.
const IMAGE_RUN_MAX: usize = 65536;
/// Batches the worker may fall behind before the EE side blocks. A frame
/// of full-screen IMAGE uploads is ~150k commands, so this holds a few
/// frames and lets the EE run ahead through GS-heavy bursts.
const QUEUE_DEPTH: usize = 256;

#[cfg(feature = "threads")]
struct Worker {
    tx: std::sync::mpsc::SyncSender<Vec<Cmd>>,
    /// Empty batch vectors handed back for reuse.
    recycle: std::sync::mpsc::Receiver<Vec<Cmd>>,
    join: Option<std::thread::JoinHandle<()>>,
}

pub struct GsFront {
    /// Renderer, when it runs on the calling thread.
    inline: Option<Gs>,
    #[cfg(feature = "threads")]
    worker: Option<Worker>,
    batch: Vec<Cmd>,
    /// HWREG words accumulated since the last other command.
    image: Vec<u64>,
    // Privileged registers (EE-visible copies).
    pub pmode: u64,
    pub smode1: u64,
    pub smode2: u64,
    pub dispfb1: u64,
    pub display1: u64,
    pub dispfb2: u64,
    pub display2: u64,
    pub bgcolor: u64,
    pub csr: u64,
    pub imr: u64,
    priv_shadow: [u64; 32],
    /// Rising edge into the EE INTC GS line (bit 0).
    pub intc_pending: bool,
    /// Latest frame composited at a vblank (worker mode, when enabled).
    latest_frame: std::sync::Arc<std::sync::Mutex<Option<SharedFrame>>>,
    /// Composite at every vblank so [`GsFront::latest_frame`] stays fresh.
    publish_frames: bool,
    /// How the published frame treats interlaced field buffers.
    pub deinterlace: super::Deinterlace,
    /// Flip which rows each field lands on (see `vblank`).
    pub swap_fields: bool,
    /// EE-side copy of the internal-2x switch (commands are sent on change).
    internal_2x: bool,
}

impl Default for GsFront {
    fn default() -> Self {
        Self::new()
    }
}

impl GsFront {
    /// Renderer on a worker thread when the `threads` feature is on,
    /// inline otherwise.
    pub fn new() -> Self {
        #[cfg(feature = "threads")]
        {
            Self::threaded()
        }
        #[cfg(not(feature = "threads"))]
        {
            Self::inline()
        }
    }

    /// Whether the renderer runs on a worker thread.
    pub fn is_threaded(&self) -> bool {
        self.inline.is_none()
    }

    /// Renderer on the calling thread (wasm, tests).
    pub fn inline() -> Self {
        Self::build(Some(Gs::new()))
    }

    #[cfg(feature = "threads")]
    pub fn threaded() -> Self {
        let mut front = Self::build(None);
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<Cmd>>(QUEUE_DEPTH);
        let (recycle_tx, recycle_rx) = std::sync::mpsc::channel::<Vec<Cmd>>();
        let latest = front.latest_frame.clone();
        let join = std::thread::Builder::new()
            .name("gs".into())
            .spawn(move || {
                let mut gs = Gs::new();
                loop {
                    let mut batch = {
                        let _p = crate::prof::scope(crate::prof::Slot::GsIdle);
                        match rx.recv() {
                            Ok(b) => b,
                            Err(_) => return, // front dropped
                        }
                    };
                    for cmd in batch.drain(..) {
                        Self::run_cmd(&mut gs, cmd, &latest);
                    }
                    let _ = recycle_tx.send(batch);
                }
            })
            .expect("failed to spawn GS thread");
        front.worker = Some(Worker { tx, recycle: recycle_rx, join: Some(join) });
        front
    }

    fn build(inline: Option<Gs>) -> Self {
        Self {
            inline,
            #[cfg(feature = "threads")]
            worker: None,
            batch: Vec::with_capacity(BATCH_MAX),
            image: Vec::new(),
            pmode: 0,
            smode1: 0,
            smode2: 0,
            dispfb1: 0,
            display1: 0,
            dispfb2: 0,
            display2: 0,
            bgcolor: 0,
            csr: 0,
            imr: 0xFF00, // all sources masked at reset
            priv_shadow: [0; 32],
            intc_pending: false,
            latest_frame: Default::default(),
            publish_frames: false,
            deinterlace: super::Deinterlace::default(),
            swap_fields: false,
            internal_2x: false,
        }
    }

    /// Execute one command against the renderer (either side).
    fn run_cmd(gs: &mut Gs, cmd: Cmd, latest: &std::sync::Mutex<Option<SharedFrame>>) {
        match cmd {
            Cmd::Reg(reg, v) => gs.write_reg(reg, v),
            Cmd::Image(data) => gs.image(&data),
            Cmd::Priv(addr, v) => gs.priv_write(addr, v),
            Cmd::Vblank(field, mode) => {
                let (w, h, rgba) = gs.framebuffer_woven(field, mode);
                *latest.lock().unwrap() = Some((w, h, std::sync::Arc::new(rgba)));
            }
            Cmd::Frame(reply) => {
                let _ = reply.send(gs.framebuffer());
            }
            Cmd::Vram(reply) => {
                let _ = reply.send(gs.vram_snapshot());
            }
            Cmd::Stats(reply) => {
                let _ = reply.send(Stats::of(gs));
            }
            Cmd::Internal2x(on) => gs.set_internal_2x(on),
            Cmd::Snapshot(reply) => {
                let _ = reply.send(postcard::to_allocvec(&*gs).unwrap_or_default());
            }
            Cmd::Restore(data) => match postcard::from_bytes(&data) {
                Ok(new) => *gs = new,
                Err(e) => tracing::error!("renderer state load failed: {e}"),
            },
            Cmd::Reset => *gs = Gs::new(),
        }
    }

    #[inline]
    fn push(&mut self, cmd: Cmd) {
        if let Some(gs) = &mut self.inline {
            Self::run_cmd(gs, cmd, &self.latest_frame);
            return;
        }
        self.flush_image();
        self.batch.push(cmd);
        if self.batch.len() >= BATCH_MAX {
            self.flush();
        }
    }

    /// Queue the accumulated HWREG words as one command.
    #[inline]
    fn flush_image(&mut self) {
        if !self.image.is_empty() {
            let data = std::mem::take(&mut self.image);
            self.batch.push(Cmd::Image(data));
        }
    }

    /// Hand the pending batch to the worker (no-op inline).
    pub fn flush(&mut self) {
        self.flush_image();
        #[cfg(feature = "threads")]
        if let Some(w) = &mut self.worker
            && !self.batch.is_empty()
        {
            let mut next = w.recycle.try_recv().unwrap_or_default();
            next.clear();
            next.reserve(BATCH_MAX);
            let batch = std::mem::replace(&mut self.batch, next);
            let _ = w.tx.send(batch);
        }
    }

    /// Composite the display at every vblank (for a live front-end).
    /// Render internally at 2x and scan out the overlay (see
    /// [`super::Gs::set_internal_2x`]).
    pub fn set_internal_2x(&mut self, on: bool) {
        if self.internal_2x == on {
            return;
        }
        self.internal_2x = on;
        self.push(Cmd::Internal2x(on));
    }

    pub fn set_publish_frames(&mut self, on: bool) {
        self.publish_frames = on;
    }

    /// Newest vblank-composited frame, if publishing is on and one exists.
    pub fn latest_frame(&self) -> Option<Frame> {
        self.latest_frame
            .lock()
            .unwrap()
            .as_ref()
            .map(|(w, h, rgba)| (*w, *h, rgba.as_ref().clone()))
    }

    /// The same frame without copying its pixels.
    pub fn latest_frame_shared(&self) -> Option<SharedFrame> {
        self.latest_frame.lock().unwrap().clone()
    }

    // --- general registers (from the GIF) ---------------------------------

    /// General register write. SIGNAL/FINISH/LABEL are resolved here so the
    /// EE sees them at a deterministic point; drawing registers stream on.
    #[inline]
    pub fn write_reg(&mut self, reg: u8, v: u64) {
        match reg {
            0x60 => self.raise_int(0), // SIGNAL
            0x61 => self.raise_int(1), // FINISH
            0x62 => {}                 // LABEL
            0x54 if self.inline.is_none() => {
                // HWREG: collect the run (see Cmd::Image).
                self.image.push(v);
                if self.image.len() >= IMAGE_RUN_MAX {
                    self.flush_image();
                }
            }
            _ => self.push(Cmd::Reg(reg, v)),
        }
    }

    // --- privileged registers -------------------------------------------

    pub fn priv_write(&mut self, addr: u32, v: u64) {
        trace!(target: "ps2_core::gs", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#018x}"), "priv write");
        // CRTC timing group: SMODE1 plus SRFSH/SYNCH1/SYNCH2/SYNCHV. These
        // carry the NTSC/PAL distinction but drive nothing here, so a log
        // target is the only way to check what a kernel programmed.
        if matches!(addr & 0x1FF0, 0x0010 | 0x0030..=0x0060) {
            debug!(target: "ps2_core::gs::crtc",
                addr = format_args!("{:#06x}", addr & 0x1FF0), value = format_args!("{v:#018x}"), "crtc reg");
        }
        match addr & 0x1FF0 {
            0x0000 => self.pmode = v,
            0x0010 => self.smode1 = v,
            0x0020 => self.smode2 = v,
            0x0070 => self.dispfb1 = v,
            0x0080 => self.display1 = v,
            0x0090 => self.dispfb2 = v,
            0x00A0 => self.display2 = v,
            0x00E0 => self.bgcolor = v,
            0x1000 => {
                // Interrupt flags are write-1-to-clear; bit 9 is reset.
                self.csr &= !(v & 0x1F);
                if v & 0x200 != 0 {
                    self.csr = 0;
                }
                return;
            }
            0x1010 => {
                self.imr = v;
                return;
            }
            _ => {
                self.priv_shadow[((addr >> 4) & 31) as usize] = v;
                return;
            }
        }
        // Display registers also reach the renderer, in order with drawing.
        self.push(Cmd::Priv(addr, v));
    }

    pub fn priv_read(&mut self, addr: u32) -> u64 {
        match addr & 0x1FF0 {
            0x0000 => self.pmode,
            0x0010 => self.smode1,
            0x0020 => self.smode2,
            0x0070 => self.dispfb1,
            0x0080 => self.display1,
            0x0090 => self.dispfb2,
            0x00A0 => self.display2,
            0x00E0 => self.bgcolor,
            // CSR: flags + FIFO empty + revision/id. REV 0x1B is what
            // retail hardware and PCSX2 both report; kernel paths gated on
            // REV == 1 are meant to stay unreachable.
            0x1000 => self.csr | 0x4000 | (0x1B << 16) | (0x55 << 24),
            0x1010 => self.imr,
            _ => self.priv_shadow[((addr >> 4) & 31) as usize],
        }
    }

    fn raise_int(&mut self, bit: u32) {
        let was = self.csr & (1 << bit) != 0;
        self.csr |= 1 << bit;
        // IMR masks sit 8 bits above the CSR flags; 1 = masked.
        if !was && self.imr & (1 << (bit + 8)) == 0 {
            self.intc_pending = true;
        }
    }

    /// Vertical sync: toggles FIELD, latches VSINT, and lets the renderer
    /// composite the frame the display shows now.
    pub fn vblank(&mut self) {
        self.csr ^= 1 << 13;
        self.raise_int(3);
        if self.publish_frames {
            // CSR FIELD=1 holds the even rows (measured on SLPS-25918: the
            // field arriving with FIELD set sits between the other field's
            // lines k-1 and k); `swap_fields` flips that.
            let odd_rows = (self.csr & (1 << 13) == 0) != self.swap_fields;
            self.push(Cmd::Vblank(odd_rows, self.deinterlace));
        }
        self.flush();
    }

    // --- synchronous reads (drain first) ----------------------------------

    /// Current display as RGBA8, as of everything written so far.
    pub fn framebuffer(&mut self) -> Frame {
        if let Some(gs) = &mut self.inline {
            return gs.framebuffer();
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.push(Cmd::Frame(tx));
        self.flush();
        rx.recv().unwrap_or((0, 0, Vec::new()))
    }

    /// Copy of VRAM, as of everything written so far.
    pub fn vram(&mut self) -> Box<[u8]> {
        if let Some(gs) = &mut self.inline {
            return gs.vram_snapshot();
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.push(Cmd::Vram(tx));
        self.flush();
        rx.recv().unwrap_or_else(|_| vec![0u8; super::VRAM_SIZE].into_boxed_slice())
    }

    /// Renderer statistics, as of everything written so far.
    pub fn stats(&mut self) -> Stats {
        if let Some(gs) = &self.inline {
            return Stats::of(gs);
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.push(Cmd::Stats(tx));
        self.flush();
        rx.recv().unwrap_or_default()
    }

    /// Direct access to an inline renderer (tests).
    pub fn inline_gs(&mut self) -> Option<&mut Gs> {
        self.inline.as_mut()
    }

    /// Everything a save state needs from the display side, as of every
    /// command written so far.
    pub fn snapshot(&mut self) -> Result<GsState, String> {
        let renderer = match &mut self.inline {
            Some(gs) => postcard::to_allocvec(&*gs).map_err(|e| e.to_string())?,
            None => {
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                self.push(Cmd::Snapshot(tx));
                self.flush();
                rx.recv().map_err(|_| "renderer stopped".to_string())?
            }
        };
        Ok(GsState {
            pmode: self.pmode,
            smode1: self.smode1,
            smode2: self.smode2,
            dispfb1: self.dispfb1,
            display1: self.display1,
            dispfb2: self.dispfb2,
            display2: self.display2,
            bgcolor: self.bgcolor,
            csr: self.csr,
            imr: self.imr,
            priv_shadow: self.priv_shadow,
            intc_pending: self.intc_pending,
            internal_2x: self.internal_2x,
            renderer,
        })
    }

    /// Blank the renderer for a power cycle. VRAM and the privileged
    /// registers are machine state and go; the renderer itself — the
    /// worker thread, and the frame slot the frontend reads through — and
    /// the display switches belong to the host and stay.
    pub fn reset(&mut self) {
        self.batch.clear();
        self.image.clear();
        self.pmode = 0;
        self.smode1 = 0;
        self.smode2 = 0;
        self.dispfb1 = 0;
        self.display1 = 0;
        self.dispfb2 = 0;
        self.display2 = 0;
        self.bgcolor = 0;
        self.csr = 0;
        self.imr = 0xFF00; // all sources masked at reset
        self.priv_shadow = [0; 32];
        self.intc_pending = false;
        match &mut self.inline {
            Some(gs) => *gs = Gs::new(),
            _ => {
                self.push(Cmd::Reset);
                self.flush();
            }
        }
        // A fresh renderer draws at 1x whatever the front last asked for,
        // so re-send the switch from a cleared copy of it.
        let internal_2x = std::mem::take(&mut self.internal_2x);
        self.set_internal_2x(internal_2x);
    }

    /// Put back a [`GsFront::snapshot`], keeping the renderer where it is.
    /// Everything that can fail happens first, so a rejected blob leaves
    /// the front exactly as it was.
    pub fn restore(&mut self, state: GsState) -> Result<(), String> {
        let decoded = match &self.inline {
            Some(_) => Some(
                postcard::from_bytes::<Gs>(&state.renderer).map_err(|e| e.to_string())?,
            ),
            None => None,
        };
        self.batch.clear();
        self.image.clear();
        self.pmode = state.pmode;
        self.smode1 = state.smode1;
        self.smode2 = state.smode2;
        self.dispfb1 = state.dispfb1;
        self.display1 = state.display1;
        self.dispfb2 = state.dispfb2;
        self.display2 = state.display2;
        self.bgcolor = state.bgcolor;
        self.csr = state.csr;
        self.imr = state.imr;
        self.priv_shadow = state.priv_shadow;
        self.intc_pending = state.intc_pending;
        self.internal_2x = state.internal_2x;
        match (&mut self.inline, decoded) {
            (Some(gs), Some(new)) => *gs = new,
            _ => {
                self.push(Cmd::Restore(state.renderer));
                self.flush();
            }
        }
        Ok(())
    }
}

#[cfg(feature = "threads")]
impl Drop for GsFront {
    fn drop(&mut self) {
        if let Some(mut w) = self.worker.take() {
            drop(w.tx); // closes the channel; the worker returns
            if let Some(j) = w.join.take() {
                let _ = j.join();
            }
        }
    }
}
