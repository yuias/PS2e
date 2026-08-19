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
use tracing::trace;

/// One composited display frame: width, height, RGBA8.
pub type Frame = (u32, u32, Vec<u8>);

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
    /// Vertical blank with the field now displayed: weave the display into
    /// the shared frame slot.
    Vblank(bool),
    /// Reply with the current display (after draining).
    Frame(std::sync::mpsc::SyncSender<Frame>),
    /// Reply with a copy of VRAM.
    Vram(std::sync::mpsc::SyncSender<Box<[u8]>>),
    /// Reply with the statistics.
    Stats(std::sync::mpsc::SyncSender<Stats>),
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
    latest_frame: std::sync::Arc<std::sync::Mutex<Option<Frame>>>,
    /// Composite at every vblank so [`GsFront::latest_frame`] stays fresh.
    publish_frames: bool,
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
        }
    }

    /// Execute one command against the renderer (either side).
    fn run_cmd(gs: &mut Gs, cmd: Cmd, latest: &std::sync::Mutex<Option<Frame>>) {
        match cmd {
            Cmd::Reg(reg, v) => gs.write_reg(reg, v),
            Cmd::Image(data) => gs.image(&data),
            Cmd::Priv(addr, v) => gs.priv_write(addr, v),
            Cmd::Vblank(field) => {
                let frame = gs.framebuffer_woven(field);
                *latest.lock().unwrap() = Some(frame);
            }
            Cmd::Frame(reply) => {
                let _ = reply.send(gs.framebuffer());
            }
            Cmd::Vram(reply) => {
                let _ = reply.send(gs.canvas.to_vec());
            }
            Cmd::Stats(reply) => {
                let _ = reply.send(Stats::of(gs));
            }
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
    pub fn set_publish_frames(&mut self, on: bool) {
        self.publish_frames = on;
    }

    /// Newest vblank-composited frame, if publishing is on and one exists.
    pub fn latest_frame(&self) -> Option<Frame> {
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
            0x0020 => self.smode2,
            0x0070 => self.dispfb1,
            0x0080 => self.display1,
            0x0090 => self.dispfb2,
            0x00A0 => self.display2,
            0x00E0 => self.bgcolor,
            // CSR: flags + FIFO empty + revision/id.
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
            self.push(Cmd::Vblank(self.csr & (1 << 13) != 0));
        }
        self.flush();
    }

    // --- synchronous reads (drain first) ----------------------------------

    /// Current display as RGBA8, as of everything written so far.
    pub fn framebuffer(&mut self) -> Frame {
        if let Some(gs) = &self.inline {
            return gs.framebuffer();
        }
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        self.push(Cmd::Frame(tx));
        self.flush();
        rx.recv().unwrap_or((0, 0, Vec::new()))
    }

    /// Copy of VRAM, as of everything written so far.
    pub fn vram(&mut self) -> Box<[u8]> {
        if let Some(gs) = &self.inline {
            return gs.canvas.to_vec();
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
