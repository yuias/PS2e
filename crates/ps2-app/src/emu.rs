//! Emulator worker thread.
//!
//! Owns the [`Ps2System`] and the audio output, paces emulation against the
//! audio buffer (wall clock when no device exists) and publishes read-only
//! snapshots for the UI. The UI never touches the system directly — it sends
//! [`Command`]s and reads [`Shared`] — so a slow repaint can no longer starve
//! the audio thread, and the frontend stays thin enough to port later (e.g.
//! a wasm build driving the same snapshots single-threaded).

use crate::audio::Audio;
use ps2_core::{EE_CLOCK_HZ, Ps2System};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

/// Emulation slice: 5ms of machine time per pacer iteration.
const SLICE: u64 = EE_CLOCK_HZ / 200;
/// Audio cushion the pacer keeps buffered (frames; ~80ms at 48kHz). Doubles
/// as the output latency, and absorbs host-side load spikes of the same
/// length.
const AUDIO_TARGET: usize = 3_840;
/// How often the composited framebuffer is rebuilt and republished. GS
/// compositing walks the whole display area, so this is decoupled from the
/// (much faster) pacer slice rate rather than done every iteration.
const FRAME_INTERVAL: Duration = Duration::from_millis(16);
/// How often the wall-clock speed estimate (EE cycles / real time) refreshes.
const SPEED_WINDOW: Duration = Duration::from_millis(500);
/// Cap on the accumulated TTY text kept for the UI panel (bytes).
const TTY_CAP: usize = 64 * 1024;

pub enum Command {
    SetRunning(bool),
    Step,
    Reset,
    Quit,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum DebuggerState {
    /// No --debug-ee / --debug-iop.
    #[default]
    None,
    Listening,
    /// --wait-debugger holds execution until the first attach.
    Waiting,
    Running,
    Halted,
}

/// Cheap per-slice snapshot for the UI.
#[derive(Clone, Default)]
pub struct Status {
    pub cycles: u64,
    pub running: bool,
    pub debugger: DebuggerState,
    /// EE cycles retired per wall-clock second, as a multiple of real time
    /// (1.0 == full speed). Updated every [`SPEED_WINDOW`].
    pub speed: f64,
    /// Stereo frames queued at the audio device.
    pub audio_buffered: usize,
    /// Callbacks that ran out of samples (audible as crackle).
    pub audio_underruns: u64,
    /// EE core: pc, 128-bit GPRs (low half first), HI/LO and COP0 regs.
    pub ee_pc: u32,
    pub ee_gpr: [[u64; 2]; 32],
    pub ee_hi: [u64; 2],
    pub ee_lo: [u64; 2],
    pub ee_cop0: [u32; 32],
    /// IOP core: pc, GPRs, HI/LO and COP0 regs.
    pub iop_pc: u32,
    pub iop_gpr: [u32; 32],
    pub iop_hi: u32,
    pub iop_lo: u32,
    pub iop_cop0: [u32; 32],
}

/// Latest composited display frame (see [`Ps2System::framebuffer`]).
#[derive(Default)]
pub struct FrameSnapshot {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// State published by the worker and inputs fed back by the UI.
#[derive(Default)]
pub struct Shared {
    pub frame: Mutex<FrameSnapshot>,
    pub status: Mutex<Status>,
    /// Accumulated kernel/game TTY text, capped to [`TTY_CAP`]. Cleared
    /// directly by the UI (no round-trip through the worker needed).
    pub tty: Mutex<String>,
    /// Digital pad bits (UI -> worker), SIO2 bit order (see [`crate::pad`]).
    pub buttons: AtomicU16,
    /// Master volume as f32 bits (UI -> worker).
    pub volume: AtomicU32,
    /// Debugger attached/halted (set by the worker) drives UI enablement.
    pub debugger_active: AtomicBool,
}

/// Everything the worker owns besides the system itself.
pub struct WorkerConfig {
    pub bios: Vec<u8>,
    /// None disables persistence (headless-style, no card mounted).
    pub memcard_path: Option<PathBuf>,
    pub debugger: Option<ps2_debug::DebugServer>,
    pub wait_debugger: bool,
    pub volume: f32,
}

pub struct Emu {
    pub shared: Arc<Shared>,
    tx: mpsc::Sender<Command>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Emu {
    pub fn send(&self, cmd: Command) {
        let _ = self.tx.send(cmd);
    }
}

impl Drop for Emu {
    /// Stop the worker; it flushes the memory card before exiting.
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Quit);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

pub fn spawn(sys: Ps2System, cfg: WorkerConfig, ctx: eframe::egui::Context) -> Emu {
    let shared = Arc::new(Shared::default());
    shared.volume.store(cfg.volume.to_bits(), Ordering::Relaxed);
    let (tx, rx) = mpsc::channel();
    let sh = shared.clone();
    let join = std::thread::Builder::new()
        .name("emu".into())
        .spawn(move || Worker::new(sys, cfg, sh, rx, ctx).run())
        .expect("failed to spawn emulator thread");
    Emu {
        shared,
        tx,
        join: Some(join),
    }
}

struct Worker {
    sys: Ps2System,
    cfg: WorkerConfig,
    shared: Arc<Shared>,
    rx: mpsc::Receiver<Command>,
    ctx: eframe::egui::Context,
    /// Created on this thread: cpal streams are not Send everywhere.
    audio: Option<Audio>,
    running: bool,
    debugger_seen: bool,
    /// Wall-clock pacer (only used when no audio device exists).
    clock: Instant,
    deficit: f64,
    last_frame_publish: Instant,
    speed_window_start: Instant,
    speed_window_cycles: u64,
}

impl Worker {
    fn new(
        sys: Ps2System,
        cfg: WorkerConfig,
        shared: Arc<Shared>,
        rx: mpsc::Receiver<Command>,
        ctx: eframe::egui::Context,
    ) -> Self {
        let now = Instant::now();
        Self {
            sys,
            cfg,
            shared,
            rx,
            ctx,
            audio: None,
            // Start running: the window exists to play, and a debugger that
            // wants the reset vector uses --wait-debugger.
            running: true,
            debugger_seen: false,
            clock: now,
            deficit: 0.0,
            last_frame_publish: now,
            speed_window_start: now,
            speed_window_cycles: 0,
        }
    }

    fn debugger_active(&self) -> bool {
        self.cfg.debugger.as_ref().is_some_and(|d| d.attached())
            || (self.cfg.wait_debugger && !self.debugger_seen)
    }

    fn run(mut self) {
        self.audio = Audio::new();
        // Frames come from the GS worker's vblank composite; asking the
        // renderer directly would stall emulation until it caught up.
        self.sys.set_publish_frames(true);
        loop {
            if !self.handle_commands() {
                break;
            }
            self.sys.bus.sio2.buttons = self.shared.buttons.load(Ordering::Relaxed);

            // While a debugger is attached (or awaited) it owns execution.
            let mut worked = false;
            if let Some(dbg) = &mut self.cfg.debugger {
                dbg.pump(&mut self.sys, SLICE);
                self.debugger_seen |= dbg.attached();
                worked = dbg.attached() && !dbg.halted();
            }

            if !self.debugger_active() && self.running {
                worked |= self.pace_slice();
            }

            self.push_audio();
            self.publish();
            self.flush_memcard();
            self.flush_tty();

            if !worked {
                // Paused / halted / buffer full: 2ms is well inside the
                // ~80ms audio cushion (48 frames drain per ms)
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        self.flush_memcard();
    }

    /// Returns false when Quit was received.
    fn handle_commands(&mut self) -> bool {
        while let Ok(cmd) = self.rx.try_recv() {
            let debugger_active = self.debugger_active();
            match cmd {
                Command::SetRunning(r) if !debugger_active => self.running = r,
                Command::Step if !debugger_active => {
                    self.running = false;
                    self.sys.step();
                }
                Command::Reset if !debugger_active => {
                    self.running = false;
                    // Ambient assets survive a reset: disc and memory card
                    // (mid-write contents included).
                    let disc = self.sys.bus.cdvd.disc.take();
                    let memcard = std::mem::take(&mut self.sys.bus.sio2.memcard);
                    self.sys = Ps2System::new(self.cfg.bios.clone()).expect("reset failed");
                    self.sys.set_publish_frames(true);
                    self.sys.bus.cdvd.disc = disc;
                    self.sys.bus.sio2.memcard = memcard;
                }
                Command::SetRunning(_) | Command::Step | Command::Reset => {}
                Command::Quit => return false,
            }
        }
        true
    }

    /// Run one slice if the pacer allows it. With an audio device the SPU2's
    /// cycle-locked 48kHz output is the clock: run whenever the buffer is
    /// below target, which also gives full-host-speed catch-up after a load
    /// spike (this core interprets well under real time, so in practice the
    /// buffer rarely reaches target and the pacer just runs flat out).
    /// Without a device, pace against the wall clock instead.
    fn pace_slice(&mut self) -> bool {
        match &self.audio {
            Some(audio) => {
                if audio.buffered_frames() < AUDIO_TARGET {
                    self.sys.run(SLICE);
                    true
                } else {
                    false
                }
            }
            None => {
                let dt = std::mem::replace(&mut self.clock, Instant::now()).elapsed();
                self.deficit += dt.as_secs_f64() * EE_CLOCK_HZ as f64;
                // Cap the backlog so a long stall doesn't fast-forward
                self.deficit = self.deficit.min(3.0 * SLICE as f64);
                if self.deficit >= SLICE as f64 {
                    self.deficit -= SLICE as f64;
                    self.sys.run(SLICE);
                    true
                } else {
                    false
                }
            }
        }
    }

    fn push_audio(&mut self) {
        let mut samples = self.sys.bus.spu2.take_output();
        if let Some(audio) = &self.audio {
            let vol = f32::from_bits(self.shared.volume.load(Ordering::Relaxed));
            for s in &mut samples {
                *s = (*s as f32 * vol) as i16;
            }
            audio.push_samples(&samples);
        }
    }

    fn publish(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_frame_publish) >= FRAME_INTERVAL {
            self.last_frame_publish = now;
            if let Some((w, h, rgba)) = self.sys.latest_frame()
                && w > 0
                && h > 0
            {
                let mut f = self.shared.frame.lock().unwrap();
                f.width = w;
                f.height = h;
                f.rgba = rgba;
            }
            self.ctx.request_repaint();
        }

        let elapsed = now.duration_since(self.speed_window_start);
        let mut speed = None;
        if elapsed >= SPEED_WINDOW {
            let delta = self.sys.cycles.saturating_sub(self.speed_window_cycles);
            speed = Some(delta as f64 / (EE_CLOCK_HZ as f64 * elapsed.as_secs_f64()));
            self.speed_window_cycles = self.sys.cycles;
            self.speed_window_start = now;
        }

        {
            let mut st = self.shared.status.lock().unwrap();
            st.cycles = self.sys.cycles;
            st.running = self.running;
            st.debugger = match &self.cfg.debugger {
                None => DebuggerState::None,
                Some(d) if d.attached() && d.halted() => DebuggerState::Halted,
                Some(d) if d.attached() => DebuggerState::Running,
                Some(_) if self.cfg.wait_debugger && !self.debugger_seen => DebuggerState::Waiting,
                Some(_) => DebuggerState::Listening,
            };
            if let Some(speed) = speed {
                st.speed = speed;
            }
            if let Some(audio) = &self.audio {
                st.audio_buffered = audio.buffered_frames();
                st.audio_underruns = audio.underruns();
            }
            st.ee_pc = self.sys.ee.pc;
            st.ee_gpr = self.sys.ee.gpr;
            st.ee_hi = self.sys.ee.hi;
            st.ee_lo = self.sys.ee.lo;
            st.ee_cop0 = self.sys.ee.cop0.regs;
            st.iop_pc = self.sys.iop.pc;
            st.iop_gpr = self.sys.iop.gpr;
            st.iop_hi = self.sys.iop.hi;
            st.iop_lo = self.sys.iop.lo;
            st.iop_cop0 = self.sys.iop.cop0;
        }
        self.shared
            .debugger_active
            .store(self.debugger_active(), Ordering::Relaxed);
    }

    /// Stream kernel TTY output to stdout (same as the headless path) and
    /// accumulate it for the UI panel, capped to [`TTY_CAP`].
    fn flush_tty(&mut self) {
        let tty = self.sys.take_tty();
        if tty.is_empty() {
            return;
        }
        crate::print_tty(&tty);
        let mut buf = self.shared.tty.lock().unwrap();
        buf.push_str(&tty);
        if buf.len() > TTY_CAP {
            let excess = buf.len() - TTY_CAP;
            // Drain to the next char boundary so multi-byte UTF-8 isn't split.
            let cut = (excess..buf.len())
                .find(|&i| buf.is_char_boundary(i))
                .unwrap_or(buf.len());
            buf.drain(..cut);
        }
    }

    fn flush_memcard(&mut self) {
        let Some(path) = &self.cfg.memcard_path else {
            return;
        };
        if self.sys.bus.sio2.memcard.dirty {
            match std::fs::write(path, &self.sys.bus.sio2.memcard.data) {
                Ok(()) => {
                    self.sys.bus.sio2.memcard.dirty = false;
                    tracing::info!("memory card saved");
                }
                Err(e) => tracing::error!("failed to save memory card: {e}"),
            }
        }
    }
}
