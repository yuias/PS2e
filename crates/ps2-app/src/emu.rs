//! Emulator worker thread.
//!
//! Owns the [`Ps2System`] and the audio output, paces emulation against the
//! audio buffer (wall clock when no device exists) and publishes read-only
//! snapshots for the UI. The UI never touches the system directly — it sends
//! [`Command`]s and reads [`Shared`] — so a slow repaint can no longer starve
//! the audio thread, and the frontend stays thin enough to port later (e.g.
//! a wasm build driving the same snapshots single-threaded).

use crate::audio::Audio;
use crate::scan;
use ps2_core::cheats::Group;
use ps2_core::{EE_CLOCK_HZ, Ps2System, Region};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

/// Emulation slice: 5ms of machine time per pacer iteration.
const SLICE: u64 = EE_CLOCK_HZ / 1000;
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

/// A disc image handed to the worker: the open file plus the name to show
/// for it. `File` carries no path of its own, so the frontend passes one.
pub struct Disc {
    pub file: std::fs::File,
    pub name: String,
    /// Cheats from the pnach file next to the image, if there is one,
    /// already carrying whatever `cheats.toml` had switched off.
    pub cheats: Vec<Group>,
}

impl Disc {
    /// Open an image, labelling it with its file name. Cheats are the
    /// caller's to supply: reading them needs `cheats.toml`, which the UI
    /// owns.
    pub fn open(path: &std::path::Path, cheats: Vec<Group>) -> std::io::Result<Self> {
        Ok(Self { file: std::fs::File::open(path)?, name: disc_name(path), cheats })
    }
}

/// The label shown for an image: its file name, falling back to the whole
/// path for the odd case of one that has none.
pub fn disc_name(path: &std::path::Path) -> String {
    path.file_name().unwrap_or(path.as_os_str()).to_string_lossy().into_owned()
}

/// What the UI shows about the disc currently in the drive.
#[derive(Clone, Default)]
pub struct DiscInfo {
    /// File name of the image, e.g. `SLPS-25418.iso`.
    pub name: String,
    /// Boot serial read off the disc, e.g. `SLPS-25418`.
    pub serial: Option<String>,
}

pub enum Command {
    SetRunning(bool),
    Step,
    Reset,
    /// Open the drive. Whatever was in it is held until the tray closes.
    OpenTray,
    /// Close the drive on a disc; `None` puts back the one that came out,
    /// so a cancelled pick changes nothing.
    CloseTray(Option<Disc>),
    /// Put a disc in the drive and power-cycle onto it.
    BootDisc(Option<Disc>),
    /// Write the machine to [`WorkerConfig::state_path`].
    SaveState,
    /// Restore it from there.
    LoadState,
    /// One pass of the memory scanner; the result lands in
    /// [`Shared::scan`].
    Scan(scan::Request),
    /// Install a cheat table, replacing the disc's. Rebuilding re-arms
    /// every one-shot command, so this is for a reload or a new file --
    /// [`Command::SetCheatEnabled`] is the way to flip one cheat.
    SetCheats(Vec<Group>),
    /// Switch one named cheat on or off, leaving the rest of the table
    /// as it stands.
    SetCheatEnabled(String, bool),
    Quit,
}

/// Debug panels the UI has open, as bits in [`Shared::panels`]. The
/// worker does the work behind a panel only while its bit is set, so a
/// closed panel costs nothing.
pub const PANEL_REGS: u8 = 1;
pub const PANEL_MEMORY: u8 = 2;

/// Bytes the memory viewer shows at once.
pub const VIEW_BYTES: usize = 256;

/// One window of memory for the viewer, refreshed with the framebuffer
/// while the panel is open.
#[derive(Clone, Default)]
pub struct MemoryView {
    pub target: Option<scan::Target>,
    pub base: u32,
    pub bytes: Vec<u8>,
}

/// Pack the viewer's request into one atomic word.
pub fn pack_view(target: scan::Target, base: u32) -> u64 {
    u64::from(base) | u64::from(target == scan::Target::Iop) << 32
}

pub fn unpack_view(word: u64) -> (scan::Target, u32) {
    (if word >> 32 != 0 { scan::Target::Iop } else { scan::Target::Ee }, word as u32)
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
    /// Video timing the machine is running at now (software can change it).
    pub region: Region,
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
    pub rgba: std::sync::Arc<Vec<u8>>,
    pub width: u32,
    pub height: u32,
    /// Bumped on every new frame so the display can skip re-uploads.
    pub seq: u64,
}

/// State published by the worker and inputs fed back by the UI.
#[derive(Default)]
pub struct Shared {
    pub frame: Mutex<FrameSnapshot>,
    pub status: Mutex<Status>,
    /// Accumulated kernel/game TTY text, capped to [`TTY_CAP`]. Cleared
    /// directly by the UI (no round-trip through the worker needed).
    pub tty: Mutex<String>,
    /// Pad state (UI -> worker) as one word, so buttons and sticks land in
    /// the same slice: see [`pack_pad`].
    pub pad: AtomicU64,
    /// Master volume as f32 bits (UI -> worker).
    pub volume: AtomicU32,
    /// Deinterlace mode index (UI -> worker), see [`deinterlace_mode`].
    pub deinterlace: std::sync::atomic::AtomicU8,
    /// Flip which rows each field lands on (UI -> worker).
    pub swap_fields: AtomicBool,
    /// Whether the disc's cheats are applied (UI -> worker).
    pub cheats: AtomicBool,
    /// Open debug panels (UI -> worker), `PANEL_*` bits.
    pub panels: std::sync::atomic::AtomicU8,
    /// The window the memory viewer wants (UI -> worker), see [`pack_view`].
    pub view: AtomicU64,
    /// That window's bytes (worker -> UI).
    pub memory: Mutex<MemoryView>,
    /// The last scanner pass (worker -> UI).
    pub scan: Mutex<scan::Result>,
    /// Render internally at 2x (UI -> worker).
    pub internal_2x: AtomicBool,
    /// Debugger attached/halted (set by the worker) drives UI enablement.
    pub debugger_active: AtomicBool,
    /// Last one-shot result worth showing in the status bar, and whether
    /// it was a failure.
    pub notice: Mutex<Option<(String, bool)>>,
    /// Disc in the drive, `None` while it is empty or the tray is open.
    pub disc: Mutex<Option<DiscInfo>>,
}

/// Deinterlace modes in UI/config order, indexed by `Shared::deinterlace`.
pub const DEINTERLACE_MODES: [ps2_core::gs::Deinterlace; 7] = [
    ps2_core::gs::Deinterlace::Weave,
    ps2_core::gs::Deinterlace::Bob,
    ps2_core::gs::Deinterlace::Blend,
    ps2_core::gs::Deinterlace::Adaptive,
    ps2_core::gs::Deinterlace::AdaptiveDebug,
    ps2_core::gs::Deinterlace::Yadif,
    ps2_core::gs::Deinterlace::Bwdif,
];

/// Stick bytes at rest: 0x7F on every axis.
pub const STICKS_CENTRED: [u8; 4] = [0x7F; 4];

/// The pad word: button bits in the low half-word (SIO2 order), then the
/// four stick bytes in the order the pad reports them (rx, ry, lx, ly).
pub fn pack_pad(buttons: u16, sticks: [u8; 4]) -> u64 {
    sticks.iter().enumerate().fold(u64::from(buttons), |w, (i, &b)| w | u64::from(b) << (16 + 8 * i))
}

pub fn unpack_pad(word: u64) -> (u16, [u8; 4]) {
    (word as u16, std::array::from_fn(|i| (word >> (16 + 8 * i)) as u8))
}

pub fn deinterlace_mode(index: u8) -> ps2_core::gs::Deinterlace {
    DEINTERLACE_MODES.get(index as usize).copied().unwrap_or_default()
}

/// Everything the worker owns besides the system itself.
pub struct WorkerConfig {
    /// None disables persistence (headless-style, no card mounted).
    pub memcard_path: Option<PathBuf>,
    /// Where the window's save state lives.
    pub state_path: PathBuf,
    /// Name of the image already in `sys`'s drive (from `--disc`), if any,
    /// and the cheats that came with it.
    pub disc_name: Option<String>,
    pub cheats: Vec<Group>,
    pub debugger: Option<ps2_debug::DebugServer>,
    pub wait_debugger: bool,
    /// Video timing region the machine starts in; re-applied on reset.
    pub region: Region,
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
    let shared = Arc::new(Shared {
        pad: AtomicU64::new(pack_pad(0, STICKS_CENTRED)),
        ..Default::default()
    });
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
    /// Disc taken out while the drive is open, put back if the pick that
    /// opened it is cancelled.
    removed: Option<Disc>,
    /// The cheat table of the disc in the drive, kept across power cycles.
    cheats: Vec<Group>,
    /// Memory scan in progress.
    scan: Option<scan::Scan>,
    /// File name of the disc in the drive; mirrors `Shared::disc` so the
    /// worker can restore it across a power cycle without reading it back.
    disc_name: Option<String>,
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
        let mut worker = Self {
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
            removed: None,
            cheats: Vec::new(),
            scan: None,
            disc_name: None,
            clock: now,
            deficit: 0.0,
            last_frame_publish: now,
            speed_window_start: now,
            speed_window_cycles: 0,
        };
        let cheats = std::mem::take(&mut worker.cfg.cheats);
        worker.install_cheats(cheats);
        worker.publish_disc(worker.cfg.disc_name.clone());
        worker
    }

    /// Adopt a disc's cheat table. The table outlives the machine: a
    /// power cycle rebuilds `sys`, so it is pushed in again there.
    fn install_cheats(&mut self, cheats: Vec<Group>) {
        self.cheats = cheats;
        self.sys.set_cheats(self.cheats.clone());
    }

    /// Refresh the disc shown by the UI. `name` is the image's file name,
    /// `None` when the drive is empty; the serial is read off the disc.
    fn publish_disc(&mut self, name: Option<String>) {
        self.disc_name = name.clone();
        let info = name.map(|name| DiscInfo {
            name,
            serial: self.sys.bus.cdvd.boot_serial(),
        });
        *self.shared.disc.lock().unwrap() = info;
    }

    fn debugger_active(&self) -> bool {
        self.cfg.debugger.as_ref().is_some_and(|d| d.attached())
            || (self.cfg.wait_debugger && !self.debugger_seen)
    }

    fn run(mut self) {
        self.audio = Audio::new(AUDIO_TARGET);
        // Frames come from the GS worker's vblank composite; asking the
        // renderer directly would stall emulation until it caught up.
        self.sys.set_publish_frames(true);
        loop {
            if !self.handle_commands() {
                break;
            }
            let (buttons, sticks) = unpack_pad(self.shared.pad.load(Ordering::Relaxed));
            self.sys.bus.sio2.buttons = buttons;
            self.sys.bus.sio2.sticks = sticks;
            self.sys.bus.gs.deinterlace = deinterlace_mode(self.shared.deinterlace.load(Ordering::Relaxed));
            self.sys.bus.gs.swap_fields = self.shared.swap_fields.load(Ordering::Relaxed);
            self.sys.set_cheats_enabled(self.shared.cheats.load(Ordering::Relaxed));
            self.sys.bus.gs.set_internal_2x(self.shared.internal_2x.load(Ordering::Relaxed));

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
                    let name = self.disc_name.clone();
                    self.power_cycle();
                    self.publish_disc(name);
                }
                Command::OpenTray if !debugger_active => {
                    let name = self.disc_name.clone().unwrap_or_default();
                    self.removed =
                        self.sys.bus.cdvd.open_tray().map(|file| Disc { file, name, cheats: self.cheats.clone() });
                    self.publish_disc(None);
                }
                Command::CloseTray(disc) if !debugger_active => {
                    // A new image brings its own cheats; the one that came
                    // out and went back in keeps the table it had.
                    if let Some(d) = &disc {
                        self.install_cheats(d.cheats.clone());
                    }
                    let disc = disc.or_else(|| self.removed.take());
                    let name = disc.as_ref().map(|d| d.name.clone());
                    self.sys.bus.cdvd.close_tray(disc.map(|d| d.file), self.sys.cycles);
                    self.removed = None;
                    self.publish_disc(name);
                }
                Command::BootDisc(disc) if !debugger_active => {
                    let name = disc.as_ref().map(|d| d.name.clone());
                    let cheats = disc.as_ref().map(|d| d.cheats.clone()).unwrap_or_default();
                    self.sys.bus.cdvd.disc = disc.map(|d| d.file);
                    self.power_cycle();
                    self.install_cheats(cheats);
                    self.publish_disc(name);
                    self.running = true;
                }
                Command::SaveState if !debugger_active => {
                    let path = self.cfg.state_path.clone();
                    let outcome = match self.sys.save_state() {
                        Ok(data) => match crate::state::write(&path, &data) {
                            Ok(len) => Ok(format!("state saved ({} MiB)", len / (1 << 20))),
                            Err(e) => Err(format!("cannot write the state: {e}")),
                        },
                        Err(e) => Err(e),
                    };
                    self.report(outcome);
                }
                Command::LoadState if !debugger_active => {
                    let path = self.cfg.state_path.clone();
                    let outcome = crate::state::read(&path)
                        .map_err(|e| format!("cannot read the state: {e}"))
                        .and_then(|d| self.sys.load_state(&d))
                        .map(|()| "state loaded".to_string());
                    self.report(outcome);
                }
                Command::Scan(req) => {
                    let ram = match req.target {
                        scan::Target::Ee => &self.sys.bus.ram[..],
                        scan::Target::Iop => &self.sys.bus.iop_ram[..],
                    };
                    let (scan, result) = scan::Scan::pass(self.scan.take(), req, ram);
                    self.scan = Some(scan);
                    *self.shared.scan.lock().unwrap() = result;
                    self.ctx.request_repaint();
                }
                Command::SetRunning(_)
                | Command::Step
                | Command::Reset
                | Command::OpenTray
                | Command::CloseTray(_)
                | Command::BootDisc(_)
                | Command::SaveState
                | Command::LoadState => {}
                Command::SetCheats(groups) => self.install_cheats(groups),
                Command::SetCheatEnabled(name, on) => self.sys.set_group_enabled(&name, on),
                Command::Quit => return false,
            }
        }
        true
    }

    /// Publish a one-shot result for the status bar, and log it.
    fn report(&self, outcome: Result<String, String>) {
        let text = match &outcome {
            Ok(msg) => {
                tracing::info!("{msg}");
                msg.clone()
            }
            Err(e) => {
                tracing::error!("{e}");
                e.clone()
            }
        };
        *self.shared.notice.lock().unwrap() = Some((text, outcome.is_err()));
        self.ctx.request_repaint();
    }

    /// Rebuild the machine from the reset vector. The disc, memory card
    /// (mid-write contents included) and mechacon NVRAM survive in the
    /// core, as they do across a real power cycle.
    fn power_cycle(&mut self) {
        self.sys.power_cycle(self.cfg.region).expect("reset failed");
        // A scan's candidates describe the machine that was just replaced.
        self.scan = None;
        *self.shared.scan.lock().unwrap() = scan::Result::default();
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
        let panels = self.shared.panels.load(Ordering::Relaxed);
        if now.duration_since(self.last_frame_publish) >= FRAME_INTERVAL {
            self.last_frame_publish = now;
            if panels & PANEL_MEMORY != 0 {
                self.publish_memory();
            }
            if let Some((w, h, rgba)) = self.sys.latest_frame_shared()
                && w > 0
                && h > 0
            {
                let mut f = self.shared.frame.lock().unwrap();
                f.width = w;
                f.height = h;
                f.rgba = rgba;
                f.seq = f.seq.wrapping_add(1);
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
            st.region = self.sys.region();
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
            // Both register files, only while a panel shows them.
            if panels & PANEL_REGS != 0 {
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
        }
        self.shared
            .debugger_active
            .store(self.debugger_active(), Ordering::Relaxed);
    }

    /// Copy the window the viewer asked for. Addresses are RAM offsets,
    /// clamped so the window never runs off the end.
    fn publish_memory(&mut self) {
        let (target, base) = unpack_view(self.shared.view.load(Ordering::Relaxed));
        let ram = match target {
            scan::Target::Ee => &self.sys.bus.ram[..],
            scan::Target::Iop => &self.sys.bus.iop_ram[..],
        };
        let base = (base as usize & !0xF).min(ram.len() - VIEW_BYTES);
        let mut m = self.shared.memory.lock().unwrap();
        m.target = Some(target);
        m.base = base as u32;
        m.bytes.clear();
        m.bytes.extend_from_slice(&ram[base..base + VIEW_BYTES]);
    }

    /// Stream kernel TTY output to stdout (same as the headless path) and
    /// accumulate it for the UI panel, capped to [`TTY_CAP`]. The IOP
    /// console shares the stream: both are the machine talking.
    fn flush_tty(&mut self) {
        let mut tty = self.sys.take_tty();
        tty.push_str(&self.sys.take_iop_tty());
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
