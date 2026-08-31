//! Platform-independent PS2 emulator core.
//!
//! No windowing, graphics-API or I/O dependencies: everything here must stay
//! wasm-safe. The native front-end (`ps2-app`) and future wasm bindings drive
//! this crate through [`Ps2System`].

pub mod bus;
pub mod ee;
pub mod gif;
pub mod gs;
pub mod iop;
pub mod ipu;
/// Executable memory for the recompilers (x86-64 native builds only).
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub(crate) mod jit_arena;
pub mod prof;
pub mod sif;
pub mod spu2;
pub mod timers;
pub mod vif;
pub mod vu1;
/// VU1 dynamic recompiler (x86-64 native builds only).
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
pub mod vu1_jit;

use bus::Bus;

/// EE core clock in Hz (294.912 MHz). IOP runs at 1/8 of this.
pub const EE_CLOCK_HZ: u64 = 294_912_000;
/// EE cycles per IOP cycle.
pub const EE_PER_IOP: u64 = 8;
/// EE cycles per video frame on an NTSC machine (~60 Hz).
pub const EE_CYCLES_PER_FRAME: u64 = EE_CLOCK_HZ / 60;

/// Video timing region: the vertical refresh and the horizontal-blank rates
/// the timers count.
///
/// Software owns this. The kernel's `SetGsCrt` programs SMODE1's CMOD field
/// from its `pal_ntsc` argument, and [`bus::Bus`] follows that write, so the
/// region a machine is built with only holds until the first one. It is
/// still not what software *detects* as the console's region: that comes
/// from the BIOS image's own ROMVER, so a PAL title wants a PAL BIOS.
#[derive(serde::Serialize, serde::Deserialize)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[serde(rename_all = "lowercase")]
pub enum Region {
    #[default]
    Ntsc,
    Pal,
}

impl Region {
    /// EE cycles per video frame: ~60 Hz (NTSC) or 50 Hz (PAL).
    pub const fn cycles_per_frame(self) -> u64 {
        match self {
            Region::Ntsc => EE_CYCLES_PER_FRAME,
            Region::Pal => EE_CLOCK_HZ / 50,
        }
    }

    /// Frame position at which vertical blank begins; it occupies roughly
    /// the last 5% of the frame.
    pub const fn vblank_start(self) -> u64 {
        let frame = self.cycles_per_frame();
        frame - frame / 20
    }

    /// Nominal vertical refresh, for a front-end's frame-rate display.
    pub const fn refresh_hz(self) -> f64 {
        match self {
            Region::Ntsc => 60.0,
            Region::Pal => 50.0,
        }
    }

    /// BUSCLK cycles per horizontal blank, approximated: 15734 Hz (NTSC) or
    /// 15625 Hz (PAL).
    pub const fn ee_hblank_div(self) -> u64 {
        match self {
            Region::Ntsc => 9371,
            Region::Pal => 9437,
        }
    }

    /// IOP sysclock cycles per horizontal blank, at the same rates.
    pub const fn iop_hblank_div(self) -> u64 {
        match self {
            Region::Ntsc => 2343,
            Region::Pal => 2359,
        }
    }
}

impl std::str::FromStr for Region {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "ntsc" => Ok(Region::Ntsc),
            "pal" => Ok(Region::Pal),
            other => Err(format!("unknown region {other:?} (expected ntsc or pal)")),
        }
    }
}
/// Cadence of the periodic bus tick (timers, deferred DMA and SPU2
/// completions, SPU2 mixing).
const TIMER_TICK_CYCLES: u64 = 64;

/// Save-state file magic and format version. Bump the version on any
/// change to a serialized struct.
const STATE_MAGIC: &[u8; 4] = b"PS2E";
const STATE_VERSION: u16 = 9;

/// Cheap content fingerprint (FNV-1a) to flag cross-BIOS state loads.
fn bios_fingerprint(bios: &[u8]) -> u32 {
    bios.iter().fold(0x811c_9dc5u32, |h, b| (h ^ u32::from(*b)).wrapping_mul(0x0100_0193))
}

/// Top-level system: owns every component, mirrors the real console.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct Ps2System {
    pub ee: ee::Cpu,
    pub iop: iop::Cpu,
    pub bus: Bus,
    /// Total elapsed EE cycles since reset.
    pub cycles: u64,
    /// Position within the current video frame, in EE cycles.
    frame_pos: u64,
    /// EE recompiler; `None` runs the interpreter. Translated code is not
    /// state: a load starts from an empty cache.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    #[serde(skip)]
    jit: Option<ee::jit::Jit>,
}

impl Ps2System {
    /// Build a system with the given BIOS image (must be 4 MiB). The GS
    /// renderer runs on a worker thread when the `threads` feature is on.
    pub fn new(bios: Vec<u8>) -> Result<Self, String> {
        Self::new_with(bios, cfg!(feature = "threads"))
    }

    /// [`Ps2System::new`] with an explicit choice of a threaded (`true`) or
    /// inline GS renderer; the threaded one needs the `threads` feature.
    pub fn new_with(bios: Vec<u8>, gs_threaded: bool) -> Result<Self, String> {
        Self::new_with_region(bios, gs_threaded, Region::default())
    }

    /// [`Ps2System::new`] in a chosen video timing [`Region`].
    pub fn new_region(bios: Vec<u8>, region: Region) -> Result<Self, String> {
        Self::new_with_region(bios, cfg!(feature = "threads"), region)
    }

    /// [`Ps2System::new_with`] in a chosen video timing [`Region`], which
    /// holds until software programs the CRTC (and a loaded save state
    /// brings its own).
    pub fn new_with_region(
        bios: Vec<u8>,
        gs_threaded: bool,
        region: Region,
    ) -> Result<Self, String> {
        if bios.len() != bus::BIOS_SIZE {
            return Err(format!(
                "BIOS must be {} bytes, got {}",
                bus::BIOS_SIZE,
                bios.len()
            ));
        }
        // `mut` only for the recompiler hand-off just below it.
        #[cfg_attr(not(all(feature = "jit", target_arch = "x86_64")), allow(unused_mut))]
        let mut sys = Self {
            ee: ee::Cpu::new(),
            iop: iop::Cpu::new(),
            bus: Bus::new(bios, gs_threaded, region),
            cycles: 0,
            frame_pos: 0,
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            jit: Some(ee::jit::Jit::new().map_err(|e| format!("cannot allocate JIT arena: {e}"))?),
        };
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        sys.bus.vu1.set_jit(true)?;
        Ok(sys)
    }

    /// Enable or disable the EE and VU1 recompilers (a no-op without the
    /// `jit` feature). The interpreters and the recompilers are
    /// interchangeable at any instruction boundary.
    pub fn set_jit(&mut self, on: bool) -> Result<(), String> {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        {
            if on && self.jit.is_none() {
                self.jit =
                    Some(ee::jit::Jit::new().map_err(|e| format!("cannot allocate JIT arena: {e}"))?);
            } else if !on {
                self.jit = None;
            }
            self.bus.vu1.set_jit(on)?;
            Ok(())
        }
        #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
        {
            if on { Err("built without the jit feature".into()) } else { Ok(()) }
        }
    }

    /// Serialize the complete machine state. The BIOS image, disc and
    /// memory card are *not* included: they are ambient assets the
    /// frontend owns (and rolling a memory card back would corrupt real
    /// saves). The display side follows the machine, because its renderer
    /// may have to be asked across a thread for it.
    pub fn save_state(&mut self) -> Result<Vec<u8>, String> {
        let gs = self.bus.gs.snapshot()?;
        let mut out = Vec::with_capacity(48 << 20);
        out.extend_from_slice(STATE_MAGIC);
        out.extend_from_slice(&STATE_VERSION.to_le_bytes());
        out.extend_from_slice(&bios_fingerprint(&self.bus.bios).to_le_bytes());
        let out = postcard::to_extend(&*self, out).map_err(|e| format!("serialize failed: {e}"))?;
        postcard::to_extend(&gs, out).map_err(|e| format!("serialize failed: {e}"))
    }

    /// Restore a state from [`Ps2System::save_state`], carrying over the
    /// BIOS, disc, memory card and the running renderer.
    pub fn load_state(&mut self, data: &[u8]) -> Result<(), String> {
        let (header, body) = data.split_at_checked(10).ok_or("state file too short")?;
        if &header[..4] != STATE_MAGIC {
            return Err("not a PS2e save state".into());
        }
        let version = u16::from_le_bytes(header[4..6].try_into().unwrap());
        if version != STATE_VERSION {
            return Err(format!(
                "state version {version} not supported (expected {STATE_VERSION})"
            ));
        }
        if u32::from_le_bytes(header[6..10].try_into().unwrap())
            != bios_fingerprint(&self.bus.bios)
        {
            tracing::warn!("state was saved with a different BIOS image; expect instability");
        }
        let (mut sys, rest) = postcard::take_from_bytes::<Ps2System>(body)
            .map_err(|e| format!("deserialize failed: {e}"))?;
        let gs: gs::front::GsState =
            postcard::from_bytes(rest).map_err(|e| format!("deserialize failed: {e}"))?;
        // Ambient assets and the live renderer come from the running
        // machine, not from the file.
        sys.bus.bios = std::mem::take(&mut self.bus.bios);
        sys.bus.gs = std::mem::replace(&mut self.bus.gs, gs::front::GsFront::inline());
        sys.bus.cdvd.carry_over(&mut self.bus.cdvd);
        sys.bus.sio2.memcard = std::mem::take(&mut self.bus.sio2.memcard);
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        {
            sys.jit = self.jit.take();
            // Micro memory came from the file, so nothing translated
            // against the running machine's copy is still valid, and the
            // generation counters on either side say nothing about it.
            sys.bus.vu1.jit = self.bus.vu1.jit.take();
            sys.bus.vu1.flush_jit();
        }
        sys.bus.after_load();
        sys.bus.gs.restore(gs)?;
        *self = sys;
        Ok(())
    }

    /// Recompiler counters: (blocks compiled, invalidated, run, interpreter
    /// steps taken by the dispatcher); zeros without a recompiler.
    pub fn jit_stats(&self) -> (u64, u64, u64, u64) {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if let Some(j) = &self.jit {
            return (j.blocks_compiled, j.blocks_invalidated, j.blocks_run, j.interp_steps);
        }
        (0, 0, 0, 0)
    }

    /// VU1 recompiler counters: (blocks compiled, run, cache flushes,
    /// programs handed back to the interpreter); zeros without one.
    pub fn vu1_jit_stats(&self) -> (u64, u64, u64, u64) {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if let Some(j) = &self.bus.vu1.jit {
            return (j.blocks_compiled, j.blocks_run, j.flushes, j.bails);
        }
        (0, 0, 0, 0)
    }

    /// Whether the EE recompiler is active.
    pub fn jit_enabled(&self) -> bool {
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        {
            self.jit.is_some()
        }
        #[cfg(not(all(feature = "jit", target_arch = "x86_64")))]
        {
            false
        }
    }

    /// Execute one EE instruction, stepping the IOP at the 8:1 clock ratio.
    pub fn step(&mut self) {
        self.bus.now = self.cycles;
        // Idle-loop skip: while the EE spins in the kernel idle thread only
        // an interrupt can move it, so let the rest of the machine run and
        // resume stepping (into the exception) once one is pending. Only
        // the IOP, timers and vblank can raise one, so the check runs after
        // those rather than every cycle.
        if !self.ee.idle {
            self.ee.step(&mut self.bus);
        }
        self.machine_cycle();
    }

    /// Video timing region this machine was built in.
    pub fn region(&self) -> Region {
        self.bus.region
    }

    /// EE cycles per video frame in this machine's region.
    #[inline]
    fn frame_cycles(&self) -> u64 {
        self.bus.region.cycles_per_frame()
    }

    /// Frame position at which vertical blank begins.
    #[inline]
    fn vblank_start(&self) -> u64 {
        self.bus.region.vblank_start()
    }

    /// Bring `frame_pos` back inside the frame. Software can change the
    /// region mid-frame (see [`bus::Bus::write_gs_priv`]), leaving a
    /// position the new, shorter frame has already passed; one subtraction
    /// always suffices, since the difference between the two frame lengths
    /// is smaller than either of them.
    #[inline]
    fn wrap_frame_pos(&mut self) {
        let frame = self.frame_cycles();
        if self.frame_pos >= frame {
            self.frame_pos -= frame;
        }
    }

    /// Everything but the EE for one cycle: the IOP slot, timers, vblank
    /// edges, and the idle wake-up check.
    #[inline]
    fn machine_cycle(&mut self) {
        let mut event = false;
        // The EE retires one issue group per cycle (dual-issued pairs count
        // one; see `ee::issue`); wait states are not modeled.
        if self.cycles.is_multiple_of(EE_PER_IOP) {
            let _g = prof::scope(prof::Slot::Iop);
            self.bus.now = self.cycles;
            // Same idle-loop skip as the EE, for the IOP kernel's `j .`.
            if !self.iop.idle || self.iop.interrupt_pending(&self.bus) {
                self.iop.idle = false;
                self.iop.step(&mut self.bus);
            }
            event = true;
        }
        if self.cycles.is_multiple_of(TIMER_TICK_CYCLES) && self.cycles >= self.bus.timers_due {
            let _g = prof::scope(prof::Slot::Timers);
            self.bus.now = self.cycles;
            self.bus.tick_timers();
            event = true;
        }
        // Counted rather than derived with `%`: this runs per instruction.
        if self.frame_pos == self.vblank_start() {
            self.bus.vblank(true);
            event = true;
        } else if self.frame_pos == 0 && self.cycles != 0 {
            self.bus.vblank(false);
            event = true;
        }
        self.frame_pos += 1;
        self.wrap_frame_pos();
        self.cycles += 1;
        if self.ee.idle && event && self.ee.interrupt_pending(&self.bus) {
            self.ee.idle = false;
        }
    }

    /// Run for approximately `cycles` EE cycles.
    ///
    /// With the recompiler, whole blocks retire before the rest of the
    /// machine catches up by the same number of cycles. Without it (or
    /// while the EE idles) the sequence is that of repeated
    /// [`Ps2System::step`], but the eight EE cycles between IOP slots are
    /// grouped so the per-cycle checks (`%`, vblank edge, wake-up) are
    /// hoisted, and an idle EE skips a whole group at once when no vblank
    /// edge falls inside it.
    pub fn run(&mut self, cycles: u64) {
        // One EE scope per slice: nested IOP/timer/DMA scopes hand back here.
        let _g = prof::scope(prof::Slot::Ee);
        let target = self.cycles + cycles;
        while self.cycles < target {
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            if !self.ee.idle
                && let Some(jit) = &mut self.jit
            {
                self.bus.now = self.cycles;
                // Linked blocks run until about this many cycles retired;
                // the IOP, timers and vblank then catch up. Longer chains
                // amortise the dispatcher, shorter ones keep the cores
                // closer in step. 512 is faster again but moves the
                // disc-less BIOS OSD, so the gate puts the line here.
                const CHAIN_BUDGET: u64 = 256;
                let budget = (target - self.cycles).min(CHAIN_BUDGET) as u32;
                let n = jit.run(&mut self.ee, &mut self.bus, budget);
                self.advance(n as u64);
                continue;
            }
            // Both cores idle: jump to the next timer tick or vblank edge,
            // the only things that can wake either of them.
            if self.ee.idle && self.iop.idle && !self.iop.interrupt_pending(&self.bus) {
                let k = self.idle_skip(target - self.cycles - 1);
                self.jump(k);
                self.step();
                continue;
            }
            if !self.cycles.is_multiple_of(EE_PER_IOP) || target - self.cycles < EE_PER_IOP {
                self.step();
                continue;
            }
            // Cycle 0 of the group carries the IOP slot and timers.
            self.step();
            // Cycles 1..7: EE only, plus a vblank edge if one lands here.
            let vbl_edge = self.frame_pos + EE_PER_IOP > self.vblank_start()
                && self.frame_pos <= self.vblank_start();
            let wrap = self.frame_pos + EE_PER_IOP >= self.frame_cycles();
            if self.ee.idle && !vbl_edge && !wrap {
                self.frame_pos += EE_PER_IOP - 1;
                self.cycles += EE_PER_IOP - 1;
                continue;
            }
            for _ in 1..EE_PER_IOP {
                self.bus.now = self.cycles;
                if !self.ee.idle {
                    self.ee.step(&mut self.bus);
                }
                if self.frame_pos == self.vblank_start() {
                    self.bus.vblank(true);
                    self.wake_idle_ee();
                } else if self.frame_pos == 0 && self.cycles != 0 {
                    self.bus.vblank(false);
                    self.wake_idle_ee();
                }
                self.frame_pos += 1;
                self.wrap_frame_pos();
                self.cycles += 1;
            }
        }
    }

    /// With the IOP idle (and no interrupt for it pending) nothing can
    /// change its state until the next timer tick or vblank edge: every
    /// other IOP interrupt source is an IOP or EE access, and neither core
    /// runs during a skip. Returns how many cycles can be jumped before the
    /// next `machine_cycle` must run (0 = it must run now); `limit` bounds
    /// the jump.
    #[inline]
    fn idle_skip(&self, limit: u64) -> u64 {
        // First tick-aligned cycle at or after the bus's next due time.
        let due = self.bus.timers_due.max(self.cycles);
        let to_timer = due.div_ceil(TIMER_TICK_CYCLES) * TIMER_TICK_CYCLES - self.cycles;
        let vbl_start = self.vblank_start();
        let to_vblank = if self.frame_pos < vbl_start {
            vbl_start - self.frame_pos
        } else {
            self.frame_cycles().saturating_sub(self.frame_pos)
        };
        to_timer.min(to_vblank).min(limit)
    }

    /// Jump `k` cycles that carry no IOP slot, timer tick or vblank edge
    /// (`k` never crosses a frame boundary; landing on it is the vblank-end
    /// edge, which `machine_cycle` fires at `frame_pos == 0`).
    #[inline]
    fn jump(&mut self, k: u64) {
        self.cycles += k;
        self.frame_pos += k;
        self.wrap_frame_pos();
    }

    /// Run the rest of the machine for `n` cycles after the EE retired that
    /// many instructions: IOP slots and timers keep their cadence, vblank
    /// edges land on the exact cycle, and cycles with nothing due are
    /// skipped in bulk.
    fn advance(&mut self, mut n: u64) {
        while n > 0 {
            if self.iop.idle && !self.iop.interrupt_pending(&self.bus) {
                let k = self.idle_skip(n - 1);
                self.jump(k);
                self.machine_cycle();
                n -= k + 1;
                continue;
            }
            if !self.cycles.is_multiple_of(EE_PER_IOP) || n < EE_PER_IOP {
                self.machine_cycle();
                n -= 1;
                continue;
            }
            // Groups carrying nothing but their IOP slot run as one loop.
            // Only the idle-EE wake-up is left out, and it is a no-op with
            // the EE running.
            if !self.ee.idle {
                let g = self.quiet_iop_groups(n / EE_PER_IOP);
                if g > 0 {
                    self.run_iop_groups(g);
                    n -= g * EE_PER_IOP;
                    continue;
                }
            }
            self.machine_cycle();
            n -= 1;
            // Cycles 1..7 of the group can only see a vblank edge.
            let vbl_edge = self.frame_pos + EE_PER_IOP > self.vblank_start()
                && self.frame_pos <= self.vblank_start();
            let wrap = self.frame_pos + EE_PER_IOP >= self.frame_cycles();
            if !vbl_edge && !wrap {
                self.frame_pos += EE_PER_IOP - 1;
                self.cycles += EE_PER_IOP - 1;
                n -= EE_PER_IOP - 1;
            } else {
                for _ in 1..EE_PER_IOP {
                    self.machine_cycle();
                }
                n -= EE_PER_IOP - 1;
            }
        }
    }

    /// How many whole IOP groups from a group-aligned `self.cycles` carry
    /// nothing but their IOP slot: no timer tick that is due and no vblank
    /// edge. Both land on group starts (the tick cadence and the frame
    /// edges are multiples of [`EE_PER_IOP`]), so counting groups is exact.
    #[inline]
    fn quiet_iop_groups(&self, limit: u64) -> u64 {
        let vbl_start = self.vblank_start();
        // An edge on the current cycle is `machine_cycle`'s to fire.
        if self.frame_pos == 0 || self.frame_pos == vbl_start {
            return 0;
        }
        let to_vblank = if self.frame_pos < vbl_start {
            vbl_start - self.frame_pos
        } else {
            self.frame_cycles().saturating_sub(self.frame_pos)
        };
        let due = self.bus.timers_due.max(self.cycles);
        let to_timer = due.div_ceil(TIMER_TICK_CYCLES) * TIMER_TICK_CYCLES - self.cycles;
        (to_timer / EE_PER_IOP).min(to_vblank / EE_PER_IOP).min(limit)
    }

    /// `g` IOP slots, [`EE_PER_IOP`] cycles apart, with nothing else due.
    fn run_iop_groups(&mut self, g: u64) {
        let _guard = prof::scope(prof::Slot::Iop);
        for _ in 0..g {
            self.bus.now = self.cycles;
            // Same idle-loop skip as `machine_cycle`.
            if !self.iop.idle || self.iop.interrupt_pending(&self.bus) {
                self.iop.idle = false;
                self.iop.step(&mut self.bus);
            }
            self.cycles += EE_PER_IOP;
            self.frame_pos += EE_PER_IOP;
        }
        // The batch may end exactly on the frame edge; `machine_cycle`
        // recognises the vblank-end edge by a wrapped position, not by the
        // frame length.
        self.wrap_frame_pos();
    }

    #[inline]
    fn wake_idle_ee(&mut self) {
        if self.ee.idle && self.ee.interrupt_pending(&self.bus) {
            self.ee.idle = false;
        }
    }

    /// Attach the mechacon NVRAM image at `path` (created with factory
    /// defaults when missing). The OSD's configuration — language, clock
    /// settings, and the first-boot "initialized" flag that decides whether
    /// the boot runs the setup wizard with the PS/PS2 logo screens — lives
    /// there and persists across runs.
    pub fn load_nvram(&mut self, path: std::path::PathBuf) {
        self.bus.cdvd.load_nvram(path);
    }

    /// Drain kernel TTY output captured since the last call.
    pub fn take_tty(&mut self) -> String {
        core::mem::take(&mut self.bus.tty_buffer)
    }

    /// Current display output as RGBA8: (width, height, pixels). Waits for
    /// the renderer to catch up with everything written so far.
    pub fn framebuffer(&mut self) -> gs::Frame {
        self.bus.gs.framebuffer()
    }

    /// Composite the display at every vblank so [`Ps2System::latest_frame`]
    /// can serve a live front-end without stalling emulation.
    pub fn set_publish_frames(&mut self, on: bool) {
        self.bus.gs.set_publish_frames(on);
    }

    /// Newest vblank-composited frame (see [`Ps2System::set_publish_frames`]).
    pub fn latest_frame(&self) -> Option<gs::Frame> {
        self.bus.gs.latest_frame()
    }

    /// The composited frame without copying its pixels.
    pub fn latest_frame_shared(&self) -> Option<gs::SharedFrame> {
        self.bus.gs.latest_frame_shared()
    }
}

#[cfg(test)]
mod state_tests {
    use super::*;

    /// A state must round-trip into a machine that keeps running from the
    /// same place, with the ambient assets left alone.
    #[test]
    fn state_round_trips() {
        let bios = vec![0u8; bus::BIOS_SIZE];
        let mut a = Ps2System::new_with(bios.clone(), false).unwrap();
        a.run(200_000);
        let cycles = a.cycles;
        a.bus.ram[0x1000] = 0xA5;
        let blob = a.save_state().unwrap();

        let mut b = Ps2System::new_with(bios, false).unwrap();
        b.load_state(&blob).unwrap();
        assert_eq!(b.cycles, cycles);
        assert_eq!(b.ee.pc, a.ee.pc);
        assert_eq!(b.iop.pc, a.iop.pc);
        assert_eq!(b.bus.ram[0x1000], 0xA5);
        assert_eq!(b.bus.gs.vram(), a.bus.gs.vram());

        // Both step on identically from here.
        a.run(50_000);
        b.run(50_000);
        assert_eq!(a.ee.pc, b.ee.pc);
        assert_eq!(a.cycles, b.cycles);
        assert_eq!(a.bus.ram, b.bus.ram);
    }

    #[test]
    fn a_foreign_blob_is_rejected() {
        let mut sys = Ps2System::new_with(vec![0u8; bus::BIOS_SIZE], false).unwrap();
        assert!(sys.load_state(b"nope").is_err());
        assert!(sys.load_state(&[0u8; 64]).is_err());
    }
}
