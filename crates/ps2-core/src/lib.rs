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
pub mod prof;
pub mod sif;
pub mod spu2;
pub mod timers;
pub mod vif;
pub mod vu1;

use bus::Bus;

/// EE core clock in Hz (294.912 MHz). IOP runs at 1/8 of this.
pub const EE_CLOCK_HZ: u64 = 294_912_000;
/// EE cycles per IOP cycle.
pub const EE_PER_IOP: u64 = 8;
/// EE cycles per video frame (~60 Hz).
pub const EE_CYCLES_PER_FRAME: u64 = EE_CLOCK_HZ / 60;
/// Vertical blank occupies roughly the last 5% of the frame.
const VBLANK_CYCLES: u64 = EE_CYCLES_PER_FRAME / 20;

/// Top-level system: owns every component, mirrors the real console.
pub struct Ps2System {
    pub ee: ee::Cpu,
    pub iop: iop::Cpu,
    pub bus: Bus,
    /// Total elapsed EE cycles since reset.
    pub cycles: u64,
    /// Position within the current video frame, in EE cycles.
    frame_pos: u64,
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
        if bios.len() != bus::BIOS_SIZE {
            return Err(format!(
                "BIOS must be {} bytes, got {}",
                bus::BIOS_SIZE,
                bios.len()
            ));
        }
        Ok(Self {
            ee: ee::Cpu::new(),
            iop: iop::Cpu::new(),
            bus: Bus::new(bios, gs_threaded),
            cycles: 0,
            frame_pos: 0,
        })
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
        let mut event = false;
        // 1 cycle per instruction for now; wait states and dual-issue
        // approximation come later.
        if self.cycles.is_multiple_of(EE_PER_IOP) {
            let _g = prof::scope(prof::Slot::Iop);
            // Same idle-loop skip as the EE, for the IOP kernel's `j .`.
            if !self.iop.idle || self.iop.interrupt_pending(&self.bus) {
                self.iop.idle = false;
                self.iop.step(&mut self.bus);
            }
            event = true;
        }
        if self.cycles.is_multiple_of(64) {
            let _g = prof::scope(prof::Slot::Timers);
            self.bus.tick_timers();
            event = true;
        }
        // Counted rather than derived with `%`: this runs per instruction.
        if self.frame_pos == EE_CYCLES_PER_FRAME - VBLANK_CYCLES {
            self.bus.vblank(true);
            event = true;
        } else if self.frame_pos == 0 && self.cycles != 0 {
            self.bus.vblank(false);
            event = true;
        }
        self.frame_pos += 1;
        if self.frame_pos == EE_CYCLES_PER_FRAME {
            self.frame_pos = 0;
        }
        self.cycles += 1;
        if self.ee.idle && event && self.ee.interrupt_pending(&self.bus) {
            self.ee.idle = false;
        }
    }

    /// Run for approximately `cycles` EE cycles.
    ///
    /// Same sequence as repeated [`Ps2System::step`], but the eight EE cycles
    /// between IOP slots are grouped so the per-cycle checks (`%`, vblank
    /// edge, wake-up) are hoisted, and an idle EE skips a whole group at
    /// once when no vblank edge falls inside it.
    pub fn run(&mut self, cycles: u64) {
        // One EE scope per slice: nested IOP/timer/DMA scopes hand back here.
        let _g = prof::scope(prof::Slot::Ee);
        let target = self.cycles + cycles;
        while self.cycles < target {
            if !self.cycles.is_multiple_of(EE_PER_IOP) || target - self.cycles < EE_PER_IOP {
                self.step();
                continue;
            }
            // Cycle 0 of the group carries the IOP slot and timers.
            self.step();
            // Cycles 1..7: EE only, plus a vblank edge if one lands here.
            let vbl_edge = self.frame_pos + EE_PER_IOP > EE_CYCLES_PER_FRAME - VBLANK_CYCLES
                && self.frame_pos <= EE_CYCLES_PER_FRAME - VBLANK_CYCLES;
            let wrap = self.frame_pos + EE_PER_IOP >= EE_CYCLES_PER_FRAME;
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
                if self.frame_pos == EE_CYCLES_PER_FRAME - VBLANK_CYCLES {
                    self.bus.vblank(true);
                    self.wake_idle_ee();
                } else if self.frame_pos == 0 && self.cycles != 0 {
                    self.bus.vblank(false);
                    self.wake_idle_ee();
                }
                self.frame_pos += 1;
                if self.frame_pos == EE_CYCLES_PER_FRAME {
                    self.frame_pos = 0;
                }
                self.cycles += 1;
            }
        }
    }

    #[inline]
    fn wake_idle_ee(&mut self) {
        if self.ee.idle && self.ee.interrupt_pending(&self.bus) {
            self.ee.idle = false;
        }
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
}
