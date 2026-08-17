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
    /// Build a system with the given BIOS image (must be 4 MiB).
    pub fn new(bios: Vec<u8>) -> Result<Self, String> {
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
            bus: Bus::new(bios),
            cycles: 0,
            frame_pos: 0,
        })
    }

    /// Execute one EE instruction, stepping the IOP at the 8:1 clock ratio.
    pub fn step(&mut self) {
        self.bus.now = self.cycles;
        self.ee.step(&mut self.bus);
        // 1 cycle per instruction for now; wait states and dual-issue
        // approximation come later.
        if self.cycles.is_multiple_of(EE_PER_IOP) {
            let _g = prof::scope(prof::Slot::Iop);
            self.iop.step(&mut self.bus);
        }
        if self.cycles.is_multiple_of(64) {
            let _g = prof::scope(prof::Slot::Timers);
            self.bus.tick_timers();
        }
        // Counted rather than derived with `%`: this runs per instruction.
        if self.frame_pos == EE_CYCLES_PER_FRAME - VBLANK_CYCLES {
            self.bus.vblank(true);
        } else if self.frame_pos == 0 && self.cycles != 0 {
            self.bus.vblank(false);
        }
        self.frame_pos += 1;
        if self.frame_pos == EE_CYCLES_PER_FRAME {
            self.frame_pos = 0;
        }
        self.cycles += 1;
    }

    /// Run for approximately `cycles` EE cycles.
    pub fn run(&mut self, cycles: u64) {
        // One EE scope per slice: nested IOP/timer/DMA scopes hand back here.
        let _g = prof::scope(prof::Slot::Ee);
        let target = self.cycles + cycles;
        while self.cycles < target {
            self.step();
        }
    }

    /// Drain kernel TTY output captured since the last call.
    pub fn take_tty(&mut self) -> String {
        core::mem::take(&mut self.bus.tty_buffer)
    }

    /// Current display output as RGBA8: (width, height, pixels).
    pub fn framebuffer(&self) -> (u32, u32, Vec<u8>) {
        self.bus.gs.framebuffer()
    }
}
