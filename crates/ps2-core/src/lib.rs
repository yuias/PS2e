//! Platform-independent PS2 emulator core.
//!
//! No windowing, graphics-API or I/O dependencies: everything here must stay
//! wasm-safe. The native front-end (`ps2-app`) and future wasm bindings drive
//! this crate through [`Ps2System`].

pub mod bus;
pub mod ee;
pub mod timers;

use bus::Bus;
use ee::Cpu;

/// EE core clock in Hz (294.912 MHz). IOP runs at 1/8 of this.
pub const EE_CLOCK_HZ: u64 = 294_912_000;

/// Top-level system: owns every component, mirrors the real console.
pub struct Ps2System {
    pub ee: Cpu,
    pub bus: Bus,
    /// Total elapsed EE cycles since reset.
    pub cycles: u64,
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
            ee: Cpu::new(),
            bus: Bus::new(bios),
            cycles: 0,
        })
    }

    /// Execute one EE instruction.
    pub fn step(&mut self) {
        self.bus.now = self.cycles;
        self.ee.step(&mut self.bus);
        // 1 cycle per instruction for now; wait states and dual-issue
        // approximation come later.
        self.cycles += 1;
    }

    /// Run for approximately `cycles` EE cycles.
    pub fn run(&mut self, cycles: u64) {
        let target = self.cycles + cycles;
        while self.cycles < target {
            self.step();
        }
    }

    /// Drain kernel TTY output captured since the last call.
    pub fn take_tty(&mut self) -> String {
        core::mem::take(&mut self.bus.tty_buffer)
    }
}
