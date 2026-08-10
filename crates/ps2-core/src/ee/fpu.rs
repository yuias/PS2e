//! EE COP1: single-precision FPU.
//!
//! The real FPU is not IEEE 754 compliant (no NaN/Inf, flush-to-zero,
//! different rounding). Host f32 arithmetic is used as an approximation for
//! now; results are clamped away from NaN/Inf where it matters.

pub struct Fpu {
    pub regs: [f32; 32],
    pub acc: f32,
    /// FCR31 control/status register.
    pub fcr31: u32,
    /// Result of the last c.cond.s comparison (FCR31 bit 23).
    pub condition: bool,
}

impl Default for Fpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Fpu {
    pub fn new() -> Self {
        Self {
            regs: [0.0; 32],
            acc: 0.0,
            fcr31: 0x0100_0001,
            condition: false,
        }
    }

    /// FCR0 (revision) reads; FCR31 carries the condition bit.
    pub fn read_control(&self, reg: usize) -> u32 {
        match reg {
            0 => 0x0000_2E30,
            31 => (self.fcr31 & !(1 << 23)) | ((self.condition as u32) << 23),
            _ => 0,
        }
    }

    pub fn write_control(&mut self, reg: usize, v: u32) {
        if reg == 31 {
            self.fcr31 = v;
            self.condition = v & (1 << 23) != 0;
        }
    }

    /// Clamp away NaN/Inf, mimicking the hardware's lack of them.
    pub fn clamp(v: f32) -> f32 {
        if v.is_nan() {
            0.0
        } else if v == f32::INFINITY {
            f32::MAX
        } else if v == f32::NEG_INFINITY {
            f32::MIN
        } else {
            v
        }
    }
}
