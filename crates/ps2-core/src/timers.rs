//! EE timers (T0-T3): 16-bit counters on BUSCLK (EE clock / 2).
//!
//! Counts are computed lazily from the cycle counter instead of ticking —
//! reads reconstruct the value from (now - base). Gates and interrupts are
//! not wired up yet; MODE's interrupt flags are write-1-to-clear.

use tracing::trace;

/// BUSCLK runs at half the EE clock.
const BUSCLK_SHIFT: u64 = 1;
/// HBLANK rate approximation: BUSCLK / 9371 (NTSC).
const HBLANK_DIV: u64 = 9371;

#[derive(Default, Clone, Copy)]
struct Timer {
    /// COUNT value at `base_cycle`.
    base: u16,
    /// EE cycle when COUNT was last written.
    base_cycle: u64,
    mode: u32,
    comp: u16,
    hold: u16,
    /// Cycle stamp of the last interrupt check, for edge detection.
    last_check: u64,
}

impl Timer {
    fn count(&self, now: u64) -> u16 {
        let busclk = (now.saturating_sub(self.base_cycle)) >> BUSCLK_SHIFT;
        let ticks = match self.mode & 3 {
            0 => busclk,
            1 => busclk / 16,
            2 => busclk / 256,
            _ => busclk / HBLANK_DIV,
        };
        self.base.wrapping_add(ticks as u16)
    }
}

pub struct Timers {
    timers: [Timer; 4],
}

impl Default for Timers {
    fn default() -> Self {
        Self::new()
    }
}

impl Timers {
    pub fn new() -> Self {
        Self {
            timers: [Timer::default(); 4],
        }
    }

    /// `addr` is the physical address in 0x1000_0000..0x1000_1FFF.
    pub fn read(&self, addr: u32, now: u64) -> u32 {
        let (t, reg) = Self::decode(addr);
        let timer = &self.timers[t];
        let v = match reg {
            0x00 => timer.count(now) as u32,
            0x10 => timer.mode,
            0x20 => timer.comp as u32,
            0x30 => timer.hold as u32,
            _ => 0,
        };
        trace!(target: "ps2_core::timers", t, reg, value = format_args!("{v:#x}"), "read");
        v
    }

    pub fn write(&mut self, addr: u32, v: u32, now: u64) {
        let (t, reg) = Self::decode(addr);
        trace!(target: "ps2_core::timers", t, reg, value = format_args!("{v:#x}"), "write");
        let timer = &mut self.timers[t];
        match reg {
            0x00 => {
                timer.base = v as u16;
                timer.base_cycle = now;
            }
            0x10 => {
                // Bits 10/11 (equal/overflow flags) are write-1-to-clear.
                let flags = timer.mode & 0xC00 & !(v & 0xC00);
                timer.mode = (v & !0xC00) | flags;
                // Changing the clock source restarts counting from the
                // current value so the lazy count stays consistent.
                timer.base = timer.count(now);
                timer.base_cycle = now;
            }
            0x20 => timer.comp = v as u16,
            0x30 => timer.hold = v as u16,
            _ => {}
        }
    }

    fn decode(addr: u32) -> (usize, u32) {
        let t = ((addr >> 11) & 3) as usize;
        (t, addr & 0x30)
    }

    /// Edge-detect compare matches since the last call; returns an INTC bit
    /// mask (bit 9+t per timer). Sets the mode equal-flag as on hardware.
    pub fn check_irqs(&mut self, now: u64) -> u32 {
        let mut intc = 0;
        for (t, timer) in self.timers.iter_mut().enumerate() {
            // CMPE: compare interrupt enabled.
            if timer.mode & (1 << 7) == 0 {
                continue;
            }
            let before = timer.count(timer.last_check);
            let after = timer.count(now);
            timer.last_check = now;
            let crossed = if before <= after {
                before < timer.comp && timer.comp <= after
            } else {
                // 16-bit wraparound between checks.
                timer.comp > before || timer.comp <= after
            };
            if crossed {
                timer.mode |= 1 << 10; // equal flag
                if timer.mode & (1 << 6) != 0 {
                    // ZRET: restart counting from zero on compare.
                    timer.base = 0;
                    timer.base_cycle = now;
                }
                intc |= 1 << (9 + t);
            }
        }
        intc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_follows_busclk() {
        let mut t = Timers::new();
        t.write(0x1000_0000, 0, 1000);
        // 2000 EE cycles later = 1000 BUSCLK ticks.
        assert_eq!(t.read(0x1000_0000, 3000), 1000);
    }

    #[test]
    fn prescaler_divides() {
        let mut t = Timers::new();
        t.write(0x1000_0010, 2, 0); // /256
        t.write(0x1000_0000, 0, 0);
        assert_eq!(t.read(0x1000_0000, 512 * 256 * 2), 512);
    }

    #[test]
    fn count_write_resets_base() {
        let mut t = Timers::new();
        t.write(0x1000_0000, 100, 0);
        assert_eq!(t.read(0x1000_0000, 200), 200);
    }
}
