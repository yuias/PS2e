//! EE timers (T0-T3): 16-bit counters on BUSCLK (EE clock / 2).
//!
//! Counts are computed lazily from the cycle counter instead of ticking —
//! reads reconstruct the value from (now - base). Gates and interrupts are
//! not wired up yet; MODE's interrupt flags are write-1-to-clear.

use crate::Region;
use tracing::trace;
use serde::{Deserialize, Serialize};

/// BUSCLK runs at half the EE clock.
const BUSCLK_SHIFT: u64 = 1;

#[derive(Serialize, Deserialize)]
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
    /// EE cycles per COUNT tick for the selected clock.
    fn cycles_per_tick(&self, region: Region) -> u64 {
        (match self.mode & 3 {
            0 => 1,
            1 => 16,
            2 => 256,
            _ => region.ee_hblank_div(),
        }) << BUSCLK_SHIFT
    }

    fn count(&self, now: u64, region: Region) -> u16 {
        let busclk = (now.saturating_sub(self.base_cycle)) >> BUSCLK_SHIFT;
        let ticks = match self.mode & 3 {
            0 => busclk,
            1 => busclk / 16,
            2 => busclk / 256,
            _ => busclk / region.ee_hblank_div(),
        };
        self.base.wrapping_add(ticks as u16)
    }
}

#[derive(Serialize, Deserialize)]
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
    pub fn read(&self, addr: u32, now: u64, region: Region) -> u32 {
        let (t, reg) = Self::decode(addr);
        let timer = &self.timers[t];
        let v = match reg {
            0x00 => timer.count(now, region) as u32,
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
                // Writing MODE clears the counter, as on hardware.
                timer.base = 0;
                timer.base_cycle = now;
                timer.last_check = now;
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

    /// Earliest cycle at which a timer can reach its compare value (the
    /// only event [`Timers::check_irqs`] reacts to), or `u64::MAX`. A
    /// check at or after that cycle sees the crossing; earlier checks have
    /// nothing to find.
    /// Restart every count from where it stands, so a change of clock rate
    /// does not reinterpret the span already elapsed.
    pub fn rebase(&mut self, now: u64, region: Region) {
        for timer in &mut self.timers {
            timer.base = timer.count(now, region);
            timer.base_cycle = now;
        }
    }

    pub fn next_event(&self, now: u64, region: Region) -> u64 {
        let mut due = u64::MAX;
        for timer in &self.timers {
            if timer.mode & (1 << 7) == 0 || timer.mode & (1 << 10) != 0 {
                continue;
            }
            let p = timer.cycles_per_tick(region);
            let ticks = u64::from(timer.comp.wrapping_sub(timer.count(now, region)));
            // Equal right now: the crossing (if any) is found by this
            // check; the next one is a full wrap away.
            let ticks = if ticks == 0 { 0x1_0000 } else { ticks };
            let phase = now.saturating_sub(timer.base_cycle) % p;
            due = due.min(now + ticks * p - phase);
        }
        due
    }

    /// Edge-detect compare matches since the last call; returns an INTC bit
    /// mask (bit 9+t per timer). Sets the mode equal-flag as on hardware.
    pub fn check_irqs(&mut self, now: u64, region: Region) -> u32 {
        let mut intc = 0;
        for (t, timer) in self.timers.iter_mut().enumerate() {
            // CUE (bit 7) gates counting. The equal-flag (EQUF, bit 10)
            // latches: no new compare event until a MODE write with bit 10
            // rearms it.
            if timer.mode & (1 << 7) == 0 || timer.mode & (1 << 10) != 0 {
                continue;
            }
            let before = timer.count(timer.last_check, region);
            let after = timer.count(now, region);
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
                // CMPE (bit 8) gates the INTC line, not the flag. The kernel
                // free-runs T3 with MODE 0xC83 (CMPE clear) and only enables
                // the interrupt (0x583) while its callback queue is loaded;
                // raising INTC on a disabled compare invokes the callback
                // dispatcher with an empty queue, which jumps through a NULL
                // handler on the 16-bit wrap ~4.2 s after kernel init.
                if timer.mode & (1 << 8) != 0 {
                    intc |= 1 << (9 + t);
                }
            }
        }
        intc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: Region = Region::Ntsc;

    #[test]
    fn count_follows_busclk() {
        let mut t = Timers::new();
        t.write(0x1000_0000, 0, 1000);
        // 2000 EE cycles later = 1000 BUSCLK ticks.
        assert_eq!(t.read(0x1000_0000, 3000, R), 1000);
    }

    #[test]
    fn prescaler_divides() {
        let mut t = Timers::new();
        t.write(0x1000_0010, 2, 0); // /256
        t.write(0x1000_0000, 0, 0);
        assert_eq!(t.read(0x1000_0000, 512 * 256 * 2, R), 512);
    }

    #[test]
    fn count_write_resets_base() {
        let mut t = Timers::new();
        t.write(0x1000_0000, 100, 0);
        assert_eq!(t.read(0x1000_0000, 200, R), 200);
    }

    #[test]
    fn compare_without_cmpe_sets_flag_but_no_irq() {
        let mut t = Timers::new();
        // T3 as the kernel inits it: CUE on, hblank clock, CMPE clear.
        t.write(0x1000_1810, 0xC83, 0);
        t.write(0x1000_1820, 0xFFFF, 0);
        let at_comp = 0xFFFF * R.ee_hblank_div() * 2;
        assert_eq!(t.check_irqs(at_comp, R), 0);
        // EQUF still latches as a status flag.
        assert_ne!(t.read(0x1000_1810, at_comp, R) & (1 << 10), 0);
    }

    #[test]
    fn compare_with_cmpe_fires_once() {
        let mut t = Timers::new();
        // T3 as the dispatcher arms it: CMPE set.
        t.write(0x1000_1810, 0x583, 0);
        t.write(0x1000_1820, 100, 0);
        let past = 200 * R.ee_hblank_div() * 2;
        assert_eq!(t.check_irqs(past, R), 1 << 12);
        // Latched until rearmed by a MODE write with bit 10.
        assert_eq!(t.check_irqs(past * 2, R), 0);
        t.write(0x1000_1810, 0x583, past * 2);
        assert_eq!(t.check_irqs(past * 3, R), 1 << 12);
    }
}
