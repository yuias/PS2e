//! SPU2 voice: ADPCM block decoding, pitch stepping and the ADSR envelope.
//! Same scheme as the PS1 SPU (16-byte blocks of 28 nibbles, 4-bit shift +
//! filter, loop flags), with SPU2's 20-bit halfword addressing.

/// ADPCM prediction filter coefficients (x64), indexed by the block's
/// filter nibble.
const FILTERS: [(i32, i32); 5] = [(0, 0), (60, 0), (115, -52), (98, -55), (122, -60)];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Off,
    Attack,
    Decay,
    Sustain,
    Release,
}

#[derive(Clone, Copy)]
pub struct Voice {
    /// Halfword address of the block being played (NAX) and its loop
    /// target (LSAX).
    pub nax: u32,
    pub lsax: u32,
    /// LSAX was written by software after key-on, so block loop-start
    /// flags must not move it.
    pub lsax_pinned: bool,
    /// Decoded samples of the current block and the read position within.
    block: [i16; 28],
    pos: usize,
    /// Last two output samples, for the prediction filter.
    hist: [i32; 2],
    /// Pitch phase accumulator (12 fractional bits).
    counter: u32,
    pub phase: Phase,
    /// ADSR level, 0..0x7FFF.
    pub level: i32,
    /// Samples to wait before the next envelope step.
    env_wait: u32,
    /// Block flags of the current block (bit 0 end, bit 1 repeat).
    flags: u8,
    /// Set when the block that just ended carried the end flag (ENDX).
    pub endx: bool,
}

impl Default for Voice {
    fn default() -> Self {
        Self {
            nax: 0,
            lsax: 0,
            lsax_pinned: false,
            block: [0; 28],
            pos: 28,
            hist: [0; 2],
            counter: 0,
            phase: Phase::Off,
            level: 0,
            env_wait: 0,
            flags: 0,
            endx: false,
        }
    }
}

/// Everything a voice needs from the register file for one sample.
#[derive(Clone, Copy)]
pub struct VoiceRegs {
    pub pitch: u16,
    pub adsr1: u16,
    pub adsr2: u16,
}

impl Voice {
    /// Key on: restart from `ssa` with a fresh envelope.
    pub fn key_on(&mut self, ssa: u32) {
        self.nax = ssa & 0xF_FFFF;
        self.lsax = self.nax;
        self.lsax_pinned = false;
        self.pos = 28;
        self.hist = [0; 2];
        self.counter = 0;
        self.phase = Phase::Attack;
        self.level = 0;
        self.env_wait = 0;
        self.flags = 0;
        self.endx = false;
    }

    pub fn key_off(&mut self) {
        if self.phase != Phase::Off {
            self.phase = Phase::Release;
            self.env_wait = 0;
        }
    }

    #[cfg(test)]
    pub fn active(&self) -> bool {
        self.phase != Phase::Off
    }

    /// Advance one output sample. Returns the sample after the envelope
    /// (16-bit range) and whether a new block was fetched at `nax` (for
    /// IRQA checks by the caller, which sees the updated `nax`).
    pub fn step(&mut self, ram: &[u8], regs: VoiceRegs) -> (i32, bool) {
        if self.phase == Phase::Off {
            return (0, false);
        }
        let mut fetched = false;
        // Pitch: 0x1000 = 1 sample per sample, capped at 4x.
        self.counter += u32::from(regs.pitch.min(0x3FFF));
        while self.counter >= 0x1000 {
            self.counter -= 0x1000;
            self.pos += 1;
            if self.pos >= 28 {
                self.pos = 0;
                if self.next_block(ram) {
                    fetched = true;
                }
                if self.phase == Phase::Off {
                    return (0, fetched);
                }
            }
        }
        let sample = i32::from(self.block[self.pos.min(27)]);
        self.step_envelope(regs);
        ((sample * self.level) >> 15, fetched)
    }

    /// Finish the current block (apply its loop flags) and decode the next
    /// one. Returns false when the voice stopped instead of fetching.
    fn next_block(&mut self, ram: &[u8]) -> bool {
        // Flags of the block that just finished.
        if self.flags & 1 != 0 {
            self.endx = true;
            if self.flags & 2 != 0 {
                self.nax = self.lsax;
            } else {
                // End without repeat: jump to the loop address and go
                // silent, like the PS1 SPU.
                self.nax = self.lsax;
                self.phase = Phase::Off;
                self.level = 0;
                return false;
            }
        }
        self.decode_block(ram);
        true
    }

    fn decode_block(&mut self, ram: &[u8]) {
        let base = (self.nax as usize & 0xF_FFFF) * 2;
        let hdr = ram[base];
        self.flags = ram[base + 1];
        let shift = (hdr & 0xF).min(12) as i32;
        let (f0, f1) = FILTERS[usize::from((hdr >> 4).min(4))];
        for i in 0..28 {
            let byte = ram[base + 2 + i / 2];
            let nibble = if i & 1 == 0 { byte & 0xF } else { byte >> 4 };
            let raw = ((i32::from(nibble) << 28) >> 28) << 12; // sign-extend to 16-bit range
            let mut s = (raw >> shift) + ((self.hist[0] * f0 + self.hist[1] * f1 + 32) >> 6);
            s = s.clamp(-0x8000, 0x7FFF);
            self.hist[1] = self.hist[0];
            self.hist[0] = s;
            self.block[i] = s as i16;
        }
        // Loop-start flag latches the block address unless software pinned
        // LSAX after key-on.
        if self.flags & 4 != 0 && !self.lsax_pinned {
            self.lsax = self.nax;
        }
        self.nax = (self.nax + 8) & 0xF_FFFF;
    }

    /// PS1-style envelope stepping (nocash "ADSR operation").
    fn step_envelope(&mut self, regs: VoiceRegs) {
        if self.env_wait > 0 {
            self.env_wait -= 1;
            return;
        }
        let (shift, step, exp, decrease) = match self.phase {
            Phase::Attack => {
                let shift = i32::from((regs.adsr1 >> 10) & 0x1F);
                let step = 7 - i32::from((regs.adsr1 >> 8) & 3);
                (shift, step, regs.adsr1 & 0x8000 != 0, false)
            }
            Phase::Decay => {
                let shift = i32::from((regs.adsr1 >> 4) & 0xF);
                (shift, -8, true, true)
            }
            Phase::Sustain => {
                let shift = i32::from((regs.adsr2 >> 8) & 0x1F);
                let dec = regs.adsr2 & 0x4000 != 0;
                let step = if dec {
                    -8 + i32::from((regs.adsr2 >> 6) & 3)
                } else {
                    7 - i32::from((regs.adsr2 >> 6) & 3)
                };
                (shift, step, regs.adsr2 & 0x8000 != 0, dec)
            }
            Phase::Release => {
                let shift = i32::from(regs.adsr2 & 0x1F);
                (shift, -8, regs.adsr2 & 0x20 != 0, true)
            }
            Phase::Off => return,
        };
        let mut cycles = 1u32 << (shift - 11).max(0);
        let mut step = step << (11 - shift).max(0);
        if exp && !decrease && self.level > 0x6000 {
            cycles *= 4;
        }
        if exp && decrease {
            step = (step * self.level) >> 15;
        }
        self.env_wait = cycles - 1;
        self.level = (self.level + step).clamp(0, 0x7FFF);
        match self.phase {
            Phase::Attack if self.level >= 0x7FFF => {
                self.phase = Phase::Decay;
                self.env_wait = 0;
            }
            Phase::Decay => {
                let sustain_level = (i32::from(regs.adsr1 & 0xF) + 1) * 0x800;
                if self.level <= sustain_level {
                    self.phase = Phase::Sustain;
                    self.env_wait = 0;
                }
            }
            Phase::Release if self.level == 0 => self.phase = Phase::Off,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn regs() -> VoiceRegs {
        // Fastest attack/decay/release, sustain level max.
        VoiceRegs { pitch: 0x1000, adsr1: 0x000F, adsr2: 0x0000 }
    }

    /// One block: shift 12 (unit scale), no filter, nibbles all 1 → +0x1000
    /// each... at shift 12 that is 1<<12>>12 = 1. Use shift 0 for full scale.
    fn block(flags: u8, nibble: u8) -> [u8; 16] {
        let mut b = [nibble | (nibble << 4); 16];
        b[0] = 0x00; // shift 0, filter 0
        b[1] = flags;
        b
    }

    #[test]
    fn plays_block_and_loops_on_repeat_flag() {
        let mut ram = vec![0u8; 0x100];
        ram[0x10..0x20].copy_from_slice(&block(0x06, 0x1)); // loop start + repeat... plus end below
        ram[0x20..0x30].copy_from_slice(&block(0x03, 0x1)); // end + repeat
        let mut v = Voice::default();
        v.key_on(0x8); // halfword 8 = byte 0x10
        let r = regs();
        let mut fetched_at = vec![];
        for _ in 0..28 * 3 {
            let (_, f) = v.step(&ram, r);
            if f {
                fetched_at.push(v.nax);
            }
        }
        // First fetch decodes block @8 (nax -> 0x10), second @0x10 (nax ->
        // 0x18), then the end+repeat flag jumps back to LSAX (0x8) and the
        // third fetch decodes it again.
        assert_eq!(fetched_at, vec![0x10, 0x18, 0x10]);
        assert!(v.endx);
        assert!(v.active());
    }

    #[test]
    fn end_without_repeat_silences_voice() {
        let mut ram = vec![0u8; 0x100];
        ram[0..16].copy_from_slice(&block(0x01, 0x1));
        let mut v = Voice::default();
        v.key_on(0);
        let r = regs();
        for _ in 0..60 {
            v.step(&ram, r);
        }
        assert!(v.endx);
        assert!(!v.active());
    }

    #[test]
    fn envelope_attacks_and_releases() {
        let mut ram = vec![0u8; 0x100];
        ram[0..16].copy_from_slice(&block(0x03, 0x1)); // end + repeat: loop forever
        let mut v = Voice::default();
        v.key_on(0);
        let r = regs();
        for _ in 0..1000 {
            v.step(&ram, r);
        }
        assert!(v.level > 0x7000, "level {:#x}", v.level);
        v.key_off();
        for _ in 0..1000 {
            v.step(&ram, r);
        }
        assert_eq!(v.phase, Phase::Off);
    }
}
