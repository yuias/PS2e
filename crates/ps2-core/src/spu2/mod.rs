//! SPU2 (sound processor): register file, 2 MiB sound RAM, the transfer
//! engine (manual STD writes, DMA ch4/ch7, AutoDMA streaming), the IRQA
//! interrupt and a 48 kHz mixer for the 2x24 ADPCM voices plus AutoDMA
//! input. Transfer timing matters as much as data: libspu2 polls STATX
//! after reset and after transfers, and its AutoDMA streaming re-arms the
//! DMA from the completion interrupt, so a transfer that completes
//! instantly turns into an interrupt storm that starves every other IOP
//! thread.

mod voice;

use tracing::{debug, trace};
use voice::{Voice, VoiceRegs};

pub const SPU2_RAM_SIZE: usize = 2 * 1024 * 1024;

/// EE cycles per output sample (48 kHz at 294.912 MHz).
const EE_CYCLES_PER_SAMPLE: u64 = 6144;
/// Plain DMA moves roughly one halfword per 4 IOP cycles.
const EE_CYCLES_PER_DMA_BYTE: u64 = 16;
/// AutoDMA: a 1 KiB block holds 256 stereo samples, consumed at 48 kHz.
const ADMA_BLOCK_BYTES: u64 = 1024;
const ADMA_BLOCK_SAMPLES: u64 = 256;

const REG_VMIXL: usize = 0x188;
const REG_VMIXR: usize = 0x190;
const REG_MMIX: usize = 0x198;
const REG_ATTR: usize = 0x19A;
const REG_IRQAH: usize = 0x19C;
const REG_IRQAL: usize = 0x19E;
const REG_TSAH: usize = 0x1A8;
const REG_TSAL: usize = 0x1AA;
const REG_KON: usize = 0x1A0;
const REG_KOFF: usize = 0x1A4;
const REG_STD: usize = 0x1AC;
const REG_ADMAS: usize = 0x1B0;
/// Per-voice address block (SSAH/SSAL, LSAXH/LSAXL, NAXH/NAXL), 12 bytes each.
const REG_VADDR: usize = 0x1C0;
const REG_ENDX: usize = 0x340;
const REG_STATX: usize = 0x344;
/// Per-core volume block: MVOLL/R, EVOLL/R, AVOLL/R (external input = the
/// other core's output), BVOLL/R (AutoDMA input), MVOLXL/R.
const REG_MVOL: usize = 0x760;
const CORE_VOL_STRIDE: usize = 0x28;
const REG_IRQINFO: usize = 0x7C2;

/// MMIX gates for the dry paths we mix (no reverb): voices and AutoDMA input.
const MMIX_VOICE_DRY: u16 = 0x0C00;
const MMIX_INPUT_DRY: u16 = 0x00C0;

/// ATTR transfer modes (bits 5:4); 0 = stop, 1 = manual write.
const MODE_DMA_WRITE: u16 = 2;
const MODE_DMA_READ: u16 = 3;

#[derive(Default, Clone, Copy)]
struct Core {
    /// Halfword address the next transferred halfword lands at; latched
    /// from TSA when the address is written and advanced by transfers.
    tsa: u32,
    /// EE cycle at which the in-flight DMA completes (`None` when idle).
    dma_due: Option<u64>,
    /// AutoDMA half currently being filled (blocks alternate 0/1).
    adma_half: usize,
    /// IRQA was hit and the flag is latched until IRQ enable is dropped.
    irq_flag: bool,
    /// AutoDMA input read position (0..0x200 halfwords into the L/R areas).
    adma_pos: usize,
    voices: [Voice; 24],
}

pub struct Spu2 {
    pub ram: Box<[u8]>,
    /// Halfword register file mirroring 0x1F900000..0x1F901000.
    regs: Box<[u16]>,
    cores: [Core; 2],
    /// Set when the SPU IRQ line should be raised; drained by the bus.
    irq_edge: bool,
    /// EE cycle of the last mixed output sample.
    last_sample: u64,
    /// Mixed output, interleaved stereo 16-bit at 48 kHz; the front-end
    /// drains it.
    pub out: Vec<i16>,
}

impl Default for Spu2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Spu2 {
    pub fn new() -> Self {
        Self {
            ram: vec![0u8; SPU2_RAM_SIZE].into_boxed_slice(),
            regs: vec![0u16; 0x800].into_boxed_slice(),
            cores: [Core::default(); 2],
            irq_edge: false,
            last_sample: 0,
            out: Vec::new(),
        }
    }

    /// Take the samples mixed since the last call.
    pub fn take_output(&mut self) -> Vec<i16> {
        core::mem::take(&mut self.out)
    }

    /// Raw register file (halfwords for 0x1F900000..0x1F901000), for dumps.
    pub fn regs_bytes(&self) -> Vec<u8> {
        self.regs.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Take the pending SPU IRQ edge (IOP I_STAT bit 9).
    pub fn take_irq(&mut self) -> bool {
        core::mem::take(&mut self.irq_edge)
    }

    /// Core 1's block mirrors core 0's at +0x400; 0x760.. is global.
    fn core_of(off: usize) -> usize {
        usize::from((0x400..0x760).contains(&off))
    }

    fn reg16(&self, off: usize) -> u16 {
        self.regs[(off & 0xFFE) >> 1]
    }

    fn set_reg16(&mut self, off: usize, v: u16) {
        self.regs[(off & 0xFFE) >> 1] = v;
    }

    fn attr(&self, core: usize) -> u16 {
        self.reg16(REG_ATTR + core * 0x400)
    }

    fn mode(&self, core: usize) -> u16 {
        (self.attr(core) >> 4) & 3
    }

    fn irqa(&self, core: usize) -> u32 {
        let base = core * 0x400;
        (u32::from(self.reg16(REG_IRQAH + base) & 0xF) << 16) | u32::from(self.reg16(REG_IRQAL + base))
    }

    /// STATX mirrors the PS1 SPUSTAT layout: bits 5:0 echo ATTR, bit 6 the
    /// IRQ flag, bit 7 the DMA request (set while a DMA mode is selected),
    /// bits 8/9 read/write request, bit 10 transfer busy.
    fn statx(&self, core: usize) -> u16 {
        let attr = self.attr(core);
        let mode = (attr >> 4) & 3;
        let mut v = attr & 0x3F;
        if self.cores[core].irq_flag {
            v |= 1 << 6;
        }
        if mode == MODE_DMA_WRITE || mode == MODE_DMA_READ {
            v |= 1 << 7;
        }
        if mode == MODE_DMA_READ {
            v |= 1 << 8;
        }
        if mode == MODE_DMA_WRITE {
            v |= 1 << 9;
        }
        if self.cores[core].dma_due.is_some() {
            v |= 1 << 10;
        }
        v
    }

    fn irqinfo(&self) -> u16 {
        let mut v = 0;
        for (c, core) in self.cores.iter().enumerate() {
            if core.irq_flag {
                v |= 4 << c;
            }
        }
        v
    }

    fn read16(&self, off: usize) -> u16 {
        let core = Self::core_of(off);
        let local = (off & 0xFFE) - core * 0x400;
        match local {
            // Voice ENVX: live envelope level.
            0x000..=0x17F if local & 0xF == 0xA => {
                self.cores[core].voices[local >> 4].level as u16
            }
            // NAX: current block address.
            REG_VADDR..=0x2DF if (local - REG_VADDR) % 12 >= 8 => {
                let v = (local - REG_VADDR) / 12;
                let nax = self.cores[core].voices[v].nax;
                if (local - REG_VADDR) % 12 == 8 { (nax >> 16) as u16 } else { nax as u16 }
            }
            REG_ENDX => self.endx(core) as u16,
            o if o == REG_ENDX + 2 => (self.endx(core) >> 16) as u16,
            REG_STATX => self.statx(core),
            REG_IRQINFO => self.irqinfo(),
            REG_STD => {
                let hw = self.cores[core].tsa as usize & 0xF_FFFF;
                u16::from_le_bytes([self.ram[hw * 2], self.ram[hw * 2 + 1]])
            }
            _ => self.reg16(off),
        }
    }

    fn endx(&self, core: usize) -> u32 {
        self.cores[core]
            .voices
            .iter()
            .enumerate()
            .fold(0, |acc, (i, v)| acc | (u32::from(v.endx) << i))
    }

    fn write16(&mut self, off: usize, v: u16) {
        let off = off & 0xFFE;
        let core = Self::core_of(off);
        let local = off - core * 0x400;
        match local {
            REG_ATTR => {
                let old = self.attr(core);
                self.set_reg16(off, v);
                let new_mode = (v >> 4) & 3;
                if new_mode != (old >> 4) & 3 {
                    debug!(target: "ps2_core::spu2", core, mode = new_mode, "transfer mode");
                }
                // Dropping IRQ enable acknowledges the IRQ.
                if old & 0x40 != 0 && v & 0x40 == 0 {
                    self.cores[core].irq_flag = false;
                }
            }
            REG_TSAH | REG_TSAL => {
                self.set_reg16(off, v);
                let base = core * 0x400;
                self.cores[core].tsa = (u32::from(self.reg16(REG_TSAH + base) & 0xF) << 16)
                    | u32::from(self.reg16(REG_TSAL + base));
            }
            REG_STD => {
                // Manual write: the FIFO drains instantly, so STATX never
                // reports busy for it.
                if self.mode(core) != MODE_DMA_READ {
                    self.write_halfword(core, v);
                }
            }
            REG_KON | REG_KOFF | 0x1A2 | 0x1A6 => {
                let on = local < REG_KOFF;
                let first = if local & 2 != 0 { 16 } else { 0 };
                let base = core * 0x400 + REG_VADDR;
                for i in 0..16 {
                    let voice = first + i;
                    if v & (1 << i) == 0 || voice >= 24 {
                        continue;
                    }
                    if on {
                        let a = base + voice * 12;
                        let ssa = (u32::from(self.reg16(a) & 0xF) << 16) | u32::from(self.reg16(a + 2));
                        self.cores[core].voices[voice].key_on(ssa);
                        trace!(target: "ps2_core::spu2", core, voice, ssa = format_args!("{ssa:#x}"), "key on");
                    } else {
                        self.cores[core].voices[voice].key_off();
                    }
                }
            }
            REG_VADDR..=0x2DF => {
                self.set_reg16(off, v);
                let v_idx = (local - REG_VADDR) / 12;
                let field = (local - REG_VADDR) % 12;
                if field == 4 || field == 6 {
                    // LSAX written by software pins the loop point.
                    let a = core * 0x400 + REG_VADDR + v_idx * 12 + 4;
                    let lsax = (u32::from(self.reg16(a) & 0xF) << 16) | u32::from(self.reg16(a + 2));
                    let voice = &mut self.cores[core].voices[v_idx];
                    voice.lsax = lsax;
                    voice.lsax_pinned = true;
                }
            }
            REG_MMIX | REG_ADMAS | 0x188 | 0x18C | 0x190 | 0x194 => {
                debug!(target: "ps2_core::spu2", core, reg = format_args!("{local:#x}"), value = format_args!("{v:#06x}"), "mix control");
                self.set_reg16(off, v);
            }
            REG_MVOL..=0x7AF => {
                debug!(target: "ps2_core::spu2", reg = format_args!("{off:#x}"), value = format_args!("{v:#06x}"), "volume");
                self.set_reg16(off, v);
            }
            _ => self.set_reg16(off, v),
        }
    }

    fn write_halfword(&mut self, core: usize, v: u16) {
        let hw = self.cores[core].tsa as usize & 0xF_FFFF;
        self.ram[hw * 2..hw * 2 + 2].copy_from_slice(&v.to_le_bytes());
        self.check_irqa(hw as u32);
        self.cores[core].tsa = (self.cores[core].tsa + 1) & 0xF_FFFF;
    }

    /// A transfer touching IRQA raises the SPU interrupt for every core
    /// with IRQ enabled on that address.
    fn check_irqa(&mut self, hw: u32) {
        for c in 0..2 {
            if self.attr(c) & 0x40 != 0 && self.irqa(c) == hw && !self.cores[c].irq_flag {
                self.cores[c].irq_flag = true;
                self.irq_edge = true;
                debug!(target: "ps2_core::spu2", core = c, addr = format_args!("{hw:#x}"), "IRQA hit");
            }
        }
    }

    /// Register read of `N` bytes at byte offset `off` (0..0x1000).
    pub fn read<const N: usize>(&self, off: usize) -> u32 {
        let v = match N {
            1 => (self.read16(off) >> ((off & 1) * 8)) as u32 & 0xFF,
            2 => u32::from(self.read16(off)),
            _ => u32::from(self.read16(off)) | (u32::from(self.read16(off + 2)) << 16),
        };
        trace!(target: "ps2_core::spu2", off = format_args!("{off:#05x}"), value = format_args!("{v:#x}"), "read");
        v
    }

    /// Register write of `N` bytes at byte offset `off` (0..0x1000).
    pub fn write<const N: usize>(&mut self, off: usize, v: u32) {
        trace!(target: "ps2_core::spu2", off = format_args!("{off:#05x}"), value = format_args!("{v:#x}"), "write");
        match N {
            1 => {
                let shift = (off & 1) * 8;
                let cur = self.reg16(off);
                let new = (cur & !(0xFF << shift)) | ((v as u16 & 0xFF) << shift);
                self.write16(off, new);
            }
            2 => self.write16(off, v as u16),
            _ => {
                self.write16(off, v as u16);
                self.write16(off + 2, (v >> 16) as u16);
            }
        }
    }

    /// AutoDMA streaming enabled for `core` (ADMAS bit per core).
    fn adma_enabled(&self, core: usize) -> bool {
        self.reg16(REG_ADMAS + core * 0x400) & (1 << core) != 0
    }

    /// Kick a DMA on `core`: `data` is copied into sound RAM (`to_spu`) or
    /// filled from it. Returns the EE cycle at which the transfer completes;
    /// the caller raises the IOP DMA interrupt then. AutoDMA blocks are
    /// paced at playback speed, plain transfers at bus speed; a kick while
    /// the previous transfer is still in flight queues behind it.
    pub fn dma(&mut self, core: usize, to_spu: bool, data: &mut [u8], now: u64) -> u64 {
        let bytes = data.len() as u64;
        let start_at = self.cores[core].dma_due.map_or(now, |due| due.max(now));
        let due = if to_spu && self.adma_enabled(core) {
            // Each 1 KiB block: 512 bytes left then 512 bytes right, into
            // the core's input area (L 0x2000/R 0x2200 halfwords, +0x400
            // for core 1), alternating buffer halves.
            for block in data.chunks(ADMA_BLOCK_BYTES as usize) {
                let half = self.cores[core].adma_half;
                let base = (0x2000 + (core << 10) + half * 0x100) * 2;
                let (l, r) = block.split_at(block.len().min(512));
                self.ram[base..base + l.len()].copy_from_slice(l);
                self.ram[base + 0x400..base + 0x400 + r.len()].copy_from_slice(r);
                self.cores[core].adma_half ^= 1;
            }
            let blocks = bytes.div_ceil(ADMA_BLOCK_BYTES);
            debug!(target: "ps2_core::spu2", core, bytes, "ADMA block(s) queued");
            start_at + blocks * ADMA_BLOCK_SAMPLES * EE_CYCLES_PER_SAMPLE
        } else {
            let start = self.cores[core].tsa;
            if to_spu {
                for pair in data.chunks(2) {
                    let v = u16::from_le_bytes([pair[0], *pair.get(1).unwrap_or(&0)]);
                    self.write_halfword(core, v);
                }
            } else {
                for pair in data.chunks_mut(2) {
                    let hw = self.cores[core].tsa as usize & 0xF_FFFF;
                    pair.copy_from_slice(&self.ram[hw * 2..hw * 2 + pair.len()]);
                    self.check_irqa(hw as u32);
                    self.cores[core].tsa = (self.cores[core].tsa + 1) & 0xF_FFFF;
                }
            }
            debug!(target: "ps2_core::spu2", core, to_spu, bytes, tsa = format_args!("{start:#x}"), "DMA");
            start_at + bytes * EE_CYCLES_PER_DMA_BYTE
        };
        self.cores[core].dma_due = Some(due);
        due
    }

    /// Retire DMAs whose completion time has passed and mix the output
    /// samples due by `now`; returns the cores whose IOP DMA channel (4 for
    /// core 0, 7 for core 1) just finished.
    pub fn tick(&mut self, now: u64) -> [bool; 2] {
        let mut done = [false; 2];
        for (c, core) in self.cores.iter_mut().enumerate() {
            if core.dma_due.is_some_and(|due| due <= now) {
                core.dma_due = None;
                done[c] = true;
            }
        }
        // Bound the catch-up so a long stall cannot freeze us in the mixer.
        let mut pending = (now.saturating_sub(self.last_sample) / EE_CYCLES_PER_SAMPLE).min(4096);
        if pending > 0 {
            self.last_sample = now - now % EE_CYCLES_PER_SAMPLE;
        }
        while pending > 0 {
            self.mix_sample();
            pending -= 1;
        }
        done
    }

    /// Voice/master volume register: 15-bit signed (0x3FFF = max) when bit
    /// 15 is clear; sweep mode (bit 15 set) is approximated by full volume.
    fn volume(v: u16) -> i32 {
        if v & 0x8000 != 0 { 0x7FFF } else { i32::from((v << 1) as i16) }
    }

    /// EVOL/AVOL/BVOL are plain signed 16-bit (0x7FFF = max).
    fn volume16(v: u16) -> i32 {
        i32::from(v as i16)
    }

    /// One 48 kHz output sample: voices and AutoDMA input per core, master
    /// volume, core 0 folded into core 1 (its "external input", AVOL).
    fn mix_sample(&mut self) {
        let mut core_out = [[0i32; 2]; 2];
        for c in 0..2 {
            let base = c * 0x400;
            let attr = self.attr(c);
            let mmix = self.reg16(REG_MMIX + base);
            let irq_enabled = attr & 0x40 != 0;
            let irqa = self.irqa(c);
            let vmixl = u32::from(self.reg16(REG_VMIXL + base)) | (u32::from(self.reg16(REG_VMIXL + base + 2)) << 16);
            let vmixr = u32::from(self.reg16(REG_VMIXR + base)) | (u32::from(self.reg16(REG_VMIXR + base + 2)) << 16);
            let (mut l, mut r) = (0i32, 0i32);
            for v in 0..24 {
                let vb = base + v * 0x10;
                let regs = VoiceRegs {
                    pitch: self.reg16(vb + 4),
                    adsr1: self.reg16(vb + 6),
                    adsr2: self.reg16(vb + 8),
                };
                let (sample, fetched) = {
                    let Self { ram, cores, .. } = self;
                    cores[c].voices[v].step(ram, regs)
                };
                if fetched && irq_enabled {
                    // The block just fetched spans nax-8..nax.
                    let start = self.cores[c].voices[v].nax.wrapping_sub(8) & 0xF_FFFF;
                    if irqa.wrapping_sub(start) & 0xF_FFFF < 8 && !self.cores[c].irq_flag {
                        self.cores[c].irq_flag = true;
                        self.irq_edge = true;
                        debug!(target: "ps2_core::spu2", core = c, voice = v, addr = format_args!("{irqa:#x}"), "IRQA hit by voice");
                    }
                }
                if sample == 0 || mmix & MMIX_VOICE_DRY == 0 {
                    continue;
                }
                if vmixl & (1 << v) != 0 {
                    l += (sample * Self::volume(self.reg16(vb))) >> 15;
                }
                if vmixr & (1 << v) != 0 {
                    r += (sample * Self::volume(self.reg16(vb + 2))) >> 15;
                }
            }
            let vol_base = REG_MVOL + c * CORE_VOL_STRIDE;
            if self.adma_enabled(c) {
                let pos = self.cores[c].adma_pos;
                let lb = (0x2000 + (c << 10) + pos) * 2;
                let rb = lb + 0x400;
                let il = i32::from(i16::from_le_bytes([self.ram[lb], self.ram[lb + 1]]));
                let ir = i32::from(i16::from_le_bytes([self.ram[rb], self.ram[rb + 1]]));
                self.cores[c].adma_pos = (pos + 1) & 0x1FF;
                if mmix & MMIX_INPUT_DRY != 0 {
                    l += (il * Self::volume16(self.reg16(vol_base + 12))) >> 15;
                    r += (ir * Self::volume16(self.reg16(vol_base + 14))) >> 15;
                }
            }
            let mvoll = Self::volume(self.reg16(vol_base));
            let mvolr = Self::volume(self.reg16(vol_base + 2));
            core_out[c] = [(l * mvoll) >> 15, (r * mvolr) >> 15];
        }
        let vol_base = REG_MVOL + CORE_VOL_STRIDE;
        let avoll = Self::volume16(self.reg16(vol_base + 8));
        let avolr = Self::volume16(self.reg16(vol_base + 10));
        let l = core_out[1][0] + ((core_out[0][0] * avoll) >> 15);
        let r = core_out[1][1] + ((core_out[0][1] * avolr) >> 15);
        self.out.push(l.clamp(-0x8000, 0x7FFF) as i16);
        self.out.push(r.clamp(-0x8000, 0x7FFF) as i16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statx_follows_attr_and_reset_reads_idle() {
        let mut spu = Spu2::new();
        spu.write::<2>(REG_ATTR, 0x8000);
        assert_eq!(spu.read::<2>(REG_STATX) & 0x7FF, 0);
        spu.write::<2>(REG_ATTR, 0x8020);
        assert_eq!(spu.read::<2>(REG_STATX) & 0x280, 0x280);
    }

    #[test]
    fn manual_write_lands_at_tsa_and_advances() {
        let mut spu = Spu2::new();
        spu.write::<2>(REG_TSAH, 0x1);
        spu.write::<2>(REG_TSAL, 0x0004);
        spu.write::<2>(REG_ATTR, 0x8010);
        spu.write::<2>(REG_STD, 0xABCD);
        spu.write::<2>(REG_STD, 0x1234);
        let at = (0x10004usize) * 2;
        assert_eq!(&spu.ram[at..at + 4], &[0xCD, 0xAB, 0x34, 0x12]);
        assert_eq!(spu.read::<2>(REG_STATX) & 0x400, 0);
    }

    #[test]
    fn dma_write_hits_irqa_and_completes_later() {
        let mut spu = Spu2::new();
        spu.write::<2>(REG_IRQAH + 0x400, 0);
        spu.write::<2>(REG_IRQAL + 0x400, 0x0102);
        spu.write::<2>(REG_ATTR + 0x400, 0x8060);
        spu.write::<2>(REG_TSAH + 0x400, 0);
        spu.write::<2>(REG_TSAL + 0x400, 0x0100);
        let mut data = [1u8; 16];
        let due = spu.dma(1, true, &mut data, 1000);
        assert!(due > 1000);
        assert!(spu.take_irq());
        assert_eq!(spu.read::<2>(REG_IRQINFO), 8);
        assert_eq!(spu.tick(due - 1), [false, false]);
        assert_eq!(spu.tick(due), [false, true]);
        // Dropping IRQ enable clears the flag.
        spu.write::<2>(REG_ATTR + 0x400, 0x8020);
        assert_eq!(spu.read::<2>(REG_IRQINFO), 0);
    }

    #[test]
    fn keyed_voice_reaches_the_output() {
        let mut spu = Spu2::new();
        // One looping block of constant +0x1000-ish samples at halfword 0x1000.
        let base = 0x1000 * 2;
        spu.ram[base] = 0x00; // shift 0, no filter
        spu.ram[base + 1] = 0x03; // end + repeat
        for b in &mut spu.ram[base + 2..base + 16] {
            *b = 0x11;
        }
        // Voice 0 of core 0: full volume, unity pitch, instant attack, max
        // sustain; routed dry to both channels; master volumes up.
        spu.write::<2>(0x000, 0x3FFF);
        spu.write::<2>(0x002, 0x3FFF);
        spu.write::<2>(0x004, 0x1000);
        spu.write::<2>(0x006, 0x000F);
        spu.write::<2>(0x008, 0x0000);
        spu.write::<2>(REG_VADDR, 0);
        spu.write::<2>(REG_VADDR + 2, 0x1000);
        spu.write::<2>(REG_VMIXL, 1);
        spu.write::<2>(REG_VMIXR, 1);
        spu.write::<2>(REG_MMIX, u32::from(MMIX_VOICE_DRY));
        spu.write::<2>(REG_MVOL, 0x3FFF);
        spu.write::<2>(REG_MVOL + 2, 0x3FFF);
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE + 8, 0x7FFF);
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE + 10, 0x7FFF);
        spu.write::<2>(REG_KON, 1);
        spu.tick(100 * EE_CYCLES_PER_SAMPLE);
        let out = spu.take_output();
        assert_eq!(out.len(), 200);
        assert!(out[150..].iter().any(|&s| s > 0x100), "{:?}", &out[150..160]);
        assert_eq!(spu.read::<2>(0x00A) & 0x7FFF, 0x7FFF); // ENVX at sustain
        assert!(spu.read::<2>(REG_ENDX) & 1 != 0); // looped past the end flag
    }

    #[test]
    fn adma_paces_at_playback_rate() {
        let mut spu = Spu2::new();
        spu.write::<2>(REG_ADMAS, 1);
        let mut data = vec![0u8; 2048];
        let due = spu.dma(0, true, &mut data, 0);
        assert_eq!(due, 2 * ADMA_BLOCK_SAMPLES * EE_CYCLES_PER_SAMPLE);
    }
}
