//! SPU2 (sound processor): register file, 2 MiB sound RAM, the transfer
//! engine (manual STD writes, DMA ch4/ch7, AutoDMA streaming) and the IRQA
//! interrupt. No voice mixing yet — the point is to give IOP sound drivers
//! the status bits and completion timing they wait on: libspu2 polls STATX
//! after reset and after transfers, and its AutoDMA streaming re-arms the
//! DMA from the completion interrupt, so a transfer that completes
//! instantly turns into an interrupt storm that starves every other IOP
//! thread.

use tracing::{debug, trace};

pub const SPU2_RAM_SIZE: usize = 2 * 1024 * 1024;

/// EE cycles per output sample (48 kHz at 294.912 MHz).
const EE_CYCLES_PER_SAMPLE: u64 = 6144;
/// Plain DMA moves roughly one halfword per 4 IOP cycles.
const EE_CYCLES_PER_DMA_BYTE: u64 = 16;
/// AutoDMA: a 1 KiB block holds 256 stereo samples, consumed at 48 kHz.
const ADMA_BLOCK_BYTES: u64 = 1024;
const ADMA_BLOCK_SAMPLES: u64 = 256;

const REG_ATTR: usize = 0x19A;
const REG_IRQAH: usize = 0x19C;
const REG_IRQAL: usize = 0x19E;
const REG_TSAH: usize = 0x1A8;
const REG_TSAL: usize = 0x1AA;
const REG_STD: usize = 0x1AC;
const REG_ADMAS: usize = 0x1B0;
const REG_STATX: usize = 0x344;
const REG_IRQINFO: usize = 0x7C2;

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
}

pub struct Spu2 {
    pub ram: Box<[u8]>,
    /// Halfword register file mirroring 0x1F900000..0x1F901000.
    regs: Box<[u16]>,
    cores: [Core; 2],
    /// Set when the SPU IRQ line should be raised; drained by the bus.
    irq_edge: bool,
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
        }
    }

    /// Take the pending SPU IRQ edge (IOP I_STAT bit 9).
    pub fn take_irq(&mut self) -> bool {
        core::mem::take(&mut self.irq_edge)
    }

    fn core_of(off: usize) -> usize {
        usize::from(off >= 0x400 && off < 0x800)
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
        match off & 0xFFE {
            REG_STATX => self.statx(0),
            o if o == REG_STATX + 0x400 => self.statx(1),
            REG_IRQINFO => self.irqinfo(),
            o if o == REG_STD || o == REG_STD + 0x400 => {
                let core = Self::core_of(o);
                let hw = self.cores[core].tsa as usize & 0xF_FFFF;
                u16::from_le_bytes([self.ram[hw * 2], self.ram[hw * 2 + 1]])
            }
            _ => self.reg16(off),
        }
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

    /// Whether the DMA for `core` is still in flight.
    pub fn dma_busy(&self, core: usize) -> bool {
        self.cores[core].dma_due.is_some()
    }

    /// Kick a DMA on `core`: `data` is copied into sound RAM (`to_spu`) or
    /// filled from it. Returns the EE cycle at which the transfer completes;
    /// the caller raises the IOP DMA interrupt then. AutoDMA blocks are
    /// paced at playback speed, plain transfers at bus speed.
    pub fn dma(&mut self, core: usize, to_spu: bool, data: &mut [u8], now: u64) -> u64 {
        let bytes = data.len() as u64;
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
            now + blocks * ADMA_BLOCK_SAMPLES * EE_CYCLES_PER_SAMPLE
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
            now + bytes * EE_CYCLES_PER_DMA_BYTE
        };
        self.cores[core].dma_due = Some(due);
        due
    }

    /// Retire DMAs whose completion time has passed; returns the cores
    /// whose IOP DMA channel (4 for core 0, 7 for core 1) just finished.
    pub fn tick(&mut self, now: u64) -> [bool; 2] {
        let mut done = [false; 2];
        for (c, core) in self.cores.iter_mut().enumerate() {
            if core.dma_due.is_some_and(|due| due <= now) {
                core.dma_due = None;
                done[c] = true;
            }
        }
        done
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
    fn adma_paces_at_playback_rate() {
        let mut spu = Spu2::new();
        spu.write::<2>(REG_ADMAS, 1);
        let mut data = vec![0u8; 2048];
        let due = spu.dma(0, true, &mut data, 0);
        assert_eq!(due, 2 * ADMA_BLOCK_SAMPLES * EE_CYCLES_PER_SAMPLE);
    }
}
