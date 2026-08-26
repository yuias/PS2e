//! SPU2 (sound processor): register file, 2 MiB sound RAM, the transfer
//! engine (manual STD writes, DMA ch4/ch7, AutoDMA streaming), the IRQA
//! interrupt and a 48 kHz mixer for the 2x24 ADPCM voices plus AutoDMA
//! input, with the reverb unit (see `reverb.rs`) fed by the MMIX wet
//! paths. Transfer timing matters as much as data: libspu2 polls STATX
//! after reset and after transfers, and its AutoDMA streaming re-arms the
//! DMA from the completion interrupt, so a transfer that completes
//! instantly turns into an interrupt storm that starves every other IOP
//! thread.

mod reverb;
mod voice;

use reverb::{Reverb, ReverbRegs};
use tracing::{debug, trace};
use voice::{Voice, VoiceRegs};
use serde::{Deserialize, Serialize};

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

/// MMIX gate bits, left channel; the right channel is the bit below. Dry
/// goes to the core output, wet to the reverb input.
const MMIX_VOICE_DRY: u16 = 0x0800;
const MMIX_VOICE_WET: u16 = 0x0200;
const MMIX_INPUT_DRY: u16 = 0x0080;
const MMIX_INPUT_WET: u16 = 0x0020;
const MMIX_EXT_DRY: u16 = 0x0008;
const MMIX_EXT_WET: u16 = 0x0002;

/// Reverb registers per core: ESA, 22 address registers, EEA.
const REG_ESA: usize = 0x2E0;
const REG_EEA: usize = 0x33C;
/// Reverb coefficients (vIIR..vRIN) in the per-core volume block.
const REG_RVOL: usize = 0x774;

/// ATTR transfer modes (bits 5:4); 0 = stop, 1 = manual write.
const MODE_DMA_WRITE: u16 = 2;
const MODE_DMA_READ: u16 = 3;

#[derive(Serialize, Deserialize)]
#[derive(Default, Clone)]
struct Core {
    /// Halfword address the next transferred halfword lands at; latched
    /// from TSA when the address is written and advanced by transfers.
    tsa: u32,
    /// EE cycle at which the in-flight DMA completes (`None` when idle).
    dma_due: Option<u64>,
    /// AutoDMA blocks waiting for a free ring half (see [`Spu2::dma`]).
    adma_pending: std::collections::VecDeque<Vec<u8>>,
    /// Ring halves holding data not yet played out.
    adma_filled: [bool; 2],
    /// Ring half the next AutoDMA block is written to; blocks land in
    /// stream order, alternating halves.
    adma_write: usize,
    /// IRQA was hit and the flag is latched until IRQ enable is dropped.
    irq_flag: bool,
    /// AutoDMA input read position (0..0x200 halfwords into the L/R areas).
    adma_pos: usize,
    voices: [Voice; 24],
    reverb: Reverb,
}

#[derive(Serialize, Deserialize)]
pub struct Spu2 {
    #[serde(with = "serde_bytes")]
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
            cores: [Core::default(), Core::default()],
            irq_edge: false,
            last_sample: 0,
            out: Vec::new(),
        }
    }

    /// Earliest EE cycle at which [`Spu2::tick`] has something to do: the
    /// next output sample or a transfer completing.
    pub fn next_due(&self) -> u64 {
        let mut due = self.last_sample + EE_CYCLES_PER_SAMPLE;
        for core in &self.cores {
            if let Some(d) = core.dma_due {
                due = due.min(d);
            }
        }
        due
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

    /// Output sample index of the last mixed sample, for log timestamps.
    fn t(&self) -> u64 {
        self.last_sample / EE_CYCLES_PER_SAMPLE
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
                    debug!(target: "ps2_core::spu2", t = self.t(), core, mode = new_mode, "transfer mode");
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
                        trace!(target: "ps2_core::spu2", t = self.t(), core, voice, ssa = format_args!("{ssa:#x}"), "key on");
                    } else {
                        self.cores[core].voices[voice].key_off();
                        trace!(target: "ps2_core::spu2", t = self.t(), core, voice, "key off");
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
            REG_ADMAS => {
                debug!(target: "ps2_core::spu2", t = self.t(), core, value = format_args!("{v:#06x}"), "ADMAS");
                let was = self.adma_enabled(core);
                self.set_reg16(off, v);
                if !was && self.adma_enabled(core) {
                    // A fresh stream plays from the top of the ring so the
                    // first block is heard first and the writer stays one
                    // half ahead of the reader.
                    let c = &mut self.cores[core];
                    c.adma_pending.clear();
                    c.adma_filled = [false; 2];
                    c.adma_write = 0;
                    c.adma_pos = 0;
                }
            }
            REG_MMIX | 0x188 | 0x18C | 0x190 | 0x194 => {
                debug!(target: "ps2_core::spu2", t = self.t(), core, reg = format_args!("{local:#x}"), value = format_args!("{v:#06x}"), "mix control");
                self.set_reg16(off, v);
            }
            REG_MVOL..=0x7AF => {
                debug!(target: "ps2_core::spu2", t = self.t(), reg = format_args!("{off:#x}"), value = format_args!("{v:#06x}"), "volume");
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
                debug!(target: "ps2_core::spu2", t = self.t(), core = c, addr = format_args!("{hw:#x}"), "IRQA hit");
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
        trace!(target: "ps2_core::spu2", t = self.t(), off = format_args!("{off:#05x}"), value = format_args!("{v:#x}"), "read");
        v
    }

    /// Register write of `N` bytes at byte offset `off` (0..0x1000).
    pub fn write<const N: usize>(&mut self, off: usize, v: u32) {
        trace!(target: "ps2_core::spu2", t = self.t(), off = format_args!("{off:#05x}"), value = format_args!("{v:#x}"), "write");
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

    /// Move pending AutoDMA blocks into the ring, in stream order, as long
    /// as the write half is free (played out). Writing into the half being
    /// played only happens when the stream has run dry, as on hardware.
    fn adma_fill(&mut self, core: usize) {
        loop {
            let half = self.cores[core].adma_write;
            if self.cores[core].adma_filled[half] {
                return;
            }
            let Some(block) = self.cores[core].adma_pending.pop_front() else {
                return;
            };
            // 1 KiB block: 512 bytes left then 512 bytes right, into the
            // core's input area (L 0x2000/R 0x2200 halfwords, +0x400 for
            // core 1).
            let base = (0x2000 + (core << 10) + half * 0x100) * 2;
            let (l, r) = block.split_at(block.len().min(512));
            self.ram[base..base + l.len()].copy_from_slice(l);
            self.ram[base + 0x400..base + 0x400 + r.len()].copy_from_slice(r);
            let c = &mut self.cores[core];
            c.adma_filled[half] = true;
            c.adma_write = half ^ 1;
        }
    }

    /// EE cycle at which the last queued AutoDMA block lands in the ring:
    /// each waiting block needs one more half boundary of playback. That
    /// is when the transfer completes from the IOP's point of view, which
    /// paces the stream at exactly the playback rate.
    fn adma_due(&self, core: usize, now: u64) -> u64 {
        let c = &self.cores[core];
        let waiting = c.adma_pending.len() as u64;
        if waiting == 0 {
            return now + EE_CYCLES_PER_SAMPLE;
        }
        let until_boundary = ADMA_BLOCK_SAMPLES - (c.adma_pos as u64 & (ADMA_BLOCK_SAMPLES - 1));
        // One sample of slack so the boundary sample is mixed (and the
        // block written) before the completion is seen.
        now + (until_boundary + (waiting - 1) * ADMA_BLOCK_SAMPLES + 1) * EE_CYCLES_PER_SAMPLE
    }

    /// Kick a DMA on `core`: `data` is copied into sound RAM (`to_spu`) or
    /// filled from it. Returns the EE cycle at which the transfer completes;
    /// the caller raises the IOP DMA interrupt then. AutoDMA completes when
    /// playback has made room for the last block, plain transfers run at
    /// bus speed; a plain kick while the previous transfer is still in
    /// flight queues behind it.
    pub fn dma(&mut self, core: usize, to_spu: bool, data: &mut [u8], now: u64) -> u64 {
        let bytes = data.len() as u64;
        let due = if to_spu && self.adma_enabled(core) {
            // Blocks land in the ring as halves free up (the hardware
            // transfers behind the read position); writing them all at kick
            // time overwrote the half being played. See `adma_fill`.
            for block in data.chunks(ADMA_BLOCK_BYTES as usize) {
                self.cores[core].adma_pending.push_back(block.to_vec());
            }
            self.adma_fill(core);
            debug!(target: "ps2_core::spu2", t = self.t(), core, bytes, "ADMA block(s) queued");
            self.adma_due(core, now)
        } else {
            let start_at = self.cores[core].dma_due.map_or(now, |due| due.max(now));
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
            debug!(target: "ps2_core::spu2", t = self.t(), core, to_spu, bytes, tsa = format_args!("{start:#x}"), "DMA");
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

    /// 32-bit register pair, low halfword first (KON/KOFF/VMIX*/ENDX style).
    fn addr_reg32(&self, off: usize) -> u32 {
        u32::from(self.reg16(off)) | (u32::from(self.reg16(off + 2)) << 16)
    }

    /// 20-bit halfword address register pair (high halfword first).
    fn addr_reg(&self, off: usize) -> u32 {
        (u32::from(self.reg16(off) & 0xF) << 16) | u32::from(self.reg16(off + 2))
    }

    fn reverb_regs(&self, core: usize) -> ReverbRegs {
        let base = core * 0x400;
        let mut regs = ReverbRegs {
            esa: self.addr_reg(base + REG_ESA),
            eea: (u32::from(self.reg16(base + REG_EEA) & 0xF) << 16) | 0xFFFF,
            ..Default::default()
        };
        for (i, a) in regs.addr.iter_mut().enumerate() {
            *a = self.addr_reg(base + REG_ESA + 4 + i * 4);
        }
        for (i, v) in regs.vol.iter_mut().enumerate() {
            *v = Self::volume16(self.reg16(REG_RVOL + core * CORE_VOL_STRIDE + i * 2));
        }
        regs
    }

    /// Sum of a stereo pair into `dry`/`wet` under an MMIX gate pair
    /// (`gate` is the left bit, the right bit is the one below).
    fn route(mmix: u16, gate: u16, l: i32, r: i32, acc: &mut [i32; 2]) {
        if mmix & gate != 0 {
            acc[0] += l;
        }
        if mmix & (gate >> 1) != 0 {
            acc[1] += r;
        }
    }

    /// One 48 kHz output sample: voices and AutoDMA input per core routed
    /// dry and wet by MMIX, reverb at EVOL, master volume, core 0 folded
    /// into core 1 as its "external input" (AVOL).
    fn mix_sample(&mut self) {
        let mut ext = [0i32; 2];
        let mut final_out = [0i32; 2];
        for c in 0..2 {
            let base = c * 0x400;
            let attr = self.attr(c);
            let mmix = self.reg16(REG_MMIX + base);
            let irq_enabled = attr & 0x40 != 0;
            let irqa = self.irqa(c);
            let vmixl = self.addr_reg32(REG_VMIXL + base);
            let vmixr = self.addr_reg32(REG_VMIXR + base);
            let vmixel = self.addr_reg32(REG_VMIXL + base + 4);
            let vmixer = self.addr_reg32(REG_VMIXR + base + 4);
            let mut dry = [0i32; 2];
            let mut wet = [0i32; 2];
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
                        debug!(target: "ps2_core::spu2", t = self.t(), core = c, voice = v, addr = format_args!("{irqa:#x}"), "IRQA hit by voice");
                    }
                }
                if sample == 0 {
                    continue;
                }
                let bit = 1u32 << v;
                let l = if (vmixl | vmixel) & bit != 0 { (sample * Self::volume(self.reg16(vb))) >> 15 } else { 0 };
                let r = if (vmixr | vmixer) & bit != 0 { (sample * Self::volume(self.reg16(vb + 2))) >> 15 } else { 0 };
                if vmixl & bit != 0 && mmix & MMIX_VOICE_DRY != 0 {
                    dry[0] += l;
                }
                if vmixr & bit != 0 && mmix & (MMIX_VOICE_DRY >> 1) != 0 {
                    dry[1] += r;
                }
                if vmixel & bit != 0 && mmix & MMIX_VOICE_WET != 0 {
                    wet[0] += l;
                }
                if vmixer & bit != 0 && mmix & (MMIX_VOICE_WET >> 1) != 0 {
                    wet[1] += r;
                }
            }
            let vol_base = REG_MVOL + c * CORE_VOL_STRIDE;
            if self.adma_enabled(c) {
                let pos = self.cores[c].adma_pos;
                let lb = (0x2000 + (c << 10) + pos) * 2;
                let rb = lb + 0x400;
                let il = i32::from(i16::from_le_bytes([self.ram[lb], self.ram[lb + 1]]));
                let ir = i32::from(i16::from_le_bytes([self.ram[rb], self.ram[rb + 1]]));
                let next = (pos + 1) & 0x1FF;
                self.cores[c].adma_pos = next;
                if next & 0xFF == 0 {
                    // Left a half: it can take the next queued block.
                    self.cores[c].adma_filled[pos >> 8] = false;
                    self.adma_fill(c);
                }
                let il = (il * Self::volume16(self.reg16(vol_base + 12))) >> 15;
                let ir = (ir * Self::volume16(self.reg16(vol_base + 14))) >> 15;
                Self::route(mmix, MMIX_INPUT_DRY, il, ir, &mut dry);
                Self::route(mmix, MMIX_INPUT_WET, il, ir, &mut wet);
            }
            if c == 1 {
                let el = (ext[0] * Self::volume16(self.reg16(vol_base + 8))) >> 15;
                let er = (ext[1] * Self::volume16(self.reg16(vol_base + 10))) >> 15;
                Self::route(mmix, MMIX_EXT_DRY, el, er, &mut dry);
                Self::route(mmix, MMIX_EXT_WET, el, er, &mut wet);
            }
            // Reverb: ATTR bit 7 enables the unit; EVOL scales its output.
            if attr & 0x80 != 0 {
                let regs = self.reverb_regs(c);
                let Self { ram, cores, .. } = self;
                let rv = cores[c].reverb.sample(ram, &regs, wet[0], wet[1]);
                dry[0] += (rv[0] * Self::volume16(self.reg16(vol_base + 4))) >> 15;
                dry[1] += (rv[1] * Self::volume16(self.reg16(vol_base + 6))) >> 15;
            }
            let mvoll = Self::volume(self.reg16(vol_base));
            let mvolr = Self::volume(self.reg16(vol_base + 2));
            let out = [(dry[0] * mvoll) >> 15, (dry[1] * mvolr) >> 15];
            if c == 0 {
                ext = out;
            } else {
                final_out = out;
            }
        }
        self.out.push(final_out[0].clamp(-0x8000, 0x7FFF) as i16);
        self.out.push(final_out[1].clamp(-0x8000, 0x7FFF) as i16);
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
        spu.write::<2>(REG_MMIX, u32::from(MMIX_VOICE_DRY | MMIX_VOICE_DRY >> 1));
        // Core 0 reaches the output through core 1's external input.
        spu.write::<2>(REG_MMIX + 0x400, u32::from(MMIX_EXT_DRY | MMIX_EXT_DRY >> 1));
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE, 0x3FFF);
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE + 2, 0x3FFF);
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
        // Both halves free: a 2-block kick lands at once.
        let mut data = vec![0u8; 2048];
        let due = spu.dma(0, true, &mut data, 0);
        assert_eq!(due, EE_CYCLES_PER_SAMPLE);
        // The next kick waits for playback to free both halves: the second
        // block lands at the second boundary.
        let due = spu.dma(0, true, &mut data, 0);
        assert_eq!(due, (2 * ADMA_BLOCK_SAMPLES + 1) * EE_CYCLES_PER_SAMPLE);
    }

    #[test]
    fn adma_plays_blocks_in_order() {
        let mut spu = Spu2::new();
        spu.write::<2>(REG_ADMAS, 1);
        spu.write::<2>(REG_MMIX, u32::from(MMIX_INPUT_DRY | MMIX_INPUT_DRY >> 1));
        // Core 0 reaches the output through core 1's external input.
        spu.write::<2>(REG_MMIX + 0x400, u32::from(MMIX_EXT_DRY | MMIX_EXT_DRY >> 1));
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE, 0x3FFF);
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE + 2, 0x3FFF);
        spu.write::<2>(REG_MVOL, 0x3FFF);
        spu.write::<2>(REG_MVOL + 2, 0x3FFF);
        // BVOL for core 0 and AVOL on core 1 (core 0 feeds core 1's input).
        spu.write::<2>(REG_MVOL + 12, 0x7FFF);
        spu.write::<2>(REG_MVOL + 14, 0x7FFF);
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE + 8, 0x7FFF);
        spu.write::<2>(REG_MVOL + CORE_VOL_STRIDE + 10, 0x7FFF);
        // Block k carries the constant sample (k+1)*1000 on both channels.
        let block = |k: i16| -> Vec<u8> { (0..512).flat_map(|_| ((k + 1) * 1000).to_le_bytes()).collect() };
        let mut now = 0;
        let mut expect = vec![];
        for k in 0..8i16 {
            let mut data: Vec<u8> = [block(2 * k), block(2 * k + 1)].concat();
            let due = spu.dma(0, true, &mut data, now);
            // The IOP re-arms from the completion interrupt, a little late.
            now = due + 3000;
            spu.tick(now);
            expect.extend([(2 * k + 1) * 1000, (2 * k + 2) * 1000]);
        }
        spu.tick(now + 16 * ADMA_BLOCK_SAMPLES * EE_CYCLES_PER_SAMPLE);
        let out = spu.take_output();
        // Left channel, at 256-sample block boundaries: every block
        // appears once, in order, and no sample inside a block is stale.
        for (i, s) in out.iter().step_by(2).take(16 * 256).enumerate() {
            // Volumes: 0x7FFF/0x8000 twice then MVOL 0x7FFE/0x8000, so
            // about 0.3% below the input.
            let want = expect[i / 256];
            assert!((s - want).abs() <= want / 200 + 1, "sample {i}: {s} vs {want}");
        }
    }
}
