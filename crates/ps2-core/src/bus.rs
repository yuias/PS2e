//! EE-side system bus: memory map, MMIO dispatch, TTY capture.
//!
//! Address translation is a direct segment fold for now: the kernel's TLB
//! mappings are essentially identity, so TLB instructions record entries
//! without remapping (see ARCHITECTURE.md).

use crate::gif::Gif;
use crate::gs::Gs;
use crate::sif::Sif;
use crate::timers::Timers;
use std::collections::HashSet;
use tracing::{debug, trace, warn};

pub const RAM_SIZE: usize = 32 * 1024 * 1024;
pub const BIOS_SIZE: usize = 4 * 1024 * 1024;
pub const SPAD_SIZE: usize = 16 * 1024;
/// Shadow register file for 0x1000_0000..0x1001_0000 MMIO.
const MMIO_SIZE: usize = 0x10000;

/// Number of RDRAM devices reported by the MCH init handshake.
const RDRAM_DEVICES: u32 = 2;

/// EE DMAC channel (only SIF0/SIF1 are modeled so far).
#[derive(Default)]
pub struct EeDmaChannel {
    pub chcr: u32,
    pub madr: u32,
    pub qwc: u32,
    pub tadr: u32,
    /// Current tag asked to stop the chain after its data.
    tag_end: bool,
}

const EE_CHCR_STR: u32 = 1 << 8;
const EE_CHCR_TTE: u32 = 1 << 6;
const EE_CHCR_TIE: u32 = 1 << 7;

/// IOP DMA channel (SIF0 = ch9, SIF1 = ch10).
#[derive(Default)]
pub struct IopDmaChannel {
    pub madr: u32,
    pub bcr: u32,
    pub chcr: u32,
    pub tadr: u32,
    /// Words remaining in the current block (SIF0 send side).
    words_left: u32,
    /// Current tag asked to end the transfer after its block.
    tag_end: bool,
    /// SIF1 receive side: words remaining for the current IOP tag.
    recv_left: u32,
    recv_addr: u32,
    /// Destination address the current packet started at.
    recv_start: u32,
    recv_end: bool,
    /// Padding words to the packet's qword boundary, dropped after the data.
    recv_pad: u32,
}

const IOP_CHCR_BUSY: u32 = 1 << 24;

/// IOP root counter (0-2: 16-bit PS1-style, 3-5: 32-bit).
#[derive(Default, Clone, Copy)]
struct IopTimer {
    base: u32,
    base_cycle: u64,
    mode: u32,
    target: u32,
    last_check: u64,
}

impl IopTimer {
    /// IOP sysclock ticks are EE cycles / 8; wide timers add a prescaler.
    fn count(&self, idx: usize, now: u64) -> u32 {
        let sys = now.saturating_sub(self.base_cycle) / 8;
        let ticks = if idx >= 3 {
            match (self.mode >> 13) & 3 {
                0 => sys,
                1 => sys / 8,
                2 => sys / 16,
                _ => sys / 256,
            }
        } else {
            sys
        };
        let count = self.base as u64 + ticks;
        if idx < 3 {
            count as u32 & 0xFFFF
        } else {
            count as u32
        }
    }
}

pub struct Bus {
    pub ram: Box<[u8]>,
    pub bios: Box<[u8]>,
    pub spad: Box<[u8]>,
    /// IOP RAM as seen from the EE at 0x1C00_0000 (2 MiB).
    pub iop_ram: Box<[u8]>,
    /// Shadow storage for EE MMIO registers we don't model yet: reads return
    /// the last written value so BIOS read-modify-write sequences behave.
    mmio: Box<[u8]>,
    pub gs: Gs,
    pub gif: Gif,
    pub timers: Timers,
    pub sif: Sif,
    /// IOP scratchpad (1 KiB at 0x1F800000).
    pub iop_spad: Box<[u8]>,
    /// Shadow storage for IOP MMIO (0x1F801000..0x1F810000), same idea as
    /// the EE shadow.
    iop_mmio: Box<[u8]>,
    /// IOP interrupt controller: I_STAT / I_MASK / I_CTRL.
    pub iop_i_stat: u32,
    pub iop_i_mask: u32,
    pub iop_i_ctrl: u32,
    /// IOP root counters.
    iop_timers: [IopTimer; 6],
    /// EE INTC.
    pub intc_stat: u32,
    pub intc_mask: u32,
    /// EE DMAC: GIF (ch2), SIF0 (ch5) and SIF1 (ch6), interrupt status/mask.
    pub dma_gif: EeDmaChannel,
    pub dma_sif0: EeDmaChannel,
    pub dma_sif1: EeDmaChannel,
    pub d_stat: u32,
    pub d_mask: u32,
    /// IOP DMA: SIF0 (ch9), SIF1 (ch10), interrupt control.
    pub iop_dma_sif0: IopDmaChannel,
    pub iop_dma_sif1: IopDmaChannel,
    pub iop_dicr: u32,
    pub iop_dicr2: u32,
    /// Current EE cycle count, updated by the system before each step.
    pub now: u64,
    /// Kernel TTY output captured from the EE SIO TXFIFO (observation only).
    pub tty_buffer: String,
    /// Current TTY line, flushed to the log on '\n'.
    tty_line: String,
    /// RDRAM init handshake state (MCH_RICM/MCH_DRD).
    rdram_sdevid: u32,
    /// Unmapped addresses already reported, to keep the log readable.
    warned_unmapped: HashSet<u32>,
    /// EE TLB entries (raw registers) and a 4 KiB-granular lookup cache.
    ee_tlb: [(u32, u32, u32, u32); 48],
    /// (vaddr page | 1) -> phys page; 0 = invalid slot.
    tlb_cache: Box<[(u32, u32)]>,
    /// Deferred EE DMAC completion interrupts: (D_STAT bit, due cycle).
    /// Data moves instantly but completion must not fire inside the very
    /// instruction that started the transfer.
    dma_irq_queue: Vec<(u32, u64)>,
}

impl Bus {
    pub fn new(bios: Vec<u8>) -> Self {
        assert_eq!(bios.len(), BIOS_SIZE);
        let mut mmio = vec![0u8; MMIO_SIZE].into_boxed_slice();
        // DMAC ENABLER resets to 0x1201; the BIOS uses it as a board-revision
        // key into its RDRAM configuration table during InitRDRAM.
        write_le::<4>(&mut mmio, 0xF590, 0x1201);
        Self {
            ram: vec![0u8; RAM_SIZE].into_boxed_slice(),
            bios: bios.into_boxed_slice(),
            spad: vec![0u8; SPAD_SIZE].into_boxed_slice(),
            iop_ram: vec![0u8; 2 * 1024 * 1024].into_boxed_slice(),
            mmio,
            gs: Gs::new(),
            gif: Gif::new(),
            timers: Timers::new(),
            sif: Sif::new(),
            iop_spad: vec![0u8; 1024].into_boxed_slice(),
            iop_mmio: vec![0u8; 0x10000].into_boxed_slice(),
            iop_i_stat: 0,
            iop_i_mask: 0,
            iop_i_ctrl: 0,
            iop_timers: [IopTimer::default(); 6],
            intc_stat: 0,
            intc_mask: 0,
            dma_gif: EeDmaChannel::default(),
            dma_sif0: EeDmaChannel::default(),
            dma_sif1: EeDmaChannel::default(),
            d_stat: 0,
            d_mask: 0,
            iop_dma_sif0: IopDmaChannel::default(),
            iop_dma_sif1: IopDmaChannel::default(),
            iop_dicr: 0,
            iop_dicr2: 0,
            now: 0,
            tty_buffer: String::new(),
            tty_line: String::new(),
            rdram_sdevid: 0,
            warned_unmapped: HashSet::new(),
            ee_tlb: [(0, 0, 0, 0); 48],
            tlb_cache: vec![(0u32, 0u32); 1024].into_boxed_slice(),
            dma_irq_queue: Vec::new(),
        }
    }

    /// Queue an EE DMAC completion interrupt a little into the future.
    fn ee_dma_irq(&mut self, ch: u32) {
        self.dma_irq_queue.push((1 << ch, self.now + 1024));
    }

    /// Record a TLB entry (from tlbwi) and flush the translation cache.
    pub fn ee_tlb_write(&mut self, idx: usize, mask: u32, hi: u32, lo0: u32, lo1: u32) {
        if idx < 48 {
            self.ee_tlb[idx] = (mask, hi, lo0, lo1);
            self.tlb_cache.fill((0, 0));
        }
    }

    /// Walk the TLB for a mapped-segment address. Returns a physical
    /// address; scratchpad hits map into a reserved range above VRAM-visible
    /// physical space (0x7000_0000 window preserved).
    fn tlb_lookup(&mut self, vaddr: u32) -> u32 {
        let slot = ((vaddr >> 12) & 1023) as usize;
        let (tag, base) = self.tlb_cache[slot];
        if tag == (vaddr >> 12) | 0x8000_0000 {
            return (base << 12) | (vaddr & 0xFFF);
        }
        for &(mask, hi, lo0, lo1) in &self.ee_tlb {
            if hi == 0 && lo0 == 0 && lo1 == 0 {
                continue;
            }
            let page_size = ((mask >> 13) + 1) << 12;
            let pair_mask = !(page_size * 2 - 1);
            if (vaddr & pair_mask) != (hi & pair_mask) {
                continue;
            }
            if lo0 & 0x8000_0000 != 0 {
                // Scratchpad entry: 16 KiB window.
                return 0x7000_0000 | (vaddr & 0x3FFF);
            }
            let odd = vaddr & page_size != 0;
            let lo = if odd { lo1 } else { lo0 };
            if lo & 2 == 0 {
                continue; // invalid half
            }
            let phys = ((lo >> 6) << 12) | (vaddr & (page_size - 1));
            self.tlb_cache[slot] = ((vaddr >> 12) | 0x8000_0000, phys >> 12);
            return phys;
        }
        // No mapping: fall back to a direct fold so early boot keeps working.
        if self.warned_unmapped.insert(vaddr & !0xFFF) {
            warn!(target: "ps2_core::bus", vaddr = format_args!("{vaddr:#010x}"), "EE access with no TLB mapping (direct fold)");
        }
        vaddr & 0x1FFF_FFFF
    }

    /// Translate an EE virtual address: KSEG0/1 fold directly, everything
    /// else goes through the TLB (with common fixed mappings fast-pathed).
    #[inline]
    fn translate(&mut self, vaddr: u32) -> u32 {
        match vaddr {
            // KSEG0 / KSEG1: unmapped segments.
            0x8000_0000..=0xBFFF_FFFF => vaddr & 0x1FFF_FFFF,
            // Scratchpad window (kernel TLB entry 0, effectively fixed).
            0x7000_0000..=0x7000_3FFF => vaddr,
            // Identity-mapped low RAM (kuseg): skip the walk.
            0x0000_0000..=0x01FF_FFFF => vaddr,
            _ => self.tlb_lookup(vaddr),
        }
    }

    #[inline]
    pub fn read8(&mut self, vaddr: u32) -> u8 {
        self.read::<1>(vaddr) as u8
    }
    #[inline]
    pub fn read16(&mut self, vaddr: u32) -> u16 {
        self.read::<2>(vaddr) as u16
    }
    #[inline]
    pub fn read32(&mut self, vaddr: u32) -> u32 {
        self.read::<4>(vaddr) as u32
    }
    #[inline]
    pub fn read64(&mut self, vaddr: u32) -> u64 {
        self.read::<8>(vaddr)
    }
    /// 128-bit read (lq); address is 16-byte aligned by the caller.
    pub fn read128(&mut self, vaddr: u32) -> [u64; 2] {
        [self.read::<8>(vaddr), self.read::<8>(vaddr + 8)]
    }

    #[inline]
    pub fn write8(&mut self, vaddr: u32, v: u8) {
        self.write::<1>(vaddr, v as u64)
    }
    #[inline]
    pub fn write16(&mut self, vaddr: u32, v: u16) {
        self.write::<2>(vaddr, v as u64)
    }
    #[inline]
    pub fn write32(&mut self, vaddr: u32, v: u32) {
        self.write::<4>(vaddr, v as u64)
    }
    #[inline]
    pub fn write64(&mut self, vaddr: u32, v: u64) {
        self.write::<8>(vaddr, v)
    }
    pub fn write128(&mut self, vaddr: u32, v: [u64; 2]) {
        self.write::<8>(vaddr, v[0]);
        self.write::<8>(vaddr + 8, v[1]);
    }

    /// Instruction fetch: same path as data reads for now.
    #[inline]
    pub fn fetch32(&mut self, vaddr: u32) -> u32 {
        self.read32(vaddr)
    }

    fn read<const N: usize>(&mut self, vaddr: u32) -> u64 {
        let addr = self.translate(vaddr);
        match addr {
            0x0000_0000..=0x01FF_FFFF => read_le::<N>(&self.ram, addr as usize),
            0x7000_0000..=0x7000_3FFF => read_le::<N>(&self.spad, (addr & 0x3FFF) as usize),
            0x1000_0000..=0x1000_FFFF => self.read_mmio::<N>(addr),
            0x1100_0000..=0x1100_FFFF => {
                // VU0/VU1 code and data memory; plain storage until VUs exist.
                trace!(target: "ps2_core::bus", addr, "VU memory read (stub)");
                0
            }
            0x1200_0000..=0x1200_1FFF => self.read_gs_priv::<N>(addr),
            0x1C00_0000..=0x1C1F_FFFF => read_le::<N>(&self.iop_ram, (addr & 0x1F_FFFF) as usize),
            0x1F80_0000..=0x1F80_FFFF => {
                // IOP MMIO window as seen from the EE.
                trace!(target: "ps2_core::bus", addr, "IOP MMIO read from EE (stub)");
                0
            }
            0x1FC0_0000..=0x1FFF_FFFF => read_le::<N>(&self.bios, (addr & 0x3F_FFFF) as usize),
            // ROM1 (DVD player ROM): absent, reads like erased flash.
            0x1E00_0000..=0x1E3F_FFFF => u64::MAX >> (64 - 8 * N as u32),
            // SBUS CRT-controller command interface used by ROMGSCRT:
            // +0x06 status (bit1 = command done, bit0 = busy), +0x10 data.
            0x1A00_0000..=0x1A00_FFFF => {
                trace!(target: "ps2_core::bus::sbus", addr = format_args!("{addr:#010x}"), "SBUS read (stub)");
                match addr & 0xFF {
                    0x06 => 2,
                    _ => 0,
                }
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::bus", addr = format_args!("{addr:#010x}"), size = N, "read from unmapped address (reported once)");
                }
                0
            }
        }
    }

    fn write<const N: usize>(&mut self, vaddr: u32, v: u64) {
        let addr = self.translate(vaddr);
        match addr {
            0x0000_0000..=0x01FF_FFFF => write_le::<N>(&mut self.ram, addr as usize, v),
            0x7000_0000..=0x7000_3FFF => write_le::<N>(&mut self.spad, (addr & 0x3FFF) as usize, v),
            0x1000_0000..=0x1000_FFFF => self.write_mmio::<N>(addr, v),
            0x1100_0000..=0x1100_FFFF => {
                trace!(target: "ps2_core::bus", addr, "VU memory write (stub)");
            }
            0x1200_0000..=0x1200_1FFF => self.write_gs_priv::<N>(addr, v),
            0x1C00_0000..=0x1C1F_FFFF => {
                write_le::<N>(&mut self.iop_ram, (addr & 0x1F_FFFF) as usize, v)
            }
            0x1F80_0000..=0x1F80_FFFF => {
                trace!(target: "ps2_core::bus", addr, "IOP MMIO write from EE (stub)");
            }
            0x1FC0_0000..=0x1FFF_FFFF => {
                warn!(target: "ps2_core::bus", addr = format_args!("{addr:#010x}"), "write to BIOS ROM ignored");
            }
            0x1A00_0000..=0x1A00_FFFF => {
                trace!(target: "ps2_core::bus::sbus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "SBUS write (stub)");
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), size = N, "write to unmapped address (reported once)");
                }
            }
        }
    }

    fn read_mmio<const N: usize>(&mut self, addr: u32) -> u64 {
        let off = (addr & 0xFFFF) as usize;
        match addr & !0x3 {
            0x1000_0000..=0x1000_1FFF => self.timers.read(addr, self.now) as u64,
            // SIO_ISR: no pending serial interrupts.
            0x1000_F130 => 0,
            // MCH_RICM reads back as 0 (busy bit clear = operation done).
            0x1000_F430 => {
                trace!(target: "ps2_core::bus::mch", "RICM read -> 0");
                0
            }
            // MCH_DRD: RDRAM init handshake, mirrors the documented sequence.
            0x1000_F440 => {
                let ricm = read_le::<4>(&self.mmio, 0xF430) as u32;
                let sop = (ricm >> 6) & 0xF;
                let sa = (ricm >> 16) & 0xFFF;
                trace!(target: "ps2_core::bus::mch", ricm = format_args!("{ricm:#010x}"), sop, sa = format_args!("{sa:#x}"), "DRD read");
                if sop == 0 {
                    match sa {
                        0x21 => {
                            // INIT: each device answers once.
                            if self.rdram_sdevid < RDRAM_DEVICES {
                                self.rdram_sdevid += 1;
                                0x1F
                            } else {
                                0
                            }
                        }
                        0x23 => 0x0D0D,               // CNFGA
                        0x24 => 0x0090,               // CNFGB
                        0x40 => (ricm & 0x1F) as u64, // DEVID
                        _ => 0,
                    }
                } else {
                    0
                }
            }
            // EE DMAC: GIF (ch2), SIF0 (ch5) / SIF1 (ch6), interrupt status.
            0x1000_A000 => self.dma_gif.chcr as u64,
            0x1000_A010 => self.dma_gif.madr as u64,
            0x1000_A020 => self.dma_gif.qwc as u64,
            0x1000_A030 => self.dma_gif.tadr as u64,
            0x1000_C000 => self.dma_sif0.chcr as u64,
            0x1000_C010 => self.dma_sif0.madr as u64,
            0x1000_C020 => self.dma_sif0.qwc as u64,
            0x1000_C400 => self.dma_sif1.chcr as u64,
            0x1000_C410 => self.dma_sif1.madr as u64,
            0x1000_C420 => self.dma_sif1.qwc as u64,
            0x1000_C430 => self.dma_sif1.tadr as u64,
            0x1000_E010 => (self.d_stat | (self.d_mask << 16)) as u64,
            // EE INTC.
            0x1000_F000 => self.intc_stat as u64,
            0x1000_F010 => self.intc_mask as u64,
            // SIF registers (mailboxes, flags, control handshake).
            0x1000_F200..=0x1000_F26F => self.sif.ee_read(addr) as u64,
            // DMAC ENABLER reads back the value written to ENABLEW.
            0x1000_F520 => read_le::<4>(&self.mmio, 0xF590),
            _ => {
                let v = read_le::<N>(&self.mmio, off);
                trace!(target: "ps2_core::bus::mmio", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "MMIO read (shadow)");
                v
            }
        }
    }

    fn write_mmio<const N: usize>(&mut self, addr: u32, v: u64) {
        match addr & !0x3 {
            0x1000_0000..=0x1000_1FFF => {
                self.timers.write(addr, v as u32, self.now);
                return;
            }
            // EE SIO TXFIFO: the kernel's debug output channel. Pure
            // observation — never affects execution.
            0x1000_F180 => {
                self.tty_push(v as u8);
                return;
            }
            // GIF channel: starting a transfer runs it to completion.
            0x1000_A000 => {
                self.dma_gif.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_gif();
                }
                return;
            }
            0x1000_A010 => {
                self.dma_gif.madr = v as u32;
                return;
            }
            0x1000_A020 => {
                self.dma_gif.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_A030 => {
                self.dma_gif.tadr = v as u32;
                return;
            }
            // VIF0 (ch0) / VIF1 (ch1): no VUs yet. Complete immediately so
            // nothing waits forever on them; log so the gap stays visible.
            0x1000_8000 | 0x1000_9000 => {
                let ch = if addr & !0x3 == 0x1000_8000 { 0 } else { 1 };
                if v as u32 & EE_CHCR_STR != 0 {
                    warn!(target: "ps2_core::bus::dma", ch, "VIF DMA discarded (no VU/VIF yet)");
                    self.ee_dma_irq(ch);
                    write_le::<4>(
                        &mut self.mmio,
                        (addr & 0xFFFF) as usize,
                        v & !(EE_CHCR_STR as u64),
                    );
                } else {
                    write_le::<4>(&mut self.mmio, (addr & 0xFFFF) as usize, v);
                }
                return;
            }
            // EE DMAC SIF channels: starting a transfer pumps it to completion.
            0x1000_C000 => {
                self.dma_sif0.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_sif();
                }
                return;
            }
            0x1000_C010 => {
                self.dma_sif0.madr = v as u32;
                return;
            }
            0x1000_C020 => {
                self.dma_sif0.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_C400 => {
                self.dma_sif1.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_sif();
                }
                return;
            }
            0x1000_C410 => {
                self.dma_sif1.madr = v as u32;
                return;
            }
            0x1000_C420 => {
                self.dma_sif1.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_C430 => {
                self.dma_sif1.tadr = v as u32;
                return;
            }
            // D_STAT: low half write-1-to-clear, high half toggles the mask.
            0x1000_E010 => {
                self.d_stat &= !(v as u32 & 0xFFFF);
                self.d_mask ^= (v as u32 >> 16) & 0xFFFF;
                return;
            }
            // INTC: STAT is write-1-to-clear, MASK is write-1-to-toggle.
            0x1000_F000 => {
                self.intc_stat &= !(v as u32);
                return;
            }
            0x1000_F010 => {
                self.intc_mask ^= v as u32 & 0xFFFF;
                return;
            }
            0x1000_F200..=0x1000_F26F => {
                self.sif.ee_write(addr, v as u32);
                return;
            }
            // RDRAM controller command register: busy bit (31) self-clears.
            0x1000_F410 => {
                trace!(target: "ps2_core::bus::mch", value = format_args!("{v:#010x}"), "F410 write");
                write_le::<4>(&mut self.mmio, 0xF410, v & !0x8000_0000);
                return;
            }
            // MCH_RICM: busy bit self-clears; INIT restarts device counting.
            0x1000_F430 => {
                let sa = ((v >> 16) & 0xFFF) as u32;
                let sbc = ((v >> 6) & 0xF) as u32;
                trace!(target: "ps2_core::bus::mch", value = format_args!("{v:#010x}"), sop = sbc, sa = format_args!("{sa:#x}"), "RICM write");
                if sa == 0x21 && sbc == 1 && (read_le::<4>(&self.mmio, 0xF440) >> 7) & 1 == 0 {
                    self.rdram_sdevid = 0;
                }
                write_le::<4>(&mut self.mmio, 0xF430, v & !0x8000_0000);
                return;
            }
            _ => {}
        }
        trace!(target: "ps2_core::bus::mmio", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "MMIO write (shadow)");
        write_le::<N>(&mut self.mmio, (addr & 0xFFFF) as usize, v);
    }

    fn read_gs_priv<const N: usize>(&mut self, addr: u32) -> u64 {
        let v = self.gs.priv_read(addr & !0x7);
        if N == 8 {
            v
        } else if addr & 4 != 0 {
            v >> 32
        } else {
            v & 0xFFFF_FFFF
        }
    }

    fn write_gs_priv<const N: usize>(&mut self, addr: u32, v: u64) {
        let v = if N == 8 {
            v
        } else {
            // 32-bit access: merge into the 64-bit register.
            let cur = self.gs.priv_read(addr & !0x7);
            if addr & 4 != 0 {
                (cur & 0xFFFF_FFFF) | (v << 32)
            } else {
                (cur & !0xFFFF_FFFF) | (v & 0xFFFF_FFFF)
            }
        };
        self.gs.priv_write(addr & !0x7, v);
        self.gs_sync_int();
    }

    /// Fold a pending GS interrupt edge into EE INTC bit 0.
    fn gs_sync_int(&mut self) {
        if self.gs.intc_pending {
            self.gs.intc_pending = false;
            self.intc_stat |= 1;
        }
    }

    // --- periodic events -------------------------------------------------

    /// Edge-detect timer interrupts on both sides. Called periodically.
    pub fn tick_timers(&mut self) {
        // Deliver due deferred DMA completion interrupts.
        let mut i = 0;
        while i < self.dma_irq_queue.len() {
            if self.dma_irq_queue[i].1 <= self.now {
                self.d_stat |= self.dma_irq_queue[i].0;
                self.dma_irq_queue.swap_remove(i);
            } else {
                i += 1;
            }
        }
        self.intc_stat |= self.timers.check_irqs(self.now);
        const IRQ_BITS: [u32; 6] = [4, 5, 6, 14, 15, 16];
        let mut fired = 0u32;
        let now = self.now;
        for (t, timer) in self.iop_timers.iter_mut().enumerate() {
            // IRQ on target (bit 4).
            if timer.mode & (1 << 4) == 0 {
                continue;
            }
            let before = timer.count(t, timer.last_check);
            let after = timer.count(t, now);
            timer.last_check = now;
            let target = timer.target;
            let crossed = if before <= after {
                before < target && target <= after
            } else {
                target > before || target <= after
            };
            if crossed {
                timer.mode |= 1 << 11; // reached target
                if timer.mode & (1 << 3) != 0 {
                    timer.base = 0;
                    timer.base_cycle = now;
                }
                fired |= 1 << IRQ_BITS[t];
            }
        }
        self.iop_i_stat |= fired;
    }

    /// Vertical blank begin/end: EE INTC bits 2/3, IOP I_STAT bits 0/11,
    /// GS CSR VSINT.
    pub fn vblank(&mut self, begin: bool) {
        if begin {
            self.intc_stat |= 1 << 2;
            self.iop_i_stat |= 1 << 0;
            self.gs.vblank();
            self.gs_sync_int();
        } else {
            self.intc_stat |= 1 << 3;
            self.iop_i_stat |= 1 << 11;
        }
    }

    /// Run the GIF channel (ch2) to completion: normal or source chain.
    fn pump_gif(&mut self) {
        let mut guard = 0u32;
        while self.dma_gif.chcr & EE_CHCR_STR != 0 {
            guard += 1;
            if guard > 1_000_000 {
                warn!(target: "ps2_core::bus::dma", "GIF DMA hit its iteration limit");
                break;
            }
            if self.dma_gif.qwc > 0 {
                let q = self.ee_dma_read128(self.dma_gif.madr);
                let lo = q[0] as u64 | ((q[1] as u64) << 32);
                let hi = q[2] as u64 | ((q[3] as u64) << 32);
                self.gif.process(&mut self.gs, lo, hi);
                self.dma_gif.madr = self.dma_gif.madr.wrapping_add(16);
                self.dma_gif.qwc -= 1;
                continue;
            }
            // Block finished.
            let chain = (self.dma_gif.chcr >> 2) & 3 == 1;
            if !chain || self.dma_gif.tag_end {
                self.dma_gif.chcr &= !EE_CHCR_STR;
                self.dma_gif.tag_end = false;
                self.ee_dma_irq(2);
                debug!(target: "ps2_core::bus::dma", "GIF DMA done");
                break;
            }
            // Source-chain tag.
            let tag = self.ee_dma_read128(self.dma_gif.tadr);
            let qwc = tag[0] & 0xFFFF;
            let id = (tag[0] >> 28) & 7;
            let irq = tag[0] & 0x8000_0000 != 0;
            let addr = tag[1] & 0x7FFF_FFF0;
            match id {
                0 => {
                    self.dma_gif.madr = addr;
                    self.dma_gif.tadr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tag_end = true;
                }
                1 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tadr = self.dma_gif.madr.wrapping_add(qwc * 16);
                }
                2 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tadr = addr;
                }
                3 | 4 => {
                    self.dma_gif.madr = addr;
                    self.dma_gif.tadr = self.dma_gif.tadr.wrapping_add(16);
                }
                7 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tag_end = true;
                }
                _ => {
                    warn!(target: "ps2_core::bus::dma", id, "unhandled GIF chain tag id");
                    self.dma_gif.tag_end = true;
                }
            }
            if irq && self.dma_gif.chcr & EE_CHCR_TIE != 0 {
                self.dma_gif.tag_end = true;
            }
            if self.dma_gif.chcr & EE_CHCR_TTE != 0 {
                // TTE on the GIF channel sends the tag's upper 64 bits.
                let lo = tag[2] as u64 | ((tag[3] as u64) << 32);
                self.gif.process(&mut self.gs, lo, 0);
            }
            self.dma_gif.qwc = qwc;
        }
        self.gs_sync_int();
    }

    // --- SIF DMA ---------------------------------------------------------

    /// EE interrupt lines: INT0 = INTC, INT1 = DMAC.
    pub fn ee_int0_pending(&self) -> bool {
        self.intc_stat & self.intc_mask != 0
    }
    pub fn ee_int1_pending(&self) -> bool {
        self.d_stat & self.d_mask & 0x3FF != 0
    }

    fn ee_dma_read128(&self, addr: u32) -> [u32; 4] {
        let a = (addr & 0x1FFF_FFF0) as usize;
        if a + 16 <= RAM_SIZE {
            [
                read_le::<4>(&self.ram, a) as u32,
                read_le::<4>(&self.ram, a + 4) as u32,
                read_le::<4>(&self.ram, a + 8) as u32,
                read_le::<4>(&self.ram, a + 12) as u32,
            ]
        } else {
            warn!(target: "ps2_core::bus::sifdma", addr = format_args!("{addr:#010x}"), "EE DMA read outside RAM");
            [0; 4]
        }
    }

    /// Raise an IOP DMA completion interrupt for channel 9 or 10 via DICR2.
    fn iop_dma_irq(&mut self, ch: u32) {
        let flag = 1 << ch; // DICR2 flags for ch7..13 sit at bits 24+(ch-7)+... see below
        let bit = 1 << (24 + (ch - 7));
        let enabled = self.iop_dicr2 & (1 << (16 + (ch - 7))) != 0;
        self.iop_dicr2 |= bit;
        let _ = flag;
        if enabled {
            // IOP DMA interrupt line.
            self.iop_i_stat |= 1 << 3;
        }
        debug!(target: "ps2_core::bus::sifdma", ch, enabled, "IOP DMA complete");
    }

    /// Move as much SIF traffic as the armed channels allow. Runs transfers
    /// to completion synchronously; timing comes later if software needs it.
    pub fn pump_sif(&mut self) {
        // Safety valve against malformed chains.
        for _ in 0..4096 {
            let mut progressed = false;
            progressed |= self.pump_sif1_ee();
            progressed |= self.pump_sif1_iop();
            progressed |= self.pump_sif0_iop();
            progressed |= self.pump_sif0_ee();
            if !progressed {
                return;
            }
        }
        warn!(target: "ps2_core::bus::sifdma", "SIF pump hit its iteration limit");
    }

    /// EE SIF1 (ch6): source chain from EE RAM into fifo1.
    fn pump_sif1_ee(&mut self) -> bool {
        let mut progressed = false;
        while self.dma_sif1.chcr & EE_CHCR_STR != 0 {
            if self.dma_sif1.qwc > 0 {
                for _ in 0..self.dma_sif1.qwc {
                    let q = self.ee_dma_read128(self.dma_sif1.madr);
                    self.sif.fifo1.extend(q);
                    self.dma_sif1.madr = self.dma_sif1.madr.wrapping_add(16);
                }
                self.dma_sif1.qwc = 0;
                progressed = true;
                if self.dma_sif1.tag_end {
                    self.dma_sif1.chcr &= !EE_CHCR_STR;
                    self.dma_sif1.tag_end = false;
                    self.ee_dma_irq(6);
                    debug!(target: "ps2_core::bus::sifdma", "EE SIF1 chain done");
                }
            } else {
                // Fetch the next source-chain tag.
                let tag = self.ee_dma_read128(self.dma_sif1.tadr);
                let qwc = tag[0] & 0xFFFF;
                let id = (tag[0] >> 28) & 7;
                let irq = tag[0] & 0x8000_0000 != 0;
                let addr = tag[1] & 0x7FFF_FFF0;
                trace!(
                    target: "ps2_core::bus::sifdma",
                    tadr = format_args!("{:#010x}", self.dma_sif1.tadr),
                    qwc, id, irq,
                    "EE SIF1 tag"
                );
                if self.dma_sif1.chcr & EE_CHCR_TTE != 0 {
                    // Transfer the tag's upper 64 bits (the IOP-side tag).
                    self.sif.fifo1.push_back(tag[2]);
                    self.sif.fifo1.push_back(tag[3]);
                }
                match id {
                    0 => {
                        // refe: data at ADDR, end after this block.
                        self.dma_sif1.madr = addr;
                        self.dma_sif1.tadr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tag_end = true;
                    }
                    1 => {
                        // cnt: data follows the tag.
                        self.dma_sif1.madr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tadr = self.dma_sif1.madr.wrapping_add(qwc * 16);
                    }
                    2 => {
                        // next: data follows, next tag at ADDR.
                        self.dma_sif1.madr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tadr = addr;
                    }
                    3 | 4 => {
                        // ref/refs: data at ADDR, tags are sequential.
                        self.dma_sif1.madr = addr;
                        self.dma_sif1.tadr = self.dma_sif1.tadr.wrapping_add(16);
                    }
                    7 => {
                        // end: data follows, then stop.
                        self.dma_sif1.madr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tag_end = true;
                    }
                    _ => {
                        warn!(target: "ps2_core::bus::sifdma", id, "unhandled EE source-chain tag id");
                        self.dma_sif1.tag_end = true;
                    }
                }
                if irq && self.dma_sif1.chcr & EE_CHCR_TIE != 0 {
                    self.dma_sif1.tag_end = true;
                }
                self.dma_sif1.qwc = qwc;
                progressed = true;
                if qwc == 0 && self.dma_sif1.tag_end {
                    self.dma_sif1.chcr &= !EE_CHCR_STR;
                    self.dma_sif1.tag_end = false;
                    self.ee_dma_irq(6);
                }
            }
        }
        progressed
    }

    /// IOP SIF1 (ch10): fifo1 into IOP RAM, guided by embedded 2-word tags.
    fn pump_sif1_iop(&mut self) -> bool {
        let mut progressed = false;
        while self.iop_dma_sif1.chcr & IOP_CHCR_BUSY != 0 {
            let ch = &mut self.iop_dma_sif1;
            if ch.recv_left == 0 && ch.recv_pad == 0 {
                if self.sif.fifo1.len() < 4 {
                    break;
                }
                // The IOP-side tag occupies a full quadword in the stream:
                // {addr|flags, word count, pad, pad}, then the data follows,
                // itself padded up to a qword boundary.
                let w0 = self.sif.fifo1.pop_front().unwrap();
                let w1 = self.sif.fifo1.pop_front().unwrap();
                self.sif.fifo1.pop_front();
                self.sif.fifo1.pop_front();
                ch.recv_addr = w0 & 0xFF_FFFF;
                ch.recv_start = ch.recv_addr;
                ch.recv_left = w1;
                ch.recv_pad = (4 - (w1 & 3)) & 3;
                ch.recv_end = w0 & 0xC000_0000 != 0;
                trace!(
                    target: "ps2_core::bus::sifdma",
                    addr = format_args!("{:#010x}", ch.recv_addr),
                    words = w1,
                    pad = ch.recv_pad,
                    end = ch.recv_end,
                    "IOP SIF1 tag"
                );
                progressed = true;
            }
            while ch.recv_left > 0 && !self.sif.fifo1.is_empty() {
                let w = self.sif.fifo1.pop_front().unwrap();
                let a = (ch.recv_addr & 0x1F_FFFC) as usize;
                write_le::<4>(&mut self.iop_ram, a, w as u64);
                ch.recv_addr = ch.recv_addr.wrapping_add(4);
                ch.recv_left -= 1;
                progressed = true;
            }
            if ch.recv_left > 0 {
                break; // wait for more data
            }
            while ch.recv_pad > 0 && !self.sif.fifo1.is_empty() {
                self.sif.fifo1.pop_front();
                ch.recv_pad -= 1;
                progressed = true;
            }
            if ch.recv_pad > 0 {
                break;
            }
            if ch.recv_end {
                ch.recv_end = false;
                ch.chcr &= !IOP_CHCR_BUSY;
                let start = ch.recv_start;
                self.iop_dma_irq(10);
                // An sceSifIopReset command (cid 0x80000003) means the IOP
                // is about to reboot silently via UDNL: drop in-flight SIF
                // state so the new kernel starts with clean FIFOs.
                let cid = read_le::<4>(&self.iop_ram, ((start + 8) & 0x1F_FFFC) as usize) as u32;
                if cid == 0x8000_0003 {
                    debug!(target: "ps2_core::bus::sifdma", "IOP reset command: flushing SIF state");
                    self.sif.fifo0.clear();
                    self.sif.fifo1.clear();
                    self.iop_dma_sif0 = IopDmaChannel::default();
                    let chcr = self.iop_dma_sif1.chcr;
                    self.iop_dma_sif1 = IopDmaChannel::default();
                    self.iop_dma_sif1.chcr = chcr;
                }
            }
        }
        progressed
    }

    /// IOP SIF0 (ch9): IOP RAM into fifo0, guided by 2-word tags at TADR.
    fn pump_sif0_iop(&mut self) -> bool {
        let mut progressed = false;
        while self.iop_dma_sif0.chcr & IOP_CHCR_BUSY != 0 {
            let ch = &mut self.iop_dma_sif0;
            if ch.words_left == 0 {
                // 16-byte send block: {data addr | flags, word count,
                // EE tag lo, EE tag hi}. The EE-side destination tag rides
                // ahead of the data as its own quadword.
                let t = (ch.tadr & 0x1F_FFFC) as usize;
                let w0 = read_le::<4>(&self.iop_ram, t) as u32;
                let w1 = read_le::<4>(&self.iop_ram, t + 4) as u32;
                let ee_tag_lo = read_le::<4>(&self.iop_ram, t + 8) as u32;
                let ee_tag_hi = read_le::<4>(&self.iop_ram, t + 12) as u32;
                ch.madr = w0 & 0xFF_FFFF;
                ch.words_left = ((w1 & 0xFF_FFFF) + 3) & !3;
                ch.tag_end = w0 & 0xC000_0000 != 0;
                ch.tadr = ch.tadr.wrapping_add(16);
                self.sif.fifo0.extend([ee_tag_lo, ee_tag_hi, 0, 0]);
                trace!(
                    target: "ps2_core::bus::sifdma",
                    madr = format_args!("{:#010x}", ch.madr),
                    words = ch.words_left,
                    ee_tag = format_args!("{ee_tag_lo:08x} {ee_tag_hi:08x}"),
                    end = ch.tag_end,
                    "IOP SIF0 tag"
                );
                if ch.words_left == 0 && ch.tag_end {
                    ch.chcr &= !IOP_CHCR_BUSY;
                    ch.tag_end = false;
                    self.iop_dma_irq(9);
                    progressed = true;
                    continue;
                }
            }
            while ch.words_left > 0 {
                let a = (ch.madr & 0x1F_FFFC) as usize;
                self.sif
                    .fifo0
                    .push_back(read_le::<4>(&self.iop_ram, a) as u32);
                ch.madr = ch.madr.wrapping_add(4);
                ch.words_left -= 1;
                progressed = true;
            }
            if ch.tag_end {
                ch.tag_end = false;
                ch.chcr &= !IOP_CHCR_BUSY;
                self.iop_dma_irq(9);
            }
        }
        progressed
    }

    /// EE SIF0 (ch5): fifo0 into EE RAM as a destination chain.
    fn pump_sif0_ee(&mut self) -> bool {
        let mut progressed = false;
        while self.dma_sif0.chcr & EE_CHCR_STR != 0 {
            if self.dma_sif0.qwc == 0 {
                if self.dma_sif0.tag_end {
                    self.dma_sif0.tag_end = false;
                    self.dma_sif0.chcr &= !EE_CHCR_STR;
                    self.ee_dma_irq(5);
                    debug!(target: "ps2_core::bus::sifdma", "EE SIF0 chain done");
                    progressed = true;
                    continue;
                }
                if self.sif.fifo0.len() < 4 {
                    break;
                }
                let w0 = self.sif.fifo0.pop_front().unwrap();
                let w1 = self.sif.fifo0.pop_front().unwrap();
                self.sif.fifo0.pop_front();
                self.sif.fifo0.pop_front();
                self.dma_sif0.qwc = w0 & 0xFFFF;
                self.dma_sif0.madr = w1 & 0x1FFF_FFF0;
                let id = (w0 >> 28) & 7;
                let irq = w0 & 0x8000_0000 != 0;
                self.dma_sif0.tag_end = id == 7 || (irq && self.dma_sif0.chcr & EE_CHCR_TIE != 0);
                trace!(
                    target: "ps2_core::bus::sifdma",
                    madr = format_args!("{:#010x}", self.dma_sif0.madr),
                    qwc = self.dma_sif0.qwc,
                    end = self.dma_sif0.tag_end,
                    "EE SIF0 tag"
                );
                progressed = true;
            }
            while self.dma_sif0.qwc > 0 && self.sif.fifo0.len() >= 4 {
                let a = (self.dma_sif0.madr & 0x1FFF_FFF0) as usize;
                for i in 0..4 {
                    let w = self.sif.fifo0.pop_front().unwrap();
                    if a + 16 <= RAM_SIZE {
                        write_le::<4>(&mut self.ram, a + i * 4, w as u64);
                    }
                }
                self.dma_sif0.madr = self.dma_sif0.madr.wrapping_add(16);
                self.dma_sif0.qwc -= 1;
                progressed = true;
            }
            if self.dma_sif0.qwc > 0 {
                break; // wait for more data
            }
        }
        progressed
    }

    // --- IOP side --------------------------------------------------------

    #[inline]
    pub fn iop_read8(&mut self, vaddr: u32) -> u8 {
        self.iop_read::<1>(vaddr) as u8
    }
    #[inline]
    pub fn iop_read16(&mut self, vaddr: u32) -> u16 {
        self.iop_read::<2>(vaddr) as u16
    }
    #[inline]
    pub fn iop_read32(&mut self, vaddr: u32) -> u32 {
        self.iop_read::<4>(vaddr)
    }
    #[inline]
    pub fn iop_write8(&mut self, vaddr: u32, v: u8) {
        self.iop_write::<1>(vaddr, v as u32)
    }
    #[inline]
    pub fn iop_write16(&mut self, vaddr: u32, v: u16) {
        self.iop_write::<2>(vaddr, v as u32)
    }
    #[inline]
    pub fn iop_write32(&mut self, vaddr: u32, v: u32) {
        self.iop_write::<4>(vaddr, v)
    }

    pub fn iop_irq_pending(&self) -> bool {
        self.iop_i_ctrl & 1 != 0 && (self.iop_i_stat & self.iop_i_mask) != 0
    }

    fn iop_read<const N: usize>(&mut self, vaddr: u32) -> u32 {
        // KSEG2 (cache control etc.) is not mapped to physical memory.
        if vaddr >= 0xFFFE_0000 {
            trace!(target: "ps2_core::iop::bus", vaddr = format_args!("{vaddr:#010x}"), "KSEG2 read (stub)");
            return 0;
        }
        let addr = vaddr & 0x1FFF_FFFF;
        match addr {
            // 2 MiB RAM, mirrored through the first 8 MiB.
            0x0000_0000..=0x007F_FFFF => {
                read_le::<N>(&self.iop_ram, (addr & 0x1F_FFFF) as usize) as u32
            }
            0x1F80_0000..=0x1F80_03FF => {
                read_le::<N>(&self.iop_spad, (addr & 0x3FF) as usize) as u32
            }
            0x1F80_1000..=0x1F80_FFFF => self.iop_read_mmio::<N>(addr),
            0x1D00_0000..=0x1D00_00FF => self.sif.iop_read(addr),
            0x1F40_2000..=0x1F40_203F => {
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), "CDVD read (stub)");
                0
            }
            // ROM1 (DVD player ROM): not present; reads like erased flash so
            // presence/version checks fail instead of "succeeding" with zeros.
            0x1E00_0000..=0x1E3F_FFFF => u32::MAX,
            0x1F90_0000..=0x1F90_07FF => {
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), "SPU2 read (stub)");
                0
            }
            0x1FC0_0000..=0x1FFF_FFFF => {
                read_le::<N>(&self.bios, (addr & 0x3F_FFFF) as usize) as u32
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), size = N, "IOP read from unmapped address (reported once)");
                }
                0
            }
        }
    }

    fn iop_write<const N: usize>(&mut self, vaddr: u32, v: u32) {
        if vaddr >= 0xFFFE_0000 {
            trace!(target: "ps2_core::iop::bus", vaddr = format_args!("{vaddr:#010x}"), value = format_args!("{v:#x}"), "KSEG2 write (stub)");
            return;
        }
        let addr = vaddr & 0x1FFF_FFFF;
        match addr {
            0x0000_0000..=0x007F_FFFF => {
                write_le::<N>(&mut self.iop_ram, (addr & 0x1F_FFFF) as usize, v as u64)
            }
            0x1F80_0000..=0x1F80_03FF => {
                write_le::<N>(&mut self.iop_spad, (addr & 0x3FF) as usize, v as u64)
            }
            0x1F80_1000..=0x1F80_FFFF => self.iop_write_mmio::<N>(addr, v),
            0x1D00_0000..=0x1D00_00FF => {
                self.sif.iop_write(addr, v);
                // The IOP raising SMFLG interrupts the EE (INTC SBUS); the
                // EE handler folds the flags into its SREG array and acks.
                if addr & 0xF0 == 0x30 {
                    self.intc_stat |= 1 << 1;
                }
            }
            0x1F40_2000..=0x1F40_203F => {
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "CDVD write (stub)");
            }
            0x1F90_0000..=0x1F90_07FF => {
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "SPU2 write (stub)");
            }
            0x1FC0_0000..=0x1FFF_FFFF => {
                warn!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), "IOP write to BIOS ROM ignored");
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), size = N, "IOP write to unmapped address (reported once)");
                }
            }
        }
    }

    /// IOP timers: 0-2 at 0x1F8011x0 (16-bit), 3-5 at 0x1F8014{8,9,A}0
    /// (32-bit). Lazy counts off the EE cycle counter at the IOP clock (/8).
    fn iop_timer_index(addr: u32) -> Option<usize> {
        match addr & 0xFFF0 {
            0x1100 => Some(0),
            0x1110 => Some(1),
            0x1120 => Some(2),
            0x1480 => Some(3),
            0x1490 => Some(4),
            0x14A0 => Some(5),
            _ => None,
        }
    }

    fn iop_read_mmio<const N: usize>(&mut self, addr: u32) -> u32 {
        let off = (addr & 0xFFFF) as usize;
        if let Some(t) = Self::iop_timer_index(addr) {
            let timer = &mut self.iop_timers[t];
            return match addr & 0xF {
                0x0 => timer.count(t, self.now),
                0x4 => {
                    // Reading MODE clears the reached-target flags.
                    let v = timer.mode;
                    timer.mode &= !(0x1800);
                    v
                }
                0x8 => timer.target,
                _ => 0,
            };
        }
        match addr {
            // SIO2 (pad/memcard controller): report "no device attached".
            // CTRL's start bit self-clears on write, RECV1 says disconnected.
            0x1F80_8264 => 0xFF,    // FIFO out: empty response
            0x1F80_826C => 0x1D100, // RECV1: no device
            0x1F80_8270 => 0xF,     // RECV2: constant
            0x1F80_8274 => 0,       // RECV3
            0x1F80_1070 => self.iop_i_stat,
            0x1F80_1074 => self.iop_i_mask,
            0x1F80_1078 => {
                // I_CTRL reads clear the master enable bit, as on PS1.
                let v = self.iop_i_ctrl;
                self.iop_i_ctrl &= !1;
                v
            }
            // IOP DMA: SIF0 (ch9) and SIF1 (ch10), interrupt control.
            0x1F80_1520 => self.iop_dma_sif0.madr,
            0x1F80_1524 => self.iop_dma_sif0.bcr,
            0x1F80_1528 => self.iop_dma_sif0.chcr,
            0x1F80_152C => self.iop_dma_sif0.tadr,
            0x1F80_1530 => self.iop_dma_sif1.madr,
            0x1F80_1534 => self.iop_dma_sif1.bcr,
            0x1F80_1538 => self.iop_dma_sif1.chcr,
            0x1F80_153C => self.iop_dma_sif1.tadr,
            0x1F80_10F4 => self.iop_dicr,
            0x1F80_1574 => self.iop_dicr2,
            _ => {
                let v = read_le::<N>(&self.iop_mmio, off) as u32;
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "IOP MMIO read (shadow)");
                v
            }
        }
    }

    fn iop_write_mmio<const N: usize>(&mut self, addr: u32, v: u32) {
        if let Some(t) = Self::iop_timer_index(addr) {
            let timer = &mut self.iop_timers[t];
            match addr & 0xF {
                0x0 => {
                    timer.base = v;
                    timer.base_cycle = self.now;
                }
                0x4 => {
                    timer.mode = v;
                    // Writing MODE restarts the counter, as on hardware.
                    timer.base = 0;
                    timer.base_cycle = self.now;
                    timer.last_check = self.now;
                }
                0x8 => timer.target = v,
                _ => {}
            }
            return;
        }
        match addr {
            // SIO2 CTRL: the start bit kicks a transfer and self-clears;
            // completion raises the SIO2 interrupt.
            0x1F80_8268 => {
                write_le::<4>(&mut self.iop_mmio, 0x8268, (v & !1) as u64);
                if v & 1 != 0 {
                    trace!(target: "ps2_core::iop::bus", "SIO2 transfer (no device)");
                    self.iop_i_stat |= 1 << 17;
                }
            }
            // I_STAT write acknowledges: keeps only bits written as 1.
            0x1F80_1070 => self.iop_i_stat &= v,
            0x1F80_1074 => self.iop_i_mask = v,
            0x1F80_1078 => self.iop_i_ctrl = v,
            0x1F80_1520 => self.iop_dma_sif0.madr = v & 0xFF_FFFF,
            0x1F80_1524 => self.iop_dma_sif0.bcr = v,
            0x1F80_1528 => {
                self.iop_dma_sif0.chcr = v;
                if v & IOP_CHCR_BUSY != 0 {
                    self.pump_sif();
                }
            }
            0x1F80_152C => self.iop_dma_sif0.tadr = v & 0xFF_FFFF,
            0x1F80_1530 => self.iop_dma_sif1.madr = v & 0xFF_FFFF,
            0x1F80_1534 => self.iop_dma_sif1.bcr = v,
            0x1F80_1538 => {
                self.iop_dma_sif1.chcr = v;
                if v & IOP_CHCR_BUSY != 0 {
                    self.pump_sif();
                }
            }
            0x1F80_153C => self.iop_dma_sif1.tadr = v & 0xFF_FFFF,
            // DICR/DICR2: enables in bits 16-23, flags (W1C) in bits 24-30.
            0x1F80_10F4 => {
                self.iop_dicr =
                    (v & 0x00FF_FFFF) | (self.iop_dicr & !(v & 0x7F00_0000) & 0x7F00_0000);
            }
            0x1F80_1574 => {
                self.iop_dicr2 =
                    (v & 0x00FF_FFFF) | (self.iop_dicr2 & !(v & 0x7F00_0000) & 0x7F00_0000);
            }
            // POST: boot progress byte from the IOP BIOS.
            0x1F80_2070 => {
                debug!(target: "ps2_core::iop::bus", stage = format_args!("{:#04x}", v as u8), "POST");
            }
            _ => {
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "IOP MMIO write (shadow)");
                write_le::<N>(&mut self.iop_mmio, (addr & 0xFFFF) as usize, v as u64);
            }
        }
    }

    fn tty_push(&mut self, byte: u8) {
        let c = byte as char;
        if c == '\n' {
            debug!(target: "ps2_core::tty", "{}", self.tty_line);
            self.tty_line.clear();
        } else if byte.is_ascii() && !c.is_control() {
            self.tty_line.push(c);
        }
        self.tty_buffer.push(c);
    }
}

#[inline]
fn read_le<const N: usize>(mem: &[u8], offset: usize) -> u64 {
    let mut v = 0u64;
    for i in 0..N {
        v |= (mem[offset + i] as u64) << (8 * i);
    }
    v
}

#[inline]
fn write_le<const N: usize>(mem: &mut [u8], offset: usize, v: u64) {
    for i in 0..N {
        mem[offset + i] = (v >> (8 * i)) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus() -> Bus {
        Bus::new(vec![0u8; BIOS_SIZE])
    }

    #[test]
    fn ram_read_write_roundtrip() {
        let mut b = bus();
        b.write32(0x0010_0000, 0xDEAD_BEEF);
        assert_eq!(b.read32(0x0010_0000), 0xDEAD_BEEF);
        // KSEG0/KSEG1 mirrors reach the same storage.
        assert_eq!(b.read32(0x8010_0000), 0xDEAD_BEEF);
        assert_eq!(b.read32(0xA010_0000), 0xDEAD_BEEF);
    }

    #[test]
    fn scratchpad_is_isolated_from_ram() {
        let mut b = bus();
        b.write32(0x7000_0000, 0x1234_5678);
        assert_eq!(b.read32(0x7000_0000), 0x1234_5678);
        assert_ne!(b.read32(0x0000_0000), 0x1234_5678);
    }

    #[test]
    fn tty_capture() {
        let mut b = bus();
        for c in b"hi\n" {
            b.write8(0x1000_F180, *c);
        }
        assert_eq!(b.tty_buffer, "hi\n");
    }

    #[test]
    fn rdram_init_handshake() {
        let mut b = bus();
        // SOP=0, SA=0x21 (INIT): first two reads answer 0x1F, then 0.
        b.write32(0x1000_F430, 0x21 << 16);
        assert_eq!(b.read32(0x1000_F440), 0x1F);
        assert_eq!(b.read32(0x1000_F440), 0x1F);
        assert_eq!(b.read32(0x1000_F440), 0);
    }
}
