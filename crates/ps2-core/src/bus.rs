//! EE-side system bus: memory map, MMIO dispatch, TTY capture.
//!
//! Address translation is a direct segment fold for now: the kernel's TLB
//! mappings are essentially identity, so TLB instructions record entries
//! without remapping (see ARCHITECTURE.md).

use crate::timers::Timers;
use std::collections::HashSet;
use tracing::{debug, trace, warn};

pub const RAM_SIZE: usize = 32 * 1024 * 1024;
pub const BIOS_SIZE: usize = 4 * 1024 * 1024;
pub const SPAD_SIZE: usize = 16 * 1024;
/// Shadow register file for 0x1000_0000..0x1001_0000 MMIO.
const MMIO_SIZE: usize = 0x10000;
/// Shadow for GS privileged registers at 0x1200_0000.
const GS_PRIV_SIZE: usize = 0x2000;

/// Number of RDRAM devices reported by the MCH init handshake.
const RDRAM_DEVICES: u32 = 2;

pub struct Bus {
    pub ram: Box<[u8]>,
    pub bios: Box<[u8]>,
    pub spad: Box<[u8]>,
    /// IOP RAM as seen from the EE at 0x1C00_0000 (2 MiB).
    pub iop_ram: Box<[u8]>,
    /// Shadow storage for EE MMIO registers we don't model yet: reads return
    /// the last written value so BIOS read-modify-write sequences behave.
    mmio: Box<[u8]>,
    gs_priv: Box<[u8]>,
    pub timers: Timers,
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
            gs_priv: vec![0u8; GS_PRIV_SIZE].into_boxed_slice(),
            timers: Timers::new(),
            now: 0,
            tty_buffer: String::new(),
            tty_line: String::new(),
            rdram_sdevid: 0,
            warned_unmapped: HashSet::new(),
        }
    }

    /// Fold a virtual address to a physical one (no real TLB yet).
    #[inline]
    fn translate(vaddr: u32) -> u32 {
        match vaddr {
            // Scratchpad is only reachable through this fixed virtual window.
            0x7000_0000..=0x7000_3FFF => vaddr,
            // Kernel's uncached-accelerated mirror of main RAM.
            0x3010_0000..=0x31FF_FFFF => vaddr & 0x01FF_FFFF,
            _ => vaddr & 0x1FFF_FFFF,
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
        let addr = Self::translate(vaddr);
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
        let addr = Self::translate(vaddr);
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
            // INTC_STAT is write-1-to-clear.
            0x1000_F000 => {
                let cur = read_le::<4>(&self.mmio, 0xF000);
                write_le::<4>(&mut self.mmio, 0xF000, cur & !(v & 0xFFFF_FFFF));
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
        let off = (addr & 0x1FFF) as usize;
        match addr & !0x3 {
            // GS CSR: report FIFO empty and toggle VSYNC-ish bits off the
            // cycle counter so polling loops terminate. Real GS comes later.
            0x1200_1000 => {
                let vsync = (self.now >> 19) & 1; // arbitrary but progressing
                0x0000_0008 | (vsync << 3)
            }
            _ => {
                let v = read_le::<N>(&self.gs_priv, off);
                trace!(target: "ps2_core::bus::gs", addr = format_args!("{addr:#010x}"), "GS priv read (shadow)");
                v
            }
        }
    }

    fn write_gs_priv<const N: usize>(&mut self, addr: u32, v: u64) {
        trace!(target: "ps2_core::bus::gs", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "GS priv write (shadow)");
        write_le::<N>(&mut self.gs_priv, (addr & 0x1FFF) as usize, v);
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
