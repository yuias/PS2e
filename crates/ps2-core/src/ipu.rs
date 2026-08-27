//! IPU: the image processing unit, an MPEG-2 macroblock decoder with its own
//! bitstream reader.
//!
//! What is here is the register file, the input FIFO and the bit reader on top
//! of it, plus the commands that only move bits around: `BCLR`, `FDEC`,
//! `SETIQ`, `SETVQ` and `SETTH`. That is enough for a title to parse MPEG
//! sequence and picture headers. The commands that actually decode a picture
//! -- `IDEC`, `BDEC`, `VDEC`, `CSC` and `PACK` -- are not implemented; they
//! complete without producing data and say so, rather than leaving the EE
//! spinning on a busy bit that never clears.
//!
//! The bitstream is read most-significant bit first out of quadwords in the
//! order they arrive from memory, so the FIFO stores plain byte arrays and the
//! reader indexes them as one flat bit string.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, warn};

/// Quadwords the hardware's input FIFO holds. Ours is allowed to grow past
/// this so a DMA can complete inside the write that starts it; `IFC` still
/// reports a hardware-shaped count.
const FIFO_DEPTH: usize = 8;

/// Bits in one quadword.
const QW_BITS: u32 = 128;

/// `IPU_CTRL` fields the register write leaves alone (the live status half)
/// and the ones it sets.
const CTRL_KEEP: u32 = 0x8000_FFFF;
const CTRL_WRITE: u32 = 0x47F3_0000;
const CTRL_RESET: u32 = 1 << 30;
const CTRL_BUSY: u32 = 1 << 31;

/// The busy marker in the high word of `IPU_CMD` and `IPU_TOP`.
const BUSY32: u32 = 0x8000_0000;

#[derive(Serialize, Deserialize, Default)]
pub struct Ipu {
    /// Input FIFO, quadwords in arrival order.
    fifo_in: VecDeque<[u8; 16]>,
    /// The reader's two-quadword window, `fp` of them valid, `bp` bits into
    /// the first.
    window: [[u8; 16]; 2],
    fp: u32,
    bp: u32,
    /// `IPU_CTRL`, status half included.
    ctrl: u32,
    /// `IPU_CMD`: the last decode result and whether one is still running.
    cmd_data: u32,
    cmd_busy: bool,
    /// `IPU_TOP`: the 32 bits the bit pointer sits on, and its own busy flag.
    top: u32,
    top_busy: bool,
    /// Quantiser matrices (`SETIQ`) and the vector-quantiser CLUT (`SETVQ`).
    iq: Vec<u8>,
    niq: Vec<u8>,
    vqclut: Vec<u8>,
    /// `SETTH` thresholds.
    thresh: [u16; 2],
    /// A command that ran out of bitstream, with how far it got. It resumes
    /// when the FIFO is fed.
    pending: Option<(u32, usize)>,
}

impl Ipu {
    pub fn new() -> Self {
        Self { iq: vec![0; 64], niq: vec![0; 64], vqclut: vec![0; 32], ..Self::default() }
    }

    /// Quadwords the software may see in the FIFO.
    fn ifc(&self) -> u32 {
        self.fifo_in.len().min(FIFO_DEPTH) as u32
    }

    // --- bit reader ------------------------------------------------------

    /// Pull quadwords in until `bits` more are readable. False means the
    /// FIFO ran dry and the caller must wait for more data.
    fn fill(&mut self, bits: u32) -> bool {
        while self.fp * QW_BITS < self.bp + bits {
            match self.fifo_in.pop_front() {
                Some(q) => {
                    self.window[self.fp as usize] = q;
                    self.fp += 1;
                }
                None => return false,
            }
        }
        true
    }

    /// Step the bit pointer, sliding the window when it leaves the first
    /// quadword. False means there was not enough data to step over.
    fn advance(&mut self, bits: u32) -> bool {
        if !self.fill(bits) {
            return false;
        }
        self.bp += bits;
        if self.bp >= QW_BITS {
            self.bp -= QW_BITS;
            if self.fp == 2 {
                self.window[0] = self.window[1];
                self.fp = 1;
            } else {
                match self.fifo_in.pop_front() {
                    Some(q) => {
                        self.window[0] = q;
                        self.fp = 1;
                    }
                    None => self.fp = 0,
                }
            }
        }
        true
    }

    /// One bit of the window, counted from the bit pointer.
    fn bit(&self, i: u32) -> u32 {
        let at = self.bp + i;
        let byte = self.window[(at / QW_BITS) as usize][((at % QW_BITS) / 8) as usize];
        u32::from(byte >> (7 - (at & 7)) & 1)
    }

    /// The next `n` bits without consuming them, most significant first.
    fn peek(&mut self, n: u32) -> Option<u32> {
        if !self.fill(n) {
            return None;
        }
        Some((0..n).fold(0u32, |acc, i| (acc << 1) | self.bit(i)))
    }

    /// Read one byte and consume it.
    fn take_byte(&mut self) -> Option<u8> {
        let v = self.peek(8)? as u8;
        self.advance(8);
        Some(v)
    }

    // --- registers -------------------------------------------------------

    pub fn read32(&mut self, addr: u32) -> u32 {
        match addr & 0x3C {
            0x00 => self.cmd_data,
            0x04 => u32::from(self.cmd_busy) * BUSY32,
            0x10 => self.ctrl(),
            0x20 => (self.bp & 0x7F) | (self.ifc() << 8) | (self.fp << 16),
            0x30 => self.top,
            0x34 => u32::from(self.top_busy) * BUSY32,
            _ => 0,
        }
    }

    pub fn read64(&mut self, addr: u32) -> u64 {
        u64::from(self.read32(addr)) | u64::from(self.read32(addr | 4)) << 32
    }

    /// `IPU_CTRL` with its live status fields folded in.
    fn ctrl(&self) -> u32 {
        let busy = u32::from(self.cmd_busy || self.pending.is_some()) * CTRL_BUSY;
        (self.ctrl & !(0xFF | CTRL_BUSY)) | self.ifc() | busy
    }

    /// Returns true when a command finished, which raises the IPU interrupt.
    pub fn write32(&mut self, addr: u32, v: u32) -> bool {
        match addr & 0x3C {
            0x00 => return self.command(v),
            0x10 => {
                self.ctrl = (v & CTRL_WRITE) | (self.ctrl & CTRL_KEEP);
                if v & CTRL_RESET != 0 {
                    self.reset();
                }
            }
            _ => {}
        }
        false
    }

    pub fn write64(&mut self, addr: u32, v: u64) -> bool {
        self.write32(addr, v as u32) | self.write32(addr | 4, (v >> 32) as u32)
    }

    /// `BCLR`, and the soft reset `IPU_CTRL.RST` asks for.
    fn reset(&mut self) {
        self.fifo_in.clear();
        self.window = [[0; 16]; 2];
        self.fp = 0;
        self.bp = 0;
        self.pending = None;
        self.cmd_busy = false;
        self.top_busy = false;
    }

    // --- FIFOs -----------------------------------------------------------

    /// Feed one quadword to the input FIFO, from DMA channel 4 or a
    /// programmed write. Returns true when it let a stalled command finish.
    pub fn push_in(&mut self, q: [u8; 16]) -> bool {
        self.fifo_in.push_back(q);
        self.resume()
    }

    /// Retry the command that ran out of bitstream.
    fn resume(&mut self) -> bool {
        match self.pending {
            Some((val, progress)) => self.run(val, progress),
            None => false,
        }
    }

    // --- commands --------------------------------------------------------

    fn command(&mut self, val: u32) -> bool {
        self.cmd_busy = true;
        self.run(val, 0)
    }

    /// Execute `val`, picking up at `progress` (the meaning of which is the
    /// command's own). Returns true when it completed.
    fn run(&mut self, val: u32, progress: usize) -> bool {
        let op = val >> 28;
        let skip = val & 0x3F;
        let done = match op {
            // BCLR: drop the bitstream and restart at the given bit.
            0x0 => {
                self.reset();
                self.bp = val & 0x7F;
                true
            }
            // FDEC: skip, then hand back the 32 bits under the pointer
            // without consuming them.
            0x4 => {
                if progress == 0 && !self.advance(skip) {
                    self.stall(val, 0);
                    return false;
                }
                match self.peek(32) {
                    Some(v) => {
                        self.cmd_data = v;
                        self.top = v;
                        true
                    }
                    None => {
                        self.stall(val, 1);
                        return false;
                    }
                }
            }
            // SETIQ / SETVQ: pull a table straight out of the bitstream.
            // Bit 27 of SETIQ picks the non-intra matrix.
            0x5 | 0x6 => {
                let len = if op == 0x5 { 64 } else { 32 };
                if progress == 0 && op == 0x5 && !self.advance(skip) {
                    self.stall(val, 0);
                    return false;
                }
                let mut i = progress.max(1) - 1;
                while i < len {
                    match self.take_byte() {
                        Some(b) => {
                            match op {
                                0x5 if val & (1 << 27) != 0 => self.niq[i] = b,
                                0x5 => self.iq[i] = b,
                                _ => self.vqclut[i] = b,
                            }
                            i += 1;
                        }
                        None => {
                            self.stall(val, i + 1);
                            return false;
                        }
                    }
                }
                true
            }
            // SETTH: two thresholds, straight from the command word.
            0x9 => {
                self.thresh = [(val & 0x1FF) as u16, ((val >> 16) & 0x1FF) as u16];
                true
            }
            // IDEC, BDEC, VDEC, CSC, PACK: the decoder proper. Completing
            // empty keeps the EE moving; the picture will be missing.
            _ => {
                self.cmd_data = 0;
                warn!(target: "ps2_core::bus::ipu",
                    cmd = format_args!("{op:#x}"), "decode command not implemented");
                true
            }
        };
        if done {
            debug!(target: "ps2_core::bus::ipu",
                cmd = format_args!("{op:#x}"),
                data = format_args!("{:#010x}", self.cmd_data),
                bp = self.bp, ifc = self.ifc(), "command done");
            self.pending = None;
            self.cmd_busy = false;
            self.top_busy = false;
        }
        done
    }

    /// Park a command that ran out of bitstream; `push_in` resumes it.
    fn stall(&mut self, val: u32, progress: usize) {
        self.pending = Some((val, progress));
        self.cmd_busy = true;
        self.top_busy = val >> 28 == 0x4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quadword whose bytes count up from `base`.
    fn qw(base: u8) -> [u8; 16] {
        std::array::from_fn(|i| base.wrapping_add(i as u8))
    }

    #[test]
    fn fdec_returns_the_bits_under_the_pointer() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        // FDEC with no skip: the first four bytes, most significant first.
        assert!(ipu.write32(0, 0x4000_0000));
        assert_eq!(ipu.read32(0), 0x0001_0203);
        // It does not consume them, so a second FDEC skipping 8 bits moves
        // the window on by exactly one byte.
        assert!(ipu.write32(0, 0x4000_0008));
        assert_eq!(ipu.read32(0), 0x0102_0304);
    }

    #[test]
    fn fdec_reads_across_a_quadword_boundary() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        ipu.push_in(qw(0x10));
        // Walk to the last byte of the first quadword, so the result
        // straddles the two.
        assert!(ipu.write32(0, 0x4000_0000));
        for _ in 0..15 {
            assert!(ipu.write32(0, 0x4000_0008));
        }
        assert_eq!(ipu.read32(0), 0x0F10_1112);
    }

    #[test]
    fn a_command_stalls_until_the_fifo_is_fed() {
        let mut ipu = Ipu::new();
        // Nothing in the FIFO: the command parks and IPU_CTRL stays busy.
        assert!(!ipu.write32(0, 0x4000_0000));
        assert_ne!(ipu.read32(0x10) & CTRL_BUSY, 0);
        assert_eq!(ipu.read32(4), BUSY32);
        assert!(ipu.push_in(qw(0)));
        assert_eq!(ipu.read32(0x10) & CTRL_BUSY, 0);
        assert_eq!(ipu.read32(0), 0x0001_0203);
    }

    #[test]
    fn setiq_takes_its_matrix_from_the_bitstream() {
        let mut ipu = Ipu::new();
        for i in 0..4 {
            ipu.push_in(qw(i * 16));
        }
        assert!(ipu.write32(0, 0x5000_0000));
        assert_eq!(ipu.iq[0], 0);
        assert_eq!(ipu.iq[63], 63);
        // The intra and non-intra matrices are separate.
        assert!(ipu.niq.iter().all(|&b| b == 0));
    }

    #[test]
    fn setiq_survives_being_fed_a_quadword_at_a_time() {
        // The movie player starves the FIFO constantly, so a table load has
        // to be resumable at any byte and still land byte-perfect.
        let mut ipu = Ipu::new();
        assert!(!ipu.write32(0, 0x5000_0000));
        let mut done = false;
        for i in 0..4 {
            assert!(!done, "finished before the last quadword arrived");
            done = ipu.push_in(qw(i * 16));
        }
        assert!(done);
        assert!(ipu.iq.iter().enumerate().all(|(i, &b)| b == i as u8));
    }

    #[test]
    fn a_read_that_outruns_the_fifo_resumes_without_losing_bits() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        // Two maximum skips leave the pointer near the end of the quadword,
        // so the 32 bits FDEC wants straddle a quadword that has not
        // arrived: the command parks mid-read and finishes on the feed.
        assert!(ipu.write32(0, 0x4000_0000 | 63));
        assert!(!ipu.write32(0, 0x4000_0000 | 63));
        assert!(ipu.push_in(qw(0x10)));
        assert_eq!(ipu.read32(0x20) & 0x7F, 126);
        assert_eq!(ipu.read32(0), 0xC404_4484);
    }

    #[test]
    fn bclr_drops_the_bitstream_and_sets_the_bit_pointer() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        assert!(ipu.write32(0, 0x0000_0009));
        assert_eq!(ipu.read32(0x20) & 0x7F, 9);
        assert_eq!(ipu.ifc(), 0);
    }

    #[test]
    fn ctrl_keeps_its_status_half_across_a_write() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        // Everything settable except the reset bit, which would empty the
        // FIFO out from under the check below.
        ipu.write32(0x10, !CTRL_RESET);
        // The written half is masked to the settable bits, and IFC still
        // reports the FIFO rather than what was written.
        assert_eq!(ipu.read32(0x10) & 0xF, 1);
        assert_eq!(ipu.read32(0x10) & !0xFFFF, CTRL_WRITE & !CTRL_RESET);
    }
}
