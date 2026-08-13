//! VIF1: parses the VIF command stream (DMA channel 1 / PATH2).
//!
//! Enough for the OSD boot path: DIRECT/DIRECTHL forward embedded GIF
//! packets to the GS, UNPACK/MPG store into VU1 data/micro memory (no VU1
//! execution yet — MSCAL is logged and dropped), and the state commands
//! keep the word stream in sync so everything after them stays parseable.

use crate::gif::Gif;
use crate::gs::Gs;
use tracing::{trace, warn};

/// VU1 data memory size (16 KiB).
const VU1_DATA_SIZE: usize = 16 * 1024;
/// VU1 micro memory size (16 KiB).
const VU1_MICRO_SIZE: usize = 16 * 1024;

enum State {
    Cmd,
    /// Payload words to swallow for state commands we don't model.
    Skip(u32),
    /// Next word is the STMASK value.
    Stmask,
    Direct {
        qwords: u32,
    },
    /// Raw store into VU1 data memory; layout is only faithful for V4-32,
    /// which is fine while nothing executes on VU1.
    Unpack {
        addr: usize,
        words: u32,
    },
    Mpg {
        addr: usize,
        words: u32,
    },
}

pub struct Vif {
    state: State,
    /// STCYCL cl/wl: wl > cl shrinks UNPACK's input data length.
    cl: u32,
    wl: u32,
    /// STMASK: 16 2-bit codes (cycle x field); only code 0 reads input.
    mask: u32,
    /// Word pairs being assembled into a GIF quadword during DIRECT.
    buf: [u32; 4],
    buf_len: usize,
    pub vu1_data: Box<[u8]>,
    pub vu1_micro: Box<[u8]>,
    warned_mscal: bool,
    /// Vifcodes already reported as unhandled (warn once per command).
    warned_cmds: u128,
    /// Ring of the last command words, dumped when the stream desyncs.
    history: [u32; 8],
    hist_pos: usize,
}

impl Default for Vif {
    fn default() -> Self {
        Self::new()
    }
}

impl Vif {
    pub fn new() -> Self {
        Self {
            state: State::Cmd,
            cl: 1,
            wl: 1,
            mask: 0,
            buf: [0; 4],
            buf_len: 0,
            vu1_data: vec![0u8; VU1_DATA_SIZE].into_boxed_slice(),
            vu1_micro: vec![0u8; VU1_MICRO_SIZE].into_boxed_slice(),
            warned_mscal: false,
            warned_cmds: 0,
            history: [0; 8],
            hist_pos: 0,
        }
    }

    /// Feed one 32-bit word of the VIF1 stream.
    pub fn push_word(&mut self, gs: &mut Gs, gif: &mut Gif, w: u32) {
        match &mut self.state {
            State::Cmd => self.command(w),
            State::Skip(left) => {
                *left -= 1;
                if *left == 0 {
                    self.state = State::Cmd;
                }
            }
            State::Stmask => {
                self.mask = w;
                self.state = State::Cmd;
            }
            State::Direct { qwords } => {
                self.buf[self.buf_len] = w;
                self.buf_len += 1;
                if self.buf_len == 4 {
                    self.buf_len = 0;
                    let lo = self.buf[0] as u64 | ((self.buf[1] as u64) << 32);
                    let hi = self.buf[2] as u64 | ((self.buf[3] as u64) << 32);
                    gif.process(gs, lo, hi);
                    *qwords -= 1;
                    if *qwords == 0 {
                        self.state = State::Cmd;
                    }
                }
            }
            State::Unpack { addr, words } => {
                if *addr + 4 <= VU1_DATA_SIZE {
                    self.vu1_data[*addr..*addr + 4].copy_from_slice(&w.to_le_bytes());
                }
                *addr += 4;
                *words -= 1;
                if *words == 0 {
                    self.state = State::Cmd;
                }
            }
            State::Mpg { addr, words } => {
                if *addr + 4 <= VU1_MICRO_SIZE {
                    self.vu1_micro[*addr..*addr + 4].copy_from_slice(&w.to_le_bytes());
                }
                *addr += 4;
                *words -= 1;
                if *words == 0 {
                    self.state = State::Cmd;
                }
            }
        }
    }

    fn command(&mut self, w: u32) {
        let cmd = (w >> 24) & 0x7F;
        let imm = w & 0xFFFF;
        let num = (w >> 16) & 0xFF;
        trace!(target: "ps2_core::vif", cmd = format_args!("{cmd:#04x}"), imm, num, "vifcode");
        self.history[self.hist_pos] = w;
        self.hist_pos = (self.hist_pos + 1) % self.history.len();
        match cmd {
            0x00 => {} // NOP
            0x01 => {
                // STCYCL
                self.cl = imm & 0xFF;
                self.wl = (imm >> 8) & 0xFF;
            }
            // OFFSET/BASE/ITOP/STMOD/MSKPATH3/MARK: meaningless without a
            // VU1; accepted so the stream stays in sync.
            0x02..=0x07 => {}
            // FLUSHE/FLUSH/FLUSHA: nothing runs, so nothing to wait for.
            0x10 | 0x11 | 0x13 => {}
            0x14 | 0x15 | 0x17 => {
                // MSCAL/MSCALF/MSCNT: no VU1 execution yet.
                if !self.warned_mscal {
                    warn!(target: "ps2_core::vif", "VU1 microprogram start ignored (reported once, no VU1 yet)");
                    self.warned_mscal = true;
                }
            }
            0x20 => self.state = State::Stmask,
            0x30 | 0x31 => self.state = State::Skip(4), // STROW/STCOL
            0x4A => {
                // MPG: `num` doublewords of microcode to imm*8.
                let n = if num == 0 { 256 } else { num };
                self.state = State::Mpg {
                    addr: (imm as usize) * 8,
                    words: n * 2,
                };
            }
            0x50 | 0x51 => {
                // DIRECT/DIRECTHL: imm quadwords straight to the GIF (PATH2).
                let qwords = if imm == 0 { 0x10000 } else { imm as u32 };
                self.buf_len = 0;
                self.state = State::Direct { qwords };
            }
            0x60..=0x7F => {
                // UNPACK: vn+1 fields of (32 >> vl) bits per write. Only
                // fields that actually read the input count toward the data
                // length: wl > cl row-fills whole writes, and with the m
                // flag STMASK codes != 0 fill from row/col/protect instead.
                let vn = (cmd >> 2) & 3;
                let vl = cmd & 3;
                let masked = cmd & 0x10 != 0;
                let writes = if num == 0 { 256 } else { num };
                let mut bits = 0u32;
                for i in 0..writes {
                    let pos = if self.wl > 0 { i % self.wl } else { i };
                    if self.wl > self.cl && pos >= self.cl {
                        continue; // row-filled, no input
                    }
                    if vl == 3 {
                        bits += 16; // V4-5 packs a whole vector into 16 bits
                        continue;
                    }
                    let cycle = pos.min(3);
                    for field in 0..=vn {
                        let code = (self.mask >> ((cycle * 4 + field) * 2)) & 3;
                        if !masked || code == 0 {
                            bits += 32 >> vl;
                        }
                    }
                }
                let words = bits.div_ceil(32);
                if words == 0 {
                    return;
                }
                self.state = State::Unpack {
                    addr: ((imm & 0x3FF) as usize) * 16,
                    words,
                };
            }
            _ => {
                // Unknown command: almost always a desynced stream. Report
                // once per code with recent history so the culprit shows.
                if self.warned_cmds & (1 << cmd) == 0 {
                    self.warned_cmds |= 1 << cmd;
                    let mut recent = [0u32; 8];
                    for i in 0..8 {
                        recent[i] = self.history[(self.hist_pos + i) % 8];
                    }
                    warn!(
                        target: "ps2_core::vif",
                        cmd = format_args!("{cmd:#04x}"),
                        word = format_args!("{w:#010x}"),
                        recent = format_args!("{recent:08x?}"),
                        "unhandled vifcode (reported once)"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(vif: &mut Vif, gs: &mut Gs, gif: &mut Gif, words: &[u32]) {
        for &w in words {
            vif.push_word(gs, gif, w);
        }
    }

    #[test]
    fn direct_forwards_gif_packets() {
        let (mut vif, mut gs, mut gif) = (Vif::new(), Gs::new(), Gif::new());
        // DIRECT 2 qwords: GIFtag (1 loop, PACKED, 1 reg, A+D) + one A+D
        // write of 0x1234 to GS register 0x42 (SCISSOR_1 area — any reg).
        let tag_lo: u64 = 1 | (1 << 15) | (1 << 60);
        let tag_hi: u64 = 0xE;
        feed(
            &mut vif,
            &mut gs,
            &mut gif,
            &[
                0x5000_0002, // DIRECT, 2 qwords
                tag_lo as u32,
                (tag_lo >> 32) as u32,
                tag_hi as u32,
                (tag_hi >> 32) as u32,
                0x1234,
                0,
                0x42,
                0,
            ],
        );
        assert!(gif.idle());
    }

    #[test]
    fn unpack_consumes_the_right_word_count() {
        let (mut vif, mut gs, mut gif) = (Vif::new(), Gs::new(), Gif::new());
        // UNPACK V4-32 (cmd 0x6C), 2 writes -> 8 data words.
        feed(&mut vif, &mut gs, &mut gif, &[0x6C02_0000]);
        for i in 0..8u32 {
            vif.push_word(&mut gs, &mut gif, 0xA0 + i);
        }
        // Parser is back in command state: a NOP must not be eaten as data.
        vif.push_word(&mut gs, &mut gif, 0);
        assert!(matches!(vif.state, State::Cmd));
        assert_eq!(&vif.vu1_data[0..4], &[0xA0, 0, 0, 0]);
    }

    #[test]
    fn v4_5_and_wl_gt_cl_lengths() {
        let (mut vif, mut gs, mut gif) = (Vif::new(), Gs::new(), Gif::new());
        // V4-5 (cmd 0x6F): 4 writes at 16 bits each -> 2 words.
        feed(&mut vif, &mut gs, &mut gif, &[0x6F04_0000, 0, 0]);
        assert!(matches!(vif.state, State::Cmd));
        // STCYCL cl=1 wl=2, then V4-32 with 4 writes: input carries only
        // 2 writes -> 8 words.
        feed(&mut vif, &mut gs, &mut gif, &[0x0100_0201, 0x6C04_0000]);
        for _ in 0..8 {
            vif.push_word(&mut gs, &mut gif, 0);
        }
        assert!(matches!(vif.state, State::Cmd));
    }

    #[test]
    fn masked_unpack_reads_less_input() {
        let (mut vif, mut gs, mut gif) = (Vif::new(), Gs::new(), Gif::new());
        // STMASK with field w of every cycle taking col (code 2): V4-32
        // with m set then reads 3 fields per write -> 2 writes = 6 words.
        feed(
            &mut vif,
            &mut gs,
            &mut gif,
            &[0x2000_0000, 0x8080_8080, 0x7C02_0000],
        );
        for _ in 0..6 {
            vif.push_word(&mut gs, &mut gif, 0);
        }
        assert!(matches!(vif.state, State::Cmd));
        // Without the m flag the same unpack reads all 8 words.
        feed(&mut vif, &mut gs, &mut gif, &[0x6C02_0000]);
        for _ in 0..8 {
            vif.push_word(&mut gs, &mut gif, 0);
        }
        assert!(matches!(vif.state, State::Cmd));
    }

    #[test]
    fn mpg_stores_microcode() {
        let (mut vif, mut gs, mut gif) = (Vif::new(), Gs::new(), Gif::new());
        // MPG 1 doubleword to address 0 -> 2 data words.
        feed(&mut vif, &mut gs, &mut gif, &[0x4A01_0000, 0xDEAD, 0xBEEF]);
        assert!(matches!(vif.state, State::Cmd));
        assert_eq!(&vif.vu1_micro[0..4], &[0xAD, 0xDE, 0, 0]);
    }
}
