//! VIF1: parses the VIF command stream (DMA channel 1 / PATH2).
//!
//! Enough for the OSD boot path: DIRECT/DIRECTHL forward embedded GIF
//! packets to the GS, UNPACK/MPG fill VU1 data/micro memory with TOPS
//! double-buffering, MSCAL/MSCNT run the VU1 interpreter, and the state
//! commands keep the word stream in sync.

use crate::gif::Gif;
use crate::gs::Gs;
use crate::vu1::Vu1;
use tracing::{trace, warn};

enum State {
    Cmd,
    /// Next word is the STMASK value.
    Stmask,
    /// STROW/STCOL capture: which register file, words remaining.
    Strow(u32),
    Stcol(u32),
    Direct {
        qwords: u32,
    },
    /// Expanding UNPACK: each write produces one qword in VU1 data memory
    /// from packed input elements, row/col fills and the write mask.
    Unpack {
        addr: u32,
        vn: u32,
        vl: u32,
        masked: bool,
        usn: bool,
        writes_left: u32,
        block_pos: u32,
        buf: Vec<u8>,
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
    /// STROW/STCOL fill values.
    row: [u32; 4],
    col: [u32; 4],
    /// STMOD addition mode; only mode 0 is implemented.
    stmod: u32,
    warned_stmod: bool,
    /// Word pairs being assembled into a GIF quadword during DIRECT.
    buf: [u32; 4],
    buf_len: usize,
    /// VU1 double-buffering: TOPS = base + (dbf ? offset : 0), latched
    /// into the VU's TOP at MSCAL/MSCNT, which also flips dbf.
    base: u32,
    offset: u32,
    itops: u32,
    dbf: bool,
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
            row: [0; 4],
            col: [0; 4],
            stmod: 0,
            warned_stmod: false,
            buf: [0; 4],
            buf_len: 0,
            base: 0,
            offset: 0,
            itops: 0,
            dbf: false,
            warned_cmds: 0,
            history: [0; 8],
            hist_pos: 0,
        }
    }

    /// Current TOPS value (qword address in VU1 data memory).
    fn tops(&self) -> u32 {
        self.base + if self.dbf { self.offset } else { 0 }
    }

    /// Emit as many UNPACK writes as the buffered input allows. Leftover
    /// padding bytes are dropped when the last write lands (the stream is
    /// word-aligned per unpack).
    fn unpack_drain(&mut self, vu1: &mut Vu1) {
        let (mask, row, col) = (self.mask, self.row, self.col);
        let cl = if self.cl == 0 { 256 } else { self.cl };
        let wl = if self.wl == 0 { 256 } else { self.wl };
        let State::Unpack {
            addr,
            vn,
            vl,
            masked,
            usn,
            writes_left,
            block_pos,
            buf,
        } = &mut self.state
        else {
            return;
        };
        let (vn, vl, masked, usn) = (*vn, *vl, *masked, *usn);
        let esize = (4usize >> vl).max(1);
        let mut pos = 0usize;
        while *writes_left > 0 {
            let cycle = (*block_pos).min(3) as usize;
            let row_filled = wl > cl && *block_pos >= cl;
            // Input bytes this write needs.
            let need = if row_filled {
                0
            } else if vl == 3 {
                2
            } else if masked {
                let mut n = 0usize;
                for f in 0..4 {
                    let code = (mask >> ((cycle * 4 + f) * 2)) & 3;
                    if code == 0 && (f as u32) <= vn {
                        n += 1;
                    }
                }
                (if vn == 0 { n.min(1) } else { n }) * esize
            } else if vn == 0 {
                esize
            } else {
                (vn as usize + 1) * esize
            };
            if buf.len() - pos < need {
                break; // wait for more stream words
            }
            let take = |p: &mut usize| -> u32 {
                let v = match vl {
                    0 => u32::from_le_bytes(buf[*p..*p + 4].try_into().unwrap()),
                    1 => {
                        let v = u16::from_le_bytes(buf[*p..*p + 2].try_into().unwrap());
                        if usn { v as u32 } else { v as i16 as i32 as u32 }
                    }
                    _ => {
                        let v = buf[*p];
                        if usn { v as u32 } else { v as i8 as i32 as u32 }
                    }
                };
                *p += esize;
                v
            };
            let a = ((*addr as usize) & 0x3FF) * 16;
            let write_field = |data: &mut [u8], f: usize, v: u32| {
                data[a + f * 4..a + f * 4 + 4].copy_from_slice(&v.to_le_bytes());
            };
            if row_filled {
                for (f, r) in row.iter().enumerate() {
                    write_field(&mut vu1.data, f, *r);
                }
            } else if vl == 3 {
                // V4-5: one 16-bit RGBA5551 vector.
                let v = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as u32;
                pos += 2;
                write_field(&mut vu1.data, 0, (v & 0x1F) << 3);
                write_field(&mut vu1.data, 1, ((v >> 5) & 0x1F) << 3);
                write_field(&mut vu1.data, 2, ((v >> 10) & 0x1F) << 3);
                write_field(&mut vu1.data, 3, (v >> 15) << 7);
            } else {
                // S formats broadcast one element to every written field.
                let broadcast = if vn == 0 { Some(take(&mut pos)) } else { None };
                for f in 0..4 {
                    let code = if masked {
                        (mask >> ((cycle * 4 + f) * 2)) & 3
                    } else {
                        0
                    };
                    match code {
                        0 => {
                            if let Some(v) = broadcast {
                                write_field(&mut vu1.data, f, v);
                            } else if (f as u32) <= vn {
                                let v = take(&mut pos);
                                write_field(&mut vu1.data, f, v);
                            }
                        }
                        1 => write_field(&mut vu1.data, f, row[f]),
                        2 => write_field(&mut vu1.data, f, col[cycle]),
                        _ => {} // write-protected
                    }
                }
            }
            *addr += 1;
            *block_pos += 1;
            *writes_left -= 1;
            if *block_pos == wl {
                *block_pos = 0;
                if cl > wl {
                    *addr += cl - wl; // skipping mode: jump to the next block
                }
            }
        }
        buf.drain(..pos);
        if *writes_left == 0 {
            self.state = State::Cmd;
        }
    }

    /// Feed one 32-bit word of the VIF1 stream.
    pub fn push_word(&mut self, gs: &mut Gs, gif: &mut Gif, vu1: &mut Vu1, w: u32) {
        match &mut self.state {
            State::Cmd => self.command(gs, gif, vu1, w),
            State::Stmask => {
                self.mask = w;
                self.state = State::Cmd;
            }
            State::Strow(left) => {
                self.row[(4 - *left) as usize] = w;
                *left -= 1;
                if *left == 0 {
                    self.state = State::Cmd;
                }
            }
            State::Stcol(left) => {
                self.col[(4 - *left) as usize] = w;
                *left -= 1;
                if *left == 0 {
                    self.state = State::Cmd;
                }
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
            State::Unpack { buf, .. } => {
                buf.extend_from_slice(&w.to_le_bytes());
                self.unpack_drain(vu1);
            }
            State::Mpg { addr, words } => {
                let a = *addr & 0x3FFF;
                vu1.micro[a..a + 4].copy_from_slice(&w.to_le_bytes());
                *addr = (*addr + 4) & 0x3FFF;
                *words -= 1;
                if *words == 0 {
                    self.state = State::Cmd;
                }
            }
        }
    }

    fn command(&mut self, gs: &mut Gs, gif: &mut Gif, vu1: &mut Vu1, w: u32) {
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
            0x02 => {
                // OFFSET: also resets the double-buffer flag.
                self.offset = imm & 0x3FF;
                self.dbf = false;
            }
            0x03 => self.base = imm & 0x3FF,
            0x04 => self.itops = imm & 0x3FF,
            0x05 => {
                // STMOD: row-addition modes are not implemented yet.
                self.stmod = imm & 3;
                if self.stmod != 0 && !self.warned_stmod {
                    self.warned_stmod = true;
                    warn!(target: "ps2_core::vif", mode = self.stmod, "STMOD addition mode ignored (reported once)");
                }
            }
            // MSKPATH3/MARK: accepted so the stream stays in sync.
            0x06 | 0x07 => {}
            // FLUSHE/FLUSH/FLUSHA: execution is synchronous, nothing to wait.
            0x10 | 0x11 | 0x13 => {}
            0x14 | 0x15 => {
                // MSCAL/MSCALF: latch TOP/ITOP, flip the buffer, run.
                vu1.top = self.tops() as u16;
                vu1.itop = self.itops as u16;
                self.dbf = !self.dbf;
                vu1.start(gs, gif, imm as u16);
            }
            0x17 => {
                // MSCNT: continue after the previous program.
                vu1.top = self.tops() as u16;
                vu1.itop = self.itops as u16;
                self.dbf = !self.dbf;
                vu1.continue_run(gs, gif);
            }
            0x20 => self.state = State::Stmask,
            0x30 => self.state = State::Strow(4),
            0x31 => self.state = State::Stcol(4),
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
                // UNPACK: expand packed elements into qword writes.
                // FLG (imm bit 15) rebases the address on TOPS; usn (bit
                // 14) selects zero- vs sign-extension for 8/16-bit data.
                let mut qw = (imm & 0x3FF) as u32;
                if imm & 0x8000 != 0 {
                    qw += self.tops();
                }
                self.state = State::Unpack {
                    addr: qw,
                    vn: (cmd >> 2) & 3,
                    vl: cmd & 3,
                    masked: cmd & 0x10 != 0,
                    usn: imm & 0x4000 != 0,
                    writes_left: if num == 0 { 256 } else { num },
                    block_pos: 0,
                    buf: Vec::new(),
                };
                // Writes that take no input at all (row/col/protect) can
                // complete before any data word arrives.
                self.unpack_drain(vu1);
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

    struct Rig {
        vif: Vif,
        gs: Gs,
        gif: Gif,
        vu1: Vu1,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                vif: Vif::new(),
                gs: Gs::new(),
                gif: Gif::new(),
                vu1: Vu1::new(),
            }
        }

        fn feed(&mut self, words: &[u32]) {
            for &w in words {
                self.vif
                    .push_word(&mut self.gs, &mut self.gif, &mut self.vu1, w);
            }
        }

        fn in_cmd_state(&self) -> bool {
            matches!(self.vif.state, State::Cmd)
        }
    }

    #[test]
    fn direct_forwards_gif_packets() {
        let mut r = Rig::new();
        // DIRECT 2 qwords: GIFtag (1 loop, PACKED, 1 reg, A+D) + one A+D
        // write of 0x1234 to GS register 0x42 (SCISSOR_1 area — any reg).
        let tag_lo: u64 = 1 | (1 << 15) | (1 << 60);
        let tag_hi: u64 = 0xE;
        r.feed(&[
            0x5000_0002, // DIRECT, 2 qwords
            tag_lo as u32,
            (tag_lo >> 32) as u32,
            tag_hi as u32,
            (tag_hi >> 32) as u32,
            0x1234,
            0,
            0x42,
            0,
        ]);
        assert!(r.gif.idle());
    }

    #[test]
    fn unpack_consumes_the_right_word_count() {
        let mut r = Rig::new();
        // UNPACK V4-32 (cmd 0x6C), 2 writes -> 8 data words.
        r.feed(&[0x6C02_0000]);
        for i in 0..8u32 {
            r.feed(&[0xA0 + i]);
        }
        // Parser is back in command state: a NOP must not be eaten as data.
        r.feed(&[0]);
        assert!(r.in_cmd_state());
        assert_eq!(&r.vu1.data[0..4], &[0xA0, 0, 0, 0]);
    }

    #[test]
    fn unpack_flg_uses_tops() {
        let mut r = Rig::new();
        // BASE 0, OFFSET 4, dbf starts false -> TOPS = 0. After an MSCAL
        // (running an immediate E-bit pair) dbf flips -> TOPS = 4, so a
        // FLG unpack lands at qword 4.
        r.vu1.micro[4..8].copy_from_slice(&(1u32 << 30).to_le_bytes());
        r.feed(&[0x0300_0000, 0x0200_0004, 0x1400_0000]);
        r.feed(&[0x6C01_8000]);
        for i in 0..4u32 {
            r.feed(&[0xB0 + i]);
        }
        assert!(r.in_cmd_state());
        assert_eq!(&r.vu1.data[64..68], &[0xB0, 0, 0, 0]);
    }

    #[test]
    fn v4_5_and_wl_gt_cl_lengths() {
        let mut r = Rig::new();
        // V4-5 (cmd 0x6F): 4 writes at 16 bits each -> 2 words.
        r.feed(&[0x6F04_0000, 0, 0]);
        assert!(r.in_cmd_state());
        // STCYCL cl=1 wl=2, then V4-32 with 4 writes: input carries only
        // 2 writes -> 8 words.
        r.feed(&[0x0100_0201, 0x6C04_0000]);
        for _ in 0..8 {
            r.feed(&[0]);
        }
        assert!(r.in_cmd_state());
    }

    #[test]
    fn masked_unpack_reads_less_input() {
        let mut r = Rig::new();
        // STMASK with field w of every cycle taking col (code 2): V4-32
        // with m set then reads 3 fields per write -> 2 writes = 6 words.
        r.feed(&[0x2000_0000, 0x8080_8080, 0x7C02_0000]);
        for _ in 0..6 {
            r.feed(&[0]);
        }
        assert!(r.in_cmd_state());
        // Without the m flag the same unpack reads all 8 words.
        r.feed(&[0x6C02_0000]);
        for _ in 0..8 {
            r.feed(&[0]);
        }
        assert!(r.in_cmd_state());
    }

    #[test]
    fn mpg_stores_microcode() {
        let mut r = Rig::new();
        // MPG 1 doubleword to address 8 -> 2 data words.
        r.feed(&[0x4A01_0001, 0xDEAD, 0xBEEF]);
        assert!(r.in_cmd_state());
        assert_eq!(&r.vu1.micro[8..12], &[0xAD, 0xDE, 0, 0]);
    }
}
