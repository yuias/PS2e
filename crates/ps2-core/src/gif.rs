//! GIF: unpacks GIFtag streams (PATH3 / DMA channel 2) into GS register
//! writes. PACKED, REGLIST and IMAGE modes.

use crate::gs::Gs;
use tracing::{trace, warn};

#[derive(Default)]
pub struct Gif {
    /// Loops left in the current tag.
    nloop: u32,
    /// Register descriptors and progress within the current loop.
    nreg: u32,
    reg_index: u32,
    regs: u64,
    flg: u32,
    /// PRIM data supplied by the tag (PRE bit).
    eop: bool,
}

impl Gif {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn idle(&self) -> bool {
        self.nloop == 0
    }

    /// Feed one quadword from DMA.
    pub fn process(&mut self, gs: &mut Gs, lo: u64, hi: u64) {
        if self.nloop == 0 {
            // GIFtag.
            self.nloop = (lo & 0x7FFF) as u32;
            self.eop = lo & (1 << 15) != 0;
            self.flg = ((lo >> 58) & 3) as u32;
            self.nreg = ((lo >> 60) & 0xF) as u32;
            if self.nreg == 0 {
                self.nreg = 16;
            }
            self.regs = hi;
            self.reg_index = 0;
            if lo & (1 << 46) != 0 {
                // PRE: tag carries a PRIM write.
                gs.write_reg(0x00, (lo >> 47) & 0x7FF);
            }
            trace!(
                target: "ps2_core::gif",
                nloop = self.nloop,
                flg = self.flg,
                nreg = self.nreg,
                eop = self.eop,
                "tag"
            );
            return;
        }
        match self.flg {
            0 => {
                // PACKED: one register per quadword.
                let desc = (self.regs >> (self.reg_index * 4)) & 0xF;
                self.write_packed(gs, desc as u32, lo, hi);
                self.advance(1);
            }
            1 => {
                // REGLIST: two registers per quadword, raw 64-bit data.
                for data in [lo, hi] {
                    if self.nloop == 0 {
                        break;
                    }
                    let desc = ((self.regs >> (self.reg_index * 4)) & 0xF) as u8;
                    if desc != 0xE && desc < 0x63 {
                        gs.write_reg(desc, data);
                    }
                    self.advance(1);
                }
            }
            _ => {
                // IMAGE: raw data to HWREG.
                gs.write_reg(0x54, lo);
                gs.write_reg(0x54, hi);
                self.nloop -= 1;
            }
        }
    }

    fn advance(&mut self, _n: u32) {
        self.reg_index += 1;
        if self.reg_index >= self.nreg {
            self.reg_index = 0;
            self.nloop -= 1;
        }
    }

    fn write_packed(&mut self, gs: &mut Gs, desc: u32, lo: u64, hi: u64) {
        match desc {
            0x0 => gs.write_reg(0x00, lo & 0x7FF), // PRIM
            0x1 => {
                // RGBAQ: bytes spread across the words; Q from internal.
                let r = lo & 0xFF;
                let g = (lo >> 32) & 0xFF;
                let b = hi & 0xFF;
                let a = (hi >> 32) & 0xFF;
                let q = gs.packed_q as u64;
                gs.write_reg(
                    0x01,
                    r | (g << 8) | (b << 16) | (a << 24) | ((q & 0xFFFF_FFFF) << 32),
                );
            }
            0x2 => {
                // ST: also latches Q for the next RGBAQ.
                gs.packed_q = (hi & 0xFFFF_FFFF) as u32;
                gs.write_reg(0x02, lo);
            }
            0x3 => {
                // UV.
                let u = lo & 0x3FFF;
                let v = (lo >> 32) & 0x3FFF;
                gs.write_reg(0x03, u | (v << 16));
            }
            0x4 => {
                // XYZF2/3: F at 100-107, ADC at 111.
                let x = lo & 0xFFFF;
                let y = (lo >> 32) & 0xFFFF;
                let z = (hi >> 4) & 0xFF_FFFF;
                let f = (hi >> 36) & 0xFF;
                let reg = if hi & (1 << 47) != 0 { 0x0C } else { 0x04 };
                gs.write_reg(reg, x | (y << 16) | (z << 32) | (f << 56));
            }
            0x5 => {
                let x = lo & 0xFFFF;
                let y = (lo >> 32) & 0xFFFF;
                let z = hi & 0xFFFF_FFFF;
                let reg = if hi & (1 << 47) != 0 { 0x0D } else { 0x05 };
                gs.write_reg(reg, x | (y << 16) | (z << 32));
            }
            0x6 => gs.write_reg(0x06, lo),
            0x7 => gs.write_reg(0x07, lo),
            0x8 => gs.write_reg(0x08, lo),
            0x9 => gs.write_reg(0x09, lo),
            0xA => gs.write_reg(0x0A, (hi >> 36) & 0xFF), // FOG
            0xC => gs.write_reg(0x0C, lo),
            0xD => gs.write_reg(0x0D, lo),
            0xE => {
                // A+D: address in the high word.
                let addr = (hi & 0xFF) as u8;
                gs.write_reg(addr, lo);
            }
            0xF => {} // NOP
            _ => {
                warn!(target: "ps2_core::gif", desc, "unhandled packed descriptor");
            }
        }
    }
}
