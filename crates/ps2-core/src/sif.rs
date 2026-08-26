//! SIF: the EE<->IOP subsystem interface.
//!
//! Two mailbox registers and two flag registers, visible from both sides
//! with asymmetric write semantics: each side can only *set* its own flag
//! word and *clear* the peer's. MSCOM/MSFLG belong to the EE ("main"),
//! SMCOM/SMFLG to the IOP ("sub"). The BIOS uses these for the cooperative
//! boot handshake (EESYNC); SIF DMA comes later.

use std::collections::VecDeque;
use tracing::trace;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Sif {
    pub mscom: u32,
    pub smcom: u32,
    pub msflag: u32,
    pub smflag: u32,
    pub control: u32,
    /// SIF0 FIFO: IOP -> EE, 32-bit words.
    pub fifo0: VecDeque<u32>,
    /// SIF1 FIFO: EE -> IOP, 32-bit words.
    pub fifo1: VecDeque<u32>,
}

impl Default for Sif {
    fn default() -> Self {
        Self::new()
    }
}

impl Sif {
    pub fn new() -> Self {
        Self {
            mscom: 0,
            smcom: 0,
            msflag: 0,
            smflag: 0,
            control: 0,
            fifo0: VecDeque::new(),
            fifo1: VecDeque::new(),
        }
    }

    // --- EE side (0x1000F200..0x1000F260) --------------------------------

    pub fn ee_read(&self, addr: u32) -> u32 {
        let v = match addr & 0xF0 {
            0x00 => self.mscom,
            0x10 => self.smcom,
            0x20 => self.msflag,
            0x30 => self.smflag,
            0x40 => self.control | 0xF000_0102,
            0x60 => 0x1D00_0060,
            _ => 0,
        };
        trace!(target: "ps2_core::sif", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#010x}"), "EE read");
        v
    }

    pub fn ee_write(&mut self, addr: u32, v: u32) {
        trace!(target: "ps2_core::sif", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#010x}"), "EE write");
        match addr & 0xF0 {
            0x00 => self.mscom = v,
            0x20 => self.msflag |= v,
            0x30 => self.smflag &= !v,
            0x40 => {
                // Only the EE-owned handshake bit is directly settable.
                if v & 0x100 != 0 {
                    self.control |= 0x100;
                } else {
                    self.control &= !0x100;
                }
            }
            _ => {}
        }
    }

    // --- IOP side (0x1D000000..0x1D000060) -------------------------------

    pub fn iop_read(&self, addr: u32) -> u32 {
        let v = match addr & 0xF0 {
            0x00 => self.mscom,
            0x10 => self.smcom,
            0x20 => self.msflag,
            0x30 => self.smflag,
            0x40 => self.control | 0xF000_0002,
            // Magic self-address: SIFMAN reads this to detect the SBUS.
            0x60 => 0x1D00_0060,
            _ => 0,
        };
        trace!(target: "ps2_core::sif", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#010x}"), "IOP read");
        v
    }

    pub fn iop_write(&mut self, addr: u32, v: u32) {
        trace!(target: "ps2_core::sif", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#010x}"), "IOP write");
        match addr & 0xF0 {
            0x10 => self.smcom = v,
            0x20 => self.msflag &= !v,
            0x30 => self.smflag |= v,
            // SIFMAN sets its DMA-path-ready bits (0x20 SIF0, 0x40 SIF1,
            // 0x80 SIF2) one at a time and polls for them to stick.
            0x40 => self.control |= v & 0xF0,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mailboxes_cross_over() {
        let mut sif = Sif::new();
        sif.ee_write(0x1000_F200, 0xCAFE);
        assert_eq!(sif.iop_read(0x1D00_0000), 0xCAFE);
        sif.iop_write(0x1D00_0010, 0xBEEF);
        assert_eq!(sif.ee_read(0x1000_F210), 0xBEEF);
    }

    #[test]
    fn flags_set_own_clear_peer() {
        let mut sif = Sif::new();
        sif.ee_write(0x1000_F220, 0x1_0000);
        assert_eq!(sif.iop_read(0x1D00_0020), 0x1_0000);
        sif.iop_write(0x1D00_0020, 0x1_0000); // IOP clears EE's flag
        assert_eq!(sif.ee_read(0x1000_F220), 0);

        sif.iop_write(0x1D00_0030, 0x10000);
        assert_eq!(sif.ee_read(0x1000_F230), 0x10000);
        sif.ee_write(0x1000_F230, 0x10000); // EE clears IOP's flag
        assert_eq!(sif.iop_read(0x1D00_0030), 0);
    }
}
