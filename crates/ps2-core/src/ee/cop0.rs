//! EE System Control Coprocessor (COP0): exception state, TLB bookkeeping.

use tracing::{debug, trace};
use serde::{Deserialize, Serialize};

pub const STATUS: usize = 12;
pub const CAUSE: usize = 13;
pub const EPC: usize = 14;
pub const PRID: usize = 15;
pub const CONFIG: usize = 16;
pub const ERROR_EPC: usize = 30;

const STATUS_EXL: u32 = 1 << 1;
const STATUS_ERL: u32 = 1 << 2;
const STATUS_BEV: u32 = 1 << 22;
const STATUS_EIE: u32 = 1 << 16;

#[derive(Serialize, Deserialize)]
/// One recorded TLB entry. Recorded but not used for translation yet: the
/// kernel's mappings are identity, and the bus does a direct segment fold.
#[derive(Clone, Copy, Default)]
pub struct TlbEntry {
    pub page_mask: u32,
    pub entry_hi: u32,
    pub entry_lo0: u32,
    pub entry_lo1: u32,
}

#[derive(Serialize, Deserialize)]
pub struct Cop0 {
    pub regs: [u32; 32],
    #[serde(with = "serde_big_array::BigArray")]
    pub tlb: [TlbEntry; 48],
}

impl Default for Cop0 {
    fn default() -> Self {
        Self::new()
    }
}

impl Cop0 {
    pub fn new() -> Self {
        let mut regs = [0u32; 32];
        // Reset state: BEV=1, ERL=1 as on real hardware (0x00400004).
        regs[STATUS] = STATUS_BEV | STATUS_ERL;
        regs[PRID] = 0x0000_2E20; // EE core revision
        regs[CONFIG] = 0x0000_0440;
        Self {
            regs,
            tlb: [TlbEntry::default(); 48],
        }
    }

    /// Register read; `int0`/`int1` are the live INT0/INT1 lines merged into
    /// Cause.IP2/IP3 (they are not stored, see [`Cop0::interrupt_pending`]).
    pub fn read(&self, reg: usize, cycles: u64, int0: bool, int1: bool) -> u32 {
        match reg {
            9 => cycles as u32, // Count follows the CPU cycle counter
            CAUSE => self.regs[CAUSE] | ((int0 as u32) << 10) | ((int1 as u32) << 11),
            _ => self.regs[reg],
        }
    }

    pub fn write(&mut self, reg: usize, v: u32) {
        trace!(target: "ps2_core::ee::cop0", reg, value = format_args!("{v:#010x}"), "mtc0");
        match reg {
            PRID => {} // read-only
            CAUSE => {
                // Only the software interrupt bits are writable.
                self.regs[CAUSE] = (self.regs[CAUSE] & !0x300) | (v & 0x300);
            }
            _ => self.regs[reg] = v,
        }
    }

    /// Record the indexed TLB entry (tlbwi). No remapping yet.
    pub fn tlb_write_indexed(&mut self) {
        let idx = (self.regs[0] & 0x3F) as usize;
        if idx < self.tlb.len() {
            self.tlb[idx] = TlbEntry {
                page_mask: self.regs[5],
                entry_hi: self.regs[10],
                entry_lo0: self.regs[2],
                entry_lo1: self.regs[3],
            };
            debug!(
                target: "ps2_core::ee::cop0",
                idx,
                hi = format_args!("{:#010x}", self.regs[10]),
                lo0 = format_args!("{:#010x}", self.regs[2]),
                lo1 = format_args!("{:#010x}", self.regs[3]),
                "tlbwi (recorded, not remapped)"
            );
        }
    }

    pub fn status(&self) -> u32 {
        self.regs[STATUS]
    }

    /// Enter a general exception; returns the handler vector address.
    pub fn enter_exception(&mut self, code: u32, pc: u32, in_delay_slot: bool) -> u32 {
        // ExcCode and BD are rewritten; the software IP0/IP1 bits persist.
        let cause = (self.regs[CAUSE] & 0x300) | ((code & 0x1F) << 2);
        if in_delay_slot {
            self.regs[CAUSE] = cause | (1 << 31);
            self.regs[EPC] = pc.wrapping_sub(4);
        } else {
            self.regs[CAUSE] = cause;
            self.regs[EPC] = pc;
        }
        self.regs[STATUS] |= STATUS_EXL;
        let base = if self.regs[STATUS] & STATUS_BEV != 0 {
            0xBFC0_0200
        } else {
            0x8000_0000
        };
        // Interrupts use the dedicated V_INTERRUPT vector.
        if code == 0 {
            base + 0x200
        } else {
            base + 0x180
        }
    }

    /// Whether an interrupt should be taken given the live INT0/INT1 lines
    /// (level triggered, merged into Cause.IP2/IP3 on read rather than
    /// stored: this runs every step, so the common no-interrupt case must
    /// stay a few loads).
    #[inline]
    pub fn interrupt_pending(&self, int0: bool, int1: bool) -> bool {
        let ip = ((int0 as u32) << 10) | ((int1 as u32) << 11) | (self.regs[CAUSE] & 0x300);
        if ip == 0 {
            return false;
        }
        let status = self.regs[STATUS];
        // IE, EIE set; EXL, ERL clear.
        if status & 1 == 0 || status & STATUS_EIE == 0 || status & (STATUS_EXL | STATUS_ERL) != 0 {
            return false;
        }
        (status >> 8) & (ip >> 8) & 0xFF != 0
    }

    /// ERET: return address. ERL selects ErrorEPC; otherwise EPC is used
    /// even with EXL clear — the kernel's LoadExecPS2 path erets to EELOAD
    /// with neither bit set and relies on this.
    pub fn eret(&mut self) -> u32 {
        if self.regs[STATUS] & STATUS_ERL != 0 {
            self.regs[STATUS] &= !STATUS_ERL;
            self.regs[ERROR_EPC]
        } else {
            self.regs[STATUS] &= !STATUS_EXL;
            self.regs[EPC]
        }
    }

    pub fn set_eie(&mut self, enable: bool) {
        if enable {
            self.regs[STATUS] |= STATUS_EIE;
        } else {
            self.regs[STATUS] &= !STATUS_EIE;
        }
    }
}
