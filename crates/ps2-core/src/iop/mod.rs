//! IOP core: R3000A (MIPS I) interpreter.
//!
//! Essentially a PS1 CPU. Unlike the EE, it has architectural load delay
//! slots, a PS1-style COP0 mode stack (rfe), and cache isolation. No FPU,
//! no GTE (the IOP drops the PS1's COP2 unless running PS1 games).
//!
//! The reset vector runs the same BIOS ROM as the EE; the ROM dispatches on
//! COP0 PRId (< 0x59 selects the IOP path).

use crate::bus::Bus;
use tracing::{error, trace};

const EXC_INTERRUPT: u32 = 0;
const EXC_SYSCALL: u32 = 8;
const EXC_BREAK: u32 = 9;

const STATUS: usize = 12;
const CAUSE: usize = 13;
const EPC: usize = 14;
const PRID: usize = 15;

const STATUS_IEC: u32 = 1;
const STATUS_ISC: u32 = 1 << 16;
const STATUS_BEV: u32 = 1 << 22;

pub struct Cpu {
    pub pc: u32,
    pub next_pc: u32,
    pub gpr: [u32; 32],
    pub hi: u32,
    pub lo: u32,
    pub cop0: [u32; 32],
    /// Load whose value lands after the next instruction (delay slot).
    pending_load: Option<(usize, u32)>,
    /// Register directly written by the current instruction, to let it win
    /// over a pending load targeting the same register.
    written_reg: usize,
    next_is_delay: bool,
    current_pc: u32,
    in_delay: bool,
    /// Spinning on `j .` / `nop` (the kernel idle thread): only an interrupt
    /// can move it, so [`crate::Ps2System`] skips IOP steps until one is
    /// pending.
    pub idle: bool,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        let mut cop0 = [0u32; 32];
        cop0[STATUS] = STATUS_BEV;
        // Must be in [0x10, 0x59): below 0x59 the shared BIOS reset code
        // takes the IOP path, and 0x10+ marks the PS2-revision IOP so boot
        // picks the PS2 register tables instead of the PS1 compatibility
        // kernel (TBIN).
        cop0[PRID] = 0x0000_001F;
        Self {
            pc: 0xBFC0_0000,
            next_pc: 0xBFC0_0004,
            gpr: [0; 32],
            hi: 0,
            lo: 0,
            cop0,
            pending_load: None,
            written_reg: 0,
            next_is_delay: false,
            idle: false,
            current_pc: 0xBFC0_0000,
            in_delay: false,
        }
    }

    #[inline]
    fn set_reg(&mut self, i: usize, v: u32) {
        if i != 0 {
            self.gpr[i] = v;
            self.written_reg = i;
        }
    }

    /// Schedule a delayed load. A newer load to the same register discards
    /// the older pending value, matching hardware.
    #[inline]
    fn set_load(&mut self, i: usize, v: u32) {
        if i != 0 {
            self.pending_load = Some((i, v));
        }
    }

    #[inline]
    fn branch_to(&mut self, target: u32) {
        self.next_pc = target;
        self.next_is_delay = true;
    }

    #[inline]
    fn branch_cond(&mut self, taken: bool, imm: u32) {
        if taken {
            let offset = ((imm as i16 as i32) << 2) as u32;
            self.next_pc = self.pc.wrapping_add(offset);
        }
        self.next_is_delay = true;
    }

    fn exception(&mut self, code: u32) {
        let status = self.cop0[STATUS];
        // Push the interrupt/kernel mode stack (KUc/IEc -> KUp/IEp -> KUo/IEo).
        self.cop0[STATUS] = (status & !0x3F) | ((status << 2) & 0x3F);
        if self.in_delay {
            self.cop0[CAUSE] = (code << 2) | (1 << 31);
            self.cop0[EPC] = self.current_pc.wrapping_sub(4);
        } else {
            self.cop0[CAUSE] = code << 2;
            self.cop0[EPC] = self.current_pc;
        }
        let vector = if status & STATUS_BEV != 0 {
            0xBFC0_0180
        } else {
            0x8000_0080
        };
        trace!(
            target: "ps2_core::iop::cpu",
            code,
            pc = format_args!("{:#010x}", self.current_pc),
            "exception"
        );
        self.pc = vector;
        self.next_pc = vector.wrapping_add(4);
        self.next_is_delay = false;
    }

    fn unimplemented(&self, kind: &str, instr: u32) -> ! {
        error!(
            target: "ps2_core::iop::cpu",
            pc = format_args!("{:#010x}", self.current_pc),
            instr = format_args!("{instr:#010x}"),
            "unimplemented {kind}"
        );
        panic!(
            "unimplemented IOP {kind}: instr {instr:#010x} at pc {:#010x}",
            self.current_pc
        );
    }

    /// True when stores must be swallowed (cache isolation during cache init).
    #[inline]
    fn cache_isolated(&self) -> bool {
        self.cop0[STATUS] & STATUS_ISC != 0
    }

    /// Whether an interrupt would be taken at the next step (IEc, IM bit 10
    /// and an unmasked pending line). Used to end an idle-loop skip.
    #[inline]
    pub fn interrupt_pending(&self, bus: &Bus) -> bool {
        self.cop0[STATUS] & STATUS_IEC != 0
            && self.cop0[STATUS] & (1 << 10) != 0
            && bus.iop_irq_pending()
    }

    pub fn step(&mut self, bus: &mut Bus) {
        if self.interrupt_pending(bus) {
            // A load issued before the interrupt still writes back: the
            // handler's first instruction sees the loaded value, as on the
            // R3000, rather than the stale register.
            if let Some((reg, v)) = self.pending_load.take() {
                self.gpr[reg] = v;
            }
            self.cop0[CAUSE] = (self.cop0[CAUSE] & !0xFF) | (1 << 10);
            self.in_delay = self.next_is_delay;
            self.current_pc = self.pc;
            self.exception(EXC_INTERRUPT);
        }
        self.current_pc = self.pc;
        self.in_delay = self.next_is_delay;
        self.next_is_delay = false;
        let instr = bus.iop_read32(self.pc);
        crate::prof::count_iop(self.pc);
        self.pc = self.next_pc;
        self.next_pc = self.pc.wrapping_add(4);

        // Leave the pending load visible during execute: lwl/lwr merge with
        // it via load_merge_base (hardware forwards the in-flight value).
        let pending = self.pending_load;
        self.written_reg = 0;
        self.execute(instr, bus);
        if let Some((reg, v)) = pending {
            if self.pending_load == pending {
                self.pending_load = None;
            }
            if reg != self.written_reg {
                self.gpr[reg] = v;
            }
        }
    }

    fn execute(&mut self, instr: u32, bus: &mut Bus) {
        let op = instr >> 26;
        let rs = ((instr >> 21) & 31) as usize;
        let rt = ((instr >> 16) & 31) as usize;
        let rd = ((instr >> 11) & 31) as usize;
        let sa = (instr >> 6) & 31;
        let imm = instr & 0xFFFF;
        let simm = imm as i16 as i32 as u32;
        let addr = self.gpr[rs].wrapping_add(simm);

        match op {
            0x00 => match instr & 0x3F {
                0x00 => self.set_reg(rd, self.gpr[rt] << sa),
                0x02 => self.set_reg(rd, self.gpr[rt] >> sa),
                0x03 => self.set_reg(rd, ((self.gpr[rt] as i32) >> sa) as u32),
                0x04 => self.set_reg(rd, self.gpr[rt] << (self.gpr[rs] & 31)),
                0x06 => self.set_reg(rd, self.gpr[rt] >> (self.gpr[rs] & 31)),
                0x07 => self.set_reg(rd, ((self.gpr[rt] as i32) >> (self.gpr[rs] & 31)) as u32),
                0x08 => self.branch_to(self.gpr[rs]),
                0x09 => {
                    let target = self.gpr[rs];
                    self.set_reg(rd, self.next_pc);
                    self.branch_to(target);
                }
                0x0C => self.exception(EXC_SYSCALL),
                0x0D => self.exception(EXC_BREAK),
                0x10 => self.set_reg(rd, self.hi),
                0x11 => self.hi = self.gpr[rs],
                0x12 => self.set_reg(rd, self.lo),
                0x13 => self.lo = self.gpr[rs],
                0x18 => {
                    let prod = (self.gpr[rs] as i32 as i64) * (self.gpr[rt] as i32 as i64);
                    self.lo = prod as u32;
                    self.hi = (prod >> 32) as u32;
                }
                0x19 => {
                    let prod = (self.gpr[rs] as u64) * (self.gpr[rt] as u64);
                    self.lo = prod as u32;
                    self.hi = (prod >> 32) as u32;
                }
                0x1A => {
                    let n = self.gpr[rs] as i32;
                    let d = self.gpr[rt] as i32;
                    let (q, r) = if d == 0 {
                        (if n >= 0 { -1 } else { 1 }, n)
                    } else if n == i32::MIN && d == -1 {
                        (i32::MIN, 0)
                    } else {
                        (n / d, n % d)
                    };
                    self.lo = q as u32;
                    self.hi = r as u32;
                }
                0x1B => {
                    let n = self.gpr[rs];
                    let d = self.gpr[rt];
                    let (q, r) = match (n.checked_div(d), n.checked_rem(d)) {
                        (Some(q), Some(r)) => (q, r),
                        _ => (u32::MAX, n),
                    };
                    self.lo = q;
                    self.hi = r;
                }
                // add/addu: overflow exception not modeled, as on the EE
                0x20 | 0x21 => self.set_reg(rd, self.gpr[rs].wrapping_add(self.gpr[rt])),
                0x22 | 0x23 => self.set_reg(rd, self.gpr[rs].wrapping_sub(self.gpr[rt])),
                0x24 => self.set_reg(rd, self.gpr[rs] & self.gpr[rt]),
                0x25 => self.set_reg(rd, self.gpr[rs] | self.gpr[rt]),
                0x26 => self.set_reg(rd, self.gpr[rs] ^ self.gpr[rt]),
                0x27 => self.set_reg(rd, !(self.gpr[rs] | self.gpr[rt])),
                0x2A => self.set_reg(rd, ((self.gpr[rs] as i32) < self.gpr[rt] as i32) as u32),
                0x2B => self.set_reg(rd, (self.gpr[rs] < self.gpr[rt]) as u32),
                _ => self.unimplemented("SPECIAL", instr),
            },
            0x01 => {
                let taken = if rt & 1 == 0 {
                    (self.gpr[rs] as i32) < 0
                } else {
                    (self.gpr[rs] as i32) >= 0
                };
                // bltzal/bgezal link unconditionally.
                if rt & 0x1E == 0x10 {
                    self.set_reg(31, self.next_pc);
                }
                self.branch_cond(taken, imm);
            }
            0x02 => {
                let target = (self.pc & 0xF000_0000) | ((instr & 0x03FF_FFFF) << 2);
                self.branch_to(target);
                // `j .` with a nop delay slot is the kernel idle thread.
                if target == self.current_pc && bus.iop_read32(self.pc) == 0 {
                    self.idle = true;
                }
            }
            0x03 => {
                self.set_reg(31, self.next_pc);
                self.branch_to((self.pc & 0xF000_0000) | ((instr & 0x03FF_FFFF) << 2));
            }
            0x04 => self.branch_cond(self.gpr[rs] == self.gpr[rt], imm),
            0x05 => self.branch_cond(self.gpr[rs] != self.gpr[rt], imm),
            0x06 => self.branch_cond((self.gpr[rs] as i32) <= 0, imm),
            0x07 => self.branch_cond((self.gpr[rs] as i32) > 0, imm),
            0x08 | 0x09 => self.set_reg(rt, self.gpr[rs].wrapping_add(simm)),
            0x0A => self.set_reg(rt, ((self.gpr[rs] as i32) < simm as i32) as u32),
            0x0B => self.set_reg(rt, (self.gpr[rs] < simm) as u32),
            0x0C => self.set_reg(rt, self.gpr[rs] & imm),
            0x0D => self.set_reg(rt, self.gpr[rs] | imm),
            0x0E => self.set_reg(rt, self.gpr[rs] ^ imm),
            0x0F => self.set_reg(rt, imm << 16),
            0x10 => match rs {
                0x00 => {
                    let v = self.cop0[rd];
                    self.set_load(rt, v);
                }
                0x04 => {
                    let v = self.gpr[rt];
                    match rd {
                        CAUSE => {
                            self.cop0[CAUSE] = (self.cop0[CAUSE] & !0x300) | (v & 0x300);
                        }
                        PRID => {}
                        _ => self.cop0[rd] = v,
                    }
                }
                0x10 => match instr & 0x3F {
                    // rfe: pop the mode stack.
                    0x10 => {
                        let s = self.cop0[STATUS];
                        self.cop0[STATUS] = (s & !0xF) | ((s >> 2) & 0xF);
                    }
                    _ => self.unimplemented("COP0", instr),
                },
                _ => self.unimplemented("COP0", instr),
            },
            0x20 => {
                let v = bus.iop_read8(addr);
                self.set_load(rt, v as i8 as i32 as u32);
            }
            0x21 => {
                let v = bus.iop_read16(addr);
                self.set_load(rt, v as i16 as i32 as u32);
            }
            0x22 => {
                let shift = (addr & 3) * 8;
                let mem = bus.iop_read32(addr & !3);
                let cur = self.load_merge_base(rt);
                let mask = 0x00FF_FFFFu32.checked_shr(shift).unwrap_or(0);
                self.set_load(rt, (cur & mask) | (mem << (24 - shift)));
            }
            0x23 => {
                let v = bus.iop_read32(addr);
                self.set_load(rt, v);
            }
            0x24 => {
                let v = bus.iop_read8(addr);
                self.set_load(rt, v as u32);
            }
            0x25 => {
                let v = bus.iop_read16(addr);
                self.set_load(rt, v as u32);
            }
            0x26 => {
                let shift = (addr & 3) * 8;
                let mem = bus.iop_read32(addr & !3);
                let cur = self.load_merge_base(rt);
                let keep = if shift == 0 { 0 } else { !(u32::MAX >> shift) };
                self.set_load(rt, (cur & keep) | (mem >> shift));
            }
            0x28 => {
                if !self.cache_isolated() {
                    bus.iop_write8(addr, self.gpr[rt] as u8);
                }
            }
            0x29 => {
                if !self.cache_isolated() {
                    bus.iop_write16(addr, self.gpr[rt] as u16);
                }
            }
            0x2A => {
                if !self.cache_isolated() {
                    let shift = (addr & 3) * 8;
                    let aligned = addr & !3;
                    let mem = bus.iop_read32(aligned);
                    let keep = 0xFFFF_FF00u32.checked_shl(shift).unwrap_or(0);
                    bus.iop_write32(aligned, (self.gpr[rt] >> (24 - shift)) | (mem & keep));
                }
            }
            0x2B => {
                if !self.cache_isolated() {
                    bus.iop_write32(addr, self.gpr[rt]);
                }
            }
            0x2E => {
                if !self.cache_isolated() {
                    let shift = (addr & 3) * 8;
                    let aligned = addr & !3;
                    let mem = bus.iop_read32(aligned);
                    let keep = if shift == 0 {
                        0
                    } else {
                        u32::MAX >> (32 - shift)
                    };
                    bus.iop_write32(aligned, (self.gpr[rt] << shift) | (mem & keep));
                }
            }
            _ => self.unimplemented("opcode", instr),
        }
    }

    /// lwl/lwr merge with an in-flight load to the same register (hardware
    /// forwards the pending value so lwl+lwr pairs work back to back).
    fn load_merge_base(&self, rt: usize) -> u32 {
        match self.pending_load {
            Some((reg, v)) if reg == rt => v,
            _ => self.gpr[rt],
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Ps2System;
    use crate::bus::BIOS_SIZE;

    /// BIOS with `code` at the reset vector; only the IOP is stepped.
    fn iop_system(code: &[u32]) -> Ps2System {
        let mut bios = vec![0u8; BIOS_SIZE];
        for (i, w) in code.iter().enumerate() {
            bios[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        Ps2System::new(bios).unwrap()
    }

    fn step_iop(sys: &mut Ps2System, n: usize) {
        for _ in 0..n {
            let iop = &mut sys.iop;
            iop.step(&mut sys.bus);
        }
    }

    #[test]
    fn load_delay_slot_sees_old_value() {
        let mut sys = iop_system(&[
            0x3C01_0000, // lui $at, 0 (RAM address 0)
            0x2408_0055, // addiu $t0, $0, 0x55
            0xAC28_0000, // sw $t0, 0($at)
            0x8C29_0000, // lw $t1, 0($at)
            0x0009_5021, // addu $t2, $0, $t1  (delay slot: old $t1 = 0)
            0x0009_5821, // addu $t3, $0, $t1  (new value visible)
        ]);
        step_iop(&mut sys, 6);
        assert_eq!(sys.iop.gpr[10], 0);
        assert_eq!(sys.iop.gpr[11], 0x55);
    }

    #[test]
    fn lwl_lwr_pair_merges_unaligned_word() {
        // cdvdman copies S-command results with back-to-back lwl/lwr; the
        // lwr must see the lwl's in-flight value or byte 3 is lost.
        let mut sys = iop_system(&[
            0x3C01_0000, // lui $at, 0
            0x3C09_4433, // lui $t1, 0x4433
            0x3529_2211, // ori $t1, 0x2211
            0xAC29_0000, // sw  $t1, 0($at)
            0x3C0A_8877, // lui $t2, 0x8877
            0x354A_6655, // ori $t2, 0x6655
            0xAC2A_0004, // sw  $t2, 4($at)
            0x8828_0004, // lwl $t0, 4($at)
            0x9828_0001, // lwr $t0, 1($at)
            0x0000_0000, // nop (lwr's load delay)
        ]);
        step_iop(&mut sys, 10);
        assert_eq!(sys.iop.gpr[8], 0x5544_3322);
    }

    #[test]
    fn write_in_delay_slot_beats_pending_load() {
        let mut sys = iop_system(&[
            0x3C01_0000, // lui $at, 0
            0x2408_0055, // addiu $t0, $0, 0x55
            0xAC28_0000, // sw $t0, 0($at)
            0x8C29_0000, // lw $t1, 0($at)
            0x2409_0077, // addiu $t1, $0, 0x77 (delay slot writes $t1: wins)
        ]);
        step_iop(&mut sys, 5);
        assert_eq!(sys.iop.gpr[9], 0x77);
    }

    #[test]
    fn interrupt_completes_the_pending_load() {
        // BEV is still set at reset, so the handler lives at 0xBFC00180;
        // the gap is zero-filled (nop).
        let mut code = [0u32; 0x61];
        code[..4].copy_from_slice(&[
            0x3C01_0000, // lui $at, 0
            0x2408_0055, // addiu $t0, $0, 0x55
            0xAC28_0000, // sw $t0, 0($at)
            0x8C29_0000, // lw $t1, 0($at)  (load still in flight)
        ]);
        code[0x60] = 0x0009_5021; // addu $t2, $0, $t1
        let mut sys = iop_system(&code);
        sys.iop.cop0[super::STATUS] |= super::STATUS_IEC | (1 << 10);
        step_iop(&mut sys, 4);

        sys.bus.iop_i_stat = 1;
        sys.bus.iop_i_mask = 1;
        sys.bus.iop_i_ctrl = 1;
        step_iop(&mut sys, 1);
        assert_eq!(sys.iop.gpr[10], 0x55);
    }

    #[test]
    fn cache_isolation_swallows_stores() {
        let mut sys = iop_system(&[
            0x3C08_0001, // lui $t0, 1 (Isc)... value 0x10000
            0x4088_6000, // mtc0 $t0, $12 (Status.Isc = 1)
            0x2409_0033, // addiu $t1, $0, 0x33
            0xAC09_0004, // sw $t1, 4($0)   (swallowed)
            0x4080_6000, // mtc0 $0, $12    (isolation off)
            0x8C0A_0004, // lw $t2, 4($0)
            0x0000_0000, // nop
        ]);
        step_iop(&mut sys, 7);
        assert_eq!(sys.iop.gpr[10], 0);
    }
}
