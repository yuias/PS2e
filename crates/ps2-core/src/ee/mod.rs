//! EE core: R5900 (MIPS III/IV + 128-bit MMI) interpreter.
//!
//! Plain `match`-based interpreter, no dispatch tables or JIT. GPRs are
//! 128-bit (`[u64; 2]`, little end first); 32/64-bit ops touch only the low
//! half, as on hardware. Branch delay slots are modeled with the pc/next_pc
//! pair; the R5900 has no load delay slots.

pub mod cop0;
pub mod fpu;

use crate::bus::Bus;
use cop0::Cop0;
use fpu::Fpu;
use tracing::{error, trace, warn};

/// Exception codes (Cause.ExcCode).
const EXC_INTERRUPT: u32 = 0;
const EXC_SYSCALL: u32 = 8;
const EXC_BREAK: u32 = 9;
const EXC_TRAP: u32 = 13;

pub struct Cpu {
    pub pc: u32,
    pub next_pc: u32,
    /// 128-bit GPRs, low u64 first. gpr[0] stays zero.
    pub gpr: [[u64; 2]; 32],
    /// HI/LO for multiply pipelines 0 and 1.
    pub hi: [u64; 2],
    pub lo: [u64; 2],
    /// Shift-amount register (funnel shifts / QFSRV).
    pub sa: u32,
    pub cop0: Cop0,
    pub fpu: Fpu,
    /// VU0 macro-mode register shadow until the VUs exist.
    pub vu0_vf: [[u64; 2]; 32],
    pub vu0_ctrl: [u32; 32],
    /// The next instruction to execute sits in a branch delay slot.
    next_is_delay: bool,
    /// PC of the instruction currently executing (for diagnostics/exceptions).
    current_pc: u32,
    in_delay: bool,
    warned_vu0_macro: bool,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

impl Cpu {
    pub fn new() -> Self {
        Self {
            pc: 0xBFC0_0000,
            next_pc: 0xBFC0_0004,
            gpr: [[0; 2]; 32],
            hi: [0; 2],
            lo: [0; 2],
            sa: 0,
            cop0: Cop0::new(),
            fpu: Fpu::new(),
            vu0_vf: [[0; 2]; 32],
            vu0_ctrl: [0; 32],
            next_is_delay: false,
            current_pc: 0xBFC0_0000,
            in_delay: false,
            warned_vu0_macro: false,
        }
    }

    // --- register helpers ------------------------------------------------

    #[inline]
    fn r64(&self, i: usize) -> u64 {
        self.gpr[i][0]
    }
    #[inline]
    fn r32(&self, i: usize) -> u32 {
        self.gpr[i][0] as u32
    }
    #[inline]
    fn r128(&self, i: usize) -> [u64; 2] {
        self.gpr[i]
    }
    #[inline]
    fn set64(&mut self, i: usize, v: u64) {
        if i != 0 {
            self.gpr[i][0] = v;
        }
    }
    /// 32-bit result, sign-extended into the low 64 bits.
    #[inline]
    fn set32(&mut self, i: usize, v: u32) {
        self.set64(i, v as i32 as u64);
    }
    #[inline]
    fn set128(&mut self, i: usize, v: [u64; 2]) {
        if i != 0 {
            self.gpr[i] = v;
        }
    }

    // --- control flow ----------------------------------------------------

    #[inline]
    fn branch_to(&mut self, target: u32) {
        self.next_pc = target;
        self.next_is_delay = true;
    }

    /// Relative branch; `likely` skips the delay slot when not taken.
    #[inline]
    fn branch_cond(&mut self, taken: bool, imm: u32, likely: bool) {
        if taken {
            let offset = ((imm as i16 as i32) << 2) as u32;
            self.next_pc = self.pc.wrapping_add(offset);
            self.next_is_delay = true;
        } else if likely {
            self.pc = self.next_pc;
            self.next_pc = self.pc.wrapping_add(4);
        } else {
            self.next_is_delay = true;
        }
    }

    fn exception(&mut self, code: u32) {
        let vector = self
            .cop0
            .enter_exception(code, self.current_pc, self.in_delay);
        trace!(
            target: "ps2_core::ee::cpu",
            code,
            pc = format_args!("{:#010x}", self.current_pc),
            vector = format_args!("{vector:#010x}"),
            "exception"
        );
        self.pc = vector;
        self.next_pc = vector.wrapping_add(4);
        self.next_is_delay = false;
    }

    fn unimplemented(&self, kind: &str, instr: u32) -> ! {
        error!(
            target: "ps2_core::ee::cpu",
            pc = format_args!("{:#010x}", self.current_pc),
            instr = format_args!("{instr:#010x}"),
            "unimplemented {kind}"
        );
        panic!(
            "unimplemented EE {kind}: instr {instr:#010x} at pc {:#010x}",
            self.current_pc
        );
    }

    // --- main loop -------------------------------------------------------

    pub fn step(&mut self, bus: &mut Bus) {
        debug_assert_eq!(self.gpr[0], [0, 0]);
        if self.pc == 0 {
            panic!("EE jumped to null (previous pc {:#010x})", self.current_pc);
        }
        if self
            .cop0
            .interrupt_pending(bus.ee_int0_pending(), bus.ee_int1_pending())
        {
            self.current_pc = self.pc;
            self.in_delay = self.next_is_delay;
            self.exception(EXC_INTERRUPT);
        }
        self.current_pc = self.pc;
        self.in_delay = self.next_is_delay;
        self.next_is_delay = false;
        let instr = bus.fetch32(self.pc);
        self.pc = self.next_pc;
        self.next_pc = self.pc.wrapping_add(4);
        self.execute(instr, bus);
    }

    fn execute(&mut self, instr: u32, bus: &mut Bus) {
        let op = instr >> 26;
        let rs = ((instr >> 21) & 31) as usize;
        let rt = ((instr >> 16) & 31) as usize;
        let rd = ((instr >> 11) & 31) as usize;
        let sa = (instr >> 6) & 31;
        let imm = instr & 0xFFFF;
        let simm64 = imm as i16 as i64 as u64;
        let addr = |cpu: &Cpu| cpu.r32(rs).wrapping_add(imm as i16 as i32 as u32);

        match op {
            0x00 => self.op_special(instr, rs, rt, rd, sa),
            0x01 => self.op_regimm(instr, rs, rt, imm),
            // j / jal
            0x02 => self.branch_to((self.pc & 0xF000_0000) | ((instr & 0x03FF_FFFF) << 2)),
            0x03 => {
                self.set32(31, self.next_pc);
                self.branch_to((self.pc & 0xF000_0000) | ((instr & 0x03FF_FFFF) << 2));
            }
            0x04 => self.branch_cond(self.r64(rs) == self.r64(rt), imm, false),
            0x05 => self.branch_cond(self.r64(rs) != self.r64(rt), imm, false),
            0x06 => self.branch_cond((self.r64(rs) as i64) <= 0, imm, false),
            0x07 => self.branch_cond((self.r64(rs) as i64) > 0, imm, false),
            // addi: overflow exception not modeled (see ARCHITECTURE.md)
            0x08 | 0x09 => self.set32(rt, self.r32(rs).wrapping_add(simm64 as u32)),
            0x0A => self.set64(rt, ((self.r64(rs) as i64) < simm64 as i64) as u64),
            0x0B => self.set64(rt, (self.r64(rs) < simm64) as u64),
            0x0C => self.set64(rt, self.r64(rs) & imm as u64),
            0x0D => self.set64(rt, self.r64(rs) | imm as u64),
            0x0E => self.set64(rt, self.r64(rs) ^ imm as u64),
            0x0F => self.set32(rt, imm << 16),
            0x10 => self.op_cop0(instr, rs, rt, rd, bus),
            0x11 => self.op_cop1(instr, rs, rt, rd, sa),
            0x12 => self.op_cop2(instr, rs, rt, rd),
            0x14 => self.branch_cond(self.r64(rs) == self.r64(rt), imm, true),
            0x15 => self.branch_cond(self.r64(rs) != self.r64(rt), imm, true),
            0x16 => self.branch_cond((self.r64(rs) as i64) <= 0, imm, true),
            0x17 => self.branch_cond((self.r64(rs) as i64) > 0, imm, true),
            0x18 | 0x19 => self.set64(rt, self.r64(rs).wrapping_add(simm64)),
            0x1A => self.op_ldl(rt, addr(self), bus),
            0x1B => self.op_ldr(rt, addr(self), bus),
            0x1C => self.op_mmi(instr, rs, rt, rd, sa),
            0x1E => {
                let v = bus.read128(addr(self) & !0xF);
                self.set128(rt, v);
            }
            0x1F => bus.write128(addr(self) & !0xF, self.r128(rt)),
            0x20 => {
                let v = bus.read8(addr(self));
                self.set64(rt, v as i8 as i64 as u64);
            }
            0x21 => {
                let v = bus.read16(addr(self));
                self.set64(rt, v as i16 as i64 as u64);
            }
            0x22 => self.op_lwl(rt, addr(self), bus),
            0x23 => {
                let v = bus.read32(addr(self));
                self.set32(rt, v);
            }
            0x24 => {
                let v = bus.read8(addr(self));
                self.set64(rt, v as u64);
            }
            0x25 => {
                let v = bus.read16(addr(self));
                self.set64(rt, v as u64);
            }
            0x26 => self.op_lwr(rt, addr(self), bus),
            0x27 => {
                let v = bus.read32(addr(self));
                self.set64(rt, v as u64);
            }
            0x28 => bus.write8(addr(self), self.r64(rt) as u8),
            0x29 => bus.write16(addr(self), self.r64(rt) as u16),
            0x2A => self.op_swl(rt, addr(self), bus),
            0x2B => bus.write32(addr(self), self.r32(rt)),
            0x2C => self.op_sdl(rt, addr(self), bus),
            0x2D => self.op_sdr(rt, addr(self), bus),
            0x2E => self.op_swr(rt, addr(self), bus),
            0x2F => trace!(target: "ps2_core::ee::cpu", "cache op (nop)"),
            0x31 => {
                let v = bus.read32(addr(self));
                self.fpu.regs[rt] = f32::from_bits(v);
            }
            0x33 => {} // pref
            0x36 => {
                let v = bus.read128(addr(self) & !0xF);
                self.vu0_vf[rt] = v;
            }
            0x37 => {
                let v = bus.read64(addr(self));
                self.set64(rt, v);
            }
            0x39 => bus.write32(addr(self), self.fpu.regs[rt].to_bits()),
            0x3E => bus.write128(addr(self) & !0xF, self.vu0_vf[rt]),
            0x3F => bus.write64(addr(self), self.r64(rt)),
            _ => self.unimplemented("opcode", instr),
        }
    }

    // --- SPECIAL ---------------------------------------------------------

    fn op_special(&mut self, instr: u32, rs: usize, rt: usize, rd: usize, sa: u32) {
        match instr & 0x3F {
            0x00 => self.set32(rd, self.r32(rt) << sa),
            0x02 => self.set32(rd, self.r32(rt) >> sa),
            0x03 => self.set32(rd, ((self.r32(rt) as i32) >> sa) as u32),
            0x04 => self.set32(rd, self.r32(rt) << (self.r32(rs) & 31)),
            0x06 => self.set32(rd, self.r32(rt) >> (self.r32(rs) & 31)),
            0x07 => self.set32(rd, ((self.r32(rt) as i32) >> (self.r32(rs) & 31)) as u32),
            0x08 => self.branch_to(self.r32(rs)),
            0x09 => {
                let target = self.r32(rs);
                self.set32(rd, self.next_pc);
                self.branch_to(target);
            }
            0x0A => {
                if self.r64(rt) == 0 {
                    self.set64(rd, self.r64(rs));
                }
            }
            0x0B => {
                if self.r64(rt) != 0 {
                    self.set64(rd, self.r64(rs));
                }
            }
            0x0C => {
                trace!(
                    target: "ps2_core::ee::syscall",
                    num = format_args!("{:#x}", self.r32(3)),
                    pc = format_args!("{:#010x}", self.current_pc),
                    a0 = format_args!("{:#010x}", self.r32(4)),
                    a1 = format_args!("{:#010x}", self.r32(5)),
                    "syscall"
                );
                self.exception(EXC_SYSCALL)
            }
            0x0D => self.exception(EXC_BREAK),
            0x0F => {} // sync
            0x10 => self.set64(rd, self.hi[0]),
            0x11 => self.hi[0] = self.r64(rs),
            0x12 => self.set64(rd, self.lo[0]),
            0x13 => self.lo[0] = self.r64(rs),
            0x14 => self.set64(rd, self.r64(rt) << (self.r32(rs) & 63)),
            0x16 => self.set64(rd, self.r64(rt) >> (self.r32(rs) & 63)),
            0x17 => self.set64(rd, ((self.r64(rt) as i64) >> (self.r32(rs) & 63)) as u64),
            0x18 => self.mult(0, rd, rs, rt),
            0x19 => self.multu(0, rd, rs, rt),
            0x1A => self.div(0, rs, rt),
            0x1B => self.divu(0, rs, rt),
            0x20 | 0x21 => self.set32(rd, self.r32(rs).wrapping_add(self.r32(rt))),
            0x22 | 0x23 => self.set32(rd, self.r32(rs).wrapping_sub(self.r32(rt))),
            0x24 => self.set64(rd, self.r64(rs) & self.r64(rt)),
            0x25 => self.set64(rd, self.r64(rs) | self.r64(rt)),
            0x26 => self.set64(rd, self.r64(rs) ^ self.r64(rt)),
            0x27 => self.set64(rd, !(self.r64(rs) | self.r64(rt))),
            0x28 => self.set64(rd, self.sa as u64),
            0x29 => self.sa = self.r32(rs),
            0x2A => self.set64(rd, ((self.r64(rs) as i64) < self.r64(rt) as i64) as u64),
            0x2B => self.set64(rd, (self.r64(rs) < self.r64(rt)) as u64),
            0x2C | 0x2D => self.set64(rd, self.r64(rs).wrapping_add(self.r64(rt))),
            0x2E | 0x2F => self.set64(rd, self.r64(rs).wrapping_sub(self.r64(rt))),
            // conditional traps
            0x30 => self.trap((self.r64(rs) as i64) >= self.r64(rt) as i64),
            0x31 => self.trap(self.r64(rs) >= self.r64(rt)),
            0x32 => self.trap((self.r64(rs) as i64) < self.r64(rt) as i64),
            0x33 => self.trap(self.r64(rs) < self.r64(rt)),
            0x34 => self.trap(self.r64(rs) == self.r64(rt)),
            0x36 => self.trap(self.r64(rs) != self.r64(rt)),
            0x38 => self.set64(rd, self.r64(rt) << sa),
            0x3A => self.set64(rd, self.r64(rt) >> sa),
            0x3B => self.set64(rd, ((self.r64(rt) as i64) >> sa) as u64),
            0x3C => self.set64(rd, self.r64(rt) << (sa + 32)),
            0x3E => self.set64(rd, self.r64(rt) >> (sa + 32)),
            0x3F => self.set64(rd, ((self.r64(rt) as i64) >> (sa + 32)) as u64),
            _ => self.unimplemented("SPECIAL", instr),
        }
    }

    fn trap(&mut self, cond: bool) {
        if cond {
            self.exception(EXC_TRAP);
        }
    }

    fn op_regimm(&mut self, instr: u32, rs: usize, rt: usize, imm: u32) {
        match rt {
            0x00 => self.branch_cond((self.r64(rs) as i64) < 0, imm, false),
            0x01 => self.branch_cond((self.r64(rs) as i64) >= 0, imm, false),
            0x02 => self.branch_cond((self.r64(rs) as i64) < 0, imm, true),
            0x03 => self.branch_cond((self.r64(rs) as i64) >= 0, imm, true),
            0x10 | 0x12 => {
                self.set32(31, self.next_pc);
                self.branch_cond((self.r64(rs) as i64) < 0, imm, rt == 0x12);
            }
            0x11 | 0x13 => {
                self.set32(31, self.next_pc);
                self.branch_cond((self.r64(rs) as i64) >= 0, imm, rt == 0x13);
            }
            0x18 => self.sa = ((self.r32(rs) & 0xF) ^ (imm & 0xF)) * 8,
            0x19 => self.sa = ((self.r32(rs) & 0x7) ^ (imm & 0x7)) * 16,
            _ => self.unimplemented("REGIMM", instr),
        }
    }

    // --- multiply / divide -----------------------------------------------

    fn mult(&mut self, pipe: usize, rd: usize, rs: usize, rt: usize) {
        let prod = (self.r32(rs) as i32 as i64) * (self.r32(rt) as i32 as i64);
        self.lo[pipe] = prod as i32 as u64;
        self.hi[pipe] = (prod >> 32) as u64;
        // R5900 three-operand form: rd receives LO.
        self.set64(rd, self.lo[pipe]);
    }

    fn multu(&mut self, pipe: usize, rd: usize, rs: usize, rt: usize) {
        let prod = (self.r32(rs) as u64) * (self.r32(rt) as u64);
        self.lo[pipe] = prod as u32 as i32 as u64;
        self.hi[pipe] = ((prod >> 32) as u32) as i32 as u64;
        self.set64(rd, self.lo[pipe]);
    }

    fn div(&mut self, pipe: usize, rs: usize, rt: usize) {
        let n = self.r32(rs) as i32;
        let d = self.r32(rt) as i32;
        let (q, r) = if d == 0 {
            (if n >= 0 { -1 } else { 1 }, n)
        } else if n == i32::MIN && d == -1 {
            (i32::MIN, 0)
        } else {
            (n / d, n % d)
        };
        self.lo[pipe] = q as i64 as u64;
        self.hi[pipe] = r as i64 as u64;
    }

    fn divu(&mut self, pipe: usize, rs: usize, rt: usize) {
        let n = self.r32(rs);
        let d = self.r32(rt);
        let (q, r) = match (n.checked_div(d), n.checked_rem(d)) {
            (Some(q), Some(r)) => (q, r),
            _ => (u32::MAX, n),
        };
        self.lo[pipe] = q as i32 as u64;
        self.hi[pipe] = r as i32 as u64;
    }

    fn madd(&mut self, pipe: usize, rd: usize, rs: usize, rt: usize, unsigned: bool) {
        let acc = ((self.hi[pipe] as u32 as u64) << 32) | (self.lo[pipe] as u32 as u64);
        let prod = if unsigned {
            (self.r32(rs) as u64).wrapping_mul(self.r32(rt) as u64)
        } else {
            ((self.r32(rs) as i32 as i64).wrapping_mul(self.r32(rt) as i32 as i64)) as u64
        };
        let sum = acc.wrapping_add(prod);
        self.lo[pipe] = sum as u32 as i32 as u64;
        self.hi[pipe] = ((sum >> 32) as u32) as i32 as u64;
        self.set64(rd, self.lo[pipe]);
    }

    // --- unaligned load/store --------------------------------------------

    fn op_lwl(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 3) * 8;
        let mem = bus.read32(addr & !3);
        let mask = 0x00FF_FFFFu32.checked_shr(shift).unwrap_or(0);
        let v = (self.r32(rt) & mask) | (mem << (24 - shift));
        self.set32(rt, v);
    }

    fn op_lwr(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 3) * 8;
        let mem = bus.read32(addr & !3);
        let keep = if shift == 0 { 0 } else { !(u32::MAX >> shift) };
        let v = (self.r32(rt) & keep) | (mem >> shift);
        self.set32(rt, v);
    }

    fn op_swl(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 3) * 8;
        let aligned = addr & !3;
        let mem = bus.read32(aligned);
        let keep = 0xFFFF_FF00u32.checked_shl(shift).unwrap_or(0);
        bus.write32(aligned, (self.r32(rt) >> (24 - shift)) | (mem & keep));
    }

    fn op_swr(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 3) * 8;
        let aligned = addr & !3;
        let mem = bus.read32(aligned);
        let keep = if shift == 0 {
            0
        } else {
            u32::MAX >> (32 - shift)
        };
        bus.write32(aligned, (self.r32(rt) << shift) | (mem & keep));
    }

    fn op_ldl(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 7) * 8;
        let mem = bus.read64(addr & !7);
        let keep = 0x00FF_FFFF_FFFF_FFFFu64.checked_shr(shift).unwrap_or(0);
        self.set64(rt, (self.r64(rt) & keep) | (mem << (56 - shift)));
    }

    fn op_ldr(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 7) * 8;
        let mem = bus.read64(addr & !7);
        let keep = if shift == 0 { 0 } else { !(u64::MAX >> shift) };
        self.set64(rt, (self.r64(rt) & keep) | (mem >> shift));
    }

    fn op_sdl(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 7) * 8;
        let aligned = addr & !7;
        let mem = bus.read64(aligned);
        let keep = 0xFFFF_FFFF_FFFF_FF00u64.checked_shl(shift).unwrap_or(0);
        bus.write64(aligned, (self.r64(rt) >> (56 - shift)) | (mem & keep));
    }

    fn op_sdr(&mut self, rt: usize, addr: u32, bus: &mut Bus) {
        let shift = (addr & 7) * 8;
        let aligned = addr & !7;
        let mem = bus.read64(aligned);
        let keep = if shift == 0 {
            0
        } else {
            u64::MAX >> (64 - shift)
        };
        bus.write64(aligned, (self.r64(rt) << shift) | (mem & keep));
    }

    // --- COP0 ------------------------------------------------------------

    fn op_cop0(&mut self, instr: u32, rs: usize, rt: usize, rd: usize, bus: &mut Bus) {
        match rs {
            0x00 => {
                let v = self.cop0.read(rd, bus.now);
                self.set32(rt, v);
            }
            0x04 => {
                let v = self.r32(rt);
                self.cop0.write(rd, v);
            }
            0x08 => {
                // BC0: CPCOND0 stubbed as true (DMA "always finished").
                let cond = true;
                warn!(target: "ps2_core::ee::cpu", pc = format_args!("{:#010x}", self.current_pc), "bc0 with stubbed CPCOND0");
                match rt {
                    0 => self.branch_cond(!cond, instr & 0xFFFF, false),
                    1 => self.branch_cond(cond, instr & 0xFFFF, false),
                    2 => self.branch_cond(!cond, instr & 0xFFFF, true),
                    3 => self.branch_cond(cond, instr & 0xFFFF, true),
                    _ => self.unimplemented("BC0", instr),
                }
            }
            0x10..=0x1F => match instr & 0x3F {
                0x01 => trace!(target: "ps2_core::ee::cop0", "tlbr (nop)"),
                0x02 => {
                    self.cop0.tlb_write_indexed();
                    let idx = (self.cop0.regs[0] & 0x3F) as usize;
                    bus.ee_tlb_write(
                        idx,
                        self.cop0.regs[5],
                        self.cop0.regs[10],
                        self.cop0.regs[2],
                        self.cop0.regs[3],
                    );
                }
                0x06 => trace!(target: "ps2_core::ee::cop0", "tlbwr (nop)"),
                0x08 => trace!(target: "ps2_core::ee::cop0", "tlbp (nop)"),
                0x18 => {
                    if let Some(target) = self.cop0.eret() {
                        trace!(target: "ps2_core::ee::cop0", to = format_args!("{target:#010x}"), "eret");
                        self.pc = target;
                        self.next_pc = target.wrapping_add(4);
                        self.next_is_delay = false;
                    }
                }
                0x38 => self.cop0.set_eie(true),
                0x39 => self.cop0.set_eie(false),
                _ => self.unimplemented("COP0", instr),
            },
            _ => self.unimplemented("COP0", instr),
        }
    }

    // --- COP1 (FPU) ------------------------------------------------------

    fn op_cop1(&mut self, instr: u32, rs: usize, rt: usize, rd: usize, sa: u32) {
        let fs = rd;
        let ft = rt;
        let fd = sa as usize;
        match rs {
            0x00 => self.set32(rt, self.fpu.regs[fs].to_bits()),
            0x02 => {
                let v = self.fpu.read_control(fs);
                self.set32(rt, v);
            }
            0x04 => self.fpu.regs[fs] = f32::from_bits(self.r32(rt)),
            0x06 => {
                let v = self.r32(rt);
                self.fpu.write_control(fs, v);
            }
            0x08 => {
                let cond = self.fpu.condition;
                match rt {
                    0 => self.branch_cond(!cond, instr & 0xFFFF, false),
                    1 => self.branch_cond(cond, instr & 0xFFFF, false),
                    2 => self.branch_cond(!cond, instr & 0xFFFF, true),
                    3 => self.branch_cond(cond, instr & 0xFFFF, true),
                    _ => self.unimplemented("BC1", instr),
                }
            }
            0x10 => {
                let a = self.fpu.regs[fs];
                let b = self.fpu.regs[ft];
                match instr & 0x3F {
                    0x00 => self.fpu.regs[fd] = Fpu::clamp(a + b),
                    0x01 => self.fpu.regs[fd] = Fpu::clamp(a - b),
                    0x02 => self.fpu.regs[fd] = Fpu::clamp(a * b),
                    0x03 => self.fpu.regs[fd] = Fpu::clamp(a / b),
                    0x04 => self.fpu.regs[fd] = b.abs().sqrt(),
                    0x05 => self.fpu.regs[fd] = a.abs(),
                    0x06 => self.fpu.regs[fd] = a,
                    0x07 => self.fpu.regs[fd] = -a,
                    0x16 => self.fpu.regs[fd] = Fpu::clamp(a / b.abs().sqrt()),
                    0x18 => self.fpu.acc = Fpu::clamp(a + b),
                    0x19 => self.fpu.acc = Fpu::clamp(a - b),
                    0x1A => self.fpu.acc = Fpu::clamp(a * b),
                    0x1C => self.fpu.regs[fd] = Fpu::clamp(self.fpu.acc + a * b),
                    0x1D => self.fpu.regs[fd] = Fpu::clamp(self.fpu.acc - a * b),
                    0x1E => self.fpu.acc = Fpu::clamp(self.fpu.acc + a * b),
                    0x1F => self.fpu.acc = Fpu::clamp(self.fpu.acc - a * b),
                    0x24 => {
                        // cvt.w.s: truncate with saturation
                        let v = if a >= 2147483647.0 {
                            i32::MAX
                        } else if a <= -2147483648.0 {
                            i32::MIN
                        } else {
                            a as i32
                        };
                        self.fpu.regs[fd] = f32::from_bits(v as u32);
                    }
                    0x28 => self.fpu.regs[fd] = a.max(b),
                    0x29 => self.fpu.regs[fd] = a.min(b),
                    0x30 => self.fpu.condition = false,
                    0x32 => self.fpu.condition = a == b,
                    0x34 => self.fpu.condition = a < b,
                    0x36 => self.fpu.condition = a <= b,
                    _ => self.unimplemented("COP1.S", instr),
                }
            }
            0x14 => match instr & 0x3F {
                0x20 => {
                    let v = self.fpu.regs[fs].to_bits() as i32;
                    self.fpu.regs[fd] = v as f32;
                }
                _ => self.unimplemented("COP1.W", instr),
            },
            _ => self.unimplemented("COP1", instr),
        }
    }

    // --- COP2 (VU0 macro mode) -------------------------------------------

    fn op_cop2(&mut self, instr: u32, rs: usize, rt: usize, rd: usize) {
        match rs {
            0x01 => {
                let v = self.vu0_vf[rd];
                self.set128(rt, v);
            }
            0x02 => {
                let v = self.vu0_ctrl[rd];
                self.set32(rt, v);
            }
            0x05 => self.vu0_vf[rd] = self.r128(rt),
            0x06 => self.vu0_ctrl[rd] = self.r32(rt),
            0x10..=0x1F => {
                // VU0 macro instructions: shadow-nop until the VUs exist.
                // Warn once — sync loops execute these millions of times.
                if !self.warned_vu0_macro {
                    self.warned_vu0_macro = true;
                    warn!(
                        target: "ps2_core::ee::cpu",
                        pc = format_args!("{:#010x}", self.current_pc),
                        instr = format_args!("{instr:#010x}"),
                        "VU0 macro op (nop stub, reported once)"
                    );
                }
            }
            _ => self.unimplemented("COP2", instr),
        }
    }

    // --- MMI -------------------------------------------------------------

    fn op_mmi(&mut self, instr: u32, rs: usize, rt: usize, rd: usize, sa: u32) {
        match instr & 0x3F {
            0x00 => self.madd(0, rd, rs, rt, false),
            0x01 => self.madd(0, rd, rs, rt, true),
            0x04 => {
                // plzcw: leading sign-bit count minus one, per 32-bit half.
                let v = self.r64(rs);
                let count = |x: u32| -> u32 {
                    let x = if x as i32 >= 0 { x } else { !x };
                    x.leading_zeros().saturating_sub(1)
                };
                let lo = count(v as u32) as u64;
                let hi = count((v >> 32) as u32) as u64;
                self.set64(rd, (hi << 32) | lo);
            }
            0x08 => self.op_mmi0(instr, rs, rt, rd),
            0x09 => self.op_mmi2(instr, rs, rt, rd),
            0x10 => self.set64(rd, self.hi[1]),
            0x11 => self.hi[1] = self.r64(rs),
            0x12 => self.set64(rd, self.lo[1]),
            0x13 => self.lo[1] = self.r64(rs),
            0x18 => self.mult(1, rd, rs, rt),
            0x19 => self.multu(1, rd, rs, rt),
            0x1A => self.div(1, rs, rt),
            0x1B => self.divu(1, rs, rt),
            0x20 => self.madd(1, rd, rs, rt, false),
            0x21 => self.madd(1, rd, rs, rt, true),
            0x28 => self.op_mmi1(instr, rs, rt, rd),
            0x29 => self.op_mmi3(instr, rs, rt, rd),
            0x34 => self.per_u16(rd, rt, |v| v << (sa & 15)),
            0x36 => self.per_u16(rd, rt, |v| v >> (sa & 15)),
            0x37 => self.per_u16(rd, rt, |v| ((v as i16) >> (sa & 15)) as u16),
            0x3C => self.per_u32(rd, rt, |v| v << sa),
            0x3E => self.per_u32(rd, rt, |v| v >> sa),
            0x3F => self.per_u32(rd, rt, |v| ((v as i32) >> sa) as u32),
            _ => self.unimplemented("MMI", instr),
        }
    }

    /// Apply `f` to each 16-bit lane of rt's 128 bits.
    fn per_u16(&mut self, rd: usize, rt: usize, f: impl Fn(u16) -> u16) {
        let src = self.r128(rt);
        let mut out = [0u64; 2];
        for half in 0..2 {
            for lane in 0..4 {
                let v = (src[half] >> (16 * lane)) as u16;
                out[half] |= (f(v) as u64) << (16 * lane);
            }
        }
        self.set128(rd, out);
    }

    fn per_u32(&mut self, rd: usize, rt: usize, f: impl Fn(u32) -> u32) {
        let src = self.r128(rt);
        let mut out = [0u64; 2];
        for half in 0..2 {
            for lane in 0..2 {
                let v = (src[half] >> (32 * lane)) as u32;
                out[half] |= (f(v) as u64) << (32 * lane);
            }
        }
        self.set128(rd, out);
    }

    fn op_mmi0(&mut self, instr: u32, rs: usize, rt: usize, rd: usize) {
        let sa = (instr >> 6) & 31;
        let a = self.r128(rs);
        let b = self.r128(rt);
        match sa {
            0x00 => self.lanes_u32(rd, a, b, |x, y| x.wrapping_add(y)), // paddw
            0x01 => self.lanes_u32(rd, a, b, |x, y| x.wrapping_sub(y)), // psubw
            0x02 => {
                // pcgtw
                self.lanes_u32(rd, a, b, |x, y| {
                    (((x as i32) > y as i32) as u32).wrapping_neg()
                })
            }
            0x03 => self.lanes_u32(rd, a, b, |x, y| (x as i32).max(y as i32) as u32), // pmaxw
            0x04 => self.lanes_u16(rd, a, b, |x, y| x.wrapping_add(y)),               // paddh
            0x05 => self.lanes_u16(rd, a, b, |x, y| x.wrapping_sub(y)),               // psubh
            0x06 => {
                // pcgth
                self.lanes_u16(rd, a, b, |x, y| {
                    (((x as i16) > y as i16) as u16).wrapping_neg()
                })
            }
            0x07 => self.lanes_u16(rd, a, b, |x, y| (x as i16).max(y as i16) as u16), // pmaxh
            0x08 => self.lanes_u8(rd, a, b, |x, y| x.wrapping_add(y)),                // paddb
            0x09 => self.lanes_u8(rd, a, b, |x, y| x.wrapping_sub(y)),                // psubb
            0x0A => {
                // pcgtb
                self.lanes_u8(rd, a, b, |x, y| {
                    (((x as i8) > y as i8) as u8).wrapping_neg()
                })
            }
            // paddsw / psubsw
            0x10 => self.lanes_u32(rd, a, b, |x, y| (x as i32).saturating_add(y as i32) as u32),
            0x11 => self.lanes_u32(rd, a, b, |x, y| (x as i32).saturating_sub(y as i32) as u32),
            // pextlw: interleave 32-bit words from the low halves
            0x12 => self.set128(
                rd,
                [
                    (b[0] as u32 as u64) | ((a[0] as u32 as u64) << 32),
                    ((b[0] >> 32) as u32 as u64) | (((a[0] >> 32) as u32 as u64) << 32),
                ],
            ),
            // ppacw: pack the even 32-bit words of rt (low) and rs (high)
            0x13 => self.set128(
                rd,
                [
                    (b[0] as u32 as u64) | ((b[1] as u32 as u64) << 32),
                    (a[0] as u32 as u64) | ((a[1] as u32 as u64) << 32),
                ],
            ),
            // paddsh / psubsh
            0x14 => self.lanes_u16(rd, a, b, |x, y| (x as i16).saturating_add(y as i16) as u16),
            0x15 => self.lanes_u16(rd, a, b, |x, y| (x as i16).saturating_sub(y as i16) as u16),
            0x16 => self.set128(rd, interleave_u16(b[0], a[0])), // pextlh
            0x17 => self.set128(rd, pack_u16(b, a)),             // ppach
            // paddsb / psubsb
            0x18 => self.lanes_u8(rd, a, b, |x, y| (x as i8).saturating_add(y as i8) as u8),
            0x19 => self.lanes_u8(rd, a, b, |x, y| (x as i8).saturating_sub(y as i8) as u8),
            0x1A => self.set128(rd, interleave_u8(b[0], a[0])), // pextlb
            0x1B => self.set128(rd, pack_u8(b, a)),             // ppacb
            _ => self.unimplemented("MMI0", instr),
        }
    }

    fn op_mmi1(&mut self, instr: u32, rs: usize, rt: usize, rd: usize) {
        let sa = (instr >> 6) & 31;
        let a = self.r128(rs);
        let b = self.r128(rt);
        match sa {
            0x01 => self.lanes_u32(rd, a, b, |_, y| (y as i32).unsigned_abs()), // pabsw
            0x02 => self.lanes_u32(rd, a, b, |x, y| ((x == y) as u32).wrapping_neg()), // pceqw
            0x03 => self.lanes_u32(rd, a, b, |x, y| (x as i32).min(y as i32) as u32), // pminw
            0x05 => self.lanes_u16(rd, a, b, |_, y| (y as i16).unsigned_abs()), // pabsh
            0x06 => self.lanes_u16(rd, a, b, |x, y| ((x == y) as u16).wrapping_neg()), // pceqh
            0x07 => self.lanes_u16(rd, a, b, |x, y| (x as i16).min(y as i16) as u16), // pminh
            0x0A => self.lanes_u8(rd, a, b, |x, y| ((x == y) as u8).wrapping_neg()), // pceqb
            0x10 => self.lanes_u32(rd, a, b, |x, y| x.saturating_add(y)),       // padduw
            0x11 => self.lanes_u32(rd, a, b, |x, y| x.saturating_sub(y)),       // psubuw
            // pextuw: interleave 32-bit words from the upper halves
            0x12 => self.set128(
                rd,
                [
                    (b[1] as u32 as u64) | ((a[1] as u32 as u64) << 32),
                    ((b[1] >> 32) as u32 as u64) | (((a[1] >> 32) as u32 as u64) << 32),
                ],
            ),
            0x14 => self.lanes_u16(rd, a, b, |x, y| x.saturating_add(y)), // padduh
            0x15 => self.lanes_u16(rd, a, b, |x, y| x.saturating_sub(y)), // psubuh
            0x16 => self.set128(rd, interleave_u16(b[1], a[1])),          // pextuh
            0x18 => self.lanes_u8(rd, a, b, |x, y| x.saturating_add(y)),  // paddub
            0x19 => self.lanes_u8(rd, a, b, |x, y| x.saturating_sub(y)),  // psubub
            0x1A => self.set128(rd, interleave_u8(b[1], a[1])),           // pextub
            // qfsrv: 256-bit funnel shift right of rs:rt by the SA register
            0x1B => {
                let shift = self.sa & 0xFF;
                let lo = u128::from(b[0]) | (u128::from(b[1]) << 64);
                let hi = u128::from(a[0]) | (u128::from(a[1]) << 64);
                let v = if shift == 0 {
                    lo
                } else if shift < 128 {
                    (lo >> shift) | (hi << (128 - shift))
                } else if shift < 256 {
                    hi >> (shift - 128)
                } else {
                    0
                };
                self.set128(rd, [v as u64, (v >> 64) as u64]);
            }
            _ => self.unimplemented("MMI1", instr),
        }
    }

    fn op_mmi2(&mut self, instr: u32, rs: usize, rt: usize, rd: usize) {
        let sa = (instr >> 6) & 31;
        let a = self.r128(rs);
        let b = self.r128(rt);
        match sa {
            0x08 => self.set128(rd, self.hi),      // pmfhi
            0x09 => self.set128(rd, self.lo),      // pmflo
            0x0E => self.set128(rd, [b[0], a[0]]), // pcpyld
            // pinth: interleave rt's lower halfwords with rs's upper ones
            0x0A => self.set128(rd, interleave_u16(b[0], a[1])),
            0x12 => self.set128(rd, [a[0] & b[0], a[1] & b[1]]), // pand
            0x13 => self.set128(rd, [a[0] ^ b[0], a[1] ^ b[1]]), // pxor
            _ => self.unimplemented("MMI2", instr),
        }
    }

    fn op_mmi3(&mut self, instr: u32, rs: usize, rt: usize, rd: usize) {
        let sa = (instr >> 6) & 31;
        let a = self.r128(rs);
        let b = self.r128(rt);
        match sa {
            0x08 => self.hi = a, // pmthi
            0x09 => self.lo = a, // pmtlo
            // pinteh: even halfwords of rs (upper) and rt (lower)
            0x0A => {
                let mut out = [0u64; 2];
                for half in 0..2 {
                    for lane in 0..2 {
                        let bl = (b[half] >> (32 * lane)) & 0xFFFF;
                        let al = (a[half] >> (32 * lane)) & 0xFFFF;
                        out[half] |= (bl | (al << 16)) << (32 * lane);
                    }
                }
                self.set128(rd, out);
            }
            0x0E => self.set128(rd, [a[1], b[1]]), // pcpyud
            0x12 => self.set128(rd, [a[0] | b[0], a[1] | b[1]]), // por
            0x13 => self.set128(rd, [!(a[0] | b[0]), !(a[1] | b[1])]), // pnor
            // pcpyh: replicate the low halfword of each doubleword of rt
            0x1B => {
                let rep = |v: u64| {
                    let h = v & 0xFFFF;
                    h | (h << 16) | (h << 32) | (h << 48)
                };
                self.set128(rd, [rep(b[0]), rep(b[1])]);
            }
            _ => self.unimplemented("MMI3", instr),
        }
    }

    fn lanes_u32(&mut self, rd: usize, a: [u64; 2], b: [u64; 2], f: impl Fn(u32, u32) -> u32) {
        let mut out = [0u64; 2];
        for half in 0..2 {
            for lane in 0..2 {
                let x = (a[half] >> (32 * lane)) as u32;
                let y = (b[half] >> (32 * lane)) as u32;
                out[half] |= (f(x, y) as u64) << (32 * lane);
            }
        }
        self.set128(rd, out);
    }

    fn lanes_u16(&mut self, rd: usize, a: [u64; 2], b: [u64; 2], f: impl Fn(u16, u16) -> u16) {
        let mut out = [0u64; 2];
        for half in 0..2 {
            for lane in 0..4 {
                let x = (a[half] >> (16 * lane)) as u16;
                let y = (b[half] >> (16 * lane)) as u16;
                out[half] |= (f(x, y) as u64) << (16 * lane);
            }
        }
        self.set128(rd, out);
    }

    fn lanes_u8(&mut self, rd: usize, a: [u64; 2], b: [u64; 2], f: impl Fn(u8, u8) -> u8) {
        let mut out = [0u64; 2];
        for half in 0..2 {
            for lane in 0..8 {
                let x = (a[half] >> (8 * lane)) as u8;
                let y = (b[half] >> (8 * lane)) as u8;
                out[half] |= (f(x, y) as u64) << (8 * lane);
            }
        }
        self.set128(rd, out);
    }
}

/// Interleave the 16-bit lanes of two 64-bit halves (pextlh/pextuh shape):
/// result lanes alternate low-source, high-source.
fn interleave_u16(lo: u64, hi: u64) -> [u64; 2] {
    let mut out = [0u64; 2];
    for i in 0..4 {
        let l = (lo >> (16 * i)) & 0xFFFF;
        let h = (hi >> (16 * i)) & 0xFFFF;
        let half = i / 2;
        let pos = (i % 2) * 32;
        out[half] |= (l << pos) | (h << (pos + 16));
    }
    out
}

fn interleave_u8(lo: u64, hi: u64) -> [u64; 2] {
    let mut out = [0u64; 2];
    for i in 0..8 {
        let l = (lo >> (8 * i)) & 0xFF;
        let h = (hi >> (8 * i)) & 0xFF;
        let half = i / 4;
        let pos = (i % 4) * 16;
        out[half] |= (l << pos) | (h << (pos + 8));
    }
    out
}

/// ppach: keep the even 16-bit lanes of each source; rt fills the low half.
fn pack_u16(b: [u64; 2], a: [u64; 2]) -> [u64; 2] {
    let squeeze = |v: [u64; 2]| -> u64 {
        let mut out = 0u64;
        for i in 0..4 {
            let src = v[i / 2] >> (32 * (i % 2));
            out |= (src & 0xFFFF) << (16 * i);
        }
        out
    };
    [squeeze(b), squeeze(a)]
}

/// ppacb: keep the even 8-bit lanes of each source; rt fills the low half.
fn pack_u8(b: [u64; 2], a: [u64; 2]) -> [u64; 2] {
    let squeeze = |v: [u64; 2]| -> u64 {
        let mut out = 0u64;
        for i in 0..8 {
            let src = v[i / 4] >> (16 * (i % 4));
            out |= (src & 0xFF) << (8 * i);
        }
        out
    };
    [squeeze(b), squeeze(a)]
}

#[cfg(test)]
mod tests {
    use crate::Ps2System;
    use crate::bus::BIOS_SIZE;

    /// Build a system whose BIOS contains `code` at the reset vector.
    fn system_with(code: &[u32]) -> Ps2System {
        let mut bios = vec![0u8; BIOS_SIZE];
        for (i, w) in code.iter().enumerate() {
            bios[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        Ps2System::new(bios).unwrap()
    }

    /// Step only the EE: these tests feed it R5900-only encodings the IOP
    /// core would (rightly) reject.
    fn run_ee(sys: &mut Ps2System, n: u64) {
        for _ in 0..n {
            sys.bus.now = sys.cycles;
            sys.ee.step(&mut sys.bus);
            sys.cycles += 1;
        }
    }

    #[test]
    fn lui_ori_addiu() {
        let mut sys = system_with(&[
            0x3C08_DEAD, // lui $t0, 0xDEAD
            0x3508_BEEF, // ori $t0, $t0, 0xBEEF
            0x2509_0001, // addiu $t1, $t0, 1
        ]);
        run_ee(&mut sys, 3);
        assert_eq!(sys.ee.gpr[8][0], 0xFFFF_FFFF_DEAD_BEEF);
        assert_eq!(sys.ee.gpr[9][0], 0xFFFF_FFFF_DEAD_BEF0);
    }

    #[test]
    fn branch_delay_slot_executes() {
        let mut sys = system_with(&[
            0x1000_0002, // beq $0, $0, +2 (taken)
            0x2401_0001, // addiu $at, $0, 1   (delay slot: executes)
            0x2402_0002, // addiu $v0, $0, 2   (skipped)
            0x2403_0003, // addiu $v1, $0, 3   (branch target)
        ]);
        run_ee(&mut sys, 3);
        assert_eq!(sys.ee.gpr[1][0], 1);
        assert_eq!(sys.ee.gpr[2][0], 0);
        assert_eq!(sys.ee.gpr[3][0], 3);
    }

    #[test]
    fn likely_branch_skips_delay_slot_when_not_taken() {
        let mut sys = system_with(&[
            0x5401_0002, // bnel $0, $at, +2 (not taken: $at == 0)
            0x2402_0002, // addiu $v0, $0, 2   (delay slot: skipped)
            0x2403_0003, // addiu $v1, $0, 3
        ]);
        run_ee(&mut sys, 2);
        assert_eq!(sys.ee.gpr[2][0], 0);
        assert_eq!(sys.ee.gpr[3][0], 3);
    }

    #[test]
    fn jal_links_and_jumps() {
        let mut sys = system_with(&[
            0x0FF0_0004, // jal 0xBFC00010
            0x0000_0000, // nop (delay)
            0x0000_0000,
            0x0000_0000,
            0x2404_0007, // addiu $a0, $0, 7 (target)
        ]);
        run_ee(&mut sys, 3);
        assert_eq!(sys.ee.gpr[31][0], 0xFFFF_FFFF_BFC0_0008);
        assert_eq!(sys.ee.gpr[4][0], 7);
    }

    #[test]
    fn sq_lq_roundtrip() {
        let mut sys = system_with(&[
            0x3C08_DEAD, // lui $t0, 0xDEAD
            0x3C01_0010, // lui $at, 0x0010 (RAM address 0x00100000)
            0x7C28_0000, // sq $t0, 0($at)
            0x7829_0000, // lq $t1, 0($at)
        ]);
        run_ee(&mut sys, 4);
        assert_eq!(sys.ee.gpr[9], sys.ee.gpr[8]);
    }

    #[test]
    fn mult_div_pipelines() {
        let mut sys = system_with(&[
            0x2408_0006, // addiu $t0, $0, 6
            0x2409_0007, // addiu $t1, $0, 7
            0x0109_0018, // mult $t0, $t1
            0x0109_001A, // div $t0, $t1 (6/7 = 0 rem 6)
        ]);
        run_ee(&mut sys, 4);
        assert_eq!(sys.ee.lo[0], 0); // div overwrote lo0
        assert_eq!(sys.ee.hi[0], 6);
    }
}
