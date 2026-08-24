//! Native translation of individual EE instructions.
//!
//! Register file model: every GPR lives in `Cpu.gpr` (rbx = &Cpu), so an
//! instruction is a few loads, an ALU op and a store. Fixed roles: r12 =
//! &Bus, r13 = pending branch condition, r14 = pending jump target; rax,
//! rcx, rdx, r8-r11 are scratch (helper calls clobber them). Results are
//! sign-extended into 64 bits exactly as the interpreter's `set32` does.
//! Anything not listed here goes through [`super::interp_one`].

use std::mem::offset_of;

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, VecAssembler, x64::X64Relocation};

use super::super::fpu::Fpu;
use super::super::Cpu;
use super::helpers as h;
use crate::bus::{Bus, RAM_SIZE};
use crate::vu1::Vu1;

pub type Ops = VecAssembler<X64Relocation>;

/// Bring-up knob: `PS2E_JIT_NATIVE` = comma list of groups to translate
/// natively (`imm,special,branch,load,store,quad,cmov`); unset = all.
fn native(group: &str) -> bool {
    use std::sync::OnceLock;
    static ALLOW: OnceLock<Option<Vec<String>>> = OnceLock::new();
    match ALLOW.get_or_init(|| {
        std::env::var("PS2E_JIT_NATIVE").ok().map(|v| v.split(',').map(str::to_string).collect())
    }) {
        None => true,
        Some(list) => list.iter().any(|g| g == group),
    }
}

/// Byte offset of `gpr[i]` (low half) from the Cpu base.
#[inline]
fn gpr(i: u32) -> i32 {
    (offset_of!(Cpu, gpr) + i as usize * 16) as i32
}
fn lo(pipe: usize) -> i32 {
    (offset_of!(Cpu, lo) + pipe * 8) as i32
}
fn hi(pipe: usize) -> i32 {
    (offset_of!(Cpu, hi) + pipe * 8) as i32
}
fn fpr(i: u32) -> i32 {
    (offset_of!(Cpu, fpu) + offset_of!(Fpu, regs) + i as usize * 4) as i32
}
pub fn pc_off() -> i32 {
    offset_of!(Cpu, pc) as i32
}
pub fn next_pc_off() -> i32 {
    offset_of!(Cpu, next_pc) as i32
}
pub fn idle_off() -> i32 {
    offset_of!(Cpu, idle) as i32
}
fn sa_off() -> i32 {
    offset_of!(Cpu, sa) as i32
}
fn fpu_cond_off() -> i32 {
    (offset_of!(Cpu, fpu) + offset_of!(Fpu, condition)) as i32
}
fn fpu_acc_off() -> i32 {
    (offset_of!(Cpu, fpu) + offset_of!(Fpu, acc)) as i32
}
/// Byte offset of VU0 `vf[i]` from the Bus base.
fn vu0_vf(i: u32) -> i32 {
    (offset_of!(Bus, vu0) + offset_of!(Vu1, vf) + i as usize * 16) as i32
}

/// What a decoded instruction turned into.
pub enum Emitted {
    /// Plain instruction, control falls through.
    Plain,
    /// The instruction was not translated (caller emits the interpreter call).
    Interp,
    /// A branch/jump: the condition (r13) / target (r14) and any link
    /// register are set; the caller emits the delay slot and then
    /// [`emit_branch_end`].
    Branch(BranchKind),
}

#[derive(Clone, Copy)]
pub enum BranchKind {
    /// Unconditional to a constant target.
    Jump(u32),
    /// Unconditional to the address in r14.
    JumpReg,
    /// Conditional (r13 != 0 -> taken) to a constant target; `likely`
    /// skips the delay slot when not taken.
    Cond { target: u32, likely: bool },
}

/// Store the sign-extended 32-bit value in eax to gpr[rd] (rd != 0).
fn store32(ops: &mut Ops, rd: u32) {
    if rd != 0 {
        dynasm!(ops
            ; .arch x64
            ; movsxd rax, eax
            ; mov QWORD [rbx + gpr(rd)], rax
        );
    }
}
/// Store rax to gpr[rd] (rd != 0).
fn store64(ops: &mut Ops, rd: u32) {
    if rd != 0 {
        dynasm!(ops
            ; .arch x64
            ; mov QWORD [rbx + gpr(rd)], rax
        );
    }
}
/// rax = gpr[r] (64-bit); the zero register is a real zero.
fn load64(ops: &mut Ops, r: u32) {
    if r == 0 {
        dynasm!(ops ; .arch x64 ; xor eax, eax);
    } else {
        dynasm!(ops ; .arch x64 ; mov rax, QWORD [rbx + gpr(r)]);
    }
}
/// rcx = gpr[r] (64-bit).
fn load64_rcx(ops: &mut Ops, r: u32) {
    if r == 0 {
        dynasm!(ops ; .arch x64 ; xor ecx, ecx);
    } else {
        dynasm!(ops ; .arch x64 ; mov rcx, QWORD [rbx + gpr(r)]);
    }
}

/// Emit `instr` at `addr`. Returns how the caller should continue.
pub fn emit(ops: &mut Ops, addr: u32, instr: u32) -> Emitted {
    let op = instr >> 26;
    let rs = (instr >> 21) & 31;
    let rt = (instr >> 16) & 31;
    let rd = (instr >> 11) & 31;
    let sa = ((instr >> 6) & 31) as i8;
    let imm = instr & 0xFFFF;
    let simm = imm as u16 as i16 as i32;
    let branch_target = addr.wrapping_add(4).wrapping_add((simm << 2) as u32);

    let group = match op {
        0x00 => match instr & 0x3F {
            0x08 | 0x09 => "branch",
            0x0A | 0x0B => "cmov",
            _ => "special",
        },
        0x01..=0x07 | 0x14..=0x17 => "branch",
        0x08..=0x0F | 0x18 | 0x19 => "imm",
        0x20..=0x27 | 0x37 | 0x31 => "load",
        0x28..=0x2B | 0x3F | 0x39 => "store",
        0x1E | 0x1F | 0x36 | 0x3E => "quad",
        0x11 => "cop1",
        // mfc0 / mtc0 and the ei/di pair; eret and the TLB group divert or
        // touch state the translator does not model, so they fall back.
        0x10 if rs == 0 || rs == 4 || (rs >= 0x10 && matches!(instr & 0x3F, 0x38 | 0x39)) => "cop0",
        0x12 if rs != 0x08 => "cop2",
        0x1C => "mmi",
        0x2F => "imm", // cache: no-op
        _ => return Emitted::Interp,
    };
    if !native(group) {
        return Emitted::Interp;
    }
    match op {
        0x00 => emit_special(ops, addr, instr, rs, rt, rd, sa),
        0x01 => emit_regimm(ops, addr, instr, rs, rt, branch_target),
        0x11 => emit_cop1(ops, instr, rs, rt, rd, sa as u32, branch_target),
        0x10 => {
            match rs {
                // mfc0
                0 => {
                    call_cpu_bus_arg(ops, h::cop0_read as *const () as usize, rd);
                    store32(ops, rt);
                }
                // mtc0
                4 => {
                    dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + gpr(rt)]);
                    call_cpu_arg_eax(ops, h::cop0_write as *const () as usize, rd);
                }
                // ei / di
                _ => call_cpu_arg(
                    ops,
                    h::cop0_set_eie as *const () as usize,
                    (instr & 0x3F == 0x38) as u32,
                ),
            }
            Emitted::Plain
        }
        // COP2 macro ops (not bc2): the interpreter's handler, without the
        // per-instruction bookkeeping of a full fallback.
        0x12 => {
            call_cpu_bus_arg(ops, h::cop2 as *const () as usize, instr);
            Emitted::Plain
        }
        0x1C => emit_mmi(ops, instr, rs, rt, rd),
        0x2F => Emitted::Plain,
        // j / jal
        0x02 | 0x03 => {
            let target = (addr.wrapping_add(4) & 0xF000_0000) | ((instr & 0x03FF_FFFF) << 2);
            if op == 0x03 {
                dynasm!(ops
                    ; .arch x64
                    ; mov eax, addr.wrapping_add(8) as i32
                );
                store32(ops, 31);
            }
            Emitted::Branch(BranchKind::Jump(target))
        }
        // beq / bne / blez / bgtz and likely forms
        0x04 | 0x05 | 0x14 | 0x15 => {
            load64(ops, rs);
            load64_rcx(ops, rt);
            dynasm!(ops
                ; .arch x64
                ; xor r13d, r13d
                ; cmp rax, rcx
            );
            if op & 1 == 0 {
                dynasm!(ops ; .arch x64 ; sete r13b);
            } else {
                dynasm!(ops ; .arch x64 ; setne r13b);
            }
            Emitted::Branch(BranchKind::Cond { target: branch_target, likely: op >= 0x14 })
        }
        0x06 | 0x07 | 0x16 | 0x17 => {
            load64(ops, rs);
            dynasm!(ops
                ; .arch x64
                ; xor r13d, r13d
                ; cmp rax, 0
            );
            if op & 1 == 0 {
                dynasm!(ops ; .arch x64 ; setle r13b); // blez
            } else {
                dynasm!(ops ; .arch x64 ; setg r13b); // bgtz
            }
            Emitted::Branch(BranchKind::Cond { target: branch_target, likely: op >= 0x16 })
        }
        // addi / addiu (overflow not modelled, like the interpreter)
        0x08 | 0x09 => {
            if rt == 0 {
                return Emitted::Plain;
            }
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD [rbx + gpr(rs)]
                ; add eax, simm
            );
            store32(ops, rt);
            Emitted::Plain
        }
        // slti / sltiu
        0x0A | 0x0B => {
            if rt == 0 {
                return Emitted::Plain;
            }
            load64_rcx(ops, rs);
            dynasm!(ops
                ; .arch x64
                ; xor eax, eax
                ; cmp rcx, simm // sign-extended to 64
            );
            if op == 0x0A {
                dynasm!(ops ; .arch x64 ; setl al);
            } else {
                dynasm!(ops ; .arch x64 ; setb al);
            }
            store64(ops, rt);
            Emitted::Plain
        }
        // andi / ori / xori (zero-extended immediate)
        0x0C | 0x0D | 0x0E => {
            if rt == 0 {
                return Emitted::Plain;
            }
            load64(ops, rs);
            match op {
                0x0C => dynasm!(ops ; .arch x64 ; and rax, imm as i32),
                0x0D => dynasm!(ops ; .arch x64 ; or rax, imm as i32),
                _ => dynasm!(ops ; .arch x64 ; xor rax, imm as i32),
            }
            store64(ops, rt);
            Emitted::Plain
        }
        // lui
        0x0F => {
            if rt != 0 {
                let v = (imm << 16) as i32 as i64;
                dynasm!(ops
                    ; .arch x64
                    ; mov rax, QWORD v
                    ; mov QWORD [rbx + gpr(rt)], rax
                );
            }
            Emitted::Plain
        }
        // daddi / daddiu
        0x18 | 0x19 => {
            if rt == 0 {
                return Emitted::Plain;
            }
            load64(ops, rs);
            dynasm!(ops ; .arch x64 ; add rax, simm);
            store64(ops, rt);
            Emitted::Plain
        }
        // Loads and stores through the bus helpers.
        0x20 | 0x21 | 0x23 | 0x24 | 0x25 | 0x27 | 0x37 | 0x31 => {
            emit_load(ops, op, rs, rt, simm);
            Emitted::Plain
        }
        0x28 | 0x29 | 0x2B | 0x3F | 0x39 => {
            emit_store(ops, op, rs, rt, simm);
            Emitted::Plain
        }
        // lq / sq (128-bit, address aligned down)
        0x1E => {
            emit_addr(ops, rs, simm);
            dynasm!(ops ; .arch x64 ; and eax, !0xF);
            if rt == 0 {
                // Still performs the read (no side effects on RAM), skip.
                return Emitted::Plain;
            }
            call_bus_addr_ptr(ops, h::rd128 as *const () as usize, gpr(rt));
            Emitted::Plain
        }
        0x1F => {
            emit_addr(ops, rs, simm);
            dynasm!(ops ; .arch x64 ; and eax, !0xF);
            call_bus_addr_ptr(ops, h::wr128 as *const () as usize, gpr(rt));
            Emitted::Plain
        }
        // lqc2 / sqc2: VU0 vf registers live in the bus (r12).
        0x36 => {
            emit_addr(ops, rs, simm);
            dynasm!(ops ; .arch x64 ; and eax, !0xF);
            if rt != 0 {
                call_bus_addr_busptr(ops, h::rd128 as *const () as usize, vu0_vf(rt));
            }
            Emitted::Plain
        }
        0x3E => {
            emit_addr(ops, rs, simm);
            dynasm!(ops ; .arch x64 ; and eax, !0xF);
            call_bus_addr_busptr(ops, h::wr128 as *const () as usize, vu0_vf(rt));
            Emitted::Plain
        }
        _ => Emitted::Interp,
    }
}

/// COP1: moves, branches on the condition flag, and single-precision
/// arithmetic with the interpreter's NaN/Inf clamping.
fn emit_cop1(ops: &mut Ops, instr: u32, rs: u32, rt: u32, rd: u32, sa: u32, target: u32) -> Emitted {
    let (fs, ft, fd) = (rd, rt, sa);
    match rs {
        // mfc1
        0x00 => {
            if rt != 0 {
                dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + fpr(fs)]);
                store32(ops, rt);
            }
            Emitted::Plain
        }
        // mtc1
        0x04 => {
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD [rbx + gpr(rt)]
                ; mov DWORD [rbx + fpr(fs)], eax
            );
            Emitted::Plain
        }
        // bc1f / bc1t / bc1fl / bc1tl
        0x08 if rt < 4 => {
            dynasm!(ops ; .arch x64 ; movzx r13d, BYTE [rbx + fpu_cond_off()]);
            if rt & 1 == 0 {
                dynasm!(ops ; .arch x64 ; xor r13d, 1);
            }
            Emitted::Branch(BranchKind::Cond { target, likely: rt & 2 != 0 })
        }
        // .S arithmetic
        0x10 => match instr & 0x3F {
            0x00..=0x03 => {
                dynasm!(ops
                    ; .arch x64
                    ; movss xmm0, DWORD [rbx + fpr(fs)]
                    ; movss xmm1, DWORD [rbx + fpr(ft)]
                );
                match instr & 0x3F {
                    0x00 => dynasm!(ops ; .arch x64 ; addss xmm0, xmm1),
                    0x01 => dynasm!(ops ; .arch x64 ; subss xmm0, xmm1),
                    0x02 => dynasm!(ops ; .arch x64 ; mulss xmm0, xmm1),
                    _ => dynasm!(ops ; .arch x64 ; divss xmm0, xmm1),
                }
                emit_clamp_store(ops, fpr(fd));
                Emitted::Plain
            }
            // sqrt.s: sqrt(|ft|), no clamp
            0x04 => {
                dynasm!(ops
                    ; .arch x64
                    ; mov eax, DWORD [rbx + fpr(ft)]
                    ; and eax, 0x7FFF_FFFF
                    ; movd xmm0, eax
                    ; sqrtss xmm0, xmm0
                    ; movss DWORD [rbx + fpr(fd)], xmm0
                );
                Emitted::Plain
            }
            // abs.s / mov.s / neg.s
            0x05 | 0x06 | 0x07 => {
                dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + fpr(fs)]);
                match instr & 0x3F {
                    0x05 => dynasm!(ops ; .arch x64 ; and eax, 0x7FFF_FFFF),
                    0x07 => dynasm!(ops ; .arch x64 ; xor eax, 0x8000_0000u32 as i32),
                    _ => {}
                }
                dynasm!(ops ; .arch x64 ; mov DWORD [rbx + fpr(fd)], eax);
                Emitted::Plain
            }
            // rsqrt.s: fs / sqrt(|ft|)
            0x16 => {
                dynasm!(ops
                    ; .arch x64
                    ; mov eax, DWORD [rbx + fpr(ft)]
                    ; and eax, 0x7FFF_FFFF
                    ; movd xmm1, eax
                    ; sqrtss xmm1, xmm1
                    ; movss xmm0, DWORD [rbx + fpr(fs)]
                    ; divss xmm0, xmm1
                );
                emit_clamp_store(ops, fpr(fd));
                Emitted::Plain
            }
            // adda / suba / mula (into ACC)
            0x18 | 0x19 | 0x1A => {
                dynasm!(ops
                    ; .arch x64
                    ; movss xmm0, DWORD [rbx + fpr(fs)]
                    ; movss xmm1, DWORD [rbx + fpr(ft)]
                );
                match instr & 0x3F {
                    0x18 => dynasm!(ops ; .arch x64 ; addss xmm0, xmm1),
                    0x19 => dynasm!(ops ; .arch x64 ; subss xmm0, xmm1),
                    _ => dynasm!(ops ; .arch x64 ; mulss xmm0, xmm1),
                }
                emit_clamp_store(ops, fpu_acc_off());
                Emitted::Plain
            }
            // madd / msub (fd = ACC +- fs*ft), madda / msuba (ACC = ACC +- fs*ft)
            0x1C..=0x1F => {
                dynasm!(ops
                    ; .arch x64
                    ; movss xmm0, DWORD [rbx + fpr(fs)]
                    ; mulss xmm0, DWORD [rbx + fpr(ft)]
                    ; movss xmm1, DWORD [rbx + fpu_acc_off()]
                );
                if instr & 1 == 0 {
                    dynasm!(ops ; .arch x64 ; addss xmm1, xmm0);
                } else {
                    dynasm!(ops ; .arch x64 ; subss xmm1, xmm0);
                }
                dynasm!(ops ; .arch x64 ; movaps xmm0, xmm1);
                let dst = if instr & 0x3F >= 0x1E { fpu_acc_off() } else { fpr(fd) };
                emit_clamp_store(ops, dst);
                Emitted::Plain
            }
            // cvt.w.s
            0x24 => {
                dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + fpr(fs)]);
                call_helper_eax(ops, h::cvt_w_s as *const () as usize);
                dynasm!(ops ; .arch x64 ; mov DWORD [rbx + fpr(fd)], eax);
                Emitted::Plain
            }
            // c.f / c.eq / c.lt / c.le (NaN compares false, as in Rust)
            0x30 => {
                dynasm!(ops ; .arch x64 ; mov BYTE [rbx + fpu_cond_off()], 0);
                Emitted::Plain
            }
            0x32 | 0x34 | 0x36 => {
                dynasm!(ops
                    ; .arch x64
                    ; movss xmm0, DWORD [rbx + fpr(fs)]
                    ; ucomiss xmm0, DWORD [rbx + fpr(ft)]
                    ; setnp cl
                );
                match instr & 0x3F {
                    0x32 => dynasm!(ops ; .arch x64 ; sete al),
                    0x34 => dynasm!(ops ; .arch x64 ; setb al),
                    _ => dynasm!(ops ; .arch x64 ; setbe al),
                }
                dynasm!(ops
                    ; .arch x64
                    ; and al, cl
                    ; mov BYTE [rbx + fpu_cond_off()], al
                );
                Emitted::Plain
            }
            _ => Emitted::Interp,
        },
        // cvt.s.w
        0x14 if instr & 0x3F == 0x20 => {
            dynasm!(ops
                ; .arch x64
                ; cvtsi2ss xmm0, DWORD [rbx + fpr(fs)]
                ; movss DWORD [rbx + fpr(fd)], xmm0
            );
            Emitted::Plain
        }
        _ => Emitted::Interp,
    }
}

/// Store xmm0 to `[rbx + off]` with the FPU's clamping: NaN -> 0, +Inf ->
/// f32::MAX, -Inf -> f32::MIN (their bit patterns are one below the
/// infinities).
fn emit_clamp_store(ops: &mut Ops, off: i32) {
    let store = ops.new_dynamic_label();
    let nan = ops.new_dynamic_label();
    dynasm!(ops
        ; .arch x64
        ; movd eax, xmm0
        ; mov ecx, eax
        ; and ecx, 0x7FFF_FFFF
        ; cmp ecx, 0x7F80_0000
        ; jb =>store
        ; ja =>nan
        ; sub eax, 1
        ; jmp =>store
        ; =>nan
        ; xor eax, eax
        ; =>store
        ; mov DWORD [rbx + off], eax
    );
}

fn emit_special(ops: &mut Ops, addr: u32, instr: u32, rs: u32, rt: u32, rd: u32, sa: i8) -> Emitted {
    if !native(&format!("f{:02x}", instr & 0x3F)) {
        return Emitted::Interp;
    }
    match instr & 0x3F {
        // sll / srl / sra
        0x00 | 0x02 | 0x03 => {
            if rd == 0 {
                return Emitted::Plain;
            }
            dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + gpr(rt)]);
            match instr & 0x3F {
                0x00 => dynasm!(ops ; .arch x64 ; shl eax, sa),
                0x02 => dynasm!(ops ; .arch x64 ; shr eax, sa),
                _ => dynasm!(ops ; .arch x64 ; sar eax, sa),
            }
            store32(ops, rd);
            Emitted::Plain
        }
        // sllv / srlv / srav
        0x04 | 0x06 | 0x07 => {
            if rd == 0 {
                return Emitted::Plain;
            }
            dynasm!(ops
                ; .arch x64
                ; mov ecx, DWORD [rbx + gpr(rs)]
                ; mov eax, DWORD [rbx + gpr(rt)]
            );
            match instr & 0x3F {
                0x04 => dynasm!(ops ; .arch x64 ; shl eax, cl),
                0x06 => dynasm!(ops ; .arch x64 ; shr eax, cl),
                _ => dynasm!(ops ; .arch x64 ; sar eax, cl),
            }
            store32(ops, rd);
            Emitted::Plain
        }
        // mfsa / mtsa
        0x28 => {
            dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + sa_off()]);
            store64(ops, rd);
            Emitted::Plain
        }
        0x29 => {
            if rs == 0 {
                dynasm!(ops ; .arch x64 ; xor eax, eax);
            } else {
                dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + gpr(rs)]);
            }
            dynasm!(ops ; .arch x64 ; mov DWORD [rbx + sa_off()], eax);
            Emitted::Plain
        }
        // jr / jalr
        0x08 | 0x09 => {
            dynasm!(ops ; .arch x64 ; mov r14d, DWORD [rbx + gpr(rs)]);
            if instr & 0x3F == 0x09 {
                dynasm!(ops ; .arch x64 ; mov eax, addr.wrapping_add(8) as i32);
                store32(ops, rd);
            }
            Emitted::Branch(BranchKind::JumpReg)
        }
        // movz / movn
        0x0A | 0x0B => {
            if rd == 0 {
                return Emitted::Plain;
            }
            load64_rcx(ops, rt);
            load64(ops, rs);
            let skip = ops.new_dynamic_label();
            dynasm!(ops ; .arch x64 ; test rcx, rcx);
            if instr & 0x3F == 0x0A {
                dynasm!(ops ; .arch x64 ; jnz =>skip);
            } else {
                dynasm!(ops ; .arch x64 ; jz =>skip);
            }
            store64(ops, rd);
            dynasm!(ops ; .arch x64 ; =>skip);
            Emitted::Plain
        }
        // sync
        0x0F => Emitted::Plain,
        // mfhi / mthi / mflo / mtlo
        0x10 => {
            dynasm!(ops ; .arch x64 ; mov rax, QWORD [rbx + hi(0)]);
            store64(ops, rd);
            Emitted::Plain
        }
        0x11 => {
            load64(ops, rs);
            dynasm!(ops ; .arch x64 ; mov QWORD [rbx + hi(0)], rax);
            Emitted::Plain
        }
        0x12 => {
            dynasm!(ops ; .arch x64 ; mov rax, QWORD [rbx + lo(0)]);
            store64(ops, rd);
            Emitted::Plain
        }
        0x13 => {
            load64(ops, rs);
            dynasm!(ops ; .arch x64 ; mov QWORD [rbx + lo(0)], rax);
            Emitted::Plain
        }
        // mult / multu
        0x18 | 0x19 => {
            emit_mult(ops, rs, rt, rd, 0, instr & 0x3F == 0x18);
            Emitted::Plain
        }
        // div / divu
        0x1A | 0x1B => {
            emit_div(ops, rs, rt, 0, instr & 0x3F == 0x1A);
            Emitted::Plain
        }
        // dsllv / dsrlv / dsrav
        0x14 | 0x16 | 0x17 => {
            if rd == 0 {
                return Emitted::Plain;
            }
            dynasm!(ops ; .arch x64 ; mov ecx, DWORD [rbx + gpr(rs)]);
            load64(ops, rt);
            match instr & 0x3F {
                0x14 => dynasm!(ops ; .arch x64 ; shl rax, cl),
                0x16 => dynasm!(ops ; .arch x64 ; shr rax, cl),
                _ => dynasm!(ops ; .arch x64 ; sar rax, cl),
            }
            store64(ops, rd);
            Emitted::Plain
        }
        // add / addu / sub / subu (32-bit, sign-extended)
        0x20 | 0x21 | 0x22 | 0x23 => {
            if rd == 0 {
                return Emitted::Plain;
            }
            dynasm!(ops
                ; .arch x64
                ; mov eax, DWORD [rbx + gpr(rs)]
                ; mov ecx, DWORD [rbx + gpr(rt)]
            );
            if instr & 0x3F <= 0x21 {
                dynasm!(ops ; .arch x64 ; add eax, ecx);
            } else {
                dynasm!(ops ; .arch x64 ; sub eax, ecx);
            }
            store32(ops, rd);
            Emitted::Plain
        }
        // and / or / xor / nor
        0x24..=0x27 => {
            if rd == 0 {
                return Emitted::Plain;
            }
            load64(ops, rs);
            load64_rcx(ops, rt);
            match instr & 0x3F {
                0x24 => dynasm!(ops ; .arch x64 ; and rax, rcx),
                0x25 => dynasm!(ops ; .arch x64 ; or rax, rcx),
                0x26 => dynasm!(ops ; .arch x64 ; xor rax, rcx),
                _ => dynasm!(ops ; .arch x64 ; or rax, rcx ; not rax),
            }
            store64(ops, rd);
            Emitted::Plain
        }
        // slt / sltu
        0x2A | 0x2B => {
            if rd == 0 {
                return Emitted::Plain;
            }
            load64_rcx(ops, rs);
            load64(ops, rt);
            dynasm!(ops
                ; .arch x64
                ; cmp rcx, rax
                ; mov eax, 0 // keep the flags
            );
            if instr & 0x3F == 0x2A {
                dynasm!(ops ; .arch x64 ; setl al);
            } else {
                dynasm!(ops ; .arch x64 ; setb al);
            }
            store64(ops, rd);
            Emitted::Plain
        }
        // dadd / daddu / dsub / dsubu
        0x2C | 0x2D | 0x2E | 0x2F => {
            if rd == 0 {
                return Emitted::Plain;
            }
            load64(ops, rs);
            load64_rcx(ops, rt);
            if instr & 0x3F <= 0x2D {
                dynasm!(ops ; .arch x64 ; add rax, rcx);
            } else {
                dynasm!(ops ; .arch x64 ; sub rax, rcx);
            }
            store64(ops, rd);
            Emitted::Plain
        }
        // dsll / dsrl / dsra / dsll32 / dsrl32 / dsra32
        0x38 | 0x3A | 0x3B | 0x3C | 0x3E | 0x3F => {
            if rd == 0 {
                return Emitted::Plain;
            }
            let amount = sa + if instr & 0x3F >= 0x3C { 32 } else { 0 };
            load64(ops, rt);
            match instr & 0x3 {
                0x0 => dynasm!(ops ; .arch x64 ; shl rax, amount),
                0x2 => dynasm!(ops ; .arch x64 ; shr rax, amount),
                _ => dynasm!(ops ; .arch x64 ; sar rax, amount),
            }
            store64(ops, rd);
            Emitted::Plain
        }
        _ => Emitted::Interp,
    }
}

/// MMI: pipe-1 HI/LO moves, mult1/div1 natively; the 128-bit multimedia
/// ops through the interpreter's handler.
fn emit_mmi(ops: &mut Ops, instr: u32, rs: u32, rt: u32, rd: u32) -> Emitted {
    match instr & 0x3F {
        0x10 => {
            dynasm!(ops ; .arch x64 ; mov rax, QWORD [rbx + hi(1)]);
            store64(ops, rd);
        }
        0x11 => {
            load64(ops, rs);
            dynasm!(ops ; .arch x64 ; mov QWORD [rbx + hi(1)], rax);
        }
        0x12 => {
            dynasm!(ops ; .arch x64 ; mov rax, QWORD [rbx + lo(1)]);
            store64(ops, rd);
        }
        0x13 => {
            load64(ops, rs);
            dynasm!(ops ; .arch x64 ; mov QWORD [rbx + lo(1)], rax);
        }
        0x18 | 0x19 => emit_mult(ops, rs, rt, rd, 1, instr & 0x3F == 0x18),
        0x1A | 0x1B => emit_div(ops, rs, rt, 1, instr & 0x3F == 0x1A),
        _ => call_cpu_arg(ops, h::mmi as *const () as usize, instr),
    }
    Emitted::Plain
}

/// mult/multu: LO/HI get the sign-extended halves; rd gets LO.
fn emit_mult(ops: &mut Ops, rs: u32, rt: u32, rd: u32, pipe: usize, signed: bool) {
    dynasm!(ops
        ; .arch x64
        ; mov eax, DWORD [rbx + gpr(rs)]
        ; mov ecx, DWORD [rbx + gpr(rt)]
    );
    if signed {
        dynasm!(ops ; .arch x64 ; imul ecx);
    } else {
        dynasm!(ops ; .arch x64 ; mul ecx);
    }
    dynasm!(ops
        ; .arch x64
        ; movsxd rax, eax
        ; movsxd rdx, edx
        ; mov QWORD [rbx + lo(pipe)], rax
        ; mov QWORD [rbx + hi(pipe)], rdx
    );
    store64(ops, rd);
}

/// div/divu with the interpreter's special cases: division by zero gives
/// LO = -1 (unsigned) or ±1 by the sign of the dividend (signed) and HI =
/// dividend; MIN / -1 gives MIN, 0.
fn emit_div(ops: &mut Ops, rs: u32, rt: u32, pipe: usize, signed: bool) {
    let done = ops.new_dynamic_label();
    let by_zero = ops.new_dynamic_label();
    dynasm!(ops
        ; .arch x64
        ; mov eax, DWORD [rbx + gpr(rs)]
        ; mov ecx, DWORD [rbx + gpr(rt)]
        ; test ecx, ecx
        ; jz =>by_zero
    );
    if signed {
        let divide = ops.new_dynamic_label();
        dynasm!(ops
            ; .arch x64
            ; cmp ecx, -1
            ; jne =>divide
            ; cmp eax, 0x8000_0000u32 as i32
            ; jne =>divide
            ; xor edx, edx // MIN / -1: quotient MIN (already in eax), remainder 0
            ; jmp =>done
            ; =>divide
            ; cdq
            ; idiv ecx
            ; jmp =>done
        );
    } else {
        dynasm!(ops
            ; .arch x64
            ; xor edx, edx
            ; div ecx
            ; jmp =>done
        );
    }
    dynasm!(ops ; .arch x64 ; =>by_zero);
    if signed {
        // quotient: -1 if dividend >= 0 else 1
        dynasm!(ops
            ; .arch x64
            ; mov edx, eax
            ; sar eax, 31
            ; not eax
            ; or eax, 1 // -1 for a non-negative dividend, 1 otherwise
        );
    } else {
        dynasm!(ops
            ; .arch x64
            ; mov edx, eax
            ; mov eax, -1
        );
    }
    dynasm!(ops
        ; .arch x64
        ; =>done
        ; movsxd rax, eax
        ; movsxd rdx, edx
        ; mov QWORD [rbx + lo(pipe)], rax
        ; mov QWORD [rbx + hi(pipe)], rdx
    );
}

fn emit_regimm(ops: &mut Ops, addr: u32, _instr: u32, rs: u32, rt: u32, target: u32) -> Emitted {
    match rt {
        // bltz / bgez / bltzl / bgezl / bltzal / bgezal / bltzall / bgezall
        0x00..=0x03 | 0x10..=0x13 => {
            load64(ops, rs);
            dynasm!(ops
                ; .arch x64
                ; xor r13d, r13d
                ; cmp rax, 0
            );
            if rt & 1 == 0 {
                dynasm!(ops ; .arch x64 ; setl r13b);
            } else {
                dynasm!(ops ; .arch x64 ; setge r13b);
            }
            if rt >= 0x10 {
                dynasm!(ops ; .arch x64 ; mov eax, addr.wrapping_add(8) as i32);
                store32(ops, 31);
            }
            Emitted::Branch(BranchKind::Cond { target, likely: rt & 2 != 0 })
        }
        _ => Emitted::Interp,
    }
}

/// eax = gpr[rs] + simm (32-bit).
fn emit_addr(ops: &mut Ops, rs: u32, simm: i32) {
    dynasm!(ops
        ; .arch x64
        ; mov eax, DWORD [rbx + gpr(rs)]
        ; add eax, simm
    );
}

/// With eax = virtual address: if it is plain RAM (kuseg below 32 MiB,
/// or kseg0/kseg1 mapping there), leave rdx = host address of the byte
/// and fall through; otherwise jump to `slow`. Clobbers ecx, rdx.
fn emit_ram_fast_path(ops: &mut Ops, slow: dynasmrt::DynamicLabel) {
    let ok = ops.new_dynamic_label();
    dynasm!(ops
        ; .arch x64
        ; mov ecx, eax
        ; shr ecx, 29 // segment: 0 = low kuseg, 4 = kseg0, 5 = kseg1
        ; mov edx, eax
        ; and edx, 0x1FFF_FFFF
        ; cmp edx, RAM_SIZE as i32
        ; jae =>slow
        ; test ecx, ecx
        ; jz =>ok
        ; cmp ecx, 4
        ; je =>ok
        ; cmp ecx, 5
        ; jne =>slow
        ; =>ok
        ; add rdx, QWORD [r12 + (offset_of!(Bus, ram_ptr) as i32)]
    );
}

fn emit_load(ops: &mut Ops, op: u32, rs: u32, rt: u32, simm: i32) {
    emit_addr(ops, rs, simm);
    let slow = ops.new_dynamic_label();
    let done = ops.new_dynamic_label();
    emit_ram_fast_path(ops, slow);
    // Fast path: load straight from host RAM, already extended.
    match op {
        0x20 => dynasm!(ops ; .arch x64 ; movsx rax, BYTE [rdx]),
        0x24 => dynasm!(ops ; .arch x64 ; movzx eax, BYTE [rdx]),
        0x21 => dynasm!(ops ; .arch x64 ; movsx rax, WORD [rdx]),
        0x25 => dynasm!(ops ; .arch x64 ; movzx eax, WORD [rdx]),
        0x23 => dynasm!(ops ; .arch x64 ; movsxd rax, DWORD [rdx]),
        0x27 | 0x31 => dynasm!(ops ; .arch x64 ; mov eax, DWORD [rdx]),
        0x37 => dynasm!(ops ; .arch x64 ; mov rax, QWORD [rdx]),
        _ => unreachable!(),
    }
    dynasm!(ops
        ; .arch x64
        ; jmp =>done
        ; =>slow
    );
    let (helper, ext): (usize, fn(&mut Ops)) = match op {
        0x20 => (h::rd8 as *const () as usize, |o| dynasm!(o ; .arch x64 ; movsx rax, al)),
        0x24 => (h::rd8 as *const () as usize, |o| dynasm!(o ; .arch x64 ; movzx eax, al)),
        0x21 => (h::rd16 as *const () as usize, |o| dynasm!(o ; .arch x64 ; movsx rax, ax)),
        0x25 => (h::rd16 as *const () as usize, |o| dynasm!(o ; .arch x64 ; movzx eax, ax)),
        0x23 => (h::rd32 as *const () as usize, |o| dynasm!(o ; .arch x64 ; movsxd rax, eax)),
        0x27 => (h::rd32 as *const () as usize, |o| dynasm!(o ; .arch x64 ; mov eax, eax)),
        0x37 => (h::rd64 as *const () as usize, |_| {}),
        0x31 => (h::rd32 as *const () as usize, |_| {}), // lwc1
        _ => unreachable!(),
    };
    call_bus_addr(ops, helper);
    ext(ops);
    dynasm!(ops ; .arch x64 ; =>done);
    if op == 0x31 {
        let ft = rt;
        dynasm!(ops ; .arch x64 ; mov DWORD [rbx + fpr(ft)], eax);
        return;
    }
    if rt != 0 {
        dynasm!(ops ; .arch x64 ; mov QWORD [rbx + gpr(rt)], rax);
    }
}

fn emit_store(ops: &mut Ops, op: u32, rs: u32, rt: u32, simm: i32) {
    emit_addr(ops, rs, simm);
    let slow = ops.new_dynamic_label();
    let done = ops.new_dynamic_label();
    emit_ram_fast_path(ops, slow);
    // Fast path only when no recompiled code lives on the page (the slow
    // path records the write for invalidation).
    dynasm!(ops
        ; .arch x64
        ; mov ecx, eax
        ; and ecx, 0x1FFF_FFFF
        ; shr ecx, 12
        ; mov r8, QWORD [r12 + (offset_of!(Bus, code_pages_ptr) as i32)]
        ; cmp BYTE [r8 + rcx], 0
        ; jne =>slow
    );
    if op == 0x39 {
        dynasm!(ops
            ; .arch x64
            ; mov ecx, DWORD [rbx + fpr(rt)]
            ; mov DWORD [rdx], ecx
        );
    } else {
        load64_rcx(ops, rt);
        match op {
            0x28 => dynasm!(ops ; .arch x64 ; mov BYTE [rdx], cl),
            0x29 => dynasm!(ops ; .arch x64 ; mov WORD [rdx], cx),
            0x2B => dynasm!(ops ; .arch x64 ; mov DWORD [rdx], ecx),
            0x3F => dynasm!(ops ; .arch x64 ; mov QWORD [rdx], rcx),
            _ => unreachable!(),
        }
    }
    dynasm!(ops
        ; .arch x64
        ; jmp =>done
        ; =>slow
    );
    // Value into rcx (helpers take it as the third argument).
    if op == 0x39 {
        dynasm!(ops ; .arch x64 ; mov ecx, DWORD [rbx + fpr(rt)]);
    } else {
        load64_rcx(ops, rt);
    }
    let helper = match op {
        0x28 => h::wr8 as *const () as usize,
        0x29 => h::wr16 as *const () as usize,
        0x2B | 0x39 => h::wr32 as *const () as usize,
        0x3F => h::wr64 as *const () as usize,
        _ => unreachable!(),
    };
    call_bus_addr_val(ops, helper);
    dynasm!(ops ; .arch x64 ; =>done);
}

// --- calling convention plumbing ---------------------------------------

/// Call `f(bus, eax)`; result in rax.
fn call_bus_addr(ops: &mut Ops, f: usize) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov edx, eax
        ; mov rcx, r12
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov esi, eax
        ; mov rdi, r12
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Call `f(bus, eax, rcx)`.
fn call_bus_addr_val(ops: &mut Ops, f: usize) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov r8, rcx
        ; mov edx, eax
        ; mov rcx, r12
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rdx, rcx
        ; mov esi, eax
        ; mov rdi, r12
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Call `f(bus, eax, &bus.field_at(off))`.
fn call_bus_addr_busptr(ops: &mut Ops, f: usize, off: i32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; lea r8, [r12 + off]
        ; mov edx, eax
        ; mov rcx, r12
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; lea rdx, [r12 + off]
        ; mov esi, eax
        ; mov rdi, r12
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Call `f(cpu, bus, arg)`; result in eax.
fn call_cpu_bus_arg(ops: &mut Ops, f: usize, arg: u32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov rcx, rbx
        ; mov rdx, r12
        ; mov r8d, arg as i32
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rdi, rbx
        ; mov rsi, r12
        ; mov edx, arg as i32
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Call `f(cpu, arg, eax)`.
fn call_cpu_arg_eax(ops: &mut Ops, f: usize, arg: u32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov r8d, eax
        ; mov rcx, rbx
        ; mov edx, arg as i32
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov edx, eax
        ; mov rdi, rbx
        ; mov esi, arg as i32
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Call `f(cpu, arg)`.
fn call_cpu_arg(ops: &mut Ops, f: usize, arg: u32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov rcx, rbx
        ; mov edx, arg as i32
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rdi, rbx
        ; mov esi, arg as i32
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Call `f(eax)`; result in eax.
fn call_helper_eax(ops: &mut Ops, f: usize) {
    #[cfg(windows)]
    dynasm!(ops ; .arch x64 ; mov ecx, eax);
    #[cfg(not(windows))]
    dynasm!(ops ; .arch x64 ; mov edi, eax);
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Call `f(bus, eax, &cpu.field_at(off))`.
fn call_bus_addr_ptr(ops: &mut Ops, f: usize, off: i32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; lea r8, [rbx + off]
        ; mov edx, eax
        ; mov rcx, r12
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; lea rdx, [rbx + off]
        ; mov esi, eax
        ; mov rdi, r12
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Exit sequences. Every exit adds the block's retired count to the cycle
/// counter (r15) first. Constant targets go through a link cell: an
/// indirect jump whose cell holds either the target block's body (once
/// compiled and linked) or this block's own slow path, which stores pc and
/// returns to the dispatcher.
pub struct Exits<'a> {
    pub jit: &'a mut super::Jit,
    /// (cell index, target pc, offset of the slow path in this block).
    pub links: Vec<(usize, u32, usize)>,
}

impl Exits<'_> {
    /// Exit to constant `target` after `count` retired cycles.
    pub fn to(&mut self, ops: &mut Ops, target: u32, count: u32) {
        let (idx, cell) = self.jit.link_cell();
        dynasm!(ops
            ; .arch x64
            ; add r15d, count as i32
            ; mov rax, QWORD cell as i64
            ; jmp QWORD [rax]
        );
        self.links.push((idx, target, ops.offset().0));
        dynasm!(ops
            ; .arch x64
            ; mov DWORD [rbx + pc_off()], target as i32
            ; mov DWORD [rbx + next_pc_off()], target.wrapping_add(4) as i32
            ; mov eax, r15d
            ; jmp ->epilogue
        );
    }

    /// Exit to `target` without linking (idle loop: the dispatcher must see
    /// the flag).
    pub fn to_unlinked(&mut self, ops: &mut Ops, target: u32, count: u32) {
        dynasm!(ops
            ; .arch x64
            ; add r15d, count as i32
            ; mov DWORD [rbx + pc_off()], target as i32
            ; mov DWORD [rbx + next_pc_off()], target.wrapping_add(4) as i32
            ; mov eax, r15d
            ; jmp ->epilogue
        );
    }

    /// Exit to the address in r14d.
    pub fn to_reg(&mut self, ops: &mut Ops, count: u32) {
        dynasm!(ops
            ; .arch x64
            ; add r15d, count as i32
            ; mov DWORD [rbx + pc_off()], r14d
            ; lea eax, [r14 + 4]
            ; mov DWORD [rbx + next_pc_off()], eax
            ; mov eax, r15d
            ; jmp ->epilogue
        );
    }

    /// Exit with pc already stored by an interpreted instruction.
    pub fn to_stored(&mut self, ops: &mut Ops, count: u32) {
        dynasm!(ops
            ; .arch x64
            ; add r15d, count as i32
            ; mov eax, r15d
            ; jmp ->epilogue
        );
    }
}

/// After the delay slot of a branch: resolve it and exit with `count`
/// cycles. `fallthrough` is the address after the delay slot.
pub fn emit_branch_end(ops: &mut Ops, kind: BranchKind, fallthrough: u32, count: u32, idle: bool, exits: &mut Exits) {
    match kind {
        BranchKind::Jump(t) => {
            if idle {
                dynasm!(ops ; .arch x64 ; mov BYTE [rbx + idle_off()], 1);
                exits.to_unlinked(ops, t, count);
            } else {
                exits.to(ops, t, count);
            }
        }
        BranchKind::JumpReg => exits.to_reg(ops, count),
        BranchKind::Cond { target, .. } => {
            let not_taken = ops.new_dynamic_label();
            dynasm!(ops
                ; .arch x64
                ; test r13d, r13d
                ; jz =>not_taken
            );
            if idle {
                dynasm!(ops ; .arch x64 ; mov BYTE [rbx + idle_off()], 1);
                exits.to_unlinked(ops, target, count);
            } else {
                exits.to(ops, target, count);
            }
            dynasm!(ops ; .arch x64 ; =>not_taken);
            exits.to(ops, fallthrough, count);
        }
    }
}

/// Likely branch not taken: skip the delay slot, continue after it.
pub fn emit_likely_skip(ops: &mut Ops, fallthrough: u32, count: u32, exits: &mut Exits) {
    let taken = ops.new_dynamic_label();
    dynasm!(ops
        ; .arch x64
        ; test r13d, r13d
        ; jnz =>taken
    );
    exits.to(ops, fallthrough, count);
    dynasm!(ops ; .arch x64 ; =>taken);
}
