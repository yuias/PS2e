//! Instruction emission for the VU1 recompiler.
//!
//! Fixed register roles for the whole block, mirroring the EE's:
//!   rbx  = *mut Vu1      r12  = *mut GsFront   r13  = *mut Gif
//!   r14d = the branch target the terminator pair left, if any
//!   r15d = instruction pairs retired so far, returned in eax
//! Everything else is scratch and is clobbered across helper calls.

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, VecAssembler, x64::X64Relocation};
use std::mem::offset_of;

use crate::vu1::Vu1;

pub type Ops = VecAssembler<X64Relocation>;

// --- state offsets -------------------------------------------------------

pub fn i_off() -> i32 {
    offset_of!(Vu1, i) as i32
}
pub fn mac_off() -> i32 {
    offset_of!(Vu1, mac) as i32
}
pub fn status_off() -> i32 {
    offset_of!(Vu1, status) as i32
}
pub fn mac_seen_off() -> i32 {
    offset_of!(Vu1, mac_seen) as i32
}
pub fn status_seen_off() -> i32 {
    offset_of!(Vu1, status_seen) as i32
}
pub fn flag_pipe_off() -> i32 {
    offset_of!(Vu1, flag_pipe) as i32
}
pub fn flag_cycle_off() -> i32 {
    offset_of!(Vu1, flag_cycle) as i32
}
pub fn next_pc_off() -> i32 {
    offset_of!(Vu1, next_pc) as i32
}
pub fn jit_target_off() -> i32 {
    offset_of!(Vu1, jit_target) as i32
}
fn vi_off(r: u32) -> i32 {
    (offset_of!(Vu1, vi) + (r as usize & 0xF) * 2) as i32
}

fn vi_written_reg_off() -> i32 {
    offset_of!(Vu1, vi_written_reg) as i32
}

fn vi_hazard_reg_off() -> i32 {
    offset_of!(Vu1, vi_hazard_reg) as i32
}

fn vi_hazard_val_off() -> i32 {
    offset_of!(Vu1, vi_hazard_val) as i32
}

/// Bring-up bisection knob, the EE recompiler's `PS2E_JIT_NATIVE` for VU1.
/// Unset translates everything it can; otherwise only the named groups are
/// emitted natively and the rest fall back to the interpreter, which is how
/// a single wrong opcode gets found when a gate image moves.
fn native(group: &str) -> bool {
    static GROUPS: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
    match GROUPS.get_or_init(|| {
        std::env::var("PS2E_VU1_JIT_NATIVE")
            .ok()
            .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
    }) {
        None => true,
        Some(list) => list.iter().any(|g| g == group),
    }
}

/// Whether the lower slot's opcode diverts control.
pub fn is_branch(lower: u32) -> bool {
    matches!(lower >> 25, 0x20 | 0x21 | 0x24 | 0x25 | 0x28 | 0x29 | 0x2C..=0x2F)
}

/// The upper slot's second opcode table, shared by every `0x3C..=0x3F`
/// encoding in both pipelines.
fn op2(instr: u32) -> u32 {
    (instr & 3) | ((instr >> 4) & 0x7C)
}

/// An upper slot that provably changes nothing. Two thirds of the pairs a
/// mission runs are these, so translating them into a call would spend
/// most of the recompiler's budget on doing nothing.
fn upper_is_nop(instr: u32) -> bool {
    (0x3C..=0x3F).contains(&(instr & 0x3F)) && op2(instr) == 0x2F
}

/// The same for the lower slot. The canonical VU1 lower nop is a MOVE into
/// vf00, whose write the register file discards; WAITQ and WAITP have
/// nothing to wait for here.
fn lower_is_nop(instr: u32) -> bool {
    if instr >> 25 != 0x40 || !(0x3C..=0x3F).contains(&(instr & 0x3F)) {
        return false;
    }
    let it = (instr >> 16) & 0x1F;
    match op2(instr) {
        0x30 | 0x31 => it == 0, // MOVE, MR32
        0x3B | 0x7B => true,    // WAITQ, WAITP
        _ => false,
    }
}

/// Whether the lower slot reads the aged MAC/status flags. Kept as the
/// whole flag-op family rather than just FSAND and FMAND: it costs nothing
/// (they are 2% of executed lower slots) and does not go stale.
fn lower_reads_flags(instr: u32) -> bool {
    (0x10..=0x1F).contains(&(instr >> 25))
}

/// The VI register a lower instruction writes with ALU timing, if any:
/// the write a branch straight after it does not yet see. vi00 and a
/// field of 16, which the register file discards, count as none, and so
/// do the loads (ILW, ILWR, which the branch waits for) and the flag
/// readers (FSAND, FMAND, FCAND ..., whose result the branch does see, as
/// the BIOS's own microcode relies on).
pub fn lower_vi_dest(lower: u32) -> Option<u32> {
    let it = (lower >> 16) & 0xF;
    let is = (lower >> 11) & 0xF;
    let id = (lower >> 6) & 0xF;
    let r = match lower >> 25 {
        0x08 | 0x09 => it, // IADDIU, ISUBIU
        0x21 | 0x25 => it, // BAL, JALR
        0x40 => match lower & 0x3F {
            0x30 | 0x31 | 0x34 | 0x35 => id, // IADD, ISUB, IAND, IOR
            0x32 => it,                      // IADDI
            0x3C..=0x3F => match op2(lower) {
                0x34 | 0x36 => is,               // LQI, LQD
                0x35 | 0x37 => it,               // SQI, SQD
                0x3C | 0x68 | 0x69 => it,        // MTIR, XTOP, XITOP
                _ => 0,
            },
            _ => 0,
        },
        _ => 0,
    };
    (r != 0).then_some(r)
}

/// The VI registers a branch reads, masked to the file: `is` for the
/// jumps and the compares against zero, `it` too for IBEQ/IBNE.
pub fn branch_vi_reads(lower: u32) -> [Option<u32>; 2] {
    let it = (lower >> 16) & 0xF;
    let is = (lower >> 11) & 0xF;
    let some = |r: u32| (r != 0).then_some(r);
    match lower >> 25 {
        0x24 | 0x25 | 0x2C..=0x2F => [some(is), None],
        0x28 | 0x29 => [some(it), some(is)],
        _ => [None, None],
    }
}

/// How a pair takes part in the integer branch hazard (see the fields on
/// [`Vu1`]). The translator decides statically: a pair whose lower writes
/// the register the next lower branches on copies the old value aside
/// (`set`); the branch then reads that copy (`read`). A block's first pair
/// cannot know what ran before it, so its branch checks at run time
/// (`dynamic`) against what the previous block's exit left; a first pair
/// that is not a branch clears that instead (`clear_before`).
#[derive(Clone, Copy, Default)]
pub struct Hazard {
    pub set: Option<u32>,
    pub read: Option<u32>,
    pub dynamic: bool,
    pub clear_before: bool,
}

/// Leave the hazard record clean at a block's exit: the delay pair's
/// `set` has already spoken for the next block's first pair, or there was
/// nothing to say.
pub fn exit_hazard(ops: &mut Ops, delay_set: bool) {
    if !delay_set {
        dynasm!(ops ; .arch x64 ; mov BYTE [rbx + vi_hazard_reg_off()], 0);
    }
    clear_written(ops);
}

/// Interpreter fallbacks record their writes in `vi_written_*`; nothing
/// native shifts the record, so it must not outlive the block.
pub fn clear_written(ops: &mut Ops) {
    dynasm!(ops ; .arch x64 ; mov BYTE [rbx + vi_written_reg_off()], 0);
}

/// Load VI register `r` into eax as a branch sees it.
fn load_vi_for_branch(ops: &mut Ops, r: u32, hz: Hazard) {
    let r = r & 0xF;
    if hz.read == Some(r) {
        dynasm!(ops ; .arch x64 ; movzx eax, WORD [rbx + vi_hazard_val_off()]);
    } else if hz.dynamic && r != 0 {
        dynasm!(ops ; .arch x64
            ; movzx eax, WORD [rbx + vi_off(r)]
            ; cmp BYTE [rbx + vi_hazard_reg_off()], r as i8
            ; jne >current
            ; movzx eax, WORD [rbx + vi_hazard_val_off()]
            ; current:
        );
    } else {
        dynasm!(ops ; .arch x64 ; movzx eax, WORD [rbx + vi_off(r)]);
    }
}

// --- per-pair emission ---------------------------------------------------

/// One instruction pair, in the order the interpreter runs it: age the flag
/// pipeline, take the I bit's constant, upper, then lower.
pub fn pair(ops: &mut Ops, pc: u16, upper: u32, lower: u32, hz: Hazard) {
    let ibit = upper & (1 << 31) != 0;
    if hz.clear_before {
        dynasm!(ops ; .arch x64 ; mov BYTE [rbx + vi_hazard_reg_off()], 0);
    }
    if let Some(r) = hz.set {
        dynasm!(ops ; .arch x64
            ; movzx eax, WORD [rbx + vi_off(r)]
            ; mov WORD [rbx + vi_hazard_val_off()], ax
            ; mov BYTE [rbx + vi_hazard_reg_off()], r as i8
        );
    }
    if !ibit && lower_reads_flags(lower) {
        flag_load(ops);
    }
    if ibit {
        // The lower word is a float constant this pair's upper may use.
        dynasm!(ops ; .arch x64 ; mov DWORD [rbx + i_off()], lower as i32);
    }
    if !upper_is_nop(upper) && !upper_native(ops, upper) {
        call_upper(ops, pc, upper);
    }
    if !ibit && !lower_is_nop(lower) && !lower_native(ops, pc, lower, hz) {
        call_lower(ops, pc, lower);
        if is_branch(lower) {
            // Latch the target before the delay pair can overwrite it: an
            // E-bit pair's delay slot may itself be a branch, whose target
            // the interpreter discards.
            dynasm!(ops ; .arch x64 ; mov r14d, DWORD [rbx + jit_target_off()]);
        }
    }
    if hz.read.is_some() || hz.dynamic {
        dynasm!(ops ; .arch x64 ; mov BYTE [rbx + vi_hazard_reg_off()], 0);
    }
    flag_store(ops);
    dynasm!(ops ; .arch x64 ; add r15d, 1);
}

/// The flag readers see what the pipeline held four pairs ago.
fn flag_load(ops: &mut Ops) {
    dynasm!(ops ; .arch x64
        ; mov eax, DWORD [rbx + flag_cycle_off()]
        ; and eax, 3
        ; mov ecx, DWORD [rbx + rax*4 + flag_pipe_off()]
        ; mov WORD [rbx + mac_seen_off()], cx
        ; shr ecx, 16
        ; mov WORD [rbx + status_seen_off()], cx
    );
}

/// Hand this pair's freshly computed flags to the pipeline and step it.
fn flag_store(ops: &mut Ops) {
    dynasm!(ops ; .arch x64
        ; mov eax, DWORD [rbx + flag_cycle_off()]
        ; and eax, 3
        ; mov cx, WORD [rbx + mac_off()]
        ; mov WORD [rbx + rax*4 + flag_pipe_off()], cx
        ; mov cx, WORD [rbx + status_off()]
        ; mov WORD [rbx + rax*4 + flag_pipe_off() + 2], cx
        ; add DWORD [rbx + flag_cycle_off()], 1
    );
}

fn call_upper(ops: &mut Ops, pc: u16, instr: u32) {
    let f = super::helpers::upper as *const () as usize;
    #[cfg(windows)]
    dynasm!(ops ; .arch x64
        ; mov rcx, rbx
        ; mov edx, pc as i32
        ; mov r8d, instr as i32
    );
    #[cfg(not(windows))]
    dynasm!(ops ; .arch x64
        ; mov rdi, rbx
        ; mov esi, pc as i32
        ; mov edx, instr as i32
    );
    dynasm!(ops ; .arch x64 ; mov rax, QWORD f as i64 ; call rax);
}

fn call_lower(ops: &mut Ops, pc: u16, instr: u32) {
    let f = super::helpers::lower as *const () as usize;
    #[cfg(windows)]
    dynasm!(ops ; .arch x64
        ; mov rcx, rbx
        ; mov rdx, r12
        ; mov r8, r13
        ; mov r9d, pc as i32
        ; mov DWORD [rsp + 0x20], instr as i32
    );
    #[cfg(not(windows))]
    dynasm!(ops ; .arch x64
        ; mov rdi, rbx
        ; mov rsi, r12
        ; mov rdx, r13
        ; mov ecx, pc as i32
        ; mov r8d, instr as i32
    );
    dynasm!(ops ; .arch x64 ; mov rax, QWORD f as i64 ; call rax);
}

// --- native lower pipeline -----------------------------------------------

/// Translate the lower slot directly, returning false for an encoding this
/// does not cover — the caller then calls the interpreter for it.
///
/// A branch leaves its target in r14d (or leaves the prologue's `NO_BRANCH`
/// alone when it is not taken), which is where the block's exit reads it.
fn lower_native(ops: &mut Ops, pc: u16, instr: u32, hz: Hazard) -> bool {
    let opcode = instr >> 25;
    let it = (instr >> 16) & 0x1F;
    let is = (instr >> 11) & 0x1F;
    let imm11 = ((instr & 0x7FF) as i32) << 21 >> 21;
    // Left unmasked, like the interpreter's: the exit masks it.
    let target = (pc as i32 + 1 + imm11) as u16 as i32;
    let link = pc.wrapping_add(2) as i32;

    match opcode {
        // IADDIU / ISUBIU
        0x08 | 0x09 if native("int") => {
            if it & 0xF == 0 {
                return true; // the write goes to vi00 and is discarded
            }
            let imm15 = ((instr & 0x7FF) | ((instr >> 10) & 0x7800)) as i32;
            dynasm!(ops ; .arch x64 ; movzx eax, WORD [rbx + vi_off(is)]);
            if opcode == 0x08 {
                dynasm!(ops ; .arch x64 ; add eax, imm15);
            } else {
                dynasm!(ops ; .arch x64 ; sub eax, imm15);
            }
            dynasm!(ops ; .arch x64 ; mov WORD [rbx + vi_off(it)], ax);
            true
        }
        0x20 if native("branch") => {
            dynasm!(ops ; .arch x64 ; mov r14d, target); // B
            true
        }
        0x21 if native("branch") => {
            // BAL: the link points past the delay pair.
            if it & 0xF != 0 {
                dynasm!(ops ; .arch x64 ; mov WORD [rbx + vi_off(it)], link as i16);
            }
            dynasm!(ops ; .arch x64 ; mov r14d, target);
            true
        }
        0x24 | 0x25 if native("branch") => {
            // JR / JALR: the target is read before the link is written.
            load_vi_for_branch(ops, is, hz);
            dynasm!(ops ; .arch x64 ; mov r14d, eax);
            if opcode == 0x25 && it & 0xF != 0 {
                dynasm!(ops ; .arch x64 ; mov WORD [rbx + vi_off(it)], link as i16);
            }
            true
        }
        // IBEQ / IBNE
        0x28 | 0x29 if native("branch") => {
            load_vi_for_branch(ops, is, hz);
            dynasm!(ops ; .arch x64 ; mov ecx, eax);
            load_vi_for_branch(ops, it, hz);
            dynasm!(ops ; .arch x64 ; cmp ax, cx);
            if opcode == 0x28 {
                dynasm!(ops ; .arch x64 ; jne >skip);
            } else {
                dynasm!(ops ; .arch x64 ; je >skip);
            }
            dynasm!(ops ; .arch x64 ; mov r14d, target ; skip:);
            true
        }
        // IBLTZ / IBGTZ / IBLEZ / IBGEZ, all against zero and all signed.
        0x2C..=0x2F if native("branch") => {
            load_vi_for_branch(ops, is, hz);
            dynasm!(ops ; .arch x64
                ; movsx eax, ax
                ; test eax, eax
            );
            match opcode {
                0x2C => dynasm!(ops ; .arch x64 ; jns >skip),
                0x2D => dynasm!(ops ; .arch x64 ; jle >skip),
                0x2E => dynasm!(ops ; .arch x64 ; jg >skip),
                _ => dynasm!(ops ; .arch x64 ; js >skip),
            }
            dynasm!(ops ; .arch x64 ; mov r14d, target ; skip:);
            true
        }
        _ => lower_mem_native(ops, instr) || (opcode == 0x40 && lower_special_native(ops, instr)),
    }
}

fn lower_special_native(ops: &mut Ops, instr: u32) -> bool {
    if !native("int") {
        return false;
    }
    let it = (instr >> 16) & 0x1F;
    let is = (instr >> 11) & 0x1F;
    let id = (instr >> 6) & 0x1F;
    match instr & 0x3F {
        // IADD / ISUB / IAND / IOR, all writing the `id` register.
        funct @ (0x30 | 0x31 | 0x34 | 0x35) => {
            if id & 0xF == 0 {
                return true;
            }
            dynasm!(ops ; .arch x64
                ; movzx eax, WORD [rbx + vi_off(is)]
                ; movzx ecx, WORD [rbx + vi_off(it)]
            );
            match funct {
                0x30 => dynasm!(ops ; .arch x64 ; add eax, ecx),
                0x31 => dynasm!(ops ; .arch x64 ; sub eax, ecx),
                0x34 => dynasm!(ops ; .arch x64 ; and eax, ecx),
                _ => dynasm!(ops ; .arch x64 ; or eax, ecx),
            }
            dynasm!(ops ; .arch x64 ; mov WORD [rbx + vi_off(id)], ax);
            true
        }
        // IADDI: a 5-bit signed immediate sits in the `id` slot.
        0x32 => {
            if it & 0xF == 0 {
                return true;
            }
            let imm5 = ((id as i32) << 27) >> 27;
            dynasm!(ops ; .arch x64
                ; movzx eax, WORD [rbx + vi_off(is)]
                ; add eax, imm5
                ; mov WORD [rbx + vi_off(it)], ax
            );
            true
        }
        _ => false,
    }
}

// --- native upper pipeline -----------------------------------------------

/// Constants the translated code needs as SSE operands. One aligned block,
/// addressed through rbp, so a constant costs one instruction rather than a
/// 64-bit immediate load.
#[repr(C, align(16))]
struct Consts {
    abs: [u32; 4],
    sign: [u32; 4],
    max_finite: [u32; 4],
    min_normal: [u32; 4],
    /// One lane mask per dest nibble, x in the high bit.
    dest: [[u32; 4]; 16],
}

static CONSTS: Consts = {
    let mut dest = [[0u32; 4]; 16];
    let mut d = 0;
    while d < 16 {
        let mut f = 0;
        while f < 4 {
            if d & (8 >> f) != 0 {
                dest[d][f] = 0xFFFF_FFFF;
            }
            f += 1;
        }
        d += 1;
    }
    Consts {
        abs: [0x7FFF_FFFF; 4],
        sign: [0x8000_0000; 4],
        max_finite: [0x7F7F_FFFF; 4],
        min_normal: [0x0080_0000; 4],
        dest,
    }
};

const C_ABS: i32 = 0;
const C_SIGN: i32 = 16;
const C_MAXF: i32 = 32;
const C_MINNORM: i32 = 48;
fn c_dest(d: u32) -> i32 {
    64 + (d as i32 & 0xF) * 16
}

/// Base address of [`CONSTS`], loaded into rbp by the block prologue.
pub fn consts_addr() -> i64 {
    &CONSTS as *const Consts as i64
}

fn vf_off(r: u32) -> i32 {
    (offset_of!(Vu1, vf) + (r as usize & 0x1F) * 16) as i32
}
fn acc_off() -> i32 {
    offset_of!(Vu1, acc) as i32
}
fn q_off() -> i32 {
    offset_of!(Vu1, q) as i32
}

/// The second operand of an FMAC: a broadcast field of ft, the whole of ft,
/// or one of the scalar registers.
#[derive(Clone, Copy)]
enum BSrc {
    Bc(u32),
    T,
    Q,
    I,
}

#[derive(Clone, Copy)]
enum Fmac {
    Add,
    Sub,
    Mul,
    Madd,
    Msub,
}

/// Translate the upper slot, returning false for an encoding this does not
/// cover. Only the FMAC families that go through `vf_write`/`acc_write` are
/// here: MAX/MINI, ABS and the integer conversions skip the result clamp
/// and the flags, and are rare enough to leave to the interpreter.
fn upper_native(ops: &mut Ops, instr: u32) -> bool {
    if !native("fmac") {
        return false;
    }
    let op = instr & 0x3F;
    // The A-suffixed table mirrors the plain one and writes the accumulator.
    let (table_op, fd) = if (0x3C..=0x3F).contains(&op) {
        (op2(instr), None)
    } else {
        (op, Some((instr >> 6) & 0x1F))
    };
    match fmac_op(table_op) {
        Some((kind, b)) => {
            fmac(ops, kind, b, instr, fd);
            true
        }
        None => false,
    }
}

fn fmac_op(op: u32) -> Option<(Fmac, BSrc)> {
    use BSrc::*;
    use Fmac::*;
    Some(match op {
        0x00..=0x03 => (Add, Bc(op & 3)),
        0x04..=0x07 => (Sub, Bc(op & 3)),
        0x08..=0x0B => (Madd, Bc(op & 3)),
        0x0C..=0x0F => (Msub, Bc(op & 3)),
        0x18..=0x1B => (Mul, Bc(op & 3)),
        0x1C => (Mul, Q),
        0x1E => (Mul, I),
        0x20 => (Add, Q),
        0x21 => (Madd, Q),
        0x22 => (Add, I),
        0x23 => (Madd, I),
        0x24 => (Sub, Q),
        0x25 => (Msub, Q),
        0x26 => (Sub, I),
        0x27 => (Msub, I),
        0x28 => (Add, T),
        0x29 => (Madd, T),
        0x2A => (Mul, T),
        0x2C => (Sub, T),
        0x2D => (Msub, T),
        _ => return None,
    })
}

/// `fd` is `Some` for the register-writing form, `None` for the A-suffixed
/// one that writes the accumulator.
fn fmac(ops: &mut Ops, kind: Fmac, b: BSrc, instr: u32, fd: Option<u32>) {
    let dest = (instr >> 21) & 0xF;
    let ft = (instr >> 16) & 0x1F;
    let fs = (instr >> 11) & 0x1F;

    dynasm!(ops ; .arch x64 ; movups xmm0, [rbx + vf_off(fs)]);
    match b {
        BSrc::Bc(k) => dynasm!(ops ; .arch x64
            ; movups xmm1, [rbx + vf_off(ft)]
            ; shufps xmm1, xmm1, (k * 0x55) as i8
        ),
        BSrc::T => dynasm!(ops ; .arch x64 ; movups xmm1, [rbx + vf_off(ft)]),
        BSrc::Q => dynasm!(ops ; .arch x64
            ; movss xmm1, [rbx + q_off()]
            ; shufps xmm1, xmm1, 0
        ),
        BSrc::I => dynasm!(ops ; .arch x64
            ; movss xmm1, [rbx + i_off()]
            ; shufps xmm1, xmm1, 0
        ),
    }
    match kind {
        Fmac::Add => dynasm!(ops ; .arch x64 ; addps xmm0, xmm1),
        Fmac::Sub => dynasm!(ops ; .arch x64 ; subps xmm0, xmm1),
        Fmac::Mul => dynasm!(ops ; .arch x64 ; mulps xmm0, xmm1),
        // The product rounds to f32 before the accumulator is applied, the
        // way the interpreter's two separate Rust operations do.
        Fmac::Madd => dynasm!(ops ; .arch x64
            ; mulps xmm0, xmm1
            ; movups xmm1, [rbx + acc_off()]
            ; addps xmm1, xmm0
            ; movaps xmm0, xmm1
        ),
        Fmac::Msub => dynasm!(ops ; .arch x64
            ; mulps xmm0, xmm1
            ; movups xmm1, [rbx + acc_off()]
            ; subps xmm1, xmm0
            ; movaps xmm0, xmm1
        ),
    }
    vu_num(ops);
    update_flags(ops, dest);
    let target = match fd {
        Some(0) => return, // the register file discards a write to vf00
        Some(r) => vf_off(r),
        None => acc_off(),
    };
    if dest == 0xF {
        dynasm!(ops ; .arch x64 ; movups [rbx + target], xmm0);
    } else {
        dynasm!(ops ; .arch x64
            ; movaps xmm2, xmm0
            ; movaps xmm1, [rbp + c_dest(dest)]
            ; andps xmm2, xmm1
            ; movups xmm3, [rbx + target]
            ; andnps xmm1, xmm3
            ; orps xmm2, xmm1
            ; movups [rbx + target], xmm2
        );
    }
}

/// Round an FMAC result into the range a VU register can hold: overflow
/// saturates to the largest finite value and a denormal reads back as a
/// signed zero. Working on the magnitude bits turns "exponent all ones"
/// into "above the largest finite value" and "exponent zero" into "below
/// the smallest normal", so each is one signed compare.
fn vu_num(ops: &mut Ops) {
    dynasm!(ops ; .arch x64
        ; movaps xmm1, xmm0
        ; andps xmm1, [rbp + C_ABS]
        ; movaps xmm2, xmm1
        ; pcmpgtd xmm2, [rbp + C_MAXF]
        ; movaps xmm3, [rbp + C_MINNORM]
        ; pcmpgtd xmm3, xmm1
        ; por xmm3, xmm2
        ; pandn xmm3, xmm1
        ; pand xmm2, [rbp + C_MAXF]
        ; por xmm3, xmm2
        ; andps xmm0, [rbp + C_SIGN]
        ; por xmm0, xmm3
    );
}

/// MAC holds one nibble per flag with x in the high bit, so the lanes are
/// shuffled into the opposite order before `movmskps` reads their signs.
/// Status keeps the aggregates in bits 0..5 and their sticky copies above.
fn update_flags(ops: &mut Ops, dest: u32) {
    let d = dest as i32;
    dynasm!(ops ; .arch x64
        ; xorps xmm2, xmm2
        ; movaps xmm3, xmm0
        ; cmpeqps xmm3, xmm2
        ; pshufd xmm3, xmm3, 0x1B
        ; movmskps eax, xmm3
        ; pshufd xmm3, xmm0, 0x1B
        ; movmskps ecx, xmm3
        ; and eax, d
        ; and ecx, d
        ; shl ecx, 4
        ; or eax, ecx
        ; mov WORD [rbx + mac_off()], ax
        ; xor edx, edx
        ; test al, 0x0F
        ; setnz dl
        ; xor ecx, ecx
        ; test al, -16
        ; setnz cl
        ; lea edx, [rdx + rcx*2]
        ; movzx ecx, WORD [rbx + status_off()]
        ; and ecx, -4
        ; or ecx, edx
        ; mov edx, ecx
        ; and edx, 0x3F
        ; shl edx, 6
        ; or ecx, edx
        ; mov WORD [rbx + status_off()], cx
    );
}

// --- native data memory --------------------------------------------------

/// VU1's data memory holds this many quadwords. The recompiler only ever
/// runs VU1, whose decode keeps 14 address bits; [`super::run`] checks it.
pub const DATA_QW_MASK: i32 = 16 * 1024 / 16 - 1;

fn data_ptr_off() -> i32 {
    offset_of!(Vu1, data_ptr) as i32
}

/// Map a single-field dest mask to its field index, the same way the
/// interpreter's `field_index` does.
fn field_index(dest: u32) -> i32 {
    match dest {
        8 => 0,
        4 => 1,
        2 => 2,
        _ => 3,
    }
}

/// Leave the host address of quadword `vi[r] + imm` in rax. The register is
/// read zero-extended: the interpreter sign-extends it for some encodings
/// and not others, but the mask keeps only low bits, where the two agree.
fn data_addr(ops: &mut Ops, r: u32, imm: i32) {
    dynasm!(ops ; .arch x64 ; movzx eax, WORD [rbx + vi_off(r)]);
    if imm != 0 {
        dynasm!(ops ; .arch x64 ; add eax, imm);
    }
    dynasm!(ops ; .arch x64
        ; and eax, DATA_QW_MASK
        ; shl eax, 4
        ; add rax, QWORD [rbx + data_ptr_off()]
    );
}

/// Blend xmm0 into the quadword at `[rbx + off]` under `dest`.
fn store_masked_reg(ops: &mut Ops, off: i32, dest: u32) {
    if dest == 0xF {
        dynasm!(ops ; .arch x64 ; movups [rbx + off], xmm0);
    } else {
        dynasm!(ops ; .arch x64
            ; movaps xmm1, [rbp + c_dest(dest)]
            ; movups xmm2, [rbx + off]
            ; andps xmm0, xmm1
            ; andnps xmm1, xmm2
            ; orps xmm0, xmm1
            ; movups [rbx + off], xmm0
        );
    }
}

/// The same into the data-memory quadword rax points at.
fn store_masked_mem(ops: &mut Ops, dest: u32) {
    if dest == 0xF {
        dynasm!(ops ; .arch x64 ; movups [rax], xmm0);
    } else {
        dynasm!(ops ; .arch x64
            ; movaps xmm1, [rbp + c_dest(dest)]
            ; movups xmm2, [rax]
            ; andps xmm0, xmm1
            ; andnps xmm1, xmm2
            ; orps xmm0, xmm1
            ; movups [rax], xmm0
        );
    }
}

/// Pre-decrement form: the address is `vi[r] - 1`, and the register keeps
/// the decremented value — but only if it is not vi00, whose write the
/// register file discards while the address still uses the decrement.
fn data_addr_pre_dec(ops: &mut Ops, r: u32) {
    dynasm!(ops ; .arch x64
        ; movzx eax, WORD [rbx + vi_off(r)]
        ; sub eax, 1
    );
    if r & 0xF != 0 {
        dynasm!(ops ; .arch x64 ; mov WORD [rbx + vi_off(r)], ax);
    }
    dynasm!(ops ; .arch x64
        ; and eax, DATA_QW_MASK
        ; shl eax, 4
        ; add rax, QWORD [rbx + data_ptr_off()]
    );
}

/// Add `delta` to an integer register, discarding a write to vi00.
fn vi_bump(ops: &mut Ops, r: u32, delta: i32) {
    if r & 0xF == 0 {
        return;
    }
    dynasm!(ops ; .arch x64 ; add WORD [rbx + vi_off(r)], delta as i16);
}

/// The load/store opcodes of both lower tables. Returns false for anything
/// it does not cover.
fn lower_mem_native(ops: &mut Ops, instr: u32) -> bool {
    if !native("mem") {
        return false;
    }
    let opcode = instr >> 25;
    let dest = (instr >> 21) & 0xF;
    let it = (instr >> 16) & 0x1F;
    let is = (instr >> 11) & 0x1F;
    let imm11 = ((instr & 0x7FF) as i32) << 21 >> 21;

    match opcode {
        0x00 => {
            // LQ vf[it], imm(vi[is])
            if it == 0 {
                return true;
            }
            data_addr(ops, is, imm11);
            dynasm!(ops ; .arch x64 ; movups xmm0, [rax]);
            store_masked_reg(ops, vf_off(it), dest);
            true
        }
        0x01 => {
            // SQ vf[is], imm(vi[it])
            data_addr(ops, it, imm11);
            dynasm!(ops ; .arch x64 ; movups xmm0, [rbx + vf_off(is)]);
            store_masked_mem(ops, dest);
            true
        }
        0x04 => {
            // ILW: the low half of the named field.
            if it & 0xF == 0 {
                return true;
            }
            data_addr(ops, is, imm11);
            dynasm!(ops ; .arch x64
                ; movzx ecx, WORD [rax + field_index(dest) * 4]
                ; mov WORD [rbx + vi_off(it)], cx
            );
            true
        }
        0x05 => {
            // ISW: the whole 32-bit field, from the register `it` names.
            data_addr(ops, is, imm11);
            dynasm!(ops ; .arch x64
                ; movzx ecx, WORD [rbx + vi_off(it)]
                ; mov DWORD [rax + field_index(dest) * 4], ecx
            );
            true
        }
        0x40 => lower_mem_special(ops, instr, dest, it, is),
        _ => false,
    }
}

fn lower_mem_special(ops: &mut Ops, instr: u32, dest: u32, it: u32, is: u32) -> bool {
    if !(0x3C..=0x3F).contains(&(instr & 0x3F)) {
        return false;
    }
    match op2(instr) {
        // MOVE / MR32. A `dest` of zero would still be a store of nothing,
        // so both are already handled by `lower_is_nop` when it is vf00.
        id2 @ (0x30 | 0x31) => {
            if it == 0 {
                return true;
            }
            dynasm!(ops ; .arch x64 ; movups xmm0, [rbx + vf_off(is)]);
            if id2 == 0x31 {
                // x<-y, y<-z, z<-w, w<-x.
                dynasm!(ops ; .arch x64 ; shufps xmm0, xmm0, 0x39);
            }
            store_masked_reg(ops, vf_off(it), dest);
            true
        }
        0x34 => {
            // LQI, then post-increment the address register.
            data_addr(ops, is, 0);
            if it != 0 {
                dynasm!(ops ; .arch x64 ; movups xmm0, [rax]);
                store_masked_reg(ops, vf_off(it), dest);
            }
            vi_bump(ops, is, 1);
            true
        }
        0x35 => {
            // SQI
            data_addr(ops, it, 0);
            dynasm!(ops ; .arch x64 ; movups xmm0, [rbx + vf_off(is)]);
            store_masked_mem(ops, dest);
            vi_bump(ops, it, 1);
            true
        }
        0x36 => {
            // LQD: pre-decrement.
            data_addr_pre_dec(ops, is);
            if it != 0 {
                dynasm!(ops ; .arch x64 ; movups xmm0, [rax]);
                store_masked_reg(ops, vf_off(it), dest);
            }
            true
        }
        0x37 => {
            // SQD
            data_addr_pre_dec(ops, it);
            dynasm!(ops ; .arch x64 ; movups xmm0, [rbx + vf_off(is)]);
            store_masked_mem(ops, dest);
            true
        }
        0x3E => {
            // ILWR
            if it & 0xF == 0 {
                return true;
            }
            data_addr(ops, is, 0);
            dynasm!(ops ; .arch x64
                ; movzx ecx, WORD [rax + field_index(dest) * 4]
                ; mov WORD [rbx + vi_off(it)], cx
            );
            true
        }
        0x3F => {
            // ISWR
            data_addr(ops, is, 0);
            dynasm!(ops ; .arch x64
                ; movzx ecx, WORD [rbx + vi_off(it)]
                ; mov DWORD [rax + field_index(dest) * 4], ecx
            );
            true
        }
        // XTOP / XITOP
        id2 @ (0x68 | 0x69) => {
            if it & 0xF == 0 {
                return true;
            }
            let src = if id2 == 0x68 {
                offset_of!(Vu1, top) as i32
            } else {
                offset_of!(Vu1, itop) as i32
            };
            dynasm!(ops ; .arch x64
                ; movzx ecx, WORD [rbx + src]
                ; mov WORD [rbx + vi_off(it)], cx
            );
            true
        }
        _ => false,
    }
}
