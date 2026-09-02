//! Native translation of individual IOP (R3000A) instructions.
//!
//! Register file model: every register lives in `Cpu.gpr` (rbx = &Cpu), so
//! an instruction is a load, an ALU op and a store. Fixed roles: r12 =
//! &Bus, r13d = a branch's condition or its register target, r14d = the
//! value of the load in flight, r15d = instructions the chain retired
//! before this block. rax, rcx, rdx, r8-r11 are scratch (a helper call
//! clobbers them).
//!
//! The architectural load delay is resolved while translating: a load
//! leaves its result in r14d and the *next* instruction commits it to the
//! register file, unless that instruction writes the same register itself.
//! [`Cpu::pending_load`] therefore only ever holds a load whose delay slot
//! crosses a block boundary, and a block is never entered with one set.

use std::mem::offset_of;

use dynasm::dynasm;
use dynasmrt::{DynamicLabel, DynasmApi, DynasmLabelApi, VecAssembler, x64::X64Relocation};

use super::super::Cpu;
use super::helpers as h;
use crate::bus::Bus;

pub type Ops = VecAssembler<X64Relocation>;

const STATUS: u32 = 12;
const STATUS_ISC: i32 = 1 << 16;

#[inline]
fn gpr(i: u32) -> i32 {
    (offset_of!(Cpu, gpr) + i as usize * 4) as i32
}
fn cop0(i: u32) -> i32 {
    (offset_of!(Cpu, cop0) + i as usize * 4) as i32
}
fn hi_off() -> i32 {
    offset_of!(Cpu, hi) as i32
}
fn lo_off() -> i32 {
    offset_of!(Cpu, lo) as i32
}
pub fn pc_off() -> i32 {
    offset_of!(Cpu, pc) as i32
}
pub fn next_pc_off() -> i32 {
    offset_of!(Cpu, next_pc) as i32
}
pub fn current_pc_off() -> i32 {
    offset_of!(Cpu, current_pc) as i32
}
pub fn idle_off() -> i32 {
    offset_of!(Cpu, idle) as i32
}
fn next_is_delay_off() -> i32 {
    offset_of!(Cpu, next_is_delay) as i32
}
pub fn chain_budget_off() -> i32 {
    offset_of!(Bus, iop_chain_budget) as i32
}
fn chain_start_off() -> i32 {
    offset_of!(Bus, iop_chain_start) as i32
}
fn now_off() -> i32 {
    offset_of!(Bus, now) as i32
}

/// What a decoded instruction turned into.
pub enum Emitted {
    /// Translated; control falls through to the next instruction.
    Plain,
    /// Not translated: the caller emits the interpreter call.
    Interp,
    /// A branch or jump; the caller emits the delay slot and then
    /// [`emit_branch_end`].
    Branch(BranchKind),
}

#[derive(Clone, Copy)]
pub enum BranchKind {
    /// Unconditional, to a constant target.
    Jump(u32),
    /// Unconditional, to the address in r13d.
    JumpReg,
    /// Taken when r13d != 0, to a constant target.
    Cond(u32),
}

/// Translation state carried across the instructions of one block.
pub struct State {
    /// Register whose load result waits in r14d for the next instruction to
    /// retire. `None` between blocks.
    pub pend: Option<u32>,
    /// Instructions emitted in this block so far; r15d plus this is the
    /// chain's retired count at the instruction being translated.
    pub local: u32,
    /// Exits taken when a bus access left IOP RAM.
    pub mem_exits: Vec<MemExit>,
    /// Set when a delay slot's own bus access may end the block, so its
    /// branch must not take a link.
    pub no_link: Option<DynamicLabel>,
}

/// A block exit forced by an access outside IOP RAM.
pub struct MemExit {
    pub label: DynamicLabel,
    /// Where execution resumes; the instruction that diverted has retired.
    pub pc: u32,
    /// Address of that instruction.
    pub last: u32,
    /// Load still in flight there, to hand back to the interpreter.
    pub pend: Option<u32>,
    pub retired: u32,
}

impl State {
    pub fn new() -> Self {
        Self { pend: None, local: 0, mem_exits: Vec::new(), no_link: None }
    }
}

// --- small building blocks ------------------------------------------------

/// eax = gpr[r]; the zero register is a real zero.
fn ld_a(ops: &mut Ops, r: u32) {
    if r == 0 {
        dynasm!(ops ; .arch x64 ; xor eax, eax);
    } else {
        dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + gpr(r)]);
    }
}
/// ecx = gpr[r].
fn ld_c(ops: &mut Ops, r: u32) {
    if r == 0 {
        dynasm!(ops ; .arch x64 ; xor ecx, ecx);
    } else {
        dynasm!(ops ; .arch x64 ; mov ecx, DWORD [rbx + gpr(r)]);
    }
}
/// r10d = gpr[r]: a memory address operand, kept out of the argument
/// registers until the call sequence needs it.
fn ld_addr(ops: &mut Ops, r: u32) {
    if r == 0 {
        dynasm!(ops ; .arch x64 ; xor r10d, r10d);
    } else {
        dynasm!(ops ; .arch x64 ; mov r10d, DWORD [rbx + gpr(r)]);
    }
}
/// r11d = gpr[r]: a memory data operand.
fn ld_data(ops: &mut Ops, r: u32) {
    if r == 0 {
        dynasm!(ops ; .arch x64 ; xor r11d, r11d);
    } else {
        dynasm!(ops ; .arch x64 ; mov r11d, DWORD [rbx + gpr(r)]);
    }
}
/// gpr[r] = eax, unless r is the zero register.
fn st_a(ops: &mut Ops, r: u32) {
    if r != 0 {
        dynasm!(ops ; .arch x64 ; mov DWORD [rbx + gpr(r)], eax);
    }
}

/// Retire the instruction just emitted: the load in flight lands in its
/// register unless this instruction wrote the same one directly.
fn commit(ops: &mut Ops, st: &mut State, written: u32) {
    if let Some(r) = st.pend.take()
        && r != written
    {
        dynasm!(ops ; .arch x64 ; mov DWORD [rbx + gpr(r)], r14d);
    }
}

/// Spill the load in flight into [`Cpu::pending_load`], for an exit or an
/// interpreter call that has to see the interpreter's own representation.
pub fn spill_pend(ops: &mut Ops, pend: Option<u32>) {
    let Some(r) = pend else { return };
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov rcx, rbx
        ; mov edx, r as i32
        ; mov r8d, r14d
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rdi, rbx
        ; mov esi, r as i32
        ; mov edx, r14d
    );
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD h::set_pending as *const () as i64
        ; call rax
    );
}

/// Call `f(bus, r10d, retired)`.
fn call2(ops: &mut Ops, f: usize, local: u32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov rcx, r12
        ; mov edx, r10d
        ; lea r8d, [r15 + local as i32]
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rdi, r12
        ; mov esi, r10d
        ; lea edx, [r15 + local as i32]
    );
    dynasm!(ops ; .arch x64 ; mov rax, QWORD f as i64 ; call rax);
}

/// Call `f(bus, r10d, r11d, retired)`.
fn call3(ops: &mut Ops, f: usize, local: u32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov rcx, r12
        ; mov edx, r10d
        ; mov r8d, r11d
        ; lea r9d, [r15 + local as i32]
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rdi, r12
        ; mov esi, r10d
        ; mov edx, r11d
        ; lea ecx, [r15 + local as i32]
    );
    dynasm!(ops ; .arch x64 ; mov rax, QWORD f as i64 ; call rax);
}

/// Call `f(cpu, bus, addr, instr)`.
pub fn call_interp(ops: &mut Ops, f: usize, addr: u32, instr: u32) {
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov rcx, rbx
        ; mov rdx, r12
        ; mov r8d, addr as i32
        ; mov r9d, instr as i32
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rdi, rbx
        ; mov rsi, r12
        ; mov edx, addr as i32
        ; mov ecx, instr as i32
    );
    dynasm!(ops ; .arch x64 ; mov rax, QWORD f as i64 ; call rax);
}

/// Put [`Bus::now`] at the instruction about to run. The memory helpers do
/// this themselves; an interpreter call needs it emitted.
pub fn emit_now(ops: &mut Ops, local: u32) {
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD [r12 + chain_start_off()]
        ; lea ecx, [r15 + local as i32]
        ; lea rax, [rax + rcx * 8]
        ; mov QWORD [r12 + now_off()], rax
    );
}

/// Record the exit an out-of-RAM access jumps to, and test for it. rax
/// holds a load's tagged result, eax a store's flag.
fn mem_test(ops: &mut Ops, st: &mut State, addr: u32, load: bool) {
    let label = ops.new_dynamic_label();
    if load {
        dynasm!(ops ; .arch x64 ; shr rax, 32 ; jnz =>label);
    } else {
        dynasm!(ops ; .arch x64 ; test eax, eax ; jnz =>label);
    }
    st.mem_exits.push(MemExit {
        label,
        pc: addr.wrapping_add(4),
        last: addr,
        pend: st.pend,
        retired: st.local + 1,
    });
}

/// The same test inside a delay slot, where the block ends anyway: all it
/// has to do is keep the branch from taking its link.
fn mem_test_delay(ops: &mut Ops, st: &mut State, load: bool) {
    let label = *st.no_link.get_or_insert_with(|| ops.new_dynamic_label());
    if load {
        dynasm!(ops ; .arch x64 ; shr rax, 32 ; jnz =>label);
    } else {
        dynasm!(ops ; .arch x64 ; test eax, eax ; jnz =>label);
    }
}

// --- the translator -------------------------------------------------------

/// Emit `instr` as if fetched at `addr`, including the delayed-load commit
/// of the instruction before it. `delay` marks a branch delay slot, where
/// an exit only has to suppress the branch's link.
pub fn emit(ops: &mut Ops, st: &mut State, addr: u32, instr: u32, delay: bool) -> Emitted {
    // 13% of the stream is `nop`, which is `sll $0,$0,0`: every register
    // write below falls away, but the delayed-load commit does not.
    if instr == 0 {
        commit(ops, st, 0);
        return Emitted::Plain;
    }
    let op = instr >> 26;
    let rs = (instr >> 21) & 31;
    let rt = (instr >> 16) & 31;
    let rd = (instr >> 11) & 31;
    let sa = ((instr >> 6) & 31) as i8;
    let imm = instr & 0xFFFF;
    let simm = imm as u16 as i16 as i32;
    let link_pc = addr.wrapping_add(8) as i32;
    let rel_target = addr.wrapping_add(4).wrapping_add((simm << 2) as u32);

    match op {
        0x00 => match instr & 0x3F {
            0x00 | 0x02 | 0x03 => {
                ld_a(ops, rt);
                match instr & 0x3F {
                    0x00 => dynasm!(ops ; .arch x64 ; shl eax, sa),
                    0x02 => dynasm!(ops ; .arch x64 ; shr eax, sa),
                    _ => dynasm!(ops ; .arch x64 ; sar eax, sa),
                }
                st_a(ops, rd);
                commit(ops, st, rd);
            }
            0x04 | 0x06 | 0x07 => {
                ld_c(ops, rs);
                ld_a(ops, rt);
                match instr & 0x3F {
                    0x04 => dynasm!(ops ; .arch x64 ; shl eax, cl),
                    0x06 => dynasm!(ops ; .arch x64 ; shr eax, cl),
                    _ => dynasm!(ops ; .arch x64 ; sar eax, cl),
                }
                st_a(ops, rd);
                commit(ops, st, rd);
            }
            // jr / jalr read the target before writing the link register,
            // which matters when they are the same one.
            0x08 | 0x09 => {
                ld_a(ops, rs);
                dynasm!(ops ; .arch x64 ; mov r13d, eax);
                let written = if instr & 0x3F == 0x09 && rd != 0 {
                    dynasm!(ops ; .arch x64 ; mov DWORD [rbx + gpr(rd)], link_pc);
                    rd
                } else {
                    0
                };
                commit(ops, st, written);
                return Emitted::Branch(BranchKind::JumpReg);
            }
            0x10 | 0x12 => {
                let off = if instr & 0x3F == 0x10 { hi_off() } else { lo_off() };
                dynasm!(ops ; .arch x64 ; mov eax, DWORD [rbx + off]);
                st_a(ops, rd);
                commit(ops, st, rd);
            }
            0x11 | 0x13 => {
                let off = if instr & 0x3F == 0x11 { hi_off() } else { lo_off() };
                ld_a(ops, rs);
                dynasm!(ops ; .arch x64 ; mov DWORD [rbx + off], eax);
                commit(ops, st, 0);
            }
            0x20..=0x27 => {
                ld_a(ops, rs);
                ld_c(ops, rt);
                match instr & 0x3F {
                    0x20 | 0x21 => dynasm!(ops ; .arch x64 ; add eax, ecx),
                    0x22 | 0x23 => dynasm!(ops ; .arch x64 ; sub eax, ecx),
                    0x24 => dynasm!(ops ; .arch x64 ; and eax, ecx),
                    0x25 => dynasm!(ops ; .arch x64 ; or eax, ecx),
                    0x26 => dynasm!(ops ; .arch x64 ; xor eax, ecx),
                    _ => dynasm!(ops ; .arch x64 ; or eax, ecx ; not eax),
                }
                st_a(ops, rd);
                commit(ops, st, rd);
            }
            0x2A | 0x2B => {
                ld_a(ops, rs);
                ld_c(ops, rt);
                dynasm!(ops ; .arch x64 ; cmp eax, ecx);
                if instr & 0x3F == 0x2A {
                    dynasm!(ops ; .arch x64 ; setl al);
                } else {
                    dynasm!(ops ; .arch x64 ; setb al);
                }
                dynasm!(ops ; .arch x64 ; movzx eax, al);
                st_a(ops, rd);
                commit(ops, st, rd);
            }
            _ => return Emitted::Interp,
        },
        0x01 => {
            ld_a(ops, rs);
            dynasm!(ops ; .arch x64 ; test eax, eax);
            if rt & 1 == 0 {
                dynasm!(ops ; .arch x64 ; setl al);
            } else {
                dynasm!(ops ; .arch x64 ; setge al);
            }
            dynasm!(ops ; .arch x64 ; movzx r13d, al);
            // bltzal/bgezal link whether or not the branch is taken.
            let written = if rt & 0x1E == 0x10 {
                dynasm!(ops ; .arch x64 ; mov DWORD [rbx + gpr(31)], link_pc);
                31
            } else {
                0
            };
            commit(ops, st, written);
            return Emitted::Branch(BranchKind::Cond(rel_target));
        }
        0x02 | 0x03 => {
            let target = (addr.wrapping_add(4) & 0xF000_0000) | ((instr & 0x03FF_FFFF) << 2);
            let written = if op == 0x03 {
                dynasm!(ops ; .arch x64 ; mov DWORD [rbx + gpr(31)], link_pc);
                31
            } else {
                0
            };
            commit(ops, st, written);
            return Emitted::Branch(BranchKind::Jump(target));
        }
        0x04 | 0x05 => {
            ld_a(ops, rs);
            ld_c(ops, rt);
            dynasm!(ops ; .arch x64 ; cmp eax, ecx);
            if op == 0x04 {
                dynasm!(ops ; .arch x64 ; sete al);
            } else {
                dynasm!(ops ; .arch x64 ; setne al);
            }
            dynasm!(ops ; .arch x64 ; movzx r13d, al);
            commit(ops, st, 0);
            return Emitted::Branch(BranchKind::Cond(rel_target));
        }
        0x06 | 0x07 => {
            ld_a(ops, rs);
            dynasm!(ops ; .arch x64 ; test eax, eax);
            if op == 0x06 {
                dynasm!(ops ; .arch x64 ; setle al);
            } else {
                dynasm!(ops ; .arch x64 ; setg al);
            }
            dynasm!(ops ; .arch x64 ; movzx r13d, al);
            commit(ops, st, 0);
            return Emitted::Branch(BranchKind::Cond(rel_target));
        }
        0x08 | 0x09 => {
            ld_a(ops, rs);
            dynasm!(ops ; .arch x64 ; add eax, simm);
            st_a(ops, rt);
            commit(ops, st, rt);
        }
        0x0A | 0x0B => {
            ld_a(ops, rs);
            dynasm!(ops ; .arch x64 ; cmp eax, simm);
            if op == 0x0A {
                dynasm!(ops ; .arch x64 ; setl al);
            } else {
                dynasm!(ops ; .arch x64 ; setb al);
            }
            dynasm!(ops ; .arch x64 ; movzx eax, al);
            st_a(ops, rt);
            commit(ops, st, rt);
        }
        0x0C..=0x0E => {
            ld_a(ops, rs);
            match op {
                0x0C => dynasm!(ops ; .arch x64 ; and eax, imm as i32),
                0x0D => dynasm!(ops ; .arch x64 ; or eax, imm as i32),
                _ => dynasm!(ops ; .arch x64 ; xor eax, imm as i32),
            }
            st_a(ops, rt);
            commit(ops, st, rt);
        }
        0x0F => {
            dynasm!(ops ; .arch x64 ; mov eax, (imm << 16) as i32);
            st_a(ops, rt);
            commit(ops, st, rt);
        }
        // mfc0 is a delayed load like any other. mtc0 and rfe go to the
        // interpreter and end the block, either being able to unmask an
        // interrupt the very next instruction would take.
        0x10 if rs == 0x00 => {
            commit(ops, st, 0);
            if rt != 0 {
                dynasm!(ops ; .arch x64 ; mov r14d, DWORD [rbx + cop0(rd)]);
                st.pend = Some(rt);
            }
        }
        0x20 | 0x21 | 0x23 | 0x24 | 0x25 => {
            ld_addr(ops, rs);
            if simm != 0 {
                dynasm!(ops ; .arch x64 ; add r10d, simm);
            }
            commit(ops, st, 0);
            let f = match op {
                0x20 | 0x24 => h::rd8 as *const () as usize,
                0x21 | 0x25 => h::rd16 as *const () as usize,
                _ => h::rd32 as *const () as usize,
            };
            call2(ops, f, st.local);
            if rt != 0 {
                match op {
                    0x20 => dynasm!(ops ; .arch x64 ; movsx r14d, al),
                    0x21 => dynasm!(ops ; .arch x64 ; movsx r14d, ax),
                    0x24 => dynasm!(ops ; .arch x64 ; movzx r14d, al),
                    0x25 => dynasm!(ops ; .arch x64 ; movzx r14d, ax),
                    _ => dynasm!(ops ; .arch x64 ; mov r14d, eax),
                }
                st.pend = Some(rt);
            }
            if delay {
                mem_test_delay(ops, st, true);
            } else {
                mem_test(ops, st, addr, true);
            }
        }
        0x22 | 0x26 => {
            ld_addr(ops, rs);
            if simm != 0 {
                dynasm!(ops ; .arch x64 ; add r10d, simm);
            }
            // The merge base is the in-flight value when the pairing load
            // targets the same register, exactly as `load_merge_base` does.
            if st.pend == Some(rt) {
                dynasm!(ops ; .arch x64 ; mov r11d, r14d);
            } else {
                ld_data(ops, rt);
            }
            commit(ops, st, 0);
            let f = if op == 0x22 {
                h::lwl as *const () as usize
            } else {
                h::lwr as *const () as usize
            };
            call3(ops, f, st.local);
            if rt != 0 {
                dynasm!(ops ; .arch x64 ; mov r14d, eax);
                st.pend = Some(rt);
            }
            if delay {
                mem_test_delay(ops, st, true);
            } else {
                mem_test(ops, st, addr, true);
            }
        }
        0x28 | 0x29 | 0x2A | 0x2B | 0x2E => {
            ld_addr(ops, rs);
            if simm != 0 {
                dynasm!(ops ; .arch x64 ; add r10d, simm);
            }
            ld_data(ops, rt);
            commit(ops, st, 0);
            let f = match op {
                0x28 => h::wr8 as *const () as usize,
                0x29 => h::wr16 as *const () as usize,
                0x2A => h::swl as *const () as usize,
                0x2B => h::wr32 as *const () as usize,
                _ => h::swr as *const () as usize,
            };
            // Cache isolation swallows the store, bus access and all.
            let isolated = ops.new_dynamic_label();
            dynasm!(ops
                ; .arch x64
                ; test DWORD [rbx + cop0(STATUS)], STATUS_ISC
                ; jnz =>isolated
            );
            call3(ops, f, st.local);
            if delay {
                mem_test_delay(ops, st, false);
            } else {
                mem_test(ops, st, addr, false);
            }
            dynasm!(ops ; .arch x64 ; =>isolated);
        }
        _ => return Emitted::Interp,
    }
    Emitted::Plain
}

// --- exits ----------------------------------------------------------------

/// Exit sequences. Every exit adds the block's retired count to r15d first.
/// A constant target goes through a link cell: an indirect jump whose cell
/// holds either the target block's body, once compiled, or this block's own
/// slow path, which stores pc and returns to the dispatcher.
pub struct Exits<'a> {
    pub jit: &'a mut super::Jit,
    /// (cell index, target pc, offset of the slow path in this block).
    pub links: Vec<(usize, u32, usize)>,
}

impl Exits<'_> {
    /// Exit to constant `target` after `count` instructions, linking the
    /// jump to the target block once one exists. `last` is the address of
    /// the instruction that retired last.
    pub fn to(&mut self, ops: &mut Ops, target: u32, count: u32, last: u32) {
        let (idx, cell) = self.jit.link_cell();
        dynasm!(ops
            ; .arch x64
            ; add r15d, count as i32
            ; mov rax, QWORD cell as i64
            ; jmp QWORD [rax]
        );
        self.links.push((idx, target, ops.offset().0));
        store_pc(ops, target, last);
        dynasm!(ops ; .arch x64 ; mov eax, r15d ; jmp ->epilogue);
    }

    /// Exit to constant `target` without linking, so the dispatcher and its
    /// caller get their turn.
    pub fn to_unlinked(&mut self, ops: &mut Ops, target: u32, count: u32, last: u32) {
        dynasm!(ops ; .arch x64 ; add r15d, count as i32);
        store_pc(ops, target, last);
        dynasm!(ops ; .arch x64 ; mov eax, r15d ; jmp ->epilogue);
    }

    /// Exit to the address in r13d.
    pub fn to_reg(&mut self, ops: &mut Ops, count: u32, last: u32) {
        dynasm!(ops
            ; .arch x64
            ; add r15d, count as i32
            ; mov DWORD [rbx + pc_off()], r13d
            ; lea eax, [r13 + 4]
            ; mov DWORD [rbx + next_pc_off()], eax
            ; mov DWORD [rbx + current_pc_off()], last as i32
            ; mov eax, r15d
            ; jmp ->epilogue
        );
    }

    /// Exit on the kernel idle thread's `j .`, leaving the interpreter's
    /// state exactly: the branch has retired, its delay slot has not, and
    /// the dispatcher's caller stops stepping the IOP until an interrupt
    /// arrives. `next_is_delay` is what makes an exception taken here
    /// report the branch rather than the slot.
    pub fn to_idle(&mut self, ops: &mut Ops, delay_slot: u32, target: u32, count: u32, last: u32) {
        dynasm!(ops
            ; .arch x64
            ; add r15d, count as i32
            ; mov BYTE [rbx + idle_off()], 1
            ; mov BYTE [rbx + next_is_delay_off()], 1
            ; mov DWORD [rbx + pc_off()], delay_slot as i32
            ; mov DWORD [rbx + next_pc_off()], target as i32
            ; mov DWORD [rbx + current_pc_off()], last as i32
            ; mov eax, r15d
            ; jmp ->epilogue
        );
    }

    /// Exit with pc already stored by an interpreted instruction.
    pub fn to_stored(&mut self, ops: &mut Ops, count: u32) {
        dynasm!(ops ; .arch x64 ; add r15d, count as i32 ; mov eax, r15d ; jmp ->epilogue);
    }
}

fn store_pc(ops: &mut Ops, target: u32, last: u32) {
    dynasm!(ops
        ; .arch x64
        ; mov DWORD [rbx + pc_off()], target as i32
        ; mov DWORD [rbx + next_pc_off()], target.wrapping_add(4) as i32
        ; mov DWORD [rbx + current_pc_off()], last as i32
    );
}

/// After the delay slot: resolve the branch and leave the block. `link` is
/// false when the delay slot's own bus access already ended it, or when it
/// left a load whose delay crosses the boundary.
pub fn emit_branch_end(
    ops: &mut Ops,
    exits: &mut Exits,
    kind: BranchKind,
    fallthrough: u32,
    count: u32,
    last: u32,
    link: bool,
) {
    match kind {
        BranchKind::Jump(t) => {
            if link {
                exits.to(ops, t, count, last);
            } else {
                exits.to_unlinked(ops, t, count, last);
            }
        }
        BranchKind::JumpReg => exits.to_reg(ops, count, last),
        BranchKind::Cond(target) => {
            let not_taken = ops.new_dynamic_label();
            dynasm!(ops ; .arch x64 ; test r13d, r13d ; jz =>not_taken);
            if link {
                exits.to(ops, target, count, last);
            } else {
                exits.to_unlinked(ops, target, count, last);
            }
            dynasm!(ops ; .arch x64 ; =>not_taken);
            if link {
                exits.to(ops, fallthrough, count, last);
            } else {
                exits.to_unlinked(ops, fallthrough, count, last);
            }
        }
    }
}
