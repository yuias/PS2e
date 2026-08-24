//! EE dynamic recompiler (x86-64).
//!
//! Straight-line runs of instructions ("blocks") are translated once and
//! cached by virtual PC. A block is an `extern "C"` function taking the
//! CPU and bus pointers and returning how many cycles it retired (the
//! aligned second half of a dual-issued couple costs zero — see
//! [`super::issue`], the model shared with the interpreter);
//! [`Jit::run`] executes one block and returns the EE cycles spent so the
//! system can advance the IOP, timers and vblank by the same amount. The
//! interpreter stays the reference: instructions the translator does not
//! handle natively are executed by calling back into [`Cpu::execute`], and
//! anything that diverts control (branches, exceptions, idle detection)
//! ends the block and hands the delay slot to the interpreter.
//!
//! Register state lives in the [`Cpu`] struct throughout (rbx points at it,
//! r12 at the [`Bus`]); native code reads and writes the fields directly.
//! Self-modifying code is caught by the bus: it flags RAM pages that hold
//! translated code on write and the dispatcher drops their blocks.

mod arena;
mod emit;
mod helpers;

use std::collections::HashMap;

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, VecAssembler, x64::X64Relocation};

use super::Cpu;
use crate::bus::{Bus, RAM_SIZE};
use arena::Arena;

/// Native block entry: (cpu, bus, cycle budget) -> cycles retired by the
/// chain of linked blocks that ran.
type Entry = unsafe extern "C" fn(*mut Cpu, *mut Bus, u32) -> u32;

/// Code arena size; when full, every block is dropped and it starts over.
const ARENA_BYTES: usize = 64 << 20;
/// Longest block, in instructions; may run one further so the fall-through
/// lands 8-byte aligned (a block seam must never straddle a dual-issue
/// couple, or the two worlds would charge it differently).
const MAX_BLOCK: usize = 64;
/// Direct-mapped lookup entries (indexed by pc >> 2).
const LOOKUP_ENTRIES: usize = 1 << 16;
/// Link cells (one per constant-target exit); the cache is flushed when
/// they run out.
const LINK_CELLS: usize = 1 << 20;

struct Block {
    pc: u32,
    entry: Entry,
    /// Address of the body (after the prologue): where linked jumps land.
    body: u64,
    /// Physical byte range the block was translated from (RAM blocks).
    phys: Option<(u32, u32)>,
    /// Link cells currently pointing at this block's body.
    incoming: Vec<usize>,
    valid: bool,
}

pub struct Jit {
    arena: Arena,
    blocks: Vec<Block>,
    by_pc: HashMap<u32, u32>,
    /// pc >> 2 -> block index + 1 (0 = empty); verified against `blocks`.
    lookup: Box<[u32]>,
    /// RAM page -> blocks translated from it.
    page_blocks: Vec<Vec<u32>>,
    /// Link cells: jump targets read by block exits (see `emit::Exits`).
    cells: Box<[u64]>,
    /// Each cell's own slow path, restored when its target goes away.
    cell_slow: Box<[u64]>,
    cells_used: usize,
    /// Cells waiting for a block at this pc to be compiled.
    pending_links: HashMap<u32, Vec<usize>>,
    pub blocks_compiled: u64,
    pub blocks_invalidated: u64,
    pub blocks_run: u64,
    pub interp_steps: u64,
}

impl Jit {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            arena: Arena::new(ARENA_BYTES)?,
            blocks: Vec::new(),
            by_pc: HashMap::new(),
            lookup: vec![0u32; LOOKUP_ENTRIES].into_boxed_slice(),
            page_blocks: vec![Vec::new(); RAM_SIZE >> 12],
            cells: vec![0u64; LINK_CELLS].into_boxed_slice(),
            cell_slow: vec![0u64; LINK_CELLS].into_boxed_slice(),
            cells_used: 0,
            pending_links: HashMap::new(),
            blocks_compiled: 0,
            blocks_invalidated: 0,
            blocks_run: 0,
            interp_steps: 0,
        })
    }

    /// Execute from the current PC: a chain of linked blocks until about
    /// `budget` cycles have retired (a block may overshoot), or one
    /// interpreter step when the CPU is mid-branch or has an interrupt to
    /// take. Returns the EE cycles consumed (>= 1).
    pub fn run(&mut self, cpu: &mut Cpu, bus: &mut Bus, budget: u32) -> u32 {
        if bus.jit_flush_needed {
            bus.jit_flush_needed = false;
            self.flush(bus);
        } else if !bus.dirty_code_writes.is_empty() {
            self.invalidate_dirty(bus);
        }
        if cpu.next_is_delay || cpu.interrupt_pending(bus) {
            cpu.step(bus);
            self.interp_steps += 1;
            return 1;
        }
        self.blocks_run += 1;
        let pc = cpu.pc;
        if pc == 0 {
            panic!("EE jumped to null (previous pc {:#010x})", cpu.current_pc);
        }
        let entry = match self.find(pc) {
            Some(e) => e,
            None => self.compile(pc, bus),
        };
        // SAFETY: `entry` is a complete block in our arena; the pointers are
        // exclusively ours for the call and the block only touches the CPU
        // and bus through the helpers below.
        let mut n = unsafe { entry(cpu as *mut Cpu, bus as *mut Bus, budget.max(1)) };
        // A diverting instruction (branch, exception, idle loop) leaves the
        // interpreter's delay-slot state behind; let it finish the branch.
        while cpu.next_is_delay {
            cpu.step(bus);
            self.interp_steps += 1;
            n += 1;
        }
        n
    }

    #[inline]
    fn find(&self, pc: u32) -> Option<Entry> {
        let slot = ((pc >> 2) as usize) & (LOOKUP_ENTRIES - 1);
        let idx = self.lookup[slot];
        if idx != 0 {
            let b = &self.blocks[idx as usize - 1];
            if b.valid && b.pc == pc {
                return Some(b.entry);
            }
        }
        let &idx = self.by_pc.get(&pc)?;
        let b = &self.blocks[idx as usize];
        if !b.valid {
            return None;
        }
        Some(b.entry)
    }

    /// Drop the blocks whose code was written (8-byte granularity).
    fn invalidate_dirty(&mut self, bus: &mut Bus) {
        for a in bus.dirty_code_writes.drain(..) {
            let page = (a >> 12) as usize;
            let mut any_left = false;
            for &idx in &self.page_blocks[page] {
                let b = &mut self.blocks[idx as usize];
                if !b.valid {
                    continue;
                }
                match b.phys {
                    Some((s, e)) if s < a + 8 && a < e => {
                        b.valid = false;
                        self.by_pc.remove(&b.pc);
                        self.blocks_invalidated += 1;
                        // Anything linked here goes back to its slow path
                        // and waits for a recompile at this pc.
                        let (pc, incoming) = (b.pc, std::mem::take(&mut b.incoming));
                        for &c in &incoming {
                            self.cells[c] = self.cell_slow[c];
                        }
                        self.pending_links.entry(pc).or_default().extend(incoming);
                    }
                    _ => any_left = true,
                }
            }
            if !any_left {
                self.page_blocks[page].clear();
                bus.code_pages[page] = false;
            }
        }
    }

    /// Drop every block (arena full, TLB rewrite, ...).
    pub fn flush(&mut self, bus: &mut Bus) {
        self.arena.reset();
        self.blocks.clear();
        self.by_pc.clear();
        self.lookup.fill(0);
        for v in &mut self.page_blocks {
            v.clear();
        }
        bus.code_pages.fill(false);
        bus.dirty_code_writes.clear();
        self.cells_used = 0;
        self.pending_links.clear();
    }

    /// Allocate a link cell; returns (index, address).
    fn link_cell(&mut self) -> (usize, u64) {
        let idx = self.cells_used;
        self.cells_used += 1;
        (idx, &self.cells[idx] as *const u64 as u64)
    }

    fn compile(&mut self, pc: u32, bus: &mut Bus) -> Entry {
        // Generous upper bound per block; reset the arena rather than fail.
        if self.arena.remaining() < 64 * 1024 || self.cells_used + 256 > LINK_CELLS {
            self.flush(bus);
        }
        let base = self.arena.next_addr();
        let mut ops = VecAssembler::<X64Relocation>::new(base);
        emit_prologue(&mut ops);
        // Body: linked jumps land here; leave when the cycle budget is used.
        let body_off = ops.offset().0;
        let budget_exit = ops.new_dynamic_label();
        dynasm!(ops
            ; .arch x64
            ; cmp r15d, ebp
            ; jae =>budget_exit
        );
        let mut exits = emit::Exits { jit: self, links: Vec::new() };

        // (label, cycles retired) for interpreter calls that diverted.
        let mut interp_exits: Vec<(dynasmrt::DynamicLabel, u32)> = Vec::with_capacity(MAX_BLOCK);
        let mut addr = pc;
        let mut count = 0u32;
        let mut cycles = 0u32;
        // Previous instruction when it falls through to `addr` within this
        // block (dual-issue pairing; None across control flow).
        let mut prev: Option<u32> = None;
        loop {
            let instr = bus.fetch32(addr);
            let next = bus.fetch32(addr.wrapping_add(4));
            // The aligned second half of a hazard-free couple retires with
            // its head. Decided on the instruction words alone, exactly as
            // the interpreter does, so both worlds charge the same cycles.
            let free = addr & 7 == 4 && prev.is_some_and(|p| super::issue::dual_issue(p, instr));
            // A control-flow instruction in the delay slot is undefined
            // behaviour on MIPS; leave such branches to the interpreter.
            let mut emitted = if is_control_flow(instr) && is_control_flow(next) {
                emit::Emitted::Interp
            } else {
                emit::emit(&mut ops, addr, instr)
            };
            if let emit::Emitted::Branch(kind) = emitted {
                count += 1;
                // The branch itself may be the second half of a couple; its
                // delay slot always costs its own cycle.
                if !free {
                    cycles += 1;
                }
                let ds_addr = addr.wrapping_add(4);
                if let emit::BranchKind::Cond { likely: true, .. } = kind {
                    emit::emit_likely_skip(&mut ops, ds_addr.wrapping_add(4), cycles, &mut exits);
                }
                if let emit::Emitted::Interp = emit::emit(&mut ops, ds_addr, next) {
                    emit_interp(&mut ops, ds_addr, next, cycles + 1, &mut interp_exits);
                }
                count += 1;
                cycles += 1;
                let idle = is_idle_loop(bus, addr, instr, next);
                emit::emit_branch_end(&mut ops, kind, ds_addr.wrapping_add(4), cycles, idle, &mut exits);
                break;
            }
            count += 1;
            if !free {
                cycles += 1;
            }
            if let emit::Emitted::Interp = emitted {
                emit_interp(&mut ops, addr, instr, cycles, &mut interp_exits);
                emitted = emit::Emitted::Plain;
            }
            let _ = emitted;
            let control = is_control_flow(instr);
            prev = if control { None } else { Some(instr) };
            addr = addr.wrapping_add(4);
            if control {
                // pc/next_pc were left by the interpreted instruction.
                exits.to_stored(&mut ops, cycles);
                break;
            }
            if (count as usize >= MAX_BLOCK && addr & 7 == 0) || addr & 0xFFF == 0 {
                exits.to(&mut ops, addr, cycles);
                break;
            }
        }
        for (label, retired) in &interp_exits {
            dynasm!(ops ; .arch x64 ; =>*label);
            exits.to_stored(&mut ops, *retired);
        }
        dynasm!(ops
            ; .arch x64
            ; =>budget_exit
            ; mov DWORD [rbx + emit::pc_off()], pc as i32
            ; mov DWORD [rbx + emit::next_pc_off()], pc.wrapping_add(4) as i32
            ; mov eax, r15d
            ; jmp ->epilogue
        );
        emit_epilogue(&mut ops);
        let links = exits.links;

        let code = ops.finalize().expect("dynasm assembly failed");
        let ptr = self.arena.place(&code);
        debug_assert_eq!(ptr as usize, base);
        // SAFETY: the bytes at ptr are the function assembled above.
        let entry: Entry = unsafe { std::mem::transmute(ptr) };
        let body = (base + body_off) as u64;

        // Track the RAM range the block reads its code from, for
        // invalidation on write. ROM/other blocks are never invalidated.
        let last = pc.wrapping_add((count.max(1) - 1) * 4);
        let idx = self.blocks.len() as u32;
        let phys = match (bus.ram_phys_of(pc), bus.ram_phys_of(last)) {
            (Some(s), Some(e)) => Some((s, e + 4)),
            _ => None,
        };
        if let Some((s, e)) = phys {
            for p in (s >> 12)..=((e - 1) >> 12) {
                self.page_blocks[p as usize].push(idx);
                bus.code_pages[p as usize] = true;
            }
        }
        self.blocks.push(Block { pc, entry, body, phys, incoming: Vec::new(), valid: true });
        self.by_pc.insert(pc, idx);
        self.lookup[((pc >> 2) as usize) & (LOOKUP_ENTRIES - 1)] = idx + 1;
        self.blocks_compiled += 1;

        // Wire this block's exits to compiled targets (or park them), and
        // point exits parked on this pc at the new body.
        for (cell, target, slow_off) in links {
            let slow = (base + slow_off) as u64;
            self.cell_slow[cell] = slow;
            match self.by_pc.get(&target).map(|&i| i as usize) {
                Some(t) if self.blocks[t].valid => {
                    self.cells[cell] = self.blocks[t].body;
                    self.blocks[t].incoming.push(cell);
                }
                _ => {
                    self.cells[cell] = slow;
                    self.pending_links.entry(target).or_default().push(cell);
                }
            }
        }
        if let Some(waiting) = self.pending_links.remove(&pc) {
            for cell in waiting {
                self.cells[cell] = body;
                self.blocks[idx as usize].incoming.push(cell);
            }
        }
        entry
    }

}

/// Call the interpreter for one instruction; exit the block with
/// `retired` cycles if it diverted control.
fn emit_interp(
    ops: &mut emit::Ops,
    addr: u32,
    instr: u32,
    retired: u32,
    exits: &mut Vec<(dynasmrt::DynamicLabel, u32)>,
) {
    let exit = ops.new_dynamic_label();
    emit_call4(ops, interp_one as *const () as usize, addr, instr);
    dynasm!(ops
        ; .arch x64
        ; test eax, eax
        ; jnz =>exit
    );
    exits.push((exit, retired));
}

/// The kernel idle thread: `beq $0,$0` backwards over nothing but nops
/// (delay slot included), as the interpreter's `check_idle_loop` sees it.
fn is_idle_loop(bus: &mut Bus, addr: u32, instr: u32, delay_slot: u32) -> bool {
    if instr >> 26 != 0x04 || (instr >> 16) & 0x3FF != 0 || delay_slot != 0 {
        return false;
    }
    let off = ((instr & 0xFFFF) as u16 as i16 as i32) << 2;
    let target = addr.wrapping_add(4).wrapping_add(off as u32);
    if target > addr || addr - target > 64 {
        return false;
    }
    (target..addr).step_by(4).all(|a| bus.fetch32(a) == 0)
}

/// Instructions after which a block must end: anything that may change PC
/// other than by falling through.
fn is_control_flow(instr: u32) -> bool {
    let rs = (instr >> 21) & 0x1F;
    match instr >> 26 {
        // SPECIAL: jr, jalr, syscall, break, traps.
        0x00 => matches!(instr & 0x3F, 0x08 | 0x09 | 0x0C | 0x0D | 0x30..=0x36),
        // REGIMM branches (0x00-0x03, 0x10-0x13); j/jal, beq..bgtz, likely.
        0x01 => matches!((instr >> 16) & 0x1F, 0x00..=0x03 | 0x10..=0x13),
        0x02..=0x07 | 0x14..=0x17 => true,
        // COP0: bc0 and the TLB/eret group. mtc0 and ei/di are translated
        // in place; an interrupt they unmask waits for the block to end,
        // within the latency the chain budget already allows.
        0x10 => rs == 0x08 || (matches!(rs, 0x10..=0x1F) && !matches!(instr & 0x3F, 0x38 | 0x39)),
        // COP1 bc1, COP2 bc2.
        0x11 | 0x12 => rs == 0x08,
        _ => false,
    }
}

// --- helpers called from generated code ----------------------------------

/// Run one instruction through the interpreter as if fetched at `addr`.
/// Returns 1 when control was diverted (see [`Cpu::exec_at`]).
extern "C" fn interp_one(cpu: *mut Cpu, bus: *mut Bus, addr: u32, instr: u32) -> u32 {
    // Panics must not unwind through the native frames.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the dispatcher passes live, exclusively owned pointers
        // for the duration of the block call.
        let (cpu, bus) = unsafe { (&mut *cpu, &mut *bus) };
        cpu.exec_at(bus, addr, instr)
    }));
    match r {
        Ok(diverted) => diverted as u32,
        Err(_) => {
            eprintln!("panic inside JIT helper at pc {addr:#010x}");
            std::process::abort();
        }
    }
}

// --- x86-64 emission ------------------------------------------------------

#[cfg(windows)]
macro_rules! call_args {
    ($ops:ident, $a2:expr, $a3:expr) => {
        dynasm!($ops
            ; .arch x64
            ; mov rcx, rbx
            ; mov rdx, r12
            ; mov r8d, $a2 as i32
            ; mov r9d, $a3 as i32
        )
    };
}
#[cfg(not(windows))]
macro_rules! call_args {
    ($ops:ident, $a2:expr, $a3:expr) => {
        dynasm!($ops
            ; .arch x64
            ; mov rdi, rbx
            ; mov rsi, r12
            ; mov edx, $a2 as i32
            ; mov ecx, $a3 as i32
        )
    };
}

/// Call `f(cpu, bus, a2, a3)`.
fn emit_call4(ops: &mut VecAssembler<X64Relocation>, f: usize, a2: u32, a3: u32) {
    call_args!(ops, a2, a3);
    dynasm!(ops
        ; .arch x64
        ; mov rax, QWORD f as i64
        ; call rax
    );
}

/// Save callee-saved registers, keep the stack 16-aligned with 32 bytes of
/// shadow space (Windows needs it, SysV does not mind); rbx = cpu, r12 =
/// bus, ebp = cycle budget, r15 = cycles retired so far.
fn emit_prologue(ops: &mut VecAssembler<X64Relocation>) {
    dynasm!(ops
        ; .arch x64
        ; push rbx
        ; push rbp
        ; push r12
        ; push r13
        ; push r14
        ; push r15
        ; sub rsp, 40
    );
    #[cfg(windows)]
    dynasm!(ops
        ; .arch x64
        ; mov rbx, rcx
        ; mov r12, rdx
        ; mov ebp, r8d
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rbx, rdi
        ; mov r12, rsi
        ; mov ebp, edx
    );
    dynasm!(ops ; .arch x64 ; xor r15d, r15d);
}

fn emit_epilogue(ops: &mut VecAssembler<X64Relocation>) {
    dynasm!(ops
        ; .arch x64
        ; ->epilogue:
        ; add rsp, 40
        ; pop r15
        ; pop r14
        ; pop r13
        ; pop r12
        ; pop rbp
        ; pop rbx
        ; ret
    );
}
