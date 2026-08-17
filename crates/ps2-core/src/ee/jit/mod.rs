//! EE dynamic recompiler (x86-64).
//!
//! Straight-line runs of instructions ("blocks") are translated once and
//! cached by virtual PC. A block is an `extern "C"` function taking the
//! CPU and bus pointers and returning how many instructions it retired;
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

use std::collections::HashMap;

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, VecAssembler, x64::X64Relocation};

use super::Cpu;
use crate::bus::{Bus, RAM_SIZE};
use arena::Arena;

/// Native block entry: (cpu, bus) -> instructions retired.
type Entry = unsafe extern "C" fn(*mut Cpu, *mut Bus) -> u32;

/// Code arena size; when full, every block is dropped and it starts over.
const ARENA_BYTES: usize = 64 << 20;
/// Longest block, in instructions.
const MAX_BLOCK: usize = 64;
/// Direct-mapped lookup entries (indexed by pc >> 2).
const LOOKUP_ENTRIES: usize = 1 << 16;

struct Block {
    pc: u32,
    entry: Entry,
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
    pub blocks_compiled: u64,
    pub blocks_invalidated: u64,
}

impl Jit {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            arena: Arena::new(ARENA_BYTES)?,
            blocks: Vec::new(),
            by_pc: HashMap::new(),
            lookup: vec![0u32; LOOKUP_ENTRIES].into_boxed_slice(),
            page_blocks: vec![Vec::new(); RAM_SIZE >> 12],
            blocks_compiled: 0,
            blocks_invalidated: 0,
        })
    }

    /// Execute from the current PC: one translated block, or one
    /// interpreter step when the CPU is mid-branch or has an interrupt to
    /// take. Returns the EE cycles consumed (>= 1).
    pub fn run(&mut self, cpu: &mut Cpu, bus: &mut Bus) -> u32 {
        if bus.jit_flush_needed {
            bus.jit_flush_needed = false;
            self.flush(bus);
        } else if !bus.dirty_code_pages.is_empty() {
            self.invalidate_dirty(bus);
        }
        if cpu.next_is_delay || cpu.interrupt_pending(bus) {
            cpu.step(bus);
            return 1;
        }
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
        let mut n = unsafe { entry(cpu as *mut Cpu, bus as *mut Bus) };
        // A diverting instruction (branch, exception, idle loop) leaves the
        // interpreter's delay-slot state behind; let it finish the branch.
        while cpu.next_is_delay {
            cpu.step(bus);
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

    fn invalidate_dirty(&mut self, bus: &mut Bus) {
        for page in bus.dirty_code_pages.drain(..) {
            for idx in self.page_blocks[page as usize].drain(..) {
                let b = &mut self.blocks[idx as usize];
                if b.valid {
                    b.valid = false;
                    self.by_pc.remove(&b.pc);
                    self.blocks_invalidated += 1;
                }
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
        bus.dirty_code_pages.clear();
    }

    fn compile(&mut self, pc: u32, bus: &mut Bus) -> Entry {
        // Generous upper bound per block; reset the arena rather than fail.
        if self.arena.remaining() < 64 * 1024 {
            self.flush(bus);
        }
        let mut ops = VecAssembler::<X64Relocation>::new(self.arena.next_addr());
        emit_prologue(&mut ops);

        let mut exits: Vec<dynasmrt::DynamicLabel> = Vec::with_capacity(MAX_BLOCK);
        let mut addr = pc;
        let mut count = 0u32;
        loop {
            let instr = bus.fetch32(addr);
            // Every instruction goes through the interpreter for now; the
            // helper reports whether control was diverted (branch taken,
            // likely-branch skip, exception, idle loop) and the block exits.
            let exit = ops.new_dynamic_label();
            emit_call4(&mut ops, interp_one as *const () as usize, addr, instr);
            dynasm!(ops
                ; .arch x64
                ; test eax, eax
                ; jnz =>exit
            );
            exits.push(exit);
            count += 1;
            addr = addr.wrapping_add(4);
            if is_control_flow(instr) || count as usize >= MAX_BLOCK || addr & 0xFFF == 0 {
                break;
            }
        }
        // Fell off the end: all `count` instructions retired.
        dynasm!(ops
            ; .arch x64
            ; mov eax, count as i32
            ; jmp ->epilogue
        );
        // Diverted after instruction i: i + 1 retired.
        for (i, exit) in exits.iter().enumerate() {
            dynasm!(ops
                ; .arch x64
                ; =>*exit
                ; mov eax, (i + 1) as i32
                ; jmp ->epilogue
            );
        }
        emit_epilogue(&mut ops);

        let code = ops.finalize().expect("dynasm assembly failed");
        let ptr = self.arena.place(&code);
        // SAFETY: the bytes at ptr are the function assembled above.
        let entry: Entry = unsafe { std::mem::transmute(ptr) };

        // Track the RAM pages the block reads its code from, for
        // invalidation on write. ROM/other blocks are never invalidated.
        let mut pages = [None, None];
        for (i, a) in [pc, addr.wrapping_sub(4)].iter().enumerate() {
            if let Some(p) = bus.ram_page_of(*a) {
                pages[i] = Some(p);
            }
        }
        if pages[1] == pages[0] {
            pages[1] = None;
        }
        let idx = self.blocks.len() as u32;
        for p in pages.iter().flatten() {
            self.page_blocks[*p as usize].push(idx);
            bus.code_pages[*p as usize] = true;
        }
        self.blocks.push(Block { pc, entry, valid: true });
        self.by_pc.insert(pc, idx);
        self.lookup[((pc >> 2) as usize) & (LOOKUP_ENTRIES - 1)] = idx + 1;
        self.blocks_compiled += 1;
        entry
    }
}

/// Instructions after which a block must end: anything that may change PC
/// other than by falling through.
fn is_control_flow(instr: u32) -> bool {
    match instr >> 26 {
        // SPECIAL: jr, jalr, syscall, break, traps.
        0x00 => matches!(instr & 0x3F, 0x08 | 0x09 | 0x0C | 0x0D | 0x30..=0x36),
        // REGIMM branches (0x00-0x03, 0x10-0x13); j/jal, beq..bgtz, likely.
        0x01 => matches!((instr >> 16) & 0x1F, 0x00..=0x03 | 0x10..=0x13),
        0x02..=0x07 | 0x14..=0x17 => true,
        // COP0 (eret, tlb*), COP1 (bc1), COP2 (bc2) — treat every COP op
        // as a possible diversion.
        0x10 | 0x11 | 0x12 => true,
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
/// shadow space (Windows needs it, SysV does not mind), rbx = cpu, r12 = bus.
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
    );
    #[cfg(not(windows))]
    dynasm!(ops
        ; .arch x64
        ; mov rbx, rdi
        ; mov r12, rsi
    );
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
