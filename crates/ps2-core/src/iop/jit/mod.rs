//! IOP dynamic recompiler (x86-64).
//!
//! Straight-line runs of instructions ("blocks") are translated once and
//! cached by virtual PC. A block is an `extern "C"` function taking the CPU
//! and bus pointers and a budget in instructions, and returning how many it
//! retired; [`Jit::run`] executes one chain of directly linked blocks and
//! returns that count, so the caller can advance the rest of the machine by
//! [`EE_PER_IOP`] cycles for each of them.
//!
//! The IOP retires one instruction per IOP slot and is otherwise checked
//! once per slot for interrupts and scheduled events. A chain may only
//! stand in for that loop while nothing it does can be observed in
//! between, so a block ends the moment an access leaves IOP RAM, and on
//! `mtc0`, `rfe`, `syscall` and `break`. Everything else it can run —
//! arithmetic and RAM traffic — is invisible until the chain returns.
//!
//! Blocks are short here: the mean basic block is under five instructions,
//! so the exits link directly to one another rather than returning to a
//! dispatcher between them.
//!
//! Self-modifying code and, far more often, freshly loaded modules are
//! caught by the bus: it flags IOP RAM pages that hold translated code and
//! reports writes to them (see `Bus::iop_ram_written`).

mod emit;
mod helpers;

use std::collections::HashMap;

use dynasm::dynasm;
use dynasmrt::{DynamicLabel, DynasmApi, DynasmLabelApi, VecAssembler, x64::X64Relocation};

use super::Cpu;
use crate::bus::{Bus, IOP_RAM_SIZE};
use crate::jit_arena::Arena;
use emit::Ops;
use helpers as h;

/// Native block entry: (cpu, bus, instruction budget) -> instructions
/// retired by the chain of linked blocks that ran.
type Entry = unsafe extern "C" fn(*mut Cpu, *mut Bus, u32) -> u32;

/// Code arena size; when full, every block is dropped and it starts over.
/// The whole hot footprint of an IOP is a couple of thousand blocks.
const ARENA_BYTES: usize = 8 << 20;
/// Longest block, in instructions. Blocks end at the first branch anyway,
/// and a shorter cap keeps the entry's all-or-nothing budget test from
/// wasting a large tail when the budget runs low.
const MAX_BLOCK: u32 = 24;
/// Direct-mapped lookup entries (indexed by pc >> 2). Each holds its own
/// pc and entry point, so a hit is one cache line: the dispatcher runs
/// once per chain, and a chain is only ~16 instructions, so a second
/// dependent load into the (megabytes-long) block table was over half of
/// what the recompiler spent outside translated code.
const LOOKUP_ENTRIES: usize = 1 << 14;


/// One direct-mapped lookup slot: a null `entry` is empty.
#[derive(Clone, Copy)]
struct Slot {
    pc: u32,
    entry: usize,
}

/// A `jmp rel32` at a block exit: where its displacement field is, and
/// where the instruction after it starts (displacements are relative to
/// that). Zero sends control into the slow path that follows.
#[derive(Clone, Copy)]
struct Link {
    field: usize,
    after: usize,
}

impl Link {
    /// Point the jump at `target`, or at the slow path when it is `None`.
    ///
    /// # Safety
    /// `field` must address the displacement of a complete `jmp rel32` in
    /// the arena, and no thread may be executing that jump.
    unsafe fn patch(self, target: Option<u64>) {
        let rel = match target {
            Some(t) => (t as i64 - self.after as i64) as i32,
            None => 0,
        };
        // SAFETY: the arena is writable and the field is 4 bytes inside it.
        unsafe { std::ptr::write_unaligned(self.field as *mut i32, rel) };
    }
}

struct Block {
    pc: u32,
    entry: Entry,
    /// Address of the body (after the prologue): where linked jumps land.
    body: u64,
    /// IOP RAM byte range the block was translated from.
    phys: Option<(u32, u32)>,
    /// Jumps currently patched to land in this block's body.
    incoming: Vec<Link>,
    valid: bool,
}

pub struct Jit {
    arena: Arena,
    blocks: Vec<Block>,
    by_pc: HashMap<u32, u32>,
    /// pc >> 2 -> the block there, or an empty slot.
    lookup: Box<[Slot]>,
    /// IOP RAM page -> blocks translated from it.
    page_blocks: Vec<Vec<u32>>,
    /// Jumps waiting for a block at this pc to be compiled.
    pending_links: HashMap<u32, Vec<Link>>,
    pub blocks_compiled: u64,
    pub blocks_invalidated: u64,
    pub chains_run: u64,
    pub interp_steps: u64,
    /// Why the dispatcher was reached, counted for the bring-up loop: an
    /// access outside IOP RAM, the idle jump, a branch whose delay slot the
    /// interpreter owes, a load whose delay crosses the boundary, an
    /// interrupt to take, and a first block too long for the budget left.
    exits: [u64; 6],
}

impl Jit {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            arena: Arena::new(ARENA_BYTES)?,
            blocks: Vec::new(),
            by_pc: HashMap::new(),
            lookup: vec![Slot { pc: 0, entry: 0 }; LOOKUP_ENTRIES].into_boxed_slice(),
            page_blocks: vec![Vec::new(); IOP_RAM_SIZE >> 12],
            pending_links: HashMap::new(),
            blocks_compiled: 0,
            blocks_invalidated: 0,
            chains_run: 0,
            interp_steps: 0,
            exits: [0; 6],
        })
    }

    /// Execute from the current PC: a chain of linked blocks retiring at
    /// most `budget` instructions, or one interpreter step when the CPU is
    /// mid-branch, holding a load whose delay slot crosses the boundary, or
    /// has an interrupt to take. Returns the instructions retired (>= 1),
    /// never more than `budget`.
    pub fn run(&mut self, cpu: &mut Cpu, bus: &mut Bus, budget: u32) -> u32 {
        if bus.iop_jit_flush_needed {
            bus.iop_jit_flush_needed = false;
            self.flush(bus);
        } else if !bus.iop_dirty_code_writes.is_empty() || !bus.iop_dirty_code_pages.is_empty() {
            self.invalidate_dirty(bus);
        }
        let budget = budget.max(1);
        if cpu.next_is_delay || cpu.pending_load.is_some() || cpu.interrupt_pending(bus) {
            self.exits[if cpu.next_is_delay {
                2
            } else if cpu.pending_load.is_some() {
                3
            } else {
                4
            }] += 1;
            cpu.step(bus);
            self.interp_steps += 1;
            return 1;
        }
        let pc = cpu.pc;
        let entry = match self.find(pc) {
            Some(e) => e,
            None => self.compile(pc, bus),
        };
        bus.iop_chain_start = bus.now;
        // SAFETY: `entry` is a complete block in our arena; the pointers are
        // exclusively ours for the call and the block only touches the CPU
        // and bus through the helpers.
        let n = unsafe { entry(cpu as *mut Cpu, bus as *mut Bus, budget) };
        if n == 0 {
            // The first block is longer than the budget left. Stepping is
            // both simpler and rarer than splitting the block would be.
            self.exits[5] += 1;
            bus.now = bus.iop_chain_start;
            cpu.step(bus);
            self.interp_steps += 1;
            return 1;
        }
        self.chains_run += 1;
        if cpu.idle {
            self.exits[1] += 1;
        }
        self.exits[0] = bus.iop_mmio_exits;
        n
    }

    /// The exit reasons, named, newest count first in no particular order.
    pub fn exit_counts(&self) -> [(&'static str, u64); 6] {
        let names = ["mmio", "idle", "delay_slot", "pending_load", "interrupt", "block_too_long"];
        std::array::from_fn(|i| (names[i], self.exits[i]))
    }

    #[inline]
    fn find(&self, pc: u32) -> Option<Entry> {
        let s = self.lookup[slot_of(pc)];
        if s.entry != 0 && s.pc == pc {
            // SAFETY: a non-null slot holds the entry of a live block; it
            // is cleared when that block is invalidated or flushed.
            return Some(unsafe { std::mem::transmute::<usize, Entry>(s.entry) });
        }
        let &idx = self.by_pc.get(&pc)?;
        let b = &self.blocks[idx as usize];
        if !b.valid {
            return None;
        }
        Some(b.entry)
    }

    /// Forget a block's lookup slot, if it still holds that block.
    #[inline]
    fn drop_slot(&mut self, pc: u32) {
        let s = &mut self.lookup[slot_of(pc)];
        if s.pc == pc {
            s.entry = 0;
        }
    }

    /// Drop the blocks whose code was written: one instruction word for a
    /// store, a whole page for a transfer that landed on one.
    fn invalidate_dirty(&mut self, bus: &mut Bus) {
        let mut dropped: Vec<u32> = Vec::new();
        for p in std::mem::take(&mut bus.iop_dirty_code_pages) {
            let page = p as usize;
            for &idx in &self.page_blocks[page] {
                let b = &mut self.blocks[idx as usize];
                if !b.valid {
                    continue;
                }
                b.valid = false;
                self.by_pc.remove(&b.pc);
                self.blocks_invalidated += 1;
                let (pc, incoming) = (b.pc, std::mem::take(&mut b.incoming));
                for &l in &incoming {
                    // SAFETY: the arena is ours and no chain is running.
                    unsafe { l.patch(None) };
                }
                self.pending_links.entry(pc).or_default().extend(incoming);
                dropped.push(pc);
            }
            self.page_blocks[page].clear();
            bus.iop_code_pages[page] = false;
        }
        for a in bus.iop_dirty_code_writes.drain(..) {
            let page = (a >> 12) as usize;
            let mut any_left = false;
            for &idx in &self.page_blocks[page] {
                let b = &mut self.blocks[idx as usize];
                if !b.valid {
                    continue;
                }
                match b.phys {
                    Some((s, e)) if s < a + 4 && a < e => {
                        b.valid = false;
                        self.by_pc.remove(&b.pc);
                        self.blocks_invalidated += 1;
                        // Anything linked here goes back to its slow path
                        // and waits for a recompile at this pc.
                        let (pc, incoming) = (b.pc, std::mem::take(&mut b.incoming));
                        for &l in &incoming {
                            // SAFETY: the arena is ours and no chain is running.
                            unsafe { l.patch(None) };
                        }
                        self.pending_links.entry(pc).or_default().extend(incoming);
                        dropped.push(pc);
                    }
                    _ => any_left = true,
                }
            }
            if !any_left {
                self.page_blocks[page].clear();
                bus.iop_code_pages[page] = false;
            }
        }
        for pc in dropped {
            self.drop_slot(pc);
        }
    }

    /// Drop every block (arena full, a module loaded over one, ...).
    pub fn flush(&mut self, bus: &mut Bus) {
        self.arena.reset();
        self.blocks.clear();
        self.by_pc.clear();
        self.lookup.fill(Slot { pc: 0, entry: 0 });
        for v in &mut self.page_blocks {
            v.clear();
        }
        bus.iop_code_pages.fill(false);
        bus.iop_dirty_code_writes.clear();
        bus.iop_dirty_code_pages.clear();
        self.pending_links.clear();
    }

    fn compile(&mut self, pc: u32, bus: &mut Bus) -> Entry {
        // Generous upper bound per block; reset the arena rather than fail.
        if self.arena.remaining() < 16 * 1024 {
            self.flush(bus);
        }
        // The entry test is all-or-nothing, so the block's own length has to
        // be known before its first instruction is emitted.
        let len = scan(bus, pc);
        let base = self.arena.next_addr();
        let mut ops = Ops::new(base);
        emit_prologue(&mut ops);
        // Body: linked jumps land here, past the budget test only when the
        // whole block still fits in what the chain may retire.
        let body_off = ops.offset().0;
        let budget_exit = ops.new_dynamic_label();
        dynasm!(ops
            ; .arch x64
            ; lea eax, [r15 + len as i32]
            ; cmp eax, ebp
            ; ja =>budget_exit
        );
        let mut exits = emit::Exits::default();
        let mut st = emit::State::new();
        // (label, instructions retired) for interpreter calls that diverted.
        let mut interp_exits: Vec<(DynamicLabel, u32)> = Vec::new();
        let mut addr = pc;
        let mut count = 0u32;
        loop {
            let instr = bus.iop_fetch32(addr);
            if is_branch(instr) {
                let ds_addr = addr.wrapping_add(4);
                let ds = bus.iop_fetch32(ds_addr);
                if is_branch(ds) {
                    // A branch in a delay slot is undefined on MIPS. Let
                    // the interpreter register this one and finish the pair
                    // through the dispatcher's `next_is_delay` path.
                    emit::spill_pend(&mut ops, st.pend);
                    st.pend = None;
                    emit::emit_now(&mut ops, st.local);
                    emit::call_interp(&mut ops, h::interp_one as *const () as usize, addr, instr);
                    count += 1;
                    exits.to_stored(&mut ops, count);
                    break;
                }
                let emit::Emitted::Branch(kind) = emit::emit(&mut ops, &mut st, addr, instr, false)
                else {
                    unreachable!("is_branch disagreed with the translator at {addr:#010x}")
                };
                st.local += 1;
                count += 1;
                // The kernel idle thread stops here: the interpreter sets
                // the flag from the jump itself and its caller then leaves
                // the IOP alone, so the delay slot does not run until an
                // interrupt wakes it. Retiring it here would put the IOP
                // one slot ahead of the interpreter for the whole spin.
                if is_idle_loop(addr, instr, ds) {
                    let emit::BranchKind::Jump(target) = kind else {
                        unreachable!("the idle loop is a `j`")
                    };
                    exits.to_idle(&mut ops, ds_addr, target, count, addr);
                    break;
                }
                let interpreted_slot =
                    matches!(emit::emit(&mut ops, &mut st, ds_addr, ds, true), emit::Emitted::Interp);
                if interpreted_slot {
                    emit::spill_pend(&mut ops, st.pend);
                    st.pend = None;
                    emit::emit_now(&mut ops, st.local);
                    emit::call_interp(&mut ops, h::interp_delay_one as *const () as usize, ds_addr, ds);
                }
                st.local += 1;
                count += 1;
                if interpreted_slot {
                    // An exception raised in the delay slot has set pc to
                    // its vector; the branch must not write over it.
                    let l = ops.new_dynamic_label();
                    dynasm!(ops ; .arch x64 ; test eax, eax ; jnz =>l);
                    interp_exits.push((l, count));
                }
                let fallthrough = ds_addr.wrapping_add(4);
                // Two reasons not to link. A load left in flight crosses
                // the block boundary, so the dispatcher has to step once
                // and resolve it; and a delay slot that ends a block ends
                // it wherever it sits, so the interrupt an `mtc0` or `rfe`
                // unmasks is tested before the next instruction runs.
                let link = st.pend.is_none() && !ends_block(ds);
                if link {
                    let end = |ops: &mut Ops, exits: &mut emit::Exits, link| {
                        emit::emit_branch_end(ops, exits, kind, fallthrough, count, ds_addr, link);
                    };
                    end(&mut ops, &mut exits, true);
                    if let Some(l) = st.no_link {
                        dynasm!(ops ; .arch x64 ; =>l);
                        end(&mut ops, &mut exits, false);
                    }
                } else {
                    if let Some(l) = st.no_link {
                        dynasm!(ops ; .arch x64 ; =>l);
                    }
                    emit::spill_pend(&mut ops, st.pend);
                    emit::emit_branch_end(
                        &mut ops, &mut exits, kind, fallthrough, count, ds_addr, false,
                    );
                }
                break;
            }
            if let emit::Emitted::Interp = emit::emit(&mut ops, &mut st, addr, instr, false) {
                emit::spill_pend(&mut ops, st.pend);
                st.pend = None;
                emit::emit_now(&mut ops, st.local);
                emit::call_interp(&mut ops, h::interp_one as *const () as usize, addr, instr);
                if ends_block(instr) {
                    count += 1;
                    exits.to_stored(&mut ops, count);
                    break;
                }
                let l = ops.new_dynamic_label();
                dynasm!(ops ; .arch x64 ; test eax, eax ; jnz =>l);
                interp_exits.push((l, count + 1));
            }
            st.local += 1;
            count += 1;
            addr = addr.wrapping_add(4);
            if count >= MAX_BLOCK || addr & 0xFFF == 0 {
                let last = addr.wrapping_sub(4);
                if st.pend.is_none() {
                    exits.to(&mut ops, addr, count, last);
                } else {
                    emit::spill_pend(&mut ops, st.pend);
                    exits.to_unlinked(&mut ops, addr, count, last);
                }
                break;
            }
        }
        debug_assert_eq!(count, len, "scan and emit disagreed at {pc:#010x}");
        for (label, retired) in interp_exits {
            dynasm!(ops ; .arch x64 ; =>label);
            exits.to_stored(&mut ops, retired);
        }
        for e in std::mem::take(&mut st.mem_exits) {
            dynasm!(ops ; .arch x64 ; =>e.label);
            emit::spill_pend(&mut ops, e.pend);
            exits.to_unlinked(&mut ops, e.pc, e.retired, e.last);
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
        let entry: Entry = unsafe { std::mem::transmute::<*const u8, Entry>(ptr) };
        let body = (base + body_off) as u64;

        // Track the IOP RAM range the block reads its code from, so a
        // module load or a patch over it drops the block. BIOS blocks have
        // no range and are never invalidated.
        let last = pc.wrapping_add((count.max(1) - 1) * 4);
        let idx = self.blocks.len() as u32;
        let phys = match (iop_phys_of(pc), iop_phys_of(last)) {
            (Some(s), Some(e)) => Some((s, e + 4)),
            _ => None,
        };
        if let Some((s, e)) = phys {
            for p in (s >> 12)..=((e - 1) >> 12) {
                self.page_blocks[p as usize].push(idx);
                bus.iop_code_pages[p as usize] = true;
            }
        }
        self.blocks.push(Block { pc, entry, body, phys, incoming: Vec::new(), valid: true });
        self.by_pc.insert(pc, idx);
        self.lookup[slot_of(pc)] = Slot { pc, entry: entry as usize };
        self.blocks_compiled += 1;

        // Wire this block's exits to compiled targets (or park them), and
        // point exits parked on this pc at the new body. A parked jump has
        // a zero displacement and falls into the slow path behind it, so
        // nothing has to be written for that case.
        for (field, after, target) in links {
            let link = Link { field: base + field, after: base + after };
            match self.by_pc.get(&target).map(|&i| i as usize) {
                Some(t) if self.blocks[t].valid => {
                    // SAFETY: the arena is ours and no chain is running.
                    unsafe { link.patch(Some(self.blocks[t].body)) };
                    self.blocks[t].incoming.push(link);
                }
                _ => self.pending_links.entry(target).or_default().push(link),
            }
        }
        if let Some(waiting) = self.pending_links.remove(&pc) {
            for link in waiting {
                // SAFETY: as above.
                unsafe { link.patch(Some(body)) };
                self.blocks[idx as usize].incoming.push(link);
            }
        }
        entry
    }
}

/// Direct-mapped lookup slot for a pc.
#[inline]
fn slot_of(pc: u32) -> usize {
    ((pc >> 2) as usize) & (LOOKUP_ENTRIES - 1)
}

/// IOP RAM offset a virtual address reads its code from, if RAM at all.
fn iop_phys_of(vaddr: u32) -> Option<u32> {
    ((vaddr & 0x1FFF_FFFF) < 0x0080_0000).then_some(vaddr & 0x1F_FFFF)
}

/// How many instructions the block at `pc` covers. Mirrors the loop in
/// [`Jit::compile`] exactly; a `debug_assert` there keeps the two honest.
fn scan(bus: &mut Bus, pc: u32) -> u32 {
    let mut addr = pc;
    let mut count = 0u32;
    loop {
        let instr = bus.iop_fetch32(addr);
        if is_branch(instr) {
            let ds = bus.iop_fetch32(addr.wrapping_add(4));
            count += if is_branch(ds) || is_idle_loop(addr, instr, ds) { 1 } else { 2 };
            break;
        }
        count += 1;
        if ends_block(instr) {
            break;
        }
        addr = addr.wrapping_add(4);
        if count >= MAX_BLOCK || addr & 0xFFF == 0 {
            break;
        }
    }
    count
}

/// Instructions that set the PC themselves, delay slot and all.
fn is_branch(instr: u32) -> bool {
    match instr >> 26 {
        // SPECIAL: jr, jalr.
        0x00 => matches!(instr & 0x3F, 0x08 | 0x09),
        // REGIMM, j/jal, beq..bgtz.
        0x01..=0x07 => true,
        _ => false,
    }
}

/// Instructions the translator hands to the interpreter and then stops at:
/// they divert control (`syscall`, `break`) or can unmask an interrupt the
/// next instruction would take (`mtc0`, `rfe`), which a block cannot test
/// for. Multiply and divide also go to the interpreter, but the block runs
/// on: they touch nothing outside the CPU.
fn ends_block(instr: u32) -> bool {
    match instr >> 26 {
        0x00 => !matches!(instr & 0x3F,
            0x00 | 0x02..=0x04 | 0x06 | 0x07 | 0x10..=0x13 | 0x18..=0x1B
            | 0x20..=0x27 | 0x2A | 0x2B),
        // mfc0 is translated in place; every other COP0 form ends it.
        0x10 => (instr >> 21) & 31 != 0,
        0x08..=0x0F | 0x20..=0x26 | 0x28..=0x2B | 0x2E => false,
        _ => true,
    }
}

/// The kernel idle thread: `j .` with a `nop` in its delay slot, as the
/// interpreter's own check sees it. Only an interrupt can move it, so
/// [`crate::Ps2System`] skips IOP steps until one is pending.
fn is_idle_loop(addr: u32, instr: u32, delay_slot: u32) -> bool {
    instr >> 26 == 0x02
        && delay_slot == 0
        && (addr.wrapping_add(4) & 0xF000_0000) | ((instr & 0x03FF_FFFF) << 2) == addr
}

// --- x86-64 emission ------------------------------------------------------

/// Save callee-saved registers, keep the stack 16-aligned with 32 bytes of
/// shadow space (Windows needs it, SysV does not mind); rbx = cpu, r12 =
/// bus, r15 = instructions retired so far, ebp = the chain's budget, which
/// every block entry tests its own length against. Nothing can cut that
/// budget once the chain is running — everything that could schedule an
/// event ends the block instead — so it lives in a register rather than
/// being re-read from the bus, which is worth 7% on its own.
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

#[cfg(test)]
mod tests {
    use super::Jit;
    use crate::Ps2System;
    use crate::bus::BIOS_SIZE;

    /// A machine with `code` at the IOP reset vector.
    fn system(code: &[u32]) -> Ps2System {
        let mut bios = vec![0u8; BIOS_SIZE];
        for (i, w) in code.iter().enumerate() {
            bios[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        Ps2System::new_with(bios, false).unwrap()
    }

    /// Run `n` IOP instructions through the recompiler and the same `n`
    /// through the interpreter, and require identical state. Everything the
    /// translator can get wrong — the delayed load, a branch's delay slot,
    /// cache isolation, the merge an `lwl`/`lwr` pair does — shows up as a
    /// register or a byte of RAM that does not match.
    fn same(code: &[u32], n: u32) {
        let (mut a, mut b) = (system(code), system(code));
        let mut jit = Jit::new().unwrap();
        let mut left = n;
        while left > 0 {
            let ran = jit.run(&mut a.iop, &mut a.bus, left);
            assert!(ran > 0 && ran <= left, "chain retired {ran} of {left}");
            left -= ran;
        }
        for _ in 0..n {
            b.iop.step(&mut b.bus);
        }
        assert_eq!(a.iop.pc, b.iop.pc, "pc");
        assert_eq!(a.iop.next_pc, b.iop.next_pc, "next_pc");
        assert_eq!(a.iop.gpr, b.iop.gpr, "gpr");
        assert_eq!((a.iop.hi, a.iop.lo), (b.iop.hi, b.iop.lo), "hi/lo");
        assert_eq!(a.iop.cop0, b.iop.cop0, "cop0");
        assert_eq!(a.iop.idle, b.iop.idle, "idle");
        assert_eq!(a.bus.iop_ram, b.bus.iop_ram, "IOP RAM");
    }

    #[test]
    fn arithmetic_and_the_delayed_load() {
        same(
            &[
                0x3C01_0000, // lui   $at, 0
                0x2408_0055, // addiu $t0, $0, 0x55
                0xAC28_0000, // sw    $t0, 0($at)
                0x8C29_0000, // lw    $t1, 0($at)
                0x0009_5021, // addu  $t2, $0, $t1   (delay slot: old $t1)
                0x0009_5821, // addu  $t3, $0, $t1   (new value)
                0x012A_5824, // and   $t3, $t1, $t2
                0x3C0C_1234, // lui   $t4, 0x1234
                0x358C_5678, // ori   $t4, 0x5678
                0x000C_6902, // srl   $t5, $t4, 4
                0x018D_7023, // subu  $t6, $t4, $t5
                0x01CD_782A, // slt   $t7, $t6, $t5
            ],
            12,
        );
    }

    #[test]
    fn a_write_in_the_delay_slot_beats_the_pending_load() {
        same(
            &[
                0x3C01_0000, // lui   $at, 0
                0x2408_0055, // addiu $t0, $0, 0x55
                0xAC28_0000, // sw    $t0, 0($at)
                0x8C29_0000, // lw    $t1, 0($at)
                0x2409_0077, // addiu $t1, $0, 0x77  (wins over the load)
                0x0129_5021, // addu  $t2, $t1, $t1
            ],
            6,
        );
    }

    #[test]
    fn an_unaligned_pair_merges_through_the_in_flight_value() {
        same(
            &[
                0x3C01_0000, // lui $at, 0
                0x3C09_4433, // lui $t1, 0x4433
                0x3529_2211, // ori $t1, 0x2211
                0xAC29_0000, // sw  $t1, 0($at)
                0x3C0A_8877, // lui $t2, 0x8877
                0x354A_6655, // ori $t2, 0x6655
                0xAC2A_0004, // sw  $t2, 4($at)
                0x8828_0004, // lwl $t0, 4($at)
                0x9828_0001, // lwr $t0, 1($at)
                0x0000_0000, // nop
                0xA828_0008, // swl $t0, 8($at)
                0xB828_000D, // swr $t0, 13($at)
            ],
            12,
        );
    }

    #[test]
    fn branches_run_their_delay_slots() {
        same(
            &[
                0x2408_0003, // addiu $t0, $0, 3
                0x2409_0000, // addiu $t1, $0, 0
                // loop:
                0x2129_0001, // addi  $t1, $t1, 1  (0x08)
                0x2508_FFFF, // addiu $t0, $t0, -1
                0x1500_FFFD, // bne   $t0, $0, loop
                0x212A_0001, // addi  $t2, $t1, 1  (delay slot, always)
                0x0C00_0009, // jal   0x24
                0x0000_0000, // nop
                0x2408_00FF, // addiu $t0, $0, 0xFF (0x24)
                0x03E0_0008, // jr    $ra
                0x2409_00EE, // addiu $t1, $0, 0xEE (delay slot)
            ],
            24,
        );
    }

    #[test]
    fn a_load_in_a_delay_slot_crosses_the_block_boundary() {
        same(
            &[
                0x3C01_0000, // lui   $at, 0
                0x2408_0042, // addiu $t0, $0, 0x42
                0xAC28_0010, // sw    $t0, 0x10($at)
                0x0800_0005, // j     0x14
                0x8C29_0010, // lw    $t1, 0x10($at)  (delay slot)
                0x0129_5021, // addu  $t2, $t1, $t1   (0x14: sees the old $t1)
                0x0129_5821, // addu  $t3, $t1, $t1   (sees the loaded value)
            ],
            7,
        );
    }

    #[test]
    fn cache_isolation_swallows_a_store() {
        same(
            &[
                0x3C08_0001, // lui   $t0, 1        (Status.Isc)
                0x4088_6000, // mtc0  $t0, $12
                0x2409_0033, // addiu $t1, $0, 0x33
                0xAC09_0004, // sw    $t1, 4($0)    (swallowed)
                0x4080_6000, // mtc0  $0, $12
                0x8C0A_0004, // lw    $t2, 4($0)
                0x0000_0000, // nop
                0x4009_6000, // mfc0  $t1, $12
                0x0000_0000, // nop
            ],
            9,
        );
    }

    #[test]
    fn the_idle_jump_retires_alone() {
        // `j .` with a nop slot is the kernel idle thread: the interpreter
        // sets the flag from the jump and never runs the slot, so the
        // recompiler must not retire it either.
        let code = [
            0x2408_0001, // addiu $t0, $0, 1
            0x0BF0_0001, // j     0xBFC00004  (itself)
            0x0000_0000, // nop
        ];
        same(&code, 2);
        let mut a = system(&code);
        let mut jit = Jit::new().unwrap();
        let mut left = 2;
        while left > 0 {
            left -= jit.run(&mut a.iop, &mut a.bus, left);
        }
        assert!(a.iop.idle, "the idle flag was not set");
        assert_eq!(a.iop.pc, 0xBFC0_0008, "the delay slot must not have run");
    }

    /// An encoding neither core decodes ends the block and takes the
    /// Reserved Instruction exception, landing both cores on the vector.
    #[test]
    fn an_unassigned_encoding_takes_the_reserved_instruction_exception() {
        same(
            &[
                0x2408_0007, // addiu $t0, $0, 7
                0x0000_40EC, // unassigned SPECIAL
            ],
            2,
        );
    }

    #[test]
    fn multiply_and_divide_fall_back_without_ending_the_block() {
        same(
            &[
                0x2408_0007, // addiu $t0, $0, 7
                0x2409_0003, // addiu $t1, $0, 3
                0x0109_0018, // mult  $t0, $t1
                0x0000_5012, // mflo  $t2
                0x0109_001A, // div   $t0, $t1
                0x0000_5810, // mfhi  $t3
                0x0000_6012, // mflo  $t4
                0x014C_6821, // addu  $t5, $t2, $t4
            ],
            8,
        );
    }
}
