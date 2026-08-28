//! VU1 dynamic recompiler (x86-64).
//!
//! Translates runs of instruction pairs out of micro memory into native
//! blocks. The unit of translation is a *block*: straight-line pairs up to
//! and including the first one that diverts control or carries the E bit,
//! plus that pair's delay slot.
//!
//! Micro memory is uploaded wholesale by VIF MPG, so there is no
//! fine-grained invalidation: `Vu1::write_micro` bumps a generation counter
//! only when the bytes actually change, and a change drops the whole cache.
//!
//! The recompiler and the interpreter are interchangeable at any pair
//! boundary, which is what `Vu1::run_interp` is for — a block that meets a
//! shape it cannot express hands the rest of the program back.

mod emit;
mod helpers;
#[cfg(test)]
mod tests;

use dynasm::dynasm;
use dynasmrt::{DynasmApi, DynasmLabelApi, VecAssembler, x64::X64Relocation};
use crate::gif::Gif;
use crate::gs::GsFront;
use crate::jit_arena::Arena;
use crate::vu1::{PAIR_LIMIT, Vu1};
use emit::Ops;
use helpers::NO_BRANCH;

/// A compiled block: `(vu, gs, gif) -> status | pairs retired`.
type Entry = unsafe extern "C" fn(*mut Vu1, *mut GsFront, *mut Gif) -> u32;

/// The program reached its E bit; the dispatcher is done.
const ST_END: u32 = 1 << 31;
/// Hand the rest of the program to the interpreter, from `next_pc`.
const ST_BAIL: u32 = 1 << 30;

/// Code arena; when it fills, every block is dropped and it starts over.
/// Micro memory is 16 KB, and a generation of it compiles to a few hundred
/// KB, so this is roomy — it is committed up front, hence not the EE's 64.
const ARENA_BYTES: usize = 8 << 20;
/// Micro memory holds this many instruction pairs.
const PC_SLOTS: usize = 16 * 1024 / 8;
/// Longest straight-line run translated into one block.
const MAX_PAIRS: u32 = 64;

pub struct Vu1Jit {
    arena: Arena,
    /// pc (pair index) -> entry point; `None` is "not compiled yet".
    lookup: Box<[Option<Entry>]>,
    /// The `micro_gen` this cache was translated against.
    micro_gen: u32,
    pub blocks_compiled: u64,
    pub blocks_run: u64,
    pub flushes: u64,
    pub bails: u64,
}

impl Vu1Jit {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            arena: Arena::new(ARENA_BYTES)?,
            lookup: vec![None; PC_SLOTS].into_boxed_slice(),
            micro_gen: 0,
            blocks_compiled: 0,
            blocks_run: 0,
            flushes: 0,
            bails: 0,
        })
    }

    /// Drop every block and re-key the cache on `gen`. A state load has to
    /// do this unconditionally: micro memory came from the file, so no
    /// generation comparison can be trusted across it.
    pub(crate) fn flush_for(&mut self, generation: u32) {
        self.flush();
        self.micro_gen = generation;
    }

    /// Drop every block (micro memory changed, or the arena filled).
    fn flush(&mut self) {
        self.arena.reset();
        self.lookup.fill(None);
        self.flushes += 1;
    }
}

/// Run a microprogram from `start`, in pairs, until its E bit or the safety
/// limit. Mirrors [`Vu1::run_interp`] exactly, including where it leaves
/// `next_pc` for a later MSCNT, and including where a program that never
/// ends stops.
pub fn run(vu: &mut Vu1, gs: &mut GsFront, gif: &mut Gif, start: u16) {
    let mut jit = vu.jit.take().expect("VU1 dispatcher entered without a recompiler");
    if jit.micro_gen != vu.micro_gen {
        jit.flush();
        jit.micro_gen = vu.micro_gen;
    }
    debug_assert_eq!(vu.data_qw_mask as i32, emit::DATA_QW_MASK, "the recompiler only runs VU1");
    vu.data_ptr = vu.data.as_mut_ptr() as usize;
    let mut budget = PAIR_LIMIT;
    vu.next_pc = start & vu.pc_mask;
    loop {
        let pc = vu.next_pc;
        // A block retires at most `MAX_PAIRS + 1` pairs and cannot stop
        // part way through, so once the budget is that low the interpreter
        // counts the tail out — otherwise a runaway program would stop at a
        // different pair under each engine.
        if budget <= MAX_PAIRS + 1 {
            vu.run_interp(gs, gif, pc, budget);
            break;
        }
        let entry = match jit.lookup[pc as usize] {
            Some(e) => e,
            None => compile(&mut jit, vu, pc),
        };
        // SAFETY: `entry` is a complete block in our own arena, and the
        // three pointees outlive the call — `jit` is out of `vu` for the
        // duration, so nothing the block touches is aliased.
        let status = unsafe { entry(vu, gs, gif) };
        jit.blocks_run += 1;
        budget = budget.saturating_sub(status & !(ST_END | ST_BAIL));
        if status & ST_END != 0 {
            break;
        }
        if status & ST_BAIL != 0 {
            jit.bails += 1;
            let pc = vu.next_pc;
            vu.run_interp(gs, gif, pc, budget);
            break;
        }
    }
    vu.jit = Some(jit);
}

/// Translate the block starting at `pc` and record it.
fn compile(jit: &mut Vu1Jit, vu: &Vu1, pc: u16) -> Entry {
    // Generous upper bound per block; reset rather than fail.
    if jit.arena.remaining() < 64 * 1024 {
        jit.flush();
    }
    let base = jit.arena.next_addr();
    let mut ops = VecAssembler::<X64Relocation>::new(base);
    prologue(&mut ops);

    let mask = vu.pc_mask;
    let mut cur = pc;
    let mut n = 0u32;
    loop {
        let (lower, upper) = fetch(vu, cur);
        let ibit = upper & (1 << 31) != 0;
        let ebit = upper & (1 << 30) != 0;
        let branchy = !ibit && emit::is_branch(lower);

        if !ebit && !branchy {
            emit::pair(&mut ops, cur, upper, lower);
            n += 1;
            cur = (cur + 1) & mask;
            if n >= MAX_PAIRS {
                exit(&mut ops, mask, cur, false, 0);
                break;
            }
            continue;
        }

        // A terminator, so exactly one delay pair follows it. The
        // interpreter carries a branch or an E bit found *in* that delay
        // pair into the pair after it; rather than model that (hardware
        // leaves both undefined), leave the rest of the program to the
        // interpreter. An E-bit terminator is exempt: it returns before the
        // carried state could be acted on.
        let d = (cur + 1) & mask;
        let (dl, du) = fetch(vu, d);
        let d_ibit = du & (1 << 31) != 0;
        if !ebit && (du & (1 << 30) != 0 || (!d_ibit && emit::is_branch(dl))) {
            exit(&mut ops, mask, cur, false, ST_BAIL);
            break;
        }

        emit::pair(&mut ops, cur, upper, lower);
        emit::pair(&mut ops, d, du, dl);
        exit(&mut ops, mask, (d + 1) & mask, branchy, if ebit { ST_END } else { 0 });
        break;
    }

    epilogue(&mut ops);
    let code = ops.finalize().expect("dynasm assembly failed");
    let ptr = jit.arena.place(&code);
    debug_assert_eq!(ptr as usize, base);
    // SAFETY: the arena is executable and now holds a complete block.
    let entry: Entry = unsafe { std::mem::transmute::<*const u8, Entry>(ptr) };
    jit.lookup[pc as usize] = Some(entry);
    jit.blocks_compiled += 1;
    entry
}

fn fetch(vu: &Vu1, pc: u16) -> (u32, u32) {
    let a = (pc & vu.pc_mask) as usize * 8;
    let lower = u32::from_le_bytes(vu.micro[a..a + 4].try_into().unwrap());
    let upper = u32::from_le_bytes(vu.micro[a + 4..a + 8].try_into().unwrap());
    (lower, upper)
}

/// Leave `next_pc` where the interpreter would and return `status` with the
/// retired pair count. With `branchy`, r14d holds the terminator's pending
/// branch: `NO_BRANCH` means it was not taken and `fallthrough` stands.
fn exit(ops: &mut Ops, mask: u16, fallthrough: u16, branchy: bool, status: u32) {
    if branchy {
        dynasm!(ops ; .arch x64
            ; cmp r14d, NO_BRANCH as i32
            ; je >not_taken
            ; and r14d, mask as i32
            ; mov WORD [rbx + emit::next_pc_off()], r14w
            ; jmp >done
            ; not_taken:
        );
    }
    dynasm!(ops ; .arch x64
        ; mov WORD [rbx + emit::next_pc_off()], fallthrough as i16
        ; done:
        ; mov eax, r15d
    );
    if status != 0 {
        dynasm!(ops ; .arch x64 ; or eax, status as i32);
    }
    dynasm!(ops ; .arch x64 ; jmp ->epilogue);
}

fn prologue(ops: &mut Ops) {
    dynasm!(ops ; .arch x64
        ; push rbx ; push rbp ; push r12 ; push r13 ; push r14 ; push r15
        // Six pushes plus this leave rsp 16-aligned at every inner call,
        // and cover Windows' 32-byte shadow space with room for a fifth
        // argument on the stack.
        ; sub rsp, 40
    );
    #[cfg(windows)]
    dynasm!(ops ; .arch x64 ; mov rbx, rcx ; mov r12, rdx ; mov r13, r8);
    #[cfg(not(windows))]
    dynasm!(ops ; .arch x64 ; mov rbx, rdi ; mov r12, rsi ; mov r13, rdx);
    dynasm!(ops ; .arch x64
        ; xor r15d, r15d
        ; mov r14d, NO_BRANCH as i32
        ; mov rbp, QWORD emit::consts_addr()
    );
}

fn epilogue(ops: &mut Ops) {
    dynasm!(ops ; .arch x64
        ; ->epilogue:
        ; add rsp, 40
        ; pop r15 ; pop r14 ; pop r13 ; pop r12 ; pop rbp ; pop rbx
        ; ret
    );
}
