//! VU1 dynamic recompiler (x86-64).
//!
//! Translates runs of instruction pairs out of micro memory into native
//! blocks. The unit of translation is a *block*: straight-line pairs up to
//! and including the first one that diverts control or carries the E bit,
//! plus that pair's delay slot.
//!
//! Micro memory is uploaded wholesale by VIF MPG, so there is no
//! fine-grained invalidation: `Vu1::write_micro` bumps a generation counter
//! only when the bytes actually change. A game that alternates a handful of
//! programs would then retranslate all of them on every switch, so the
//! cache is kept per *image* — keyed on the contents of micro memory rather
//! than on the generation — and a switch back to one still held costs
//! nothing.
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

/// Code arena per cached image; when one fills, that image's blocks are
/// dropped and it starts over. Micro memory is 16 KB and a generation of it
/// compiles to a few hundred KB, so this is roomy — it is committed up
/// front, hence not the EE's 64.
const ARENA_BYTES: usize = 2 << 20;
/// Images kept translated at once. Ace Combat 5's mission alternates seven.
const IMAGES: usize = 8;
/// Micro memory holds this many instruction pairs.
const PC_SLOTS: usize = 16 * 1024 / 8;
/// Longest straight-line run translated into one block.
const MAX_PAIRS: u32 = 64;

/// The translation of one micro memory image.
struct Image {
    arena: Arena,
    /// pc (pair index) -> entry point; `None` is "not compiled yet".
    lookup: Box<[Option<Entry>]>,
    /// Hash of the micro memory this was translated from.
    hash: u64,
    /// When this image was last selected, for choosing what to evict.
    used: u64,
}

impl Image {
    fn new(hash: u64, used: u64) -> std::io::Result<Self> {
        Ok(Self {
            arena: Arena::new(ARENA_BYTES)?,
            lookup: vec![None; PC_SLOTS].into_boxed_slice(),
            hash,
            used,
        })
    }

    fn reset(&mut self) {
        self.arena.reset();
        self.lookup.fill(None);
    }
}

pub struct Vu1Jit {
    images: Vec<Image>,
    /// Index into `images` of the one micro memory currently holds.
    cur: usize,
    /// The `micro_gen` `cur` was chosen for; a change re-hashes.
    micro_gen: u32,
    clock: u64,
    pub blocks_compiled: u64,
    pub blocks_run: u64,
    pub flushes: u64,
    pub bails: u64,
}

impl Vu1Jit {
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            images: vec![Image::new(0, 0)?],
            cur: 0,
            micro_gen: 0,
            clock: 0,
            blocks_compiled: 0,
            blocks_run: 0,
            flushes: 0,
            bails: 0,
        })
    }

    /// Drop every image and re-key on `gen`. A state load has to do this
    /// unconditionally: micro memory came from the file, so neither the
    /// generation nor a remembered hash can be trusted across it.
    pub(crate) fn flush_for(&mut self, generation: u32) {
        self.images.truncate(1);
        self.images[0].reset();
        self.images[0].hash = 0;
        self.cur = 0;
        self.micro_gen = generation;
        self.flushes += 1;
    }

    /// Point `cur` at the translation of the image micro memory now holds,
    /// translating into a fresh slot when it is one we have not seen.
    fn select(&mut self, micro: &[u8]) -> std::io::Result<()> {
        let hash = image_hash(micro);
        self.clock += 1;
        if let Some(i) = self.images.iter().position(|im| im.hash == hash) {
            self.cur = i;
            self.images[i].used = self.clock;
            return Ok(());
        }
        self.flushes += 1;
        if self.images.len() < IMAGES {
            self.images.push(Image::new(hash, self.clock)?);
            self.cur = self.images.len() - 1;
            return Ok(());
        }
        // Full: reuse the slot selected longest ago.
        let i = (0..self.images.len()).min_by_key(|&i| self.images[i].used).unwrap();
        self.images[i].reset();
        self.images[i].hash = hash;
        self.images[i].used = self.clock;
        self.cur = i;
        Ok(())
    }

    /// Drop the current image's blocks: its arena filled.
    fn spill(&mut self) {
        self.images[self.cur].reset();
        self.flushes += 1;
    }
}

/// Hash all of micro memory, a `u64` at a time.
fn image_hash(micro: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in micro.chunks_exact(8) {
        h = (h ^ u64::from_le_bytes(w.try_into().unwrap())).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Run a microprogram from `start`, in pairs, until its E bit or the safety
/// limit. Mirrors [`Vu1::run_interp`] exactly, including where it leaves
/// `next_pc` for a later MSCNT, and including where a program that never
/// ends stops.
pub fn run(vu: &mut Vu1, gs: &mut GsFront, gif: &mut Gif, start: u16) {
    let mut jit = vu.jit.take().expect("VU1 dispatcher entered without a recompiler");
    if jit.micro_gen != vu.micro_gen {
        jit.select(&vu.micro).expect("VU1 code arena");
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
        let entry = match jit.images[jit.cur].lookup[pc as usize] {
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
    if jit.images[jit.cur].arena.remaining() < 64 * 1024 {
        jit.spill();
    }
    let base = jit.images[jit.cur].arena.next_addr();
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
    let ptr = jit.images[jit.cur].arena.place(&code);
    debug_assert_eq!(ptr as usize, base);
    // SAFETY: the arena is executable and now holds a complete block.
    let entry: Entry = unsafe { std::mem::transmute::<*const u8, Entry>(ptr) };
    jit.images[jit.cur].lookup[pc as usize] = Some(entry);
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
