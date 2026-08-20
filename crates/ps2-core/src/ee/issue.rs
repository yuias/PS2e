//! Dual-issue cost model for the R5900.
//!
//! The EE retires up to two instructions per cycle; charging one cycle per
//! instruction made fixed CPU delay loops (PS2LOGO paces its boot animation
//! with them) run ~2x long, so their vsync catch-up skipped whole scenes.
//!
//! The interpreter executes an accepted couple inside one [`super::Cpu::step`]
//! and the JIT charges the second half zero cycles, so both worlds count
//! identically — the bit-identical frame protocol and `--no-jit` debugging
//! depend on that.
//!
//! The model is deliberately conservative and position-based so a JIT block
//! boundary can never split a couple that the interpreter would pair:
//! - only the 8-byte-aligned couple (addr % 8 == 0 with addr % 8 == 4),
//!   executed back to back, may pair — alignment and sequencing (no delay
//!   slot, no intervening control flow) are the caller's responsibility;
//! - the halves are simple ALU ops (shift, add/sub, logic, slt, lui, nop),
//!   at most one of them a load/store (one LS pipe), and the second may be
//!   a branch or jump (branch pipe; its delay slot then costs 1 as usual);
//! - no RAW or WAW hazard between the halves;
//! - the first half never diverts control: branches lead nothing, and the
//!   trapping add forms, delay slots, mult/div, MMI, 128-bit lq/sq and all
//!   coprocessor ops never pair at all.

/// Register/memory use of one pairable instruction; `None` if it never pairs.
struct Use {
    /// Bitmask of GPRs read.
    reads: u32,
    /// GPR written (0 = none; writes to $zero are no-ops anyway).
    write: u32,
    /// Touches memory (one load/store per cycle).
    mem: bool,
    /// Branch or jump: may only be the second half of a couple.
    branch: bool,
}

fn issue_use(instr: u32) -> Option<Use> {
    let rs = (instr >> 21) & 31;
    let rt = (instr >> 16) & 31;
    let rd = (instr >> 11) & 31;
    let bit = |r: u32| 1u32 << r;
    let u = |reads, write, mem| Some(Use { reads, write, mem, branch: false });
    let b = |reads, write| Some(Use { reads, write, mem: false, branch: true });
    match instr >> 26 {
        0x00 => match instr & 0x3F {
            // sll/srl/sra and the dsll/dsrl/dsra(32) forms (nop included).
            0x00 | 0x02 | 0x03 | 0x38 | 0x3A | 0x3B | 0x3C | 0x3E | 0x3F => {
                u(bit(rt), rd, false)
            }
            // sllv/srlv/srav, dsllv/dsrlv/dsrav.
            0x04 | 0x06 | 0x07 | 0x14 | 0x16 | 0x17 => u(bit(rs) | bit(rt), rd, false),
            // jr / jalr.
            0x08 => b(bit(rs), 0),
            0x09 => b(bit(rs), rd),
            // addu/subu/and/or/xor/nor/slt/sltu/daddu/dsubu; the trapping
            // add/sub/dadd/dsub forms are excluded (they may divert).
            0x21 | 0x23..=0x27 | 0x2A | 0x2B | 0x2D | 0x2F => u(bit(rs) | bit(rt), rd, false),
            _ => None,
        },
        // bltz/bgez and the likely forms (the link forms never pair).
        0x01 if (instr >> 16) & 0x1F < 4 => b(bit(rs), 0),
        // j / jal.
        0x02 => b(0, 0),
        0x03 => b(0, 31),
        // beq/bne/blez/bgtz and the likely forms.
        0x04 | 0x05 | 0x14 | 0x15 => b(bit(rs) | bit(rt), 0),
        0x06 | 0x07 | 0x16 | 0x17 => b(bit(rs), 0),
        // addiu/slti/sltiu/andi/ori/xori and daddiu (addi/daddi may trap).
        0x09..=0x0E | 0x19 => u(bit(rs), rt, false),
        // lui.
        0x0F => u(0, rt, false),
        // lwl/lwr and ldl/ldr merge with rt.
        0x22 | 0x26 | 0x1A | 0x1B => u(bit(rs) | bit(rt), rt, true),
        // lb/lh/lw/lbu/lhu/lwu/ld.
        0x20 | 0x21 | 0x23..=0x25 | 0x27 | 0x37 => u(bit(rs), rt, true),
        // sb/sh/swl/sw/sdl/sdr/swr/sd.
        0x28..=0x2E | 0x3F => u(bit(rs) | bit(rt), 0, true),
        _ => None,
    }
}

/// Whether `first` may lead a couple at all — a cheap pre-check so callers
/// can skip fetching the partner word.
#[inline]
pub fn may_lead(first: u32) -> bool {
    issue_use(first).is_some_and(|u| !u.branch)
}

/// Whether `second` retires in the same cycle as `first`. Pure function of
/// the two instruction words; see the module doc for the caller's alignment
/// and sequencing obligations.
#[inline]
pub fn dual_issue(first: u32, second: u32) -> bool {
    let (Some(a), Some(b)) = (issue_use(first), issue_use(second)) else {
        return false;
    };
    // Branches never lead (the head must fall through) and the single LS
    // pipe takes one memory op per cycle.
    if a.branch || (a.mem && b.mem) {
        return false;
    }
    // RAW or WAW on the first half's destination stalls the second.
    a.write == 0 || ((b.reads >> a.write) & 1 == 0 && a.write != b.write)
}

#[cfg(test)]
mod tests {
    use super::dual_issue;

    const NOP: u32 = 0;
    /// addiu rt, rs, imm
    fn addiu(rt: u32, rs: u32, imm: u16) -> u32 {
        (0x09 << 26) | (rs << 21) | (rt << 16) | imm as u32
    }
    /// or rd, rs, rt
    fn or(rd: u32, rs: u32, rt: u32) -> u32 {
        (rs << 21) | (rt << 16) | (rd << 11) | 0x25
    }
    /// lw rt, imm(rs)
    fn lw(rt: u32, rs: u32, imm: u16) -> u32 {
        (0x23 << 26) | (rs << 21) | (rt << 16) | imm as u32
    }
    /// sw rt, imm(rs)
    fn sw(rt: u32, rs: u32, imm: u16) -> u32 {
        (0x2B << 26) | (rs << 21) | (rt << 16) | imm as u32
    }
    /// bne rs, rt, off
    fn bne(rs: u32, rt: u32, off: u16) -> u32 {
        (0x05 << 26) | (rs << 21) | (rt << 16) | off as u32
    }

    #[test]
    fn independent_alu_pairs() {
        assert!(dual_issue(NOP, NOP));
        assert!(dual_issue(addiu(2, 2, 1), addiu(3, 3, 1)));
        assert!(dual_issue(addiu(2, 0, 5), or(4, 5, 6)));
    }

    #[test]
    fn hazards_do_not_pair() {
        // RAW: second reads the first's destination.
        assert!(!dual_issue(addiu(2, 0, 1), or(3, 2, 4)));
        assert!(!dual_issue(addiu(2, 0, 1), lw(3, 2, 0)));
        assert!(!dual_issue(addiu(2, 0, 1), sw(2, 3, 0)));
        // WAW: both write the same register.
        assert!(!dual_issue(addiu(2, 0, 1), addiu(2, 3, 1)));
        // Writes to $zero never hazard.
        assert!(dual_issue(addiu(0, 1, 1), or(2, 3, 4)));
    }

    #[test]
    fn one_memory_op_per_couple() {
        assert!(dual_issue(addiu(2, 0, 1), lw(3, 4, 0)));
        assert!(dual_issue(NOP, sw(2, 3, 0)));
        assert!(dual_issue(lw(2, 3, 0), NOP));
        assert!(dual_issue(sw(2, 3, 0), addiu(4, 5, 1)));
        assert!(!dual_issue(lw(2, 3, 0), lw(4, 5, 0)));
        assert!(!dual_issue(lw(2, 3, 0), sw(4, 5, 0)));
    }

    #[test]
    fn branches_pair_second_only() {
        assert!(dual_issue(NOP, bne(2, 3, 4)));
        assert!(dual_issue(lw(2, 3, 0), bne(4, 5, 6)));
        // jr $ra / jal
        assert!(dual_issue(NOP, (31 << 21) | 0x08));
        assert!(dual_issue(addiu(2, 0, 1), 0x0C00_0000 | 0x40));
        // RAW into the branch condition.
        assert!(!dual_issue(addiu(2, 0, 1), bne(2, 3, 4)));
        // Branches never lead.
        assert!(!dual_issue(bne(2, 3, 4), NOP));
    }

    #[test]
    fn diverting_ops_never_pair() {
        // mult
        assert!(!dual_issue(NOP, (2 << 21) | (3 << 16) | 0x18));
        // syscall
        assert!(!dual_issue(NOP, 0x0C));
        // addi (may trap on overflow)
        assert!(!dual_issue(NOP, (0x08 << 26) | (2 << 16) | 1));
    }
}
