//! Differential tests: the recompiler must leave exactly the state the
//! interpreter leaves, for the same microprogram and the same starting
//! registers. Every native instruction added to `emit` is guarded by these.

use super::*;
use crate::gif::Gif;
use crate::gs::GsFront;

/// Everything a microprogram can leave behind.
///
/// `mac_seen`/`status_seen` are deliberately absent: they are pipeline
/// scratch, refreshed from `flag_pipe` before every pair, so their value
/// between programs is not observable.
#[derive(PartialEq, Eq, Debug)]
struct Snapshot {
    vf: [[u32; 4]; 32],
    vi: [u16; 16],
    acc: [u32; 4],
    q: u32,
    i: u32,
    r: u32,
    p: u32,
    mac: u16,
    status: u16,
    clip: u32,
    next_pc: u16,
    flag_pipe: [[u16; 2]; 4],
    flag_cycle: u32,
    data: Vec<u8>,
}

fn snapshot(vu: &Vu1) -> Snapshot {
    Snapshot {
        vf: vu.vf,
        vi: vu.vi,
        acc: vu.acc,
        q: vu.q.to_bits(),
        i: vu.i.to_bits(),
        r: vu.r,
        p: vu.p.to_bits(),
        mac: vu.mac,
        status: vu.status,
        clip: vu.clip,
        next_pc: vu.next_pc,
        flag_pipe: vu.flag_pipe,
        flag_cycle: vu.flag_cycle,
        data: vu.data.to_vec(),
    }
}

/// Build a VU1, load `pairs` as `(upper, lower)` from pair 0, seed the
/// registers deterministically, and run from pair 0.
fn run(pairs: &[(u32, u32)], jit: bool) -> Snapshot {
    run_seeded(pairs, jit, false)
}

/// With `wild`, the registers and data memory are seeded with raw bit
/// patterns instead of ordinary floats. LQ and MOVE copy bits, so infinities,
/// NaNs and denormals do reach the FMACs, and they are exactly what the
/// result clamp and the flag rules are there for.
fn run_seeded(pairs: &[(u32, u32)], jit: bool, wild: bool) -> Snapshot {
    let mut vu = Vu1::new();
    for (n, &(u, l)) in pairs.iter().enumerate() {
        let mut w = [0u8; 8];
        w[..4].copy_from_slice(&l.to_le_bytes());
        w[4..].copy_from_slice(&u.to_le_bytes());
        vu.write_micro(n * 8, &w);
    }
    // A spread of ordinary values: small integers, negatives, fractions.
    for r in 1..32 {
        let f = r as f32;
        vu.vf[r] = [
            (f * 0.5).to_bits(),
            (-f * 1.25).to_bits(),
            (f * f).to_bits(),
            (1.0 / f).to_bits(),
        ];
    }
    if wild {
        const ODD: [u32; 8] = [
            0x7F80_0000, // +inf
            0xFF80_0000, // -inf
            0x7FC0_0000, // NaN
            0x0000_0001, // smallest denormal
            0x807F_FFFF, // largest negative denormal
            0x8000_0000, // -0
            0x7F7F_FFFF, // largest finite
            0x0080_0000, // smallest normal
        ];
        for r in 1..32 {
            for f in 0..4 {
                vu.vf[r][f] = ODD[(r * 4 + f) % ODD.len()];
            }
        }
    }
    for r in 1..16 {
        vu.vi[r] = (r as u16) * 7;
    }
    vu.q = 2.5;
    vu.i = -0.75;
    vu.r = 0x3F80_1234;
    vu.top = 64;
    vu.itop = 8;
    for (n, b) in vu.data.iter_mut().enumerate() {
        *b = (n as u8).wrapping_mul(31).wrapping_add(7);
    }
    if jit {
        vu.set_jit(true).expect("arena");
    }
    let (mut gs, mut gif) = (GsFront::inline(), Gif::new());
    vu.start(&mut gs, &mut gif, 0);
    snapshot(&vu)
}

#[track_caller]
fn agree(pairs: &[(u32, u32)]) {
    assert_eq!(run(pairs, false), run(pairs, true));
}

#[track_caller]
fn agree_wild(pairs: &[(u32, u32)]) {
    assert_eq!(run_seeded(pairs, false, true), run_seeded(pairs, true, true));
}

const NOP_UPPER: u32 = 0x3C | (0x2F & 3) | ((0x2F & 0x7C) << 4);
const NOP_LOWER: u32 = 0x8000_033C;
/// NOP/NOP with the E bit: ends the program.
const END: (u32, u32) = (NOP_UPPER | (1 << 30), NOP_LOWER);

fn upper(op: u32, dest: u32, ft: u32, fs: u32, fd: u32) -> u32 {
    (dest << 21) | (ft << 16) | (fs << 11) | (fd << 6) | op
}

#[test]
fn straight_line_arithmetic_agrees() {
    agree(&[
        (upper(0x28, 0xF, 2, 3, 4), NOP_LOWER),  // ADD vf4, vf3, vf2
        (upper(0x2A, 0xE, 5, 4, 6), NOP_LOWER),  // MUL.xyz vf6, vf4, vf5
        (upper(0x2C, 0x8, 7, 6, 8), NOP_LOWER),  // SUB.x vf8, vf6, vf7
        (upper(0x29, 0xF, 9, 8, 10), NOP_LOWER), // MADD vf10, vf8, vf9
        END,
    ]);
}

#[test]
fn the_i_bit_hands_its_constant_to_the_same_pair() {
    // MULi vf5, vf3, I with the constant in the lower slot.
    agree(&[
        (upper(0x1E, 0xF, 0, 3, 5) | (1 << 31), 0x4048_0000),
        (upper(0x22, 0xF, 0, 5, 6), NOP_LOWER), // ADDi vf6, vf5, I
        END,
    ]);
}

#[test]
fn loads_and_stores_agree() {
    agree(&[
        // LQ vf5, 3(vi02) / SQ vf5, 9(vi03) / ILW / ISW / LQI / SQD
        (NOP_UPPER, (0x00 << 25) | (0xF << 21) | (5 << 16) | (2 << 11) | 3),
        (NOP_UPPER, (0x01 << 25) | (0xF << 21) | (3 << 16) | (5 << 11) | 9),
        (NOP_UPPER, (0x04 << 25) | (0x4 << 21) | (6 << 16) | (2 << 11) | 5),
        (NOP_UPPER, (0x05 << 25) | (0x4 << 21) | (6 << 16) | (3 << 11) | 5),
        (NOP_UPPER, 0x8000_0000 | (0xF << 21) | (7 << 16) | (4 << 11) | (0x0D << 6) | 0x3C),
        END,
    ]);
}

#[test]
fn a_taken_branch_and_its_delay_pair_agree() {
    // IBEQ vi01, vi01, +2 — always taken, so the pair after the delay slot
    // must be skipped.
    let ibeq = (0x28 << 25) | (1 << 16) | (1 << 11) | 2;
    agree(&[
        (upper(0x28, 0xF, 2, 3, 4), ibeq),
        (upper(0x28, 0xF, 2, 3, 5), NOP_LOWER), // delay pair: runs
        (upper(0x28, 0xF, 2, 3, 6), NOP_LOWER), // skipped
        (upper(0x28, 0xF, 2, 3, 7), NOP_LOWER), // branch target
        END,
    ]);
}

#[test]
fn an_untaken_branch_falls_through() {
    // IBEQ vi01, vi02, +2 — vi01 != vi02, so it is not taken.
    let ibeq = (0x28 << 25) | (2 << 16) | (1 << 11) | 2;
    agree(&[
        (upper(0x28, 0xF, 2, 3, 4), ibeq),
        (upper(0x28, 0xF, 2, 3, 5), NOP_LOWER),
        (upper(0x28, 0xF, 2, 3, 6), NOP_LOWER),
        END,
    ]);
}

#[test]
fn a_counted_loop_agrees() {
    // vi05 = 4; loop: vi05 -= 1; IBNE vi05, vi00, loop
    let iaddiu = (0x08 << 25) | (5 << 16) | (0 << 11) | 4;
    let iaddi = 0x8000_0000 | ((-1i32 as u32 & 0x1F) << 6) | (5 << 16) | (5 << 11) | 0x32;
    let ibne = (0x29 << 25) | (0 << 16) | (5 << 11) | ((-2i32 as u32) & 0x7FF);
    agree(&[
        (NOP_UPPER, iaddiu),
        (upper(0x28, 0xF, 2, 3, 4), iaddi),
        (upper(0x2A, 0xF, 4, 4, 4), ibne),
        (upper(0x28, 0xF, 2, 3, 7), NOP_LOWER), // delay pair
        END,
    ]);
}

#[test]
fn jr_takes_its_target_from_an_integer_register() {
    // vi06 = 4; JR vi06.
    let iaddiu = (0x08 << 25) | (6 << 16) | (0 << 11) | 4;
    let jr = (0x24 << 25) | (6 << 11);
    agree(&[
        (NOP_UPPER, iaddiu),
        (upper(0x28, 0xF, 2, 3, 4), jr),
        (upper(0x28, 0xF, 2, 3, 5), NOP_LOWER), // delay pair
        (upper(0x28, 0xF, 2, 3, 6), NOP_LOWER), // skipped
        (upper(0x28, 0xF, 2, 3, 7), NOP_LOWER), // target
        END,
    ]);
}

#[test]
fn the_flag_pipeline_stays_four_pairs_deep() {
    // A compare, three unrelated FMACs, then FSAND reads the aged status.
    let fsand = (0x16 << 25) | (7 << 16) | 0x3F;
    let fmand = (0x1A << 25) | (8 << 16) | (1 << 11);
    agree(&[
        (upper(0x2C, 0xF, 1, 1, 4), NOP_LOWER), // SUB vf4, vf1, vf1 -> zero flags
        (upper(0x28, 0xF, 2, 3, 5), NOP_LOWER),
        (upper(0x28, 0xF, 2, 3, 6), NOP_LOWER),
        (upper(0x28, 0xF, 2, 3, 7), fsand),
        (upper(0x28, 0xF, 2, 3, 8), fmand),
        END,
    ]);
}

#[test]
fn a_branch_inside_a_delay_pair_falls_back_to_the_interpreter() {
    // Two branches back to back: the interpreter carries the second one
    // past the pair after it, which the recompiler declines to model.
    let b1 = (0x20 << 25) | 2;
    let b2 = (0x20 << 25) | 3;
    agree(&[
        (upper(0x28, 0xF, 2, 3, 4), b1),
        (upper(0x28, 0xF, 2, 3, 5), b2),
        (upper(0x28, 0xF, 2, 3, 6), NOP_LOWER),
        (upper(0x28, 0xF, 2, 3, 7), NOP_LOWER),
        (upper(0x28, 0xF, 2, 3, 8), NOP_LOWER),
        (upper(0x28, 0xF, 2, 3, 9), NOP_LOWER),
        END,
    ]);
}

#[test]
fn a_run_longer_than_one_block_agrees() {
    let mut p: Vec<(u32, u32)> = (0..MAX_PAIRS * 3)
        .map(|n| {
            let r = (n % 20 + 2) as u32;
            (upper(0x28, 0xF, r, r + 1, r + 2), NOP_LOWER)
        })
        .collect();
    p.push(END);
    agree(&p);
}

/// Random straight-line programs over every FMAC encoding, including the
/// ones the interpreter reports as unimplemented and runs as a nop.
#[test]
fn random_upper_pipeline_programs_agree() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..64 {
        let mut p: Vec<(u32, u32)> = (0..40)
            .map(|_| {
                let u = (next() as u32) & 0x3FFF_FFFF;
                (u, NOP_LOWER)
            })
            .collect();
        p.push(END);
        agree(&p);
    }
}

/// The same over the lower pipeline, with the branch opcodes filtered out
/// so the programs terminate.
#[test]
fn random_lower_pipeline_programs_agree() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..64 {
        let mut p: Vec<(u32, u32)> = Vec::new();
        while p.len() < 40 {
            let l = next() as u32;
            if super::emit::is_branch(l) {
                continue;
            }
            p.push((upper(0x28, 0xF, 2, 3, 4), l));
        }
        p.push(END);
        agree(&p);
    }
}

/// The lower pipeline's special table (opcode 0x40), where IADD/ISUB/IAND/
/// IOR, MOVE and XITOP are emitted natively. A uniform 32-bit draw reaches
/// it once in 128, too rarely to cover 64 sub-opcodes, so pin the opcode
/// and fuzz the rest.
#[test]
fn random_lower_special_programs_agree() {
    let mut seed = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..64 {
        let mut p: Vec<(u32, u32)> = Vec::new();
        while p.len() < 40 {
            let l = 0x8000_0000 | (next() as u32 & 0x01FF_FFFF);
            if super::emit::is_branch(l) {
                continue;
            }
            p.push((upper(0x28, 0xF, 2, 3, 4), l));
        }
        p.push(END);
        agree(&p);
    }
}

/// Random forward branches. Targets stay inside the program and never point
/// backwards, so every program terminates; the tail is all E-bit pairs so a
/// branch that jumps past the end still stops.
#[test]
fn random_forward_branch_programs_agree() {
    let mut seed = 0xD1B5_4A32_D192_ED03u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    const BRANCHES: [u32; 8] = [0x20, 0x21, 0x24, 0x25, 0x28, 0x29, 0x2C, 0x2F];
    for _ in 0..96 {
        let mut p: Vec<(u32, u32)> = Vec::new();
        for pair in 0..24u32 {
            let u = upper(0x28, 0xF, 2, 3, (pair % 20) + 4);
            let r = next();
            let l = if r & 3 == 0 {
                let op = BRANCHES[(r >> 8) as usize % BRANCHES.len()];
                let fwd = (((r >> 16) & 3) + 1) as u32;
                // JR and JALR read vi01, which the seeding leaves at 7 —
                // still inside the program, still ahead of pair 0.
                (op << 25) | (1 << 16) | (1 << 11) | fwd
            } else {
                NOP_LOWER
            };
            p.push((u, l));
        }
        for _ in 0..8 {
            p.push(END);
        }
        agree(&p);
    }
}

/// The same random FMAC programs over infinities, NaNs and denormals.
#[test]
fn random_upper_pipeline_programs_agree_on_wild_bits() {
    let mut seed = 0x0EB4_4C7B_1A55_9931u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..64 {
        let mut p: Vec<(u32, u32)> = (0..40)
            .map(|_| ((next() as u32) & 0x3FFF_FFFF, NOP_LOWER))
            .collect();
        p.push(END);
        agree_wild(&p);
    }
}

/// Random lower slots over wild bit patterns: the load and store paths copy
/// bits, so this is what puts infinities and NaNs into vf through LQ, MOVE
/// and MR32 rather than through arithmetic.
#[test]
fn random_lower_pipeline_programs_agree_on_wild_bits() {
    let mut seed = 0x1D8E_4E27_C47D_124Fu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for _ in 0..64 {
        let mut p: Vec<(u32, u32)> = Vec::new();
        while p.len() < 40 {
            let l = next() as u32;
            if super::emit::is_branch(l) {
                continue;
            }
            p.push((upper(0x28, 0xF, 2, 3, 4), l));
        }
        p.push(END);
        agree_wild(&p);
    }
}

/// The pointer-walking loads and stores, whose address register is updated
/// as well as read.
#[test]
fn the_walking_loads_and_stores_agree() {
    let lq = |id2: u32, dest: u32, it: u32, is: u32| {
        0x8000_0000 | (dest << 21) | (it << 16) | (is << 11) | (((id2 & 0x7C) << 4) | (id2 & 3))
    };
    agree(&[
        (NOP_UPPER, lq(0x34, 0xF, 5, 2)),  // LQI  vf5, (vi02++)
        (NOP_UPPER, lq(0x36, 0xF, 6, 3)),  // LQD  vf6, (--vi03)
        (NOP_UPPER, lq(0x35, 0xF, 4, 5)),  // SQI  vf5, (vi04++)
        (NOP_UPPER, lq(0x37, 0xE, 7, 6)),  // SQD  vf6, (--vi07)
        (NOP_UPPER, lq(0x31, 0xD, 8, 5)),  // MR32 vf8, vf5
        (NOP_UPPER, lq(0x68, 0x0, 9, 0)),  // XTOP vi09
        (NOP_UPPER, lq(0x3E, 0x2, 10, 2)), // ILWR vi10, (vi02)
        (NOP_UPPER, lq(0x3F, 0x2, 10, 3)), // ISWR vi10, (vi03)
        END,
    ]);
}
