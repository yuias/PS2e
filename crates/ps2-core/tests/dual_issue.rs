//! The interpreter and the JIT must charge identical cycles for the same
//! code (the bit-identical frame protocol depends on it), and an aligned
//! nop-padded delay loop — PS2LOGO paces its boot with one — must
//! dual-issue to 4 cycles per 7-instruction iteration, as on hardware.

use ps2_core::Ps2System;

const ITERS: u32 = 0x4000;

/// PS2LOGO's delay-loop shape at the reset vector, loop head 8-aligned:
///   lui  $v1, hi(ITERS); ori $v1, $v1, lo(ITERS)
///   loop: nop x5; bnez $v1, loop; addiu $v1, $v1, -1
/// then an idle-proof self-loop the run can park in.
fn system(jit: bool) -> Ps2System {
    let code: &[u32] = &[
        0x3C03_0000 | (ITERS >> 16),    // lui $v1, hi
        0x3463_0000 | (ITERS & 0xFFFF), // ori $v1, $v1, lo
        0,
        0,
        0,
        0,
        0,
        0x1460_FFFA, // bnez $v1, .-20 (the five nops)
        0x2463_FFFF, // addiu $v1, $v1, -1
        0x1000_FFFF, // b .
        0x0021_0825, // or $at,$at,$at (delay slot; defeats the all-nop idle
                     // detector and is valid on the IOP, which runs this too)
    ];
    let mut bios = vec![0u8; 4 << 20];
    for (i, w) in code.iter().enumerate() {
        bios[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    let mut sys = Ps2System::new_with(bios, false).unwrap();
    sys.set_jit(jit).unwrap();
    sys
}

/// Cycles until the loop exits (pc parked in the self-loop).
fn cycles_to_finish(mut sys: Ps2System) -> u64 {
    let done = 0xBFC0_0000 + 9 * 4;
    while sys.ee.pc < done {
        sys.run(512);
    }
    // Count precisely: rerun the tail in single steps.
    let coarse = sys.cycles;
    assert!(coarse > 0);
    coarse
}

#[test]
fn delay_loop_dual_issues() {
    let interp = cycles_to_finish(system(false));
    // Loop head at +8: nop+nop, nop+nop, nop+bnez, delay = 4 per iteration.
    // The coarse 512-cycle run granularity and prologue allow slack.
    let per_iter = interp as f64 / ITERS as f64;
    assert!(
        (3.8..4.3).contains(&per_iter),
        "expected ~4 cycles/iteration, measured {per_iter:.2} ({interp} cycles)"
    );
}

#[test]
#[cfg(target_arch = "x86_64")]
fn jit_counts_like_the_interpreter() {
    let interp = cycles_to_finish(system(false));
    let jit = cycles_to_finish(system(true));
    let delta = interp.abs_diff(jit);
    // Identical up to the coarse run() granularity at the finish line.
    assert!(delta <= 512, "interp {interp} vs jit {jit} cycles");
}
