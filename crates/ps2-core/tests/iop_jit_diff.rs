//! Differential check: the IOP recompiler against the IOP interpreter, two
//! identical machines side by side, compared often enough to name the slice
//! a divergence appeared in.
//!
//! The two are interchangeable at any instruction boundary, so every
//! observable — both cores, both RAMs — must match at every checkpoint.
//! `SLICE` and `ROUNDS` set the granularity and the length of the run.

use ps2_core::Ps2System;

/// First differing byte, for pointing at the subsystem that wrote it.
fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    if a == b {
        return None;
    }
    a.iter().zip(b).position(|(x, y)| x != y)
}

#[test]
#[ignore = "needs a BIOS image and runs for a while"]
fn iop_jit_matches_the_interpreter() {
    let bios = std::fs::read("../../assets/SCPH-50000.bin").expect("assets/SCPH-50000.bin");
    let mut a = Ps2System::new_with(bios.clone(), false).unwrap();
    let mut b = Ps2System::new_with(bios, false).unwrap();
    a.set_iop_jit(true).unwrap();
    b.set_iop_jit(false).unwrap();

    let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let slice = env("SLICE", 1_000_000);
    let rounds = env("ROUNDS", 3000);
    for round in 0..rounds {
        a.run(slice);
        b.run(slice);
        let at = format!("round {round}, cycle {}", b.cycles);
        assert_eq!(a.cycles, b.cycles, "cycles diverged at {at}");
        assert_eq!(
            (a.iop.pc, a.iop.next_pc),
            (b.iop.pc, b.iop.next_pc),
            "IOP pc diverged at {at}"
        );
        assert_eq!(a.iop.gpr, b.iop.gpr, "IOP registers diverged at {at}");
        assert_eq!((a.iop.hi, a.iop.lo), (b.iop.hi, b.iop.lo), "IOP hi/lo diverged at {at}");
        assert_eq!(a.iop.cop0, b.iop.cop0, "IOP cop0 diverged at {at}");
        assert_eq!(a.iop.idle, b.iop.idle, "IOP idle flag diverged at {at}");
        assert_eq!(a.ee.pc, b.ee.pc, "EE pc diverged at {at}");
        assert_eq!(a.ee.gpr, b.ee.gpr, "EE registers diverged at {at}");
        if let Some(i) = first_diff(&a.bus.iop_ram, &b.bus.iop_ram) {
            panic!("IOP RAM diverged at {at}, first at {i:#x}");
        }
        if let Some(i) = first_diff(&a.bus.ram, &b.bus.ram) {
            panic!("EE RAM diverged at {at}, first at {i:#x}");
        }
    }
    let (compiled, invalidated, chains, interp) = a.iop_jit_stats();
    eprintln!(
        "IOP jit: compiled={compiled} invalidated={invalidated} chains={chains} interp={interp}"
    );
    assert!(chains > 0, "the recompiler never ran");
}
