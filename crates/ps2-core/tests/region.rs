//! The video timing region drives the vblank cadence.
//!
//! Both machines run from a zeroed "BIOS", so the EE only executes nops
//! from the reset vector: the frame edges are the only thing moving.

use ps2_core::{Ps2System, Region, bus::BIOS_SIZE};

/// EE cycles until the vblank-start interrupt latches, from reset.
fn cycles_to_first_vblank(region: Region) -> u64 {
    let mut sys = Ps2System::new_with_region(vec![0u8; BIOS_SIZE], false, region).unwrap();
    for n in 1.. {
        sys.step();
        if sys.bus.intc_stat & (1 << 2) != 0 {
            return n;
        }
    }
    unreachable!()
}

#[test]
fn vblank_lands_at_the_region_frame_edge() {
    for region in [Region::Ntsc, Region::Pal] {
        // Vertical blank starts at 95% of the frame; the edge fires on the
        // cycle after `frame_pos` reaches it.
        assert_eq!(cycles_to_first_vblank(region), region.vblank_start() + 1);
    }
    // 50 Hz frames are the longer ones.
    assert!(Region::Pal.cycles_per_frame() > Region::Ntsc.cycles_per_frame());
}

/// EE cycle of each vblank-start latch, for `frames` frames from reset.
fn vblank_cycles(region: Region, frames: usize) -> Vec<u64> {
    let mut sys = Ps2System::new_with_region(vec![0u8; BIOS_SIZE], false, region).unwrap();
    let mut at = Vec::new();
    // Slices far shorter than a frame, so no two edges hide in one of them.
    while at.len() < frames {
        sys.run(4096);
        if sys.bus.intc_stat & (1 << 2) != 0 {
            sys.bus.intc_stat &= !(1 << 2); // nothing acknowledges it here
            at.push(sys.cycles);
        }
    }
    at
}

/// PAL only: the longer frame is what a wrap bug in the cycle-skipping
/// paths would show up on, and every other test already runs NTSC frames.
#[test]
fn the_pal_frame_cadence_holds_across_frames() {
    let region = Region::Pal;
    let at = vblank_cycles(region, 3);
    for pair in at.windows(2) {
        let gap = pair[1] - pair[0];
        // One slice of slack: the latch is observed at a slice end.
        assert!(
            gap.abs_diff(region.cycles_per_frame()) <= 4096,
            "{gap} cycles between vblanks, want {}",
            region.cycles_per_frame()
        );
    }
}

#[test]
fn a_loaded_state_brings_its_own_region() {
    let mut pal = Ps2System::new_with_region(vec![0u8; BIOS_SIZE], false, Region::Pal).unwrap();
    pal.run(1000);
    let saved = pal.save_state().unwrap();
    let mut ntsc = Ps2System::new_with_region(vec![0u8; BIOS_SIZE], false, Region::Ntsc).unwrap();
    ntsc.load_state(&saved).unwrap();
    assert_eq!(ntsc.region(), Region::Pal);
}

#[test]
fn default_region_is_ntsc() {
    let sys = Ps2System::new_with(vec![0u8; BIOS_SIZE], false).unwrap();
    assert_eq!(sys.region(), Region::Ntsc);
}
