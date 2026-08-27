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

#[test]
fn default_region_is_ntsc() {
    let sys = Ps2System::new_with(vec![0u8; BIOS_SIZE], false).unwrap();
    assert_eq!(sys.region(), Region::Ntsc);
}
