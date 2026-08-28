//! Thunks the translated code calls back into.
//!
//! Every one takes raw pointers: the dispatcher guarantees the pointees are
//! live and not otherwise borrowed for as long as a block runs.

use crate::gif::Gif;
use crate::gs::GsFront;
use crate::vu1::Vu1;

/// `jit_target` when the pair left no pending branch.
pub const NO_BRANCH: u32 = 0xFFFF_FFFF;

/// Upper (FMAC) pipeline, straight into the interpreter's decoder.
pub extern "C" fn upper(vu: *mut Vu1, pc: u32, instr: u32) {
    // SAFETY: see module docs.
    unsafe { (*vu).exec_upper(pc as u16, instr) }
}

/// Lower pipeline. A branch leaves its target in `jit_target`, where the
/// block's exit picks it up after the delay pair has run.
pub extern "C" fn lower(vu: *mut Vu1, gs: *mut GsFront, gif: *mut Gif, pc: u32, instr: u32) {
    // SAFETY: see module docs.
    unsafe {
        let vu = &mut *vu;
        let mut branch = None;
        vu.exec_lower(&mut *gs, &mut *gif, pc as u16, instr, &mut branch);
        vu.jit_target = branch.map_or(NO_BRANCH, u32::from);
    }
}
