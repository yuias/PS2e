//! `extern "C"` entry points the generated code calls for anything that
//! needs the bus. They take raw pointers because they are called from
//! native frames; the dispatcher guarantees the pointees are live and not
//! otherwise borrowed for the duration of the block.

use super::super::Cpu;
use crate::bus::Bus;

/// mfc0: needs the cycle counter and live interrupt lines.
pub extern "C" fn cop0_read(cpu: *mut Cpu, bus: *mut Bus, rd: u32) -> u32 {
    // SAFETY: see module docs.
    unsafe {
        let bus = &mut *bus;
        (*cpu).cop0.read(rd as usize, bus.now, bus.ee_int0_pending(), bus.ee_int1_pending())
    }
}

/// mtc0: the register file's own write rules (read-only PRId, the
/// partially writable Cause) live in [`crate::ee::cop0`].
pub extern "C" fn cop0_write(cpu: *mut Cpu, rd: u32, v: u32) {
    // SAFETY: see module docs.
    unsafe { (*cpu).cop0.write(rd as usize, v) }
}

/// ei / di.
pub extern "C" fn cop0_set_eie(cpu: *mut Cpu, enable: u32) {
    // SAFETY: see module docs.
    unsafe { (*cpu).cop0.set_eie(enable != 0) }
}

/// MMI (128-bit multimedia) instruction, straight to the interpreter's
/// handler: none of them divert control.
pub extern "C" fn mmi(cpu: *mut Cpu, instr: u32) {
    let (rs, rt, rd, sa) = ((instr >> 21) & 31, (instr >> 16) & 31, (instr >> 11) & 31, (instr >> 6) & 31);
    // SAFETY: see module docs.
    unsafe { (*cpu).op_mmi(instr, rs as usize, rt as usize, rd as usize, sa) }
}

/// COP2 macro instruction other than bc2 (VU0 register moves and micro
/// ops), straight to the interpreter's handler.
pub extern "C" fn cop2(cpu: *mut Cpu, bus: *mut Bus, instr: u32) {
    let (rs, rt, rd) = ((instr >> 21) & 31, (instr >> 16) & 31, (instr >> 11) & 31);
    // SAFETY: see module docs.
    unsafe { (*cpu).op_cop2(instr, rs as usize, rt as usize, rd as usize, &mut *bus) }
}

pub extern "C" fn rd8(bus: *mut Bus, addr: u32) -> u64 {
    // SAFETY: see module docs.
    unsafe { (*bus).read8(addr) as u64 }
}
pub extern "C" fn rd16(bus: *mut Bus, addr: u32) -> u64 {
    // SAFETY: see module docs.
    unsafe { (*bus).read16(addr) as u64 }
}
pub extern "C" fn rd32(bus: *mut Bus, addr: u32) -> u64 {
    // SAFETY: see module docs.
    unsafe { (*bus).read32(addr) as u64 }
}
pub extern "C" fn rd64(bus: *mut Bus, addr: u32) -> u64 {
    // SAFETY: see module docs.
    unsafe { (*bus).read64(addr) }
}
/// lq: `out` points at the destination GPR (both halves).
pub extern "C" fn rd128(bus: *mut Bus, addr: u32, out: *mut [u64; 2]) {
    // SAFETY: see module docs; `out` is a GPR slot inside the Cpu.
    unsafe { *out = (*bus).read128(addr) }
}
pub extern "C" fn wr8(bus: *mut Bus, addr: u32, v: u64) {
    // SAFETY: see module docs.
    unsafe { (*bus).write8(addr, v as u8) }
}
pub extern "C" fn wr16(bus: *mut Bus, addr: u32, v: u64) {
    // SAFETY: see module docs.
    unsafe { (*bus).write16(addr, v as u16) }
}
pub extern "C" fn wr32(bus: *mut Bus, addr: u32, v: u64) {
    // SAFETY: see module docs.
    unsafe { (*bus).write32(addr, v as u32) }
}
pub extern "C" fn wr64(bus: *mut Bus, addr: u32, v: u64) {
    // SAFETY: see module docs.
    unsafe { (*bus).write64(addr, v) }
}
/// cvt.w.s with the interpreter's saturation (NaN -> 0).
pub extern "C" fn cvt_w_s(bits: u32) -> u32 {
    let a = f32::from_bits(bits);
    let v = if a >= 2147483647.0 {
        i32::MAX
    } else if a <= -2147483648.0 {
        i32::MIN
    } else {
        a as i32
    };
    v as u32
}

/// sq: `src` points at the source GPR.
pub extern "C" fn wr128(bus: *mut Bus, addr: u32, src: *const [u64; 2]) {
    // SAFETY: see module docs.
    unsafe { (*bus).write128(addr, *src) }
}
