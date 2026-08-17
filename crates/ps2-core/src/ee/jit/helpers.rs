//! `extern "C"` entry points the generated code calls for anything that
//! needs the bus. They take raw pointers because they are called from
//! native frames; the dispatcher guarantees the pointees are live and not
//! otherwise borrowed for the duration of the block.

use crate::bus::Bus;

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
