//! `extern "C"` entry points the generated IOP code calls for anything that
//! needs the bus. They take raw pointers because they are called from
//! native frames; the dispatcher guarantees the pointees are live and not
//! otherwise borrowed for the duration of the block.
//!
//! Every memory helper also reports whether the access left IOP RAM, in bit
//! 32 of a load's result or bit 0 of a store's. Only such an access can
//! move state the chain's caller re-tests once per instruction — the timer
//! due time, either core's interrupt lines — so a block ends on one, and a
//! chain that touched nothing but RAM is equivalent to the same
//! instructions stepped a group at a time.

use super::super::Cpu;
use crate::EE_PER_IOP;
use crate::bus::Bus;

/// The 8 MiB RAM mirror is plain memory; every other address ends the block.
#[inline]
fn diverts(bus: &mut Bus, vaddr: u32) -> u64 {
    let out = (vaddr & 0x1FFF_FFFF) >= 0x0080_0000;
    bus.iop_mmio_exits += u64::from(out);
    u64::from(out) << 32
}

/// Place [`Bus::now`] at the instruction being executed: the chain started
/// at a known cycle and each instruction before this one took
/// [`EE_PER_IOP`] of them.
#[inline]
fn at(bus: &mut Bus, retired: u32) {
    bus.now = bus.iop_chain_start + u64::from(retired) * EE_PER_IOP;
}

pub extern "C" fn rd8(bus: *mut Bus, addr: u32, retired: u32) -> u64 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    u64::from(bus.iop_read8(addr)) | diverts(bus, addr)
}

pub extern "C" fn rd16(bus: *mut Bus, addr: u32, retired: u32) -> u64 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    u64::from(bus.iop_read16(addr)) | diverts(bus, addr)
}

pub extern "C" fn rd32(bus: *mut Bus, addr: u32, retired: u32) -> u64 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    u64::from(bus.iop_read32(addr)) | diverts(bus, addr)
}

/// lwl: `cur` is the merge base, the in-flight value when a load to the
/// same register is still in the delay slot.
pub extern "C" fn lwl(bus: *mut Bus, addr: u32, cur: u32, retired: u32) -> u64 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    let shift = (addr & 3) * 8;
    let mem = bus.iop_read32(addr & !3);
    let mask = 0x00FF_FFFFu32.checked_shr(shift).unwrap_or(0);
    u64::from((cur & mask) | (mem << (24 - shift))) | diverts(bus, addr)
}

pub extern "C" fn lwr(bus: *mut Bus, addr: u32, cur: u32, retired: u32) -> u64 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    let shift = (addr & 3) * 8;
    let mem = bus.iop_read32(addr & !3);
    let keep = if shift == 0 { 0 } else { !(u32::MAX >> shift) };
    u64::from((cur & keep) | (mem >> shift)) | diverts(bus, addr)
}

pub extern "C" fn wr8(bus: *mut Bus, addr: u32, v: u32, retired: u32) -> u32 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    bus.iop_write8(addr, v as u8);
    (diverts(bus, addr) >> 32) as u32
}

pub extern "C" fn wr16(bus: *mut Bus, addr: u32, v: u32, retired: u32) -> u32 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    bus.iop_write16(addr, v as u16);
    (diverts(bus, addr) >> 32) as u32
}

pub extern "C" fn wr32(bus: *mut Bus, addr: u32, v: u32, retired: u32) -> u32 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    bus.iop_write32(addr, v);
    (diverts(bus, addr) >> 32) as u32
}

pub extern "C" fn swl(bus: *mut Bus, addr: u32, v: u32, retired: u32) -> u32 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    let shift = (addr & 3) * 8;
    let aligned = addr & !3;
    let mem = bus.iop_read32(aligned);
    let keep = 0xFFFF_FF00u32.checked_shl(shift).unwrap_or(0);
    bus.iop_write32(aligned, (v >> (24 - shift)) | (mem & keep));
    (diverts(bus, addr) >> 32) as u32
}

pub extern "C" fn swr(bus: *mut Bus, addr: u32, v: u32, retired: u32) -> u32 {
    // SAFETY: see module docs.
    let bus = unsafe { &mut *bus };
    at(bus, retired);
    let shift = (addr & 3) * 8;
    let aligned = addr & !3;
    let mem = bus.iop_read32(aligned);
    let keep = if shift == 0 { 0 } else { u32::MAX >> (32 - shift) };
    bus.iop_write32(aligned, (v << shift) | (mem & keep));
    (diverts(bus, addr) >> 32) as u32
}

/// Hand a load whose delay slot crosses the block boundary back to the
/// interpreter's own representation ([`Cpu::pending_load`]'s layout is not
/// something generated code should write).
pub extern "C" fn set_pending(cpu: *mut Cpu, reg: u32, v: u32) {
    // SAFETY: see module docs.
    unsafe { (*cpu).pending_load = Some((reg as usize, v)) }
}

/// [`interp_one`] for an instruction sitting in a branch's delay slot.
pub extern "C" fn interp_delay_one(cpu: *mut Cpu, bus: *mut Bus, addr: u32, instr: u32) -> u32 {
    interp(cpu, bus, addr, instr, true)
}

/// Run one instruction through the interpreter as if fetched at `addr`.
/// Returns 1 when control was diverted (see [`Cpu::exec_at`]).
pub extern "C" fn interp_one(cpu: *mut Cpu, bus: *mut Bus, addr: u32, instr: u32) -> u32 {
    interp(cpu, bus, addr, instr, false)
}

fn interp(cpu: *mut Cpu, bus: *mut Bus, addr: u32, instr: u32, in_delay: bool) -> u32 {
    // Panics must not unwind through the native frames.
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the dispatcher passes live, exclusively owned pointers for
        // the duration of the block call.
        let (cpu, bus) = unsafe { (&mut *cpu, &mut *bus) };
        let diverted = cpu.exec_at(bus, addr, instr, in_delay);
        // The translator resolves load delays itself and only ever calls
        // back for instructions that issue none; one left here would be
        // invisible to it.
        debug_assert!(
            cpu.pending_load.is_none(),
            "IOP fallback left a load in flight at {addr:#010x}"
        );
        diverted
    }));
    match r {
        Ok(diverted) => diverted as u32,
        Err(_) => {
            eprintln!("panic inside IOP JIT helper at pc {addr:#010x}");
            std::process::abort();
        }
    }
}
