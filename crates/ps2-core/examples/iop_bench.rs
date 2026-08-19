//! IOP interpreter micro-benchmark: the ee_bench memcpy loop planted in
//! IOP RAM and stepped directly, bypassing the EE. Prints ns per IOP
//! instruction. `cargo run --release -p ps2-core --example iop_bench`.

use std::time::Instant;

use ps2_core::{Ps2System, bus::BIOS_SIZE};

fn lui(rt: u32, imm: u32) -> u32 {
    (0x0F << 26) | (rt << 16) | (imm & 0xFFFF)
}
fn addiu(rt: u32, rs: u32, imm: i32) -> u32 {
    (0x09 << 26) | (rs << 21) | (rt << 16) | (imm as u32 & 0xFFFF)
}
fn lw(rt: u32, off: i32, base: u32) -> u32 {
    (0x23 << 26) | (base << 21) | (rt << 16) | (off as u32 & 0xFFFF)
}
fn sw(rt: u32, off: i32, base: u32) -> u32 {
    (0x2B << 26) | (base << 21) | (rt << 16) | (off as u32 & 0xFFFF)
}
fn addu(rd: u32, rs: u32, rt: u32) -> u32 {
    (rs << 21) | (rt << 16) | (rd << 11) | 0x21
}
fn bne(rs: u32, rt: u32, off: i32) -> u32 {
    (0x05 << 26) | (rs << 21) | (rt << 16) | (off as u32 & 0xFFFF)
}
fn j(target: u32) -> u32 {
    (0x02 << 26) | ((target >> 2) & 0x03FF_FFFF)
}

fn main() {
    let bios = vec![0u8; BIOS_SIZE];
    let mut sys = Ps2System::new(bios).expect("bios");
    // Program at IOP RAM 0x1000: t0 = 0x10000 (src), t1 = 0x20000 (dst).
    let base = 0x1000u32;
    let prog = [
        lui(8, 0x0001),
        lui(9, 0x0002),
        addiu(10, 0, 1024),
        lw(11, 0, 8),
        addiu(8, 8, 4),
        sw(11, 0, 9),
        addiu(9, 9, 4),
        addu(12, 12, 11),
        addiu(10, 10, -1),
        bne(10, 0, -7),
        0,
        j(base),
        0,
    ];
    for (i, w) in prog.iter().enumerate() {
        sys.bus.iop_write32(base + i as u32 * 4, *w);
    }
    sys.iop.pc = base;
    sys.iop.next_pc = base + 4;
    for _ in 0..1_000_000 {
        sys.iop.step(&mut sys.bus);
    }
    let n = 50_000_000u64;
    let t = Instant::now();
    for _ in 0..n {
        sys.iop.step(&mut sys.bus);
    }
    let dt = t.elapsed();
    println!("{:.2} ns/IOP instruction ({} in {:.2} s)", dt.as_nanos() as f64 / n as f64, n, dt.as_secs_f64());
}
