//! Interpreter micro-benchmark: a small MIPS loop planted at the BIOS
//! reset vector, run for a fixed number of EE cycles. Prints ns per EE
//! step (the IOP executes the same words at 1/8 rate, as on hardware).
//! `cargo run --release -p ps2-core --example ee_bench`.

use std::time::Instant;

use ps2_core::{Ps2System, bus::BIOS_SIZE};

/// Assemble a handful of I/R-type words (enough for a memcpy-style loop).
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
    let mut bios = vec![0u8; BIOS_SIZE];
    // t0 = 0x80010000 (src), t1 = 0x80020000 (dst), t2 = 1024 words.
    // loop: lw t3,0(t0); addiu t0,4; sw t3,0(t1); addiu t1,4; addu t4,t4,t3;
    //       addiu t2,-1; bne t2,zero,loop; nop
    // then reset the pointers/counter and jump back.
    let prog = [
        lui(8, 0x8001),
        lui(9, 0x8002),
        addiu(10, 0, 1024),
        // loop at +12
        lw(11, 0, 8),
        addiu(8, 8, 4),
        sw(11, 0, 9),
        addiu(9, 9, 4),
        addu(12, 12, 11),
        addiu(10, 10, -1),
        bne(10, 0, -7),
        0, // nop
        j(0xBFC0_0000),
        0,
    ];
    for (i, w) in prog.iter().enumerate() {
        bios[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
    }
    let mut sys = Ps2System::new(bios).expect("bios");
    // Warm up (fills the fetch/TLB caches), then time.
    sys.run(5_000_000);
    let cycles = 200_000_000u64;
    let t = Instant::now();
    sys.run(cycles);
    let dt = t.elapsed();
    println!(
        "{:.2} ns/EE step  ({:.1} M steps/s, {} cycles in {:.2} s)",
        dt.as_nanos() as f64 / cycles as f64,
        cycles as f64 / dt.as_secs_f64() / 1e6,
        cycles,
        dt.as_secs_f64()
    );
}
