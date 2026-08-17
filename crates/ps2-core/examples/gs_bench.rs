//! Rasterizer micro-benchmark: full-screen textured sprites and triangles
//! through the common Amagami/OSD pipeline setups. Prints ns per shaded
//! pixel per case. Run with `cargo run --release -p ps2-core --example gs_bench`.

use std::time::Instant;

use ps2_core::gs::Gs;

const W: u64 = 640;
const H: u64 = 224;
const XOFF: u64 = 1728 * 16;
const YOFF: u64 = 1936 * 16;

fn setup() -> Gs {
    let mut gs = Gs::new();
    // Texture: 256x256 PSMT8 at bp 4480 with a CSM1 CLUT at bp 10108;
    // 32-bit texture at bp 2240 (640x224); 16-bit CLUT variant not covered.
    for y in 0..256 {
        for x in 0..256 {
            gs.write_psmt8(4480, 4, x, y, (x ^ y) as u8);
        }
    }
    for e in 0..256u32 {
        gs.write_psmct32(10108, 1, e & 0xF, e >> 4, 0x8000_0000 | e * 0x0001_0101);
    }
    for y in 0..H as u32 {
        for x in 0..W as u32 {
            gs.write_psmct32(2240, 10, x, y, 0x8060_7080 ^ (x * 3 + y * 7));
        }
    }
    gs.write_reg(0x1A, 1); // PRMODECONT: PRIM
    gs.write_reg(0x4C, 210 | (10 << 16)); // FRAME_1: 6720, fbw 10, PSMCT32
    gs.write_reg(0x4E, 280 | (1 << 24)); // ZBUF_1: 8960, Z24, writes on
    gs.write_reg(0x47, 0x30000); // TEST_1: ZTE ALWAYS
    gs.write_reg(0x40, (W - 1) << 16 | (H - 1) << 48); // SCISSOR_1
    gs.write_reg(0x18, XOFF | (YOFF << 32)); // XYOFFSET_1
    gs.write_reg(0x42, 0x44); // ALPHA_1: (Cs - Cd) * As + Cd
    gs.write_reg(0x3B, 0x80); // TEXA
    gs.write_reg(0x01, 0x3F80_0000_8080_8080); // RGBAQ: unity modulate, q = 1
    gs
}

fn tex0_psmt8() -> u64 {
    // tbp 4480, tbw 4, PSMT8, 256x256, TCC, cbp 10108, CSM1, CLD 1
    4480 | (4 << 14) | (0x13 << 20) | (8 << 26) | (8 << 30) | (1 << 34) | (10108 << 37) | (1 << 61)
}

fn tex0_psmct32() -> u64 {
    // tbp 2240, tbw 10, PSMCT32, 1024x256, TCC
    2240 | (10 << 14) | (10 << 26) | (8 << 30) | (1 << 34)
}

fn xyz(x: u64, y: u64, z: u64) -> u64 {
    (XOFF + x * 16) | ((YOFF + y * 16) << 16) | (z << 32)
}

/// Full-screen sprite with STQ coordinates.
fn sprite(gs: &mut Gs) {
    gs.write_reg(0x02, 0x3F80_0000 << 32); // ST: (0, 1)
    gs.write_reg(0x03, 0);
    gs.write_reg(0x05, xyz(0, 0, 0xFF_FFFF));
    gs.write_reg(0x02, 0x3F80_0000); // ST: (1, 0)
    gs.write_reg(0x03, 0);
    gs.write_reg(0x05, xyz(W, H, 0xFF_FFFF));
}

/// Two triangles covering the screen, gouraud + STQ.
fn quad(gs: &mut Gs) {
    let corners = [(0, 0), (W, 0), (0, H), (W, 0), (W, H), (0, H)];
    for (x, y) in corners {
        let s = if x == 0 { 0.0f32 } else { 1.0 };
        let t = if y == 0 { 0.0f32 } else { 1.0 };
        gs.write_reg(0x02, s.to_bits() as u64 | ((t.to_bits() as u64) << 32));
        gs.write_reg(0x01, 0x3F80_0000_8080_8080 ^ (x + y));
        gs.write_reg(0x05, xyz(x, y, 0xFF_FFFF));
    }
}

fn bench(name: &str, gs: &mut Gs, prim: u64, tex1: u64, tex0: u64, draw: fn(&mut Gs)) {
    gs.write_reg(0x14, tex1); // TEX1_1
    gs.write_reg(0x06, tex0); // TEX0_1
    // Warm up, then time.
    gs.write_reg(0x00, prim);
    draw(gs);
    let before = gs.pixels_shaded;
    let t = Instant::now();
    let iters = 20;
    for _ in 0..iters {
        gs.write_reg(0x00, prim);
        draw(gs);
    }
    let dt = t.elapsed();
    let px = gs.pixels_shaded - before;
    println!(
        "{name:<34} {:7.2} ns/px  ({px} px, {:.1} ms)",
        dt.as_nanos() as f64 / px as f64,
        dt.as_secs_f64() * 1e3
    );
}

fn main() {
    let mut gs = setup();
    // PRIM bits: 6 sprite / 3 triangle, IIP 3, TME 4, ABE 6, FST 8 (STQ = 0).
    bench("sprite flat untextured", &mut gs, 0x6, 0, 0, sprite);
    bench("sprite psmt8 nearest", &mut gs, 0x16, 0, tex0_psmt8(), sprite);
    bench("sprite psmt8 bilinear", &mut gs, 0x16, 0x20, tex0_psmt8(), sprite);
    bench("sprite psmt8 bilinear + blend", &mut gs, 0x56, 0x20, tex0_psmt8(), sprite);
    bench("sprite psmct32 bilinear + blend", &mut gs, 0x56, 0x20, tex0_psmct32(), sprite);
    bench("tri gouraud untextured", &mut gs, 0xB, 0, 0, quad);
    bench("tri psmt8 bilinear + blend", &mut gs, 0x5B, 0x20, tex0_psmt8(), quad);
}
