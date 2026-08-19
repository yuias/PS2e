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
    for y in 0..64u32 {
        for x in 0..64u32 {
            gs.write_psmct16(12320, 1, x, y, 0x02, (0x8000 | (x * 31 / 63) | ((y * 31 / 63) << 5) | (((x ^ y) & 31) << 10)) as u16);
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

fn tex0_psmct16() -> u64 {
    // tbp 12320, tbw 1, PSMCT16, 64x64, TCC
    12320 | (1 << 14) | (0x02 << 20) | (6 << 26) | (6 << 30) | (1 << 34)
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

/// Many small rotated particle quads (two triangles each) over the screen,
/// the way the game's effects draw their 64x64 PSMCT16 sprites.
fn particles(gs: &mut Gs) {
    let mut seed = 12345u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as u64
    };
    for _ in 0..1500 {
        let cx = (rnd() % (W - 64)) as f32 + 32.0;
        let cy = (rnd() % (H - 64)) as f32 + 32.0;
        let ang = (rnd() % 628) as f32 / 100.0;
        let r = 24.0f32;
        let (sn, cs) = ang.sin_cos();
        let corner = |dx: f32, dy: f32| -> (u64, u64) {
            let x = cx + dx * cs - dy * sn;
            let y = cy + dx * sn + dy * cs;
            (x.max(0.0) as u64, y.max(0.0) as u64)
        };
        let pts = [(-r, -r, 0.0f32, 0.0f32), (r, -r, 1.0, 0.0), (-r, r, 0.0, 1.0), (r, -r, 1.0, 0.0), (r, r, 1.0, 1.0), (-r, r, 0.0, 1.0)];
        for (dx, dy, s, t) in pts {
            let (x, y) = corner(dx, dy);
            gs.write_reg(0x02, s.to_bits() as u64 | ((t.to_bits() as u64) << 32));
            gs.write_reg(0x01, 0x3F80_0000_8000_0000 | 0x80); // RGBAQ: black, alpha 128, q 1
            gs.write_reg(0x05, xyz(x.min(W - 1), y.min(H - 1), 0x80_0000));
        }
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
    // Amagami's in-game setup: Z writes masked (ZTE ALWAYS), no frame mask,
    // modulate by a non-neutral colour; its blended layers alpha-test.
    gs.write_reg(0x4E, 280 | (1 << 24) | (1 << 32)); // ZBUF_1: ZMSK
    gs.write_reg(0x01, 0x3F80_0000_8060_7080); // RGBAQ: tinted modulate
    bench("game sprite psmt8 nearest", &mut gs, 0x16, 0, tex0_psmt8(), sprite);
    gs.write_reg(0x01, 0x3F80_0000_8080_8080); // RGBAQ: unity
    gs.write_reg(0x47, 0x30000 | 1 | (6 << 1) | (0 << 4)); // TEST_1: ATE GREATER 0
    bench("game sprite psmt8 nearest+blend+ate", &mut gs, 0x56, 0, tex0_psmt8(), sprite);
    gs.write_reg(0x47, 0x30000);
    bench("game sprite psmct32 bilinear+blend", &mut gs, 0x56, 0x20, tex0_psmct32(), sprite);
    bench("game sprite psmt8 bilinear", &mut gs, 0x16, 0x20, tex0_psmt8(), sprite);
    bench("game tri psmt8 bilinear + blend", &mut gs, 0x5B, 0x20, tex0_psmt8(), quad);
    // Particles: Z test GEQUAL with writes masked, blend, rotated PSMCT16.
    gs.write_reg(0x47, 0x50000); // TEST_1: ZTE GEQUAL
    bench("game particles psmct16 bilinear+blend+z", &mut gs, 0x5B, 0x20, tex0_psmct16(), particles);
    gs.write_reg(0x47, 0x30000); // ZTE ALWAYS (masked: no Z traffic)
    bench("game particles psmct16 bilinear+blend", &mut gs, 0x5B, 0x20, tex0_psmct16(), particles);
    bench("game particles psmct16 bilinear", &mut gs, 0x1B, 0x20, tex0_psmct16(), particles);
    bench("game particles psmct16 nearest", &mut gs, 0x1B, 0, tex0_psmct16(), particles);
}
