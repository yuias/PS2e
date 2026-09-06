//! The sample-domain half of the decoder: the inverse DCT (ISO/IEC 13818-2
//! Annex A) and the IPU's colour-space conversion from 4:2:0 YCbCr to the
//! GS pixel formats.

use std::sync::LazyLock;

/// `C(u) cos((2x+1) u pi / 16) / 2`, indexed `[x][u]`: one dimension of the
/// separable transform, normalisation folded in.
static COS: LazyLock<[[f64; 8]; 8]> = LazyLock::new(|| {
    std::array::from_fn(|x| {
        std::array::from_fn(|u| {
            let c = if u == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
            c * (((2 * x + 1) * u) as f64 * std::f64::consts::PI / 16.0).cos() / 2.0
        })
    })
});

/// Inverse DCT of one block in place: coefficients in natural order
/// (`row * 8 + column`) in, samples saturated to the 9-bit range the
/// standard requires out. Double precision, so the IEEE 1180 accuracy
/// bound is met with room to spare.
pub fn idct(block: &mut [i32; 64]) {
    let cos = &*COS;
    let mut rows = [[0f64; 8]; 8];
    for v in 0..8 {
        for x in 0..8 {
            rows[v][x] = (0..8).map(|u| f64::from(block[v * 8 + u]) * cos[x][u]).sum();
        }
    }
    for y in 0..8 {
        for x in 0..8 {
            let s: f64 = (0..8).map(|v| rows[v][x] * cos[y][v]).sum();
            block[y * 8 + x] = (s.round() as i32).clamp(-256, 255);
        }
    }
}

/// One decoded macroblock as 8-bit samples: the IPU's `CSC` input layout,
/// 384 bytes in the order Y (16x16), Cb (8x8), Cr (8x8).
pub struct Samples {
    pub y: [u8; 256],
    pub cb: [u8; 64],
    pub cr: [u8; 64],
}

impl Samples {
    pub fn from_bytes(b: &[u8; 384]) -> Self {
        Self {
            y: b[..256].try_into().unwrap(),
            cb: b[256..320].try_into().unwrap(),
            cr: b[320..].try_into().unwrap(),
        }
    }
}

/// How the converter settles alpha and sign for a macroblock.
#[derive(Clone, Copy)]
pub struct Convert {
    /// `SETTH` thresholds: below `th[0]` on all three channels the pixel is
    /// cleared entirely, below `th[1]` its alpha drops to 0x40.
    pub th: [u16; 2],
    /// `IDEC.SGN`: flip the sign of every colour channel.
    pub sgn: bool,
}

/// Convert one macroblock to RGB32 (`A << 24 | B << 16 | G << 8 | R`),
/// raster order. The coefficients are the fixed-point BT.601 set the IPU
/// applies, in 1/128 units, with the chroma sample of each 2x2 cell used
/// for all four of its pixels.
pub fn to_rgb32(s: &Samples, cv: Convert) -> [u32; 256] {
    let mut out = [0u32; 256];
    for (i, o) in out.iter_mut().enumerate() {
        let (row, col) = (i / 16, i % 16);
        let y = i32::from(s.y[i]) - 16;
        let cb = i32::from(s.cb[(row / 2) * 8 + col / 2]) - 128;
        let cr = i32::from(s.cr[(row / 2) * 8 + col / 2]) - 128;
        let chan = |v: i32| ((v + 64) >> 7).clamp(0, 255) as u32;
        let r = chan(149 * y + 204 * cr);
        let g = chan(149 * y - 50 * cb - 104 * cr);
        let b = chan(149 * y + 258 * cb);
        let below = |t: u16| r < u32::from(t) && g < u32::from(t) && b < u32::from(t);
        let a = if cv.th[0] > 0 && below(cv.th[0]) {
            *o = 0;
            continue;
        } else if cv.th[1] > 0 && below(cv.th[1]) {
            0x40
        } else {
            0x80
        };
        let flip = if cv.sgn { 0x80 } else { 0 };
        *o = a << 24 | (b ^ flip) << 16 | (g ^ flip) << 8 | (r ^ flip);
    }
    out
}

/// The 4x4 ordered-dither offsets applied before truncating to 5 bits
/// (the GS's default DIMX matrix, which the IPU shares).
const DITHER: [[i32; 4]; 4] = [[-4, 0, -3, 1], [2, -2, 3, -1], [-3, 1, -4, 0], [3, -1, 2, -2]];

/// RGB32 to the GS's 16-bit `A1 B5 G5 R5`, optionally dithered. Alpha
/// keeps only its top bit: the half-transparent 0x40 has nowhere to go.
pub fn to_rgb16(rgb32: &[u32; 256], dte: bool) -> [u16; 256] {
    let mut out = [0u16; 256];
    for (i, &p) in rgb32.iter().enumerate() {
        let d = if dte { DITHER[(i / 16) & 3][i & 3] } else { 0 };
        let five = |shift: u32| ((((p >> shift) & 0xFF) as i32 + d).clamp(0, 255) >> 3) as u16;
        out[i] = ((p >> 31) as u16) << 15 | five(16) << 10 | five(8) << 5 | five(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_dc_only_block_is_flat() {
        // F(0,0) = 1024 is mid-grey for an intra block: every sample 128.
        let mut b = [0i32; 64];
        b[0] = 1024;
        idct(&mut b);
        assert!(b.iter().all(|&s| s == 128), "{b:?}");
        // The output saturates at the 9-bit range.
        b = [0; 64];
        b[0] = 4000;
        idct(&mut b);
        assert!(b.iter().all(|&s| s == 255));
        b = [0; 64];
        b[0] = -4000;
        idct(&mut b);
        assert!(b.iter().all(|&s| s == -256));
    }

    #[test]
    fn a_horizontal_cosine_splits_the_block_left_and_right() {
        let mut b = [0i32; 64];
        b[1] = 200;
        idct(&mut b);
        for row in b.chunks(8) {
            assert!(row[..4].iter().all(|&s| s > 0) && row[4..].iter().all(|&s| s < 0));
            // Symmetric about the centre, and every row identical.
            assert_eq!(row[0], -row[7]);
            assert_eq!(row, &b[..8]);
        }
    }

    fn flat(y: u8, cb: u8, cr: u8) -> Samples {
        Samples { y: [y; 256], cb: [cb; 64], cr: [cr; 64] }
    }

    #[test]
    fn video_range_black_and_white_land_on_the_limits() {
        let cv = Convert { th: [0, 0], sgn: false };
        assert!(to_rgb32(&flat(16, 128, 128), cv).iter().all(|&p| p == 0x8000_0000));
        assert!(to_rgb32(&flat(235, 128, 128), cv).iter().all(|&p| p == 0x80FF_FFFF));
        // Mid-grey and a pure red cast, per the BT.601 coefficients.
        assert_eq!(to_rgb32(&flat(128, 128, 128), cv)[0], 0x8082_8282);
        assert_eq!(to_rgb32(&flat(128, 128, 255), cv)[0], 0x8082_1BFF);
    }

    #[test]
    fn thresholds_and_sign_apply_after_conversion() {
        let dark = flat(40, 128, 128);
        let cv = Convert { th: [0, 0], sgn: false };
        let p = to_rgb32(&dark, cv)[0];
        assert_eq!(p, 0x801C_1C1C);
        // Below TH1 on every channel: half alpha. Below TH0: cleared.
        assert_eq!(to_rgb32(&dark, Convert { th: [0, 0x20], ..cv })[0], 0x401C_1C1C);
        assert_eq!(to_rgb32(&dark, Convert { th: [0x20, 0x40], ..cv })[0], 0);
        assert_eq!(to_rgb32(&dark, Convert { th: [0x10, 0x20], ..cv })[0], 0x401C_1C1C);
        assert_eq!(to_rgb32(&dark, Convert { sgn: true, ..cv })[0], 0x809C_9C9C);
    }

    #[test]
    fn rgb16_keeps_five_bits_and_the_top_of_alpha() {
        let mut px = [0x80FF_8000u32; 256];
        px[1] = 0x4000_00FF;
        let out = to_rgb16(&px, false);
        assert_eq!(out[0], 1 << 15 | 31 << 10 | 16 << 5);
        assert_eq!(out[1], 31);
        // Dithering nudges a value that sits just under a 5-bit boundary
        // up in the cells whose offset is positive, and never past 255.
        let px = [0x80FF_7F7Fu32; 256];
        let out = to_rgb16(&px, true);
        assert_eq!(out[0] & 0x1F, 15);
        assert_eq!(out[3] & 0x1F, 16);
        assert_eq!(out[0] >> 10 & 0x1F, 31);
    }
}
