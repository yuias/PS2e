//! GS local memory addressing.
//!
//! VRAM is kept the way the hardware organizes it — 8 KiB pages of 32
//! 256-byte blocks, blocks of 64-byte columns — so that formats overlaying
//! one region see each other's data: CLUTs packed four blocks apart, 8-bit
//! textures parked over a Z buffer, IMAGE uploads read back as another
//! format. Block tables follow the GS User's Manual (as in PCSX2's
//! GSLocalMemory); the intra-block column layouts are the closed forms of
//! its column tables. Buffers are addressed by block pointer `bp`, width
//! `bw` in 64-pixel units and pixel `(x, y)`; results wrap in 4 MiB.

use super::VRAM_SIZE;

/// 32-bit page: 64x32 pixels, 8x4 blocks of 8x8. Also used by PSMT8H/4HL/4HH.
const BLOCK32: [[u32; 8]; 4] = [
    [0, 1, 4, 5, 16, 17, 20, 21],
    [2, 3, 6, 7, 18, 19, 22, 23],
    [8, 9, 12, 13, 24, 25, 28, 29],
    [10, 11, 14, 15, 26, 27, 30, 31],
];
/// 16-bit page: 64x64 pixels, 4x8 blocks of 16x8.
const BLOCK16: [[u32; 4]; 8] = [
    [0, 2, 8, 10],
    [1, 3, 9, 11],
    [4, 6, 12, 14],
    [5, 7, 13, 15],
    [16, 18, 24, 26],
    [17, 19, 25, 27],
    [20, 22, 28, 30],
    [21, 23, 29, 31],
];
const BLOCK16S: [[u32; 4]; 8] = [
    [0, 2, 16, 18],
    [1, 3, 17, 19],
    [8, 10, 24, 26],
    [9, 11, 25, 27],
    [4, 6, 20, 22],
    [5, 7, 21, 23],
    [12, 14, 28, 30],
    [13, 15, 29, 31],
];
/// 8-bit page: 128x64 pixels, 8x4 blocks of 16x16 (same block order as 32-bit).
const BLOCK8: [[u32; 8]; 4] = BLOCK32;
/// 4-bit page: 128x128 pixels, 4x8 blocks of 32x16 (same block order as 16-bit).
const BLOCK4: [[u32; 4]; 8] = BLOCK16;
/// The Z formats use the colour tables with block bits 3 and 4 flipped.
const Z_FLIP: u32 = 24;

const BYTE_MASK: usize = VRAM_SIZE - 1;

/// Byte address of a 32-bit pixel (PSMCT32/24, PSMT8H/4HL/4HH; `z` for PSMZ32/24).
#[inline(always)]
pub(super) fn addr32(bp: u32, bw: u32, x: u32, y: u32, z: bool) -> usize {
    let bw = bw.max(1);
    let block = bp
        .wrapping_add(((y >> 5) * bw + (x >> 6)) * 32)
        .wrapping_add(BLOCK32[((y >> 3) & 3) as usize][((x >> 3) & 7) as usize] ^ if z { Z_FLIP } else { 0 });
    let col = ((y & 7) >> 1) * 16 + ((x & 7) >> 1) * 4 + (y & 1) * 2 + (x & 1);
    ((block as usize) * 256 + col as usize * 4) & BYTE_MASK
}

/// Byte address of a 16-bit pixel (`s` for PSMCT16S/PSMZ16S, `z` for the Z formats).
#[inline(always)]
pub(super) fn addr16(bp: u32, bw: u32, x: u32, y: u32, s: bool, z: bool) -> usize {
    let bw = bw.max(1);
    let table = if s { &BLOCK16S } else { &BLOCK16 };
    let block = bp
        .wrapping_add(((y >> 6) * bw + (x >> 6)) * 32)
        .wrapping_add(table[((y >> 3) & 7) as usize][((x >> 4) & 3) as usize] ^ if z { Z_FLIP } else { 0 });
    let col = ((y & 7) >> 1) * 32 + (y & 1) * 4 + ((x & 7) >> 1) * 8 + (x & 1) * 2 + ((x & 15) >> 3);
    ((block as usize) * 256 + col as usize * 2) & BYTE_MASK
}

/// Byte address of an 8-bit texel (PSMT8).
#[inline(always)]
pub(super) fn addr8(bp: u32, bw: u32, x: u32, y: u32) -> usize {
    let bw = bw.max(1);
    let block = bp
        .wrapping_add(((y >> 6) * (bw >> 1) + (x >> 7)) * 32)
        .wrapping_add(BLOCK8[((y >> 4) & 3) as usize][((x >> 4) & 7) as usize]);
    // Columns are 16x4; odd column pairs swap their 4-texel halves.
    let c = (y & 15) >> 2;
    let ry = y & 3;
    let swap = ((ry >> 1) ^ (c & 1)) & 1;
    let xs = (x & 15) ^ (swap << 2);
    let col = c * 64 + (ry & 1) * 8 + (ry >> 1) + ((xs >> 1) & 3) * 16 + (xs & 1) * 4 + (xs >> 3) * 2;
    ((block as usize) * 256 + col as usize) & BYTE_MASK
}

/// Nibble address of a 4-bit texel (PSMT4): byte `>> 1`, low nibble when even.
#[inline(always)]
pub(super) fn addr4(bp: u32, bw: u32, x: u32, y: u32) -> usize {
    let bw = bw.max(1);
    let block = bp
        .wrapping_add(((y >> 7) * (bw >> 1) + (x >> 7)) * 32)
        .wrapping_add(BLOCK4[((y >> 4) & 7) as usize][((x >> 5) & 3) as usize]);
    // Columns are 32x4 with the same half-swap pattern as PSMT8.
    let c = (y & 15) >> 2;
    let ry = y & 3;
    let swap = ((ry >> 1) ^ (c & 1)) & 1;
    let xs = (x & 31) ^ (swap << 2);
    let col =
        c * 128 + (ry & 1) * 16 + (ry >> 1) + ((xs >> 1) & 3) * 32 + (xs & 1) * 8 + ((xs >> 3) & 3) * 2;
    ((block as usize) * 512 + col as usize) & (BYTE_MASK * 2 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Every pixel of a page maps to a distinct address inside that page.
    fn bijective(pw: u32, ph: u32, bytes_per_page: usize, f: impl Fn(u32, u32) -> usize) {
        let mut seen = HashSet::new();
        for y in 0..ph {
            for x in 0..pw {
                let a = f(x, y);
                assert!(a < bytes_per_page, "({x},{y}) -> {a}");
                assert!(seen.insert(a), "({x},{y}) collides at {a}");
            }
        }
        assert_eq!(seen.len(), (pw * ph) as usize);
    }

    #[test]
    fn page_layouts_are_bijective() {
        bijective(64, 32, 8192, |x, y| addr32(0, 1, x, y, false));
        bijective(64, 32, 8192, |x, y| addr32(0, 1, x, y, true));
        bijective(64, 64, 8192, |x, y| addr16(0, 1, x, y, false, false));
        bijective(64, 64, 8192, |x, y| addr16(0, 1, x, y, true, false));
        bijective(64, 64, 8192, |x, y| addr16(0, 1, x, y, false, true));
        bijective(128, 64, 8192, |x, y| addr8(0, 2, x, y));
        bijective(128, 128, 16384, |x, y| addr4(0, 2, x, y));
    }

    /// Spot checks against the GS column tables (word/short/byte/nibble
    /// indices inside a block).
    #[test]
    fn column_tables() {
        let col32 = |x, y| addr32(0, 1, x, y, false) / 4;
        assert_eq!((0..8).map(|x| col32(x, 0)).collect::<Vec<_>>(), [0, 1, 4, 5, 8, 9, 12, 13]);
        assert_eq!((0..8).map(|x| col32(x, 1)).collect::<Vec<_>>(), [2, 3, 6, 7, 10, 11, 14, 15]);
        assert_eq!((0..8).map(|x| col32(x, 5)).collect::<Vec<_>>(), [34, 35, 38, 39, 42, 43, 46, 47]);

        let col16 = |x, y| addr16(0, 1, x, y, false, false) / 2;
        assert_eq!(
            (0..16).map(|x| col16(x, 0)).collect::<Vec<_>>(),
            [0, 2, 8, 10, 16, 18, 24, 26, 1, 3, 9, 11, 17, 19, 25, 27]
        );
        assert_eq!(
            (0..16).map(|x| col16(x, 3)).collect::<Vec<_>>(),
            [36, 38, 44, 46, 52, 54, 60, 62, 37, 39, 45, 47, 53, 55, 61, 63]
        );

        let col8 = |x, y| addr8(0, 2, x, y);
        assert_eq!(
            (0..16).map(|x| col8(x, 0)).collect::<Vec<_>>(),
            [0, 4, 16, 20, 32, 36, 48, 52, 2, 6, 18, 22, 34, 38, 50, 54]
        );
        assert_eq!(
            (0..16).map(|x| col8(x, 2)).collect::<Vec<_>>(),
            [33, 37, 49, 53, 1, 5, 17, 21, 35, 39, 51, 55, 3, 7, 19, 23]
        );
        assert_eq!(
            (0..16).map(|x| col8(x, 6)).collect::<Vec<_>>(),
            [65, 69, 81, 85, 97, 101, 113, 117, 67, 71, 83, 87, 99, 103, 115, 119]
        );
        assert_eq!(
            (0..16).map(|x| col8(x, 12)).collect::<Vec<_>>(),
            [224, 228, 240, 244, 192, 196, 208, 212, 226, 230, 242, 246, 194, 198, 210, 214]
        );

        let col4 = |x, y| addr4(0, 2, x, y);
        assert_eq!(
            (0..32).map(|x| col4(x, 0)).collect::<Vec<_>>(),
            [
                0, 8, 32, 40, 64, 72, 96, 104, 2, 10, 34, 42, 66, 74, 98, 106, 4, 12, 36, 44, 68,
                76, 100, 108, 6, 14, 38, 46, 70, 78, 102, 110
            ]
        );
        assert_eq!(
            (0..32).map(|x| col4(x, 3)).collect::<Vec<_>>(),
            [
                81, 89, 113, 121, 17, 25, 49, 57, 83, 91, 115, 123, 19, 27, 51, 59, 85, 93, 117,
                125, 21, 29, 53, 61, 87, 95, 119, 127, 23, 31, 55, 63
            ]
        );
        assert_eq!(
            (0..32).map(|x| col4(x, 5)).collect::<Vec<_>>(),
            [
                208, 216, 240, 248, 144, 152, 176, 184, 210, 218, 242, 250, 146, 154, 178, 186,
                212, 220, 244, 252, 148, 156, 180, 188, 214, 222, 246, 254, 150, 158, 182, 190
            ]
        );
    }

    /// Block placement: a 16x16 32-bit image (a CLUT) occupies exactly four
    /// consecutive blocks, and buffer width scales the page stride.
    #[test]
    fn block_placement() {
        let blocks: HashSet<usize> =
            (0..16).flat_map(|y| (0..16).map(move |x| addr32(10108, 1, x, y, false) / 256)).collect();
        assert_eq!(blocks, (10108..10112).collect());
        assert_eq!(addr32(0, 10, 64, 0, false) / 256, 32);
        assert_eq!(addr32(0, 10, 0, 32, false) / 256, 320);
        assert_eq!(addr32(0, 10, 0, 0, true) / 256, 24);
        assert_eq!(addr16(0, 10, 0, 64, false, false) / 256, 320);
        assert_eq!(addr8(0, 10, 0, 64) / 256, 160);
        assert_eq!(addr4(0, 10, 0, 128) / 512, 160);
    }
}
