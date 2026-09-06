//! MPEG-2 variable-length codes, from ISO/IEC 13818-2 Annex B.
//!
//! Each table is written as the standard prints it, one `(code, length,
//! value)` per row, and expanded at first use into a flat array indexed by
//! the next `bits` bits of the stream. Building the array checks that no
//! code is a prefix of another, so a mistyped row fails loudly.

// The codes are grouped in fours from the left, as the standard prints
// them, so a row can be checked against the page.
#![allow(clippy::unusual_byte_groupings)]

use std::sync::LazyLock;

/// One lookup result: how many bits the code took (0 for an invalid code)
/// and what it meant.
#[derive(Clone, Copy, Default)]
pub struct Code {
    pub len: u8,
    pub val: u16,
}

pub struct Table {
    /// Bits of lookahead the array is indexed by; the longest code.
    pub bits: u32,
    entries: Box<[Code]>,
}

impl Table {
    fn build(bits: u32, rows: &[(u16, u8, u16)]) -> Table {
        let mut entries = vec![Code::default(); 1 << bits].into_boxed_slice();
        for &(code, len, val) in rows {
            let shift = bits - u32::from(len);
            let base = usize::from(code) << shift;
            for e in &mut entries[base..base + (1 << shift)] {
                assert_eq!(e.len, 0, "code {code:#b}/{len} overlaps another");
                *e = Code { len, val };
            }
        }
        Table { bits, entries }
    }

    /// Decode from `peeked`, the next `bits` bits of the stream.
    pub fn lookup(&self, peeked: u32) -> Code {
        self.entries[peeked as usize]
    }

    /// Bit patterns no code claims, in units of the longest code.
    #[cfg(test)]
    pub fn holes(&self) -> usize {
        self.entries.iter().filter(|e| e.len == 0).count()
    }
}

// --- macroblock layer ----------------------------------------------------

/// `macroblock_address_increment` value for `macroblock_stuffing`
/// (MPEG-1 only) and `macroblock_escape`, the two rows of Table B.1 that
/// carry no increment of their own.
pub const MBA_STUFFING: u16 = 34;
pub const MBA_ESCAPE: u16 = 35;

/// Table B.1, `macroblock_address_increment`.
pub static MBA: LazyLock<Table> = LazyLock::new(|| {
    Table::build(11, &[
        (0b1, 1, 1),
        (0b011, 3, 2),
        (0b010, 3, 3),
        (0b0011, 4, 4),
        (0b0010, 4, 5),
        (0b0001_1, 5, 6),
        (0b0001_0, 5, 7),
        (0b0000_111, 7, 8),
        (0b0000_110, 7, 9),
        (0b0000_1011, 8, 10),
        (0b0000_1010, 8, 11),
        (0b0000_1001, 8, 12),
        (0b0000_1000, 8, 13),
        (0b0000_0111, 8, 14),
        (0b0000_0110, 8, 15),
        (0b0000_0101_11, 10, 16),
        (0b0000_0101_10, 10, 17),
        (0b0000_0101_01, 10, 18),
        (0b0000_0101_00, 10, 19),
        (0b0000_0100_11, 10, 20),
        (0b0000_0100_10, 10, 21),
        (0b0000_0100_011, 11, 22),
        (0b0000_0100_010, 11, 23),
        (0b0000_0100_001, 11, 24),
        (0b0000_0100_000, 11, 25),
        (0b0000_0011_111, 11, 26),
        (0b0000_0011_110, 11, 27),
        (0b0000_0011_101, 11, 28),
        (0b0000_0011_100, 11, 29),
        (0b0000_0011_011, 11, 30),
        (0b0000_0011_010, 11, 31),
        (0b0000_0011_001, 11, 32),
        (0b0000_0011_000, 11, 33),
        (0b0000_0001_111, 11, MBA_STUFFING),
        (0b0000_0001_000, 11, MBA_ESCAPE),
    ])
});

/// `macroblock_type` flags, in the bit positions the IPU reports them.
pub const MB_INTRA: u16 = 1;
pub const MB_PATTERN: u16 = 2;
pub const MB_BACKWARD: u16 = 4;
pub const MB_FORWARD: u16 = 8;
pub const MB_QUANT: u16 = 16;

/// Table B.2, `macroblock_type` in I-pictures.
pub static MBT_I: LazyLock<Table> = LazyLock::new(|| {
    Table::build(2, &[(0b1, 1, MB_INTRA), (0b01, 2, MB_QUANT | MB_INTRA)])
});

/// Table B.3, `macroblock_type` in P-pictures.
pub static MBT_P: LazyLock<Table> = LazyLock::new(|| {
    Table::build(6, &[
        (0b1, 1, MB_FORWARD | MB_PATTERN),
        (0b01, 2, MB_PATTERN),
        (0b001, 3, MB_FORWARD),
        (0b0001_1, 5, MB_INTRA),
        (0b0001_0, 5, MB_QUANT | MB_FORWARD | MB_PATTERN),
        (0b0000_1, 5, MB_QUANT | MB_PATTERN),
        (0b0000_01, 6, MB_QUANT | MB_INTRA),
    ])
});

/// Table B.4, `macroblock_type` in B-pictures.
pub static MBT_B: LazyLock<Table> = LazyLock::new(|| {
    Table::build(6, &[
        (0b10, 2, MB_FORWARD | MB_BACKWARD),
        (0b11, 2, MB_FORWARD | MB_BACKWARD | MB_PATTERN),
        (0b010, 3, MB_BACKWARD),
        (0b011, 3, MB_BACKWARD | MB_PATTERN),
        (0b0010, 4, MB_FORWARD),
        (0b0011, 4, MB_FORWARD | MB_PATTERN),
        (0b0001_1, 5, MB_INTRA),
        (0b0001_0, 5, MB_QUANT | MB_FORWARD | MB_BACKWARD | MB_PATTERN),
        (0b0000_11, 6, MB_QUANT | MB_FORWARD | MB_PATTERN),
        (0b0000_10, 6, MB_QUANT | MB_BACKWARD | MB_PATTERN),
        (0b0000_01, 6, MB_QUANT | MB_INTRA),
    ])
});

/// Table B.5, `macroblock_type` in D-pictures.
pub static MBT_D: LazyLock<Table> = LazyLock::new(|| Table::build(1, &[(0b1, 1, MB_INTRA)]));

/// Table B.11, `dmvector`, as the signed value plus one.
pub static DMV: LazyLock<Table> = LazyLock::new(|| {
    Table::build(2, &[(0b0, 1, 1), (0b10, 2, 2), (0b11, 2, 0)])
});

/// Table B.9, `coded_block_pattern` (the 4:2:0 rows).
pub static CBP: LazyLock<Table> = LazyLock::new(|| {
    Table::build(9, &[
        (0b111, 3, 60),
        (0b1101, 4, 4),
        (0b1100, 4, 8),
        (0b1011, 4, 16),
        (0b1010, 4, 32),
        (0b1001_1, 5, 12),
        (0b1001_0, 5, 48),
        (0b1000_1, 5, 20),
        (0b1000_0, 5, 40),
        (0b0111_1, 5, 28),
        (0b0111_0, 5, 44),
        (0b0110_1, 5, 52),
        (0b0110_0, 5, 56),
        (0b0101_1, 5, 1),
        (0b0101_0, 5, 61),
        (0b0100_1, 5, 2),
        (0b0100_0, 5, 62),
        (0b0011_11, 6, 24),
        (0b0011_10, 6, 36),
        (0b0011_01, 6, 3),
        (0b0011_00, 6, 63),
        (0b0010_111, 7, 5),
        (0b0010_110, 7, 9),
        (0b0010_101, 7, 17),
        (0b0010_100, 7, 33),
        (0b0010_011, 7, 6),
        (0b0010_010, 7, 10),
        (0b0010_001, 7, 18),
        (0b0010_000, 7, 34),
        (0b0001_1111, 8, 7),
        (0b0001_1110, 8, 11),
        (0b0001_1101, 8, 19),
        (0b0001_1100, 8, 35),
        (0b0001_1011, 8, 13),
        (0b0001_1010, 8, 49),
        (0b0001_1001, 8, 21),
        (0b0001_1000, 8, 41),
        (0b0001_0111, 8, 14),
        (0b0001_0110, 8, 50),
        (0b0001_0101, 8, 22),
        (0b0001_0100, 8, 42),
        (0b0001_0011, 8, 15),
        (0b0001_0010, 8, 51),
        (0b0001_0001, 8, 23),
        (0b0001_0000, 8, 43),
        (0b0000_1111, 8, 25),
        (0b0000_1110, 8, 37),
        (0b0000_1101, 8, 26),
        (0b0000_1100, 8, 38),
        (0b0000_1011, 8, 29),
        (0b0000_1010, 8, 45),
        (0b0000_1001, 8, 53),
        (0b0000_1000, 8, 57),
        (0b0000_0111, 8, 30),
        (0b0000_0110, 8, 46),
        (0b0000_0101, 8, 54),
        (0b0000_0100, 8, 58),
        (0b0000_0011_1, 9, 31),
        (0b0000_0011_0, 9, 47),
        (0b0000_0010_1, 9, 55),
        (0b0000_0010_0, 9, 59),
        (0b0000_0001_1, 9, 27),
        (0b0000_0001_0, 9, 39),
        (0b0000_0000_1, 9, 0),
    ])
});

// --- block layer ---------------------------------------------------------

/// Table B.12, `dct_dc_size_luminance`.
pub static DC_LUMA: LazyLock<Table> = LazyLock::new(|| {
    Table::build(9, &[
        (0b100, 3, 0),
        (0b00, 2, 1),
        (0b01, 2, 2),
        (0b101, 3, 3),
        (0b110, 3, 4),
        (0b1110, 4, 5),
        (0b1111_0, 5, 6),
        (0b1111_10, 6, 7),
        (0b1111_110, 7, 8),
        (0b1111_1110, 8, 9),
        (0b1111_1111_0, 9, 10),
        (0b1111_1111_1, 9, 11),
    ])
});

/// Table B.13, `dct_dc_size_chrominance`.
pub static DC_CHROMA: LazyLock<Table> = LazyLock::new(|| {
    Table::build(10, &[
        (0b00, 2, 0),
        (0b01, 2, 1),
        (0b10, 2, 2),
        (0b110, 3, 3),
        (0b1110, 4, 4),
        (0b1111_0, 5, 5),
        (0b1111_10, 6, 6),
        (0b1111_110, 7, 7),
        (0b1111_1110, 8, 8),
        (0b1111_1111_0, 9, 9),
        (0b1111_1111_10, 10, 10),
        (0b1111_1111_11, 10, 11),
    ])
});

/// A DCT coefficient table entry: `run << 8 | level`, or one of the two
/// markers. The sign bit that follows every run/level code is not part
/// of the code and is read separately.
pub const DCT_EOB: u16 = 0xFFFF;
pub const DCT_ESCAPE: u16 = 0xFFFE;

const fn rl(run: u16, level: u16) -> u16 {
    run << 8 | level
}

/// Table B.14, DCT coefficients table zero. The `1s` code for the first
/// coefficient of a non-intra block (where End of Block cannot occur) is
/// handled by the caller; here `10` is EOB and `11s` is run 0, level 1.
static DCT_B14_ROWS: &[(u16, u8, u16)] = &[
    (0b10, 2, DCT_EOB),
    (0b11, 2, rl(0, 1)),
    (0b011, 3, rl(1, 1)),
    (0b0100, 4, rl(0, 2)),
    (0b0101, 4, rl(2, 1)),
    (0b0010_1, 5, rl(0, 3)),
    (0b0011_1, 5, rl(3, 1)),
    (0b0011_0, 5, rl(4, 1)),
    (0b0001_10, 6, rl(1, 2)),
    (0b0001_11, 6, rl(5, 1)),
    (0b0001_01, 6, rl(6, 1)),
    (0b0001_00, 6, rl(7, 1)),
    (0b0000_110, 7, rl(0, 4)),
    (0b0000_100, 7, rl(2, 2)),
    (0b0000_111, 7, rl(8, 1)),
    (0b0000_101, 7, rl(9, 1)),
    (0b0000_01, 6, DCT_ESCAPE),
    (0b0010_0110, 8, rl(0, 5)),
    (0b0010_0001, 8, rl(0, 6)),
    (0b0010_0101, 8, rl(1, 3)),
    (0b0010_0100, 8, rl(3, 2)),
    (0b0010_0111, 8, rl(10, 1)),
    (0b0010_0011, 8, rl(11, 1)),
    (0b0010_0010, 8, rl(12, 1)),
    (0b0010_0000, 8, rl(13, 1)),
    (0b0000_0010_10, 10, rl(0, 7)),
    (0b0000_0011_00, 10, rl(1, 4)),
    (0b0000_0010_11, 10, rl(2, 3)),
    (0b0000_0011_11, 10, rl(4, 2)),
    (0b0000_0010_01, 10, rl(5, 2)),
    (0b0000_0011_10, 10, rl(14, 1)),
    (0b0000_0011_01, 10, rl(15, 1)),
    (0b0000_0010_00, 10, rl(16, 1)),
    (0b0000_0001_1101, 12, rl(0, 8)),
    (0b0000_0001_1000, 12, rl(0, 9)),
    (0b0000_0001_0011, 12, rl(0, 10)),
    (0b0000_0001_0000, 12, rl(0, 11)),
    (0b0000_0001_1011, 12, rl(1, 5)),
    (0b0000_0001_0100, 12, rl(2, 4)),
    (0b0000_0001_1100, 12, rl(3, 3)),
    (0b0000_0001_0010, 12, rl(4, 3)),
    (0b0000_0001_1110, 12, rl(6, 2)),
    (0b0000_0001_0101, 12, rl(7, 2)),
    (0b0000_0001_0001, 12, rl(8, 2)),
    (0b0000_0001_1111, 12, rl(17, 1)),
    (0b0000_0001_1010, 12, rl(18, 1)),
    (0b0000_0001_1001, 12, rl(19, 1)),
    (0b0000_0001_0111, 12, rl(20, 1)),
    (0b0000_0001_0110, 12, rl(21, 1)),
    (0b0000_0000_1101_0, 13, rl(0, 12)),
    (0b0000_0000_1100_1, 13, rl(0, 13)),
    (0b0000_0000_1100_0, 13, rl(0, 14)),
    (0b0000_0000_1011_1, 13, rl(0, 15)),
    (0b0000_0000_1011_0, 13, rl(1, 6)),
    (0b0000_0000_1010_1, 13, rl(1, 7)),
    (0b0000_0000_1010_0, 13, rl(2, 5)),
    (0b0000_0000_1001_1, 13, rl(3, 4)),
    (0b0000_0000_1001_0, 13, rl(5, 3)),
    (0b0000_0000_1000_1, 13, rl(9, 2)),
    (0b0000_0000_1000_0, 13, rl(10, 2)),
    (0b0000_0000_1111_1, 13, rl(22, 1)),
    (0b0000_0000_1111_0, 13, rl(23, 1)),
    (0b0000_0000_1110_1, 13, rl(24, 1)),
    (0b0000_0000_1110_0, 13, rl(25, 1)),
    (0b0000_0000_1101_1, 13, rl(26, 1)),
    (0b0000_0000_0111_11, 14, rl(0, 16)),
    (0b0000_0000_0111_10, 14, rl(0, 17)),
    (0b0000_0000_0111_01, 14, rl(0, 18)),
    (0b0000_0000_0111_00, 14, rl(0, 19)),
    (0b0000_0000_0110_11, 14, rl(0, 20)),
    (0b0000_0000_0110_10, 14, rl(0, 21)),
    (0b0000_0000_0110_01, 14, rl(0, 22)),
    (0b0000_0000_0110_00, 14, rl(0, 23)),
    (0b0000_0000_0101_11, 14, rl(0, 24)),
    (0b0000_0000_0101_10, 14, rl(0, 25)),
    (0b0000_0000_0101_01, 14, rl(0, 26)),
    (0b0000_0000_0101_00, 14, rl(0, 27)),
    (0b0000_0000_0100_11, 14, rl(0, 28)),
    (0b0000_0000_0100_10, 14, rl(0, 29)),
    (0b0000_0000_0100_01, 14, rl(0, 30)),
    (0b0000_0000_0100_00, 14, rl(0, 31)),
    (0b0000_0000_0011_000, 15, rl(0, 32)),
    (0b0000_0000_0010_111, 15, rl(0, 33)),
    (0b0000_0000_0010_110, 15, rl(0, 34)),
    (0b0000_0000_0010_101, 15, rl(0, 35)),
    (0b0000_0000_0010_100, 15, rl(0, 36)),
    (0b0000_0000_0010_011, 15, rl(0, 37)),
    (0b0000_0000_0010_010, 15, rl(0, 38)),
    (0b0000_0000_0010_001, 15, rl(0, 39)),
    (0b0000_0000_0010_000, 15, rl(0, 40)),
    (0b0000_0000_0011_111, 15, rl(1, 8)),
    (0b0000_0000_0011_110, 15, rl(1, 9)),
    (0b0000_0000_0011_101, 15, rl(1, 10)),
    (0b0000_0000_0011_100, 15, rl(1, 11)),
    (0b0000_0000_0011_011, 15, rl(1, 12)),
    (0b0000_0000_0011_010, 15, rl(1, 13)),
    (0b0000_0000_0011_001, 15, rl(1, 14)),
    (0b0000_0000_0001_0011, 16, rl(1, 15)),
    (0b0000_0000_0001_0010, 16, rl(1, 16)),
    (0b0000_0000_0001_0001, 16, rl(1, 17)),
    (0b0000_0000_0001_0000, 16, rl(1, 18)),
    (0b0000_0000_0001_0100, 16, rl(6, 3)),
    (0b0000_0000_0001_1010, 16, rl(11, 2)),
    (0b0000_0000_0001_1001, 16, rl(12, 2)),
    (0b0000_0000_0001_1000, 16, rl(13, 2)),
    (0b0000_0000_0001_0111, 16, rl(14, 2)),
    (0b0000_0000_0001_0110, 16, rl(15, 2)),
    (0b0000_0000_0001_0101, 16, rl(16, 2)),
    (0b0000_0000_0001_1111, 16, rl(27, 1)),
    (0b0000_0000_0001_1110, 16, rl(28, 1)),
    (0b0000_0000_0001_1101, 16, rl(29, 1)),
    (0b0000_0000_0001_1100, 16, rl(30, 1)),
    (0b0000_0000_0001_1011, 16, rl(31, 1)),
];

/// The rows of Table B.15, DCT coefficients table one, that differ from
/// Table B.14. Every code of 12 bits or more is shared between the two
/// tables, so those come from B.14 unless the run/level pair was given a
/// shorter code here.
static DCT_B15_ROWS: &[(u16, u8, u16)] = &[
    (0b0110, 4, DCT_EOB),
    (0b10, 2, rl(0, 1)),
    (0b110, 3, rl(0, 2)),
    (0b0111, 4, rl(0, 3)),
    (0b1110_0, 5, rl(0, 4)),
    (0b1110_1, 5, rl(0, 5)),
    (0b0001_01, 6, rl(0, 6)),
    (0b0001_00, 6, rl(0, 7)),
    (0b1111_011, 7, rl(0, 8)),
    (0b1111_100, 7, rl(0, 9)),
    (0b0010_0011, 8, rl(0, 10)),
    (0b0010_0010, 8, rl(0, 11)),
    (0b1111_1010, 8, rl(0, 12)),
    (0b1111_1011, 8, rl(0, 13)),
    (0b1111_1110, 8, rl(0, 14)),
    (0b1111_1111, 8, rl(0, 15)),
    (0b010, 3, rl(1, 1)),
    (0b0001_10, 6, rl(1, 2)),
    (0b1111_001, 7, rl(1, 3)),
    (0b0010_0111, 8, rl(1, 4)),
    (0b0010_0000, 8, rl(1, 5)),
    (0b0010_1, 5, rl(2, 1)),
    (0b0000_111, 7, rl(2, 2)),
    (0b1111_1100, 8, rl(2, 3)),
    (0b0000_0011_00, 10, rl(2, 4)),
    (0b0011_1, 5, rl(3, 1)),
    (0b0010_0110, 8, rl(3, 2)),
    (0b0011_0, 5, rl(4, 1)),
    (0b1111_1101, 8, rl(4, 2)),
    (0b0001_11, 6, rl(5, 1)),
    (0b0000_0010_0, 9, rl(5, 2)),
    (0b0000_110, 7, rl(6, 1)),
    (0b0000_100, 7, rl(7, 1)),
    (0b0000_101, 7, rl(8, 1)),
    (0b1111_000, 7, rl(9, 1)),
    (0b1111_010, 7, rl(10, 1)),
    (0b0010_0001, 8, rl(11, 1)),
    (0b0010_0101, 8, rl(12, 1)),
    (0b0010_0100, 8, rl(13, 1)),
    (0b0000_0010_1, 9, rl(14, 1)),
    (0b0000_0011_1, 9, rl(15, 1)),
    (0b0000_0011_01, 10, rl(16, 1)),
    (0b0000_01, 6, DCT_ESCAPE),
];

pub static DCT_B14: LazyLock<Table> = LazyLock::new(|| Table::build(16, DCT_B14_ROWS));

pub static DCT_B15: LazyLock<Table> = LazyLock::new(|| {
    let mut rows = DCT_B15_ROWS.to_vec();
    rows.extend(DCT_B14_ROWS.iter().filter(|(_, len, val)| {
        *len >= 12 && !DCT_B15_ROWS.iter().any(|(_, _, v)| v == val)
    }));
    Table::build(16, &rows)
});

#[cfg(test)]
mod tests {
    use super::*;

    /// Every code space has exactly the gaps the standard leaves. Most are
    /// the all-zero pattern, kept free so no code can emulate a start
    /// code; B.1 also leaves `0000 0010 xxx` and the unused neighbours of
    /// escape and stuffing; B.15 vacates the long codes whose run/level
    /// pairs moved to short ones.
    #[test]
    fn the_tables_are_complete_prefix_codes() {
        assert_eq!(MBA.holes(), 8 + 8 + 3 + 3);
        for t in [&*MBT_I, &*MBT_P, &*MBT_B, &*MBT_D, &*CBP] {
            assert_eq!(t.holes(), 1);
        }
        for t in [&*DMV, &*DC_LUMA, &*DC_CHROMA] {
            assert_eq!(t.holes(), 0);
        }
        // B.14: the 16 sixteen-bit patterns under 0000 0000 0000.
        assert_eq!(DCT_B14.holes(), 16);
        // B.15: those, plus six 12-bit and four 13-bit codes vacated.
        assert_eq!(DCT_B15.holes(), 16 + 6 * 16 + 4 * 8);
        // Both hold every run/level pair exactly once.
        for t in [&*DCT_B14, &*DCT_B15] {
            let mut seen = std::collections::HashSet::new();
            for e in t.entries.iter().filter(|e| e.len > 0) {
                seen.insert(e.val);
            }
            assert_eq!(seen.len(), 113);
        }
    }
}
