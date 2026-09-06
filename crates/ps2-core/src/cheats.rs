//! Cheats in the pnach format: a text file of `patch=` lines, each a
//! command on EE or IOP memory applied once at start or once per frame.
//!
//! The format is the one PCSX2 documents at
//! <https://pcsx2.net/docs/advanced/writing-patches/>; this module is
//! written from that description. The `extended` code types are the RAW
//! codes of that page: plain writes (0-2), increment/decrement (3), the
//! strided multi-write (4), memory copy (5), pointer-chain write (6),
//! bitwise operations (7) and the conditional skip (D). Types the page
//! does not list are reported and skipped.
//!
//! Writes go through the debugger's `poke8` paths, which tell the
//! recompilers about pages they hold translated code for. Writing RAM
//! directly would leave stale blocks running.

use crate::bus::Bus;
use crate::Ps2System;
use tracing::{info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Ee,
    Iop,
}

/// The test a conditional (type D) applies to the value it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compare {
    Equal,
    NotEqual,
    Less,
    Greater,
    /// Not all the bits of `v` set.
    Nand,
    /// All the bits of `v` set.
    And,
    /// None of the bits of `v` set.
    Nor,
    /// Some bit of `v` set.
    Or,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bitwise {
    Or,
    And,
    Xor,
}

/// What one command does. `width` is in bytes; every multi-byte value is
/// little-endian in memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// Store `data` at `addr`.
    Write { addr: u32, data: Vec<u8> },
    /// Add `amount` (negative to decrement) to the `width`-byte value.
    Add { addr: u32, width: u8, amount: i64 },
    /// `count` 32-bit writes from `addr`, `stride` words apart, the value
    /// growing by `step` each time.
    Multi { addr: u32, count: u32, stride: u32, value: u32, step: u32 },
    /// Copy `len` bytes from `src` to `dst`.
    Copy { src: u32, dst: u32, len: u32 },
    /// Read a pointer at `addr`, follow it through `hops` (adding each
    /// offset and reading again), then write `value` at the final pointer
    /// plus `offset`.
    Pointer { addr: u32, hops: Vec<u32>, offset: u32, width: u8, value: u32 },
    /// Combine the `width`-byte value with `value`.
    Bitwise { addr: u32, width: u8, op: Bitwise, value: u32 },
    /// Read the `width`-byte value and, when the comparison fails, skip
    /// the next `skip` commands.
    Cond { addr: u32, width: u8, cmp: Compare, value: u32, skip: usize },
}

/// One command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cheat {
    pub target: Target,
    /// `place` 0 or 3: applied once, at start. Otherwise every frame.
    pub once: bool,
    pub op: Op,
}

/// A `patch=` line's fields, before the type is interpreted.
struct Line {
    once: bool,
    target: Target,
    addr: u64,
    kind: String,
    data: String,
    number: usize,
}

/// Parse a pnach file. Every line the parser rejects becomes a warning
/// naming its line number; the rest of the file is still used.
pub fn parse(text: &str) -> (Vec<Cheat>, Vec<String>) {
    let mut cheats = Vec::new();
    let mut warnings: Vec<(usize, String)> = Vec::new();
    let mut lines = Vec::new();
    for (i, raw) in text.trim_start_matches('\u{FEFF}').lines().enumerate() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('[') {
            continue;
        }
        let Some((key, rest)) = line.split_once('=') else {
            warnings.push((i + 1, "not a key=value line".into()));
            continue;
        };
        if !key.trim().eq_ignore_ascii_case("patch") {
            // gametitle=, author=, comment=, description=, gs*=: metadata.
            continue;
        }
        match parse_fields(rest, i + 1) {
            Ok(l) => lines.push(l),
            Err(e) => warnings.push((i + 1, e)),
        }
    }
    // Extended codes may continue onto following lines, so they are
    // assembled over the list rather than line by line.
    let mut it = lines.into_iter();
    while let Some(l) = it.next() {
        let number = l.number;
        match parse_command(l, &mut it) {
            Ok(c) => cheats.push(c),
            Err(e) => warnings.push((number, e)),
        }
    }
    warnings.sort_by_key(|w| w.0);
    (cheats, warnings.into_iter().map(|(n, e)| format!("line {n}: {e}")).collect())
}

/// `place,cpu,address,type,data`. Note that `place` 0 is "once at
/// start", not "off": a pnach has no way to switch a line off.
fn parse_fields(rest: &str, number: usize) -> Result<Line, String> {
    let f: Vec<&str> = rest.split(',').map(str::trim).collect();
    if f.len() != 5 {
        return Err(format!("expected 5 comma-separated fields, got {}", f.len()));
    }
    let once = match f[0] {
        "0" | "3" => true,
        "1" | "2" => false,
        p => return Err(format!("unknown place '{p}'")),
    };
    let target = match f[1].to_ascii_uppercase().as_str() {
        "EE" => Target::Ee,
        "IOP" => Target::Iop,
        c => return Err(format!("unknown cpu '{c}'")),
    };
    let addr = hex(f[2]).map_err(|e| format!("address: {e}"))?;
    Ok(Line { once, target, addr, kind: f[3].to_ascii_lowercase(), data: f[4].to_string(), number })
}

fn parse_command(l: Line, rest: &mut std::vec::IntoIter<Line>) -> Result<Cheat, String> {
    let Line { once, target, addr, kind, data, .. } = l;
    let op = match kind.as_str() {
        "byte" | "short" | "word" | "double" | "beshort" | "beword" | "bedouble" => {
            let width = match kind.trim_start_matches("be") {
                "byte" => 1,
                "short" => 2,
                "word" => 4,
                _ => 8,
            };
            let v = hex(&data).map_err(|e| format!("data: {e}"))?;
            let mut bytes = v.to_le_bytes()[..width].to_vec();
            if kind.starts_with("be") {
                bytes.reverse();
            }
            Op::Write { addr: addr as u32, data: bytes }
        }
        "bytes" => {
            let s = data.trim_start_matches("0x");
            if s.is_empty() || !s.len().is_multiple_of(2) {
                return Err("bytes: need an even number of hex digits".into());
            }
            let bytes = (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| "bytes: not hex".to_string()))
                .collect::<Result<Vec<u8>, _>>()?;
            Op::Write { addr: addr as u32, data: bytes }
        }
        "extended" => {
            let data = hex(&data).map_err(|e| format!("data: {e}"))? as u32;
            // Continuation lines of a multi-line code, as raw word pairs.
            let mut next = || -> Result<(u32, u32), String> {
                let l = rest.next().ok_or("the code needs another line and the file ended")?;
                let v = hex(&l.data).map_err(|e| format!("continuation data: {e}"))?;
                Ok((l.addr as u32, v as u32))
            };
            extended(addr as u32, data, &mut next)?
        }
        t => return Err(format!("unknown type '{t}'")),
    };
    Ok(Cheat { target, once, op })
}

/// One RAW code from its first line's two words; `next` supplies the
/// following lines' words when the type takes more.
fn extended(a: u32, d: u32, next: &mut dyn FnMut() -> Result<(u32, u32), String>) -> Result<Op, String> {
    let addr = a & 0x0FFF_FFFF;
    Ok(match a >> 28 {
        0 => Op::Write { addr, data: vec![d as u8] },
        1 => Op::Write { addr, data: d.to_le_bytes()[..2].to_vec() },
        2 => Op::Write { addr, data: d.to_le_bytes().to_vec() },
        // 30s0vvvv 0aaaaaaa: the address is in the data word; a 32-bit
        // amount does not fit and comes on a second line.
        3 => {
            let s = (a >> 20) & 0xF;
            let (width, amount) = match s {
                0 | 1 => (1, u64::from(a & 0xFF)),
                2 | 3 => (2, u64::from(a & 0xFFFF)),
                4 | 5 => (4, u64::from(next()?.0)),
                _ => return Err(format!("increment code with unknown operation {s}")),
            };
            let amount = amount as i64;
            Op::Add { addr: d & 0x0FFF_FFFF, width, amount: if s % 2 == 1 { -amount } else { amount } }
        }
        // 4aaaaaaa nnnnssss / vvvvvvvv iiiiiiii
        4 => {
            let (value, step) = next()?;
            Op::Multi { addr, count: d >> 16, stride: d & 0xFFFF, value, step }
        }
        // 5sssssss nnnnnnnn / 0ddddddd 00000000
        5 => {
            let (dst, _) = next()?;
            Op::Copy { src: addr, dst: dst & 0x0FFF_FFFF, len: d }
        }
        // 6aaaaaaa vvvvvvvv / 000snnnn t0 / t1 t2 / ... / t(n-2) iiiiiiii:
        // n pointer reads, so n-1 offsets between them and one after.
        6 => {
            let (head, first) = next()?;
            let width = match (head >> 16) & 0xF {
                0 => 1,
                1 => 2,
                2 => 4,
                s => return Err(format!("pointer code with unknown width {s}")),
            };
            let n = (head & 0xFFFF).max(1) as usize;
            let mut words = vec![first];
            while words.len() < n {
                let (x, y) = next()?;
                words.push(x);
                words.push(y);
            }
            words.truncate(n);
            let offset = words.pop().unwrap();
            Op::Pointer { addr, hops: words, offset, width, value: d }
        }
        // 7aaaaaaa 00x0vvvv
        7 => {
            let x = (d >> 20) & 0xF;
            let op = match x {
                0 | 1 => Bitwise::Or,
                2 | 3 => Bitwise::And,
                4 | 5 => Bitwise::Xor,
                _ => return Err(format!("bitwise code with unknown operation {x}")),
            };
            let width = if x.is_multiple_of(2) { 1 } else { 2 };
            Op::Bitwise { addr, width, op, value: d & if width == 1 { 0xFF } else { 0xFFFF } }
        }
        // Daaaaaaa nntsvvvv
        0xD => {
            let cmp = match (d >> 20) & 0xF {
                0 => Compare::Equal,
                1 => Compare::NotEqual,
                2 => Compare::Less,
                3 => Compare::Greater,
                4 => Compare::Nand,
                5 => Compare::And,
                6 => Compare::Nor,
                7 => Compare::Or,
                t => return Err(format!("conditional code with unknown comparison {t}")),
            };
            let width = match (d >> 16) & 0xF {
                0 => 2,
                1 => 1,
                s => return Err(format!("conditional code with unknown width {s}")),
            };
            let value = d & if width == 1 { 0xFF } else { 0xFFFF };
            Op::Cond { addr, width, cmp, value, skip: ((d >> 24) as usize).max(1) }
        }
        t => return Err(format!("extended code type {t:X} is not supported")),
    })
}

fn hex(s: &str) -> Result<u64, String> {
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(s, 16).map_err(|_| format!("'{s}' is not hex"))
}

/// The installed table and what has happened to it.
#[derive(Default)]
pub struct Table {
    cheats: Vec<Cheat>,
    /// Per command: a one-shot has fired, or a write was refused and warned.
    done: Vec<bool>,
    pub enabled: bool,
}

impl Table {
    /// Re-arm the one-shot entries, for a machine that starts over.
    pub(crate) fn rearm(&mut self) {
        self.done.fill(false);
    }

    pub fn len(&self) -> usize {
        self.cheats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cheats.is_empty()
    }

    /// Run the table once, in order, honouring conditional skips.
    fn apply(&mut self, bus: &mut Bus) {
        let mut i = 0;
        while i < self.cheats.len() {
            let c = &self.cheats[i];
            let at = i;
            i += 1;
            if self.done[at] {
                continue;
            }
            let mut mem = Memory { bus, target: c.target };
            match run(&mut mem, &c.op) {
                Ok(skip) => {
                    i += skip;
                    if c.once {
                        self.done[at] = true;
                    }
                }
                Err(addr) => {
                    warn!(target = ?c.target, addr = format_args!("{addr:#010x}"), "cheat touched an address outside RAM; dropping it");
                    self.done[at] = true;
                }
            }
        }
    }
}

/// Byte access to one CPU's memory. Every method fails with the address
/// that was not RAM.
struct Memory<'a> {
    bus: &'a mut Bus,
    target: Target,
}

impl Memory<'_> {
    fn read(&mut self, addr: u32, width: u8) -> Result<u64, u32> {
        let mut v = 0u64;
        for i in 0..u32::from(width) {
            let a = addr.wrapping_add(i);
            let b = match self.target {
                Target::Ee => self.bus.peek8(a),
                Target::Iop => self.bus.iop_peek8(a),
            };
            v |= u64::from(b.ok_or(a)?) << (8 * i);
        }
        Ok(v)
    }

    fn write(&mut self, addr: u32, width: u8, v: u64) -> Result<(), u32> {
        for i in 0..u32::from(width) {
            let a = addr.wrapping_add(i);
            let ok = match self.target {
                Target::Ee => self.bus.poke8(a, (v >> (8 * i)) as u8),
                Target::Iop => self.bus.iop_poke8(a, (v >> (8 * i)) as u8),
            };
            if !ok {
                return Err(a);
            }
        }
        Ok(())
    }
}

/// Execute one command; the result is how many commands to skip.
fn run(m: &mut Memory, op: &Op) -> Result<usize, u32> {
    match op {
        Op::Write { addr, data } => {
            for (i, &b) in data.iter().enumerate() {
                m.write(addr.wrapping_add(i as u32), 1, u64::from(b))?;
            }
        }
        Op::Add { addr, width, amount } => {
            let v = m.read(*addr, *width)?;
            m.write(*addr, *width, (v as i64).wrapping_add(*amount) as u64)?;
        }
        Op::Multi { addr, count, stride, value, step } => {
            let mut v = *value;
            for i in 0..*count {
                m.write(addr.wrapping_add(i * stride * 4), 4, u64::from(v))?;
                v = v.wrapping_add(*step);
            }
        }
        Op::Copy { src, dst, len } => {
            for i in 0..*len {
                let b = m.read(src.wrapping_add(i), 1)?;
                m.write(dst.wrapping_add(i), 1, b)?;
            }
        }
        Op::Pointer { addr, hops, offset, width, value } => {
            let mut p = m.read(*addr, 4)? as u32;
            for &t in hops {
                p = m.read(p.wrapping_add(t), 4)? as u32;
            }
            m.write(p.wrapping_add(*offset), *width, u64::from(*value))?;
        }
        Op::Bitwise { addr, width, op, value } => {
            let v = m.read(*addr, *width)?;
            let v = match op {
                Bitwise::Or => v | u64::from(*value),
                Bitwise::And => v & u64::from(*value),
                Bitwise::Xor => v ^ u64::from(*value),
            };
            m.write(*addr, *width, v)?;
        }
        Op::Cond { addr, width, cmp, value, skip } => {
            let v = m.read(*addr, *width)?;
            let x = u64::from(*value);
            let holds = match cmp {
                Compare::Equal => v == x,
                Compare::NotEqual => v != x,
                Compare::Less => v < x,
                Compare::Greater => v > x,
                Compare::Nand => v & x != x,
                Compare::And => v & x == x,
                Compare::Nor => v & x == 0,
                Compare::Or => v & x != 0,
            };
            if !holds {
                return Ok(*skip);
            }
        }
    }
    Ok(0)
}

impl Ps2System {
    /// Install a cheat table, re-arming the one-shot entries. Nothing is
    /// applied until the next frame boundary.
    pub fn set_cheats(&mut self, cheats: Vec<Cheat>) {
        if !cheats.is_empty() {
            info!(count = cheats.len(), "cheats installed");
        }
        self.cheats.done = vec![false; cheats.len()];
        self.cheats.cheats = cheats;
    }

    pub fn cheats(&self) -> &Table {
        &self.cheats
    }

    pub fn set_cheats_enabled(&mut self, on: bool) {
        self.cheats.enabled = on;
    }

    /// Called at the start of vertical blank. With no table installed this
    /// is one branch, which is all a hidden feature may cost.
    pub(crate) fn apply_cheats(&mut self) {
        if self.cheats.cheats.is_empty() || !self.cheats.enabled {
            return;
        }
        self.cheats.apply(&mut self.bus);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::BIOS_SIZE;
    use crate::Region;

    fn ee(op: Op) -> Cheat {
        Cheat { target: Target::Ee, once: false, op }
    }

    #[test]
    fn every_documented_line_shape_parses() {
        let text = "\u{FEFF}gametitle=Some Game // title\r\n\
                    [Infinite Health]\r\n\
                    author=someone\r\n\
                    \r\n\
                    patch=1,EE,00123456,word,0000abcd // hp\r\n\
                    patch=0,ee,0x00100000,short,1234\r\n\
                    patch=1,IOP,00001000,byte,7f\r\n\
                    patch=1,EE,00200000,double,0102030405060708\r\n\
                    patch=1,EE,00200008,beword,11223344\r\n\
                    patch=1,EE,00200010,bytes,deadbeef\r\n\
                    patch=1,EE,203E5320,extended,00004370\r\n\
                    patch=2,EE,10300000,extended,ffff1234\r\n\
                    patch=3,EE,00400000,extended,ab\r\n";
        let (c, w) = parse(text);
        assert!(w.is_empty(), "{w:?}");
        let write = |addr, data: &[u8], once| Cheat { target: Target::Ee, once, op: Op::Write { addr, data: data.to_vec() } };
        assert_eq!(c, vec![
            write(0x0012_3456, &[0xCD, 0xAB, 0, 0], false),
            write(0x0010_0000, &[0x34, 0x12], true),
            Cheat { target: Target::Iop, once: false, op: Op::Write { addr: 0x1000, data: vec![0x7F] } },
            write(0x0020_0000, &[8, 7, 6, 5, 4, 3, 2, 1], false),
            write(0x0020_0008, &[0x11, 0x22, 0x33, 0x44], false),
            write(0x0020_0010, &[0xDE, 0xAD, 0xBE, 0xEF], false),
            write(0x003E_5320, &[0x70, 0x43, 0, 0], false),
            write(0x0030_0000, &[0x34, 0x12], false),
            write(0x0040_0000, &[0xAB], true),
        ]);
    }

    #[test]
    fn multi_line_codes_take_their_continuation_lines() {
        let text = "patch=1,EE,30100005,extended,00100000\n\
                    patch=1,EE,3020ffff,extended,00100002\n\
                    patch=1,EE,30500000,extended,00100004\n\
                    patch=1,EE,00000010,extended,00000000\n\
                    patch=1,EE,40100000,extended,00030002\n\
                    patch=1,EE,00000001,extended,00000010\n\
                    patch=1,EE,50100000,extended,00000008\n\
                    patch=1,EE,00200000,extended,00000000\n\
                    patch=1,EE,60100000,extended,00000042\n\
                    patch=1,EE,00020003,extended,00000010\n\
                    patch=1,EE,00000020,extended,00000004\n\
                    patch=1,EE,60100000,extended,00000007\n\
                    patch=1,EE,00000001,extended,00000008\n\
                    patch=1,EE,70100000,extended,003000f0\n\
                    patch=1,EE,D0100000,extended,02100005\n\
                    patch=1,EE,D0100000,extended,000100ff\n";
        let (c, w) = parse(text);
        assert!(w.is_empty(), "{w:?}");
        assert_eq!(c, vec![
            ee(Op::Add { addr: 0x10_0000, width: 1, amount: -5 }),
            ee(Op::Add { addr: 0x10_0002, width: 2, amount: 0xFFFF }),
            ee(Op::Add { addr: 0x10_0004, width: 4, amount: -0x10 }),
            ee(Op::Multi { addr: 0x10_0000, count: 3, stride: 2, value: 1, step: 0x10 }),
            ee(Op::Copy { src: 0x10_0000, dst: 0x20_0000, len: 8 }),
            ee(Op::Pointer { addr: 0x10_0000, hops: vec![0x10, 0x20], offset: 4, width: 4, value: 0x42 }),
            ee(Op::Pointer { addr: 0x10_0000, hops: vec![], offset: 8, width: 1, value: 7 }),
            ee(Op::Bitwise { addr: 0x10_0000, width: 2, op: Bitwise::And, value: 0xF0 }),
            ee(Op::Cond { addr: 0x10_0000, width: 2, cmp: Compare::NotEqual, value: 5, skip: 2 }),
            ee(Op::Cond { addr: 0x10_0000, width: 1, cmp: Compare::Equal, value: 0xFF, skip: 1 }),
        ]);
    }

    #[test]
    fn bad_lines_are_reported_by_number_and_the_rest_kept() {
        let text = "patch=1,EE,00100000,word\n\
                    patch=9,EE,00100000,word,1\n\
                    patch=1,VU0,00100000,word,1\n\
                    patch=1,EE,zz,word,1\n\
                    patch=1,EE,00100000,float,1\n\
                    patch=1,EE,E0100000,extended,1\n\
                    patch=1,EE,00100000,bytes,abc\n\
                    what is this\n\
                    patch=1,EE,00100000,byte,5\n\
                    patch=1,EE,30400000,extended,00100000\n";
        let (c, w) = parse(text);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].op, Op::Write { addr: 0x10_0000, data: vec![5] });
        let lines: Vec<u32> = w.iter().map(|m| m.split(&[' ', ':']).nth(1).unwrap().parse().unwrap()).collect();
        assert_eq!(lines, [1, 2, 3, 4, 5, 6, 7, 8, 10]);
        assert!(w[5].contains("type E"), "{}", w[5]);
        assert!(w[4].contains("float"));
        assert!(w[8].contains("another line"), "{}", w[8]);
    }

    fn bus() -> Bus {
        Bus::new(vec![0u8; BIOS_SIZE], false, Region::Ntsc)
    }

    fn table(cheats: Vec<Cheat>) -> Table {
        Table { done: vec![false; cheats.len()], cheats, enabled: true }
    }

    #[test]
    fn each_code_type_does_what_the_page_says() {
        let mut b = bus();
        b.write32(0x10_0000, 0x0000_00F0);
        b.write32(0x10_0010, 0x0010_0100); // pointer to the block below
        b.write32(0x10_0100, 0x0010_0200); // hop 0 lands here
        let mut t = table(vec![
            ee(Op::Add { addr: 0x10_0000, width: 1, amount: -1 }),
            ee(Op::Add { addr: 0x10_0002, width: 2, amount: 0x102 }),
            ee(Op::Bitwise { addr: 0x10_0000, width: 1, op: Bitwise::Xor, value: 0xFF }),
            ee(Op::Multi { addr: 0x10_0020, count: 3, stride: 2, value: 7, step: 1 }),
            ee(Op::Copy { src: 0x10_0020, dst: 0x10_0040, len: 16 }),
            ee(Op::Pointer { addr: 0x10_0010, hops: vec![0], offset: 4, width: 2, value: 0xBEEF }),
        ]);
        t.apply(&mut b);
        assert_eq!(b.read32(0x10_0000), 0x0102_0010); // F0-1 = EF, ^FF = 10; +0x102 above
        assert_eq!(b.read32(0x10_0020), 7);
        assert_eq!(b.read32(0x10_0028), 8);
        assert_eq!(b.read32(0x10_0030), 9);
        assert_eq!(b.read32(0x10_0040), 7);
        assert_eq!(b.read32(0x10_0048), 8);
        assert_eq!(b.read32(0x10_0204), 0xBEEF);
    }

    #[test]
    fn a_failed_condition_skips_the_commands_it_counts() {
        let mut b = bus();
        b.write32(0x10_0000, 0x1234);
        let write = |addr, v| ee(Op::Write { addr, data: vec![v] });
        let mut t = table(vec![
            ee(Op::Cond { addr: 0x10_0000, width: 2, cmp: Compare::Equal, value: 0x1234, skip: 1 }),
            write(0x10_0010, 1), // taken
            ee(Op::Cond { addr: 0x10_0000, width: 2, cmp: Compare::Less, value: 0x1234, skip: 2 }),
            write(0x10_0011, 2), // skipped
            write(0x10_0012, 3), // skipped
            write(0x10_0013, 4), // taken
            ee(Op::Cond { addr: 0x10_0000, width: 1, cmp: Compare::And, value: 0x30, skip: 1 }),
            write(0x10_0014, 5), // taken: 0x34 has both bits
            ee(Op::Cond { addr: 0x10_0000, width: 1, cmp: Compare::Nor, value: 0x04, skip: 9 }),
            write(0x10_0015, 6), // skipped, and the skip runs off the end
        ]);
        t.apply(&mut b);
        assert_eq!(b.read32(0x10_0010), 0x0400_0001);
        assert_eq!(b.read32(0x10_0014), 0x0000_0005);
    }

    #[test]
    fn a_command_outside_ram_is_dropped_not_retried() {
        let mut b = bus();
        let mut t = table(vec![
            ee(Op::Write { addr: 0x1000_F000, data: vec![1] }),
            ee(Op::Write { addr: 0x10_0000, data: vec![1] }),
        ]);
        t.apply(&mut b);
        assert_eq!(t.done, [true, false]);
        assert_eq!(b.read32(0x10_0000), 1);
    }
}
