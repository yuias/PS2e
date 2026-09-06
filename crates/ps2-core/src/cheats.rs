//! Cheats in the pnach format: a text file of `patch=` lines, each a
//! write to EE or IOP memory applied once at start or once per frame.
//!
//! The format is the one PCSX2 documents at
//! <https://pcsx2.net/docs/advanced/writing-patches/>; this parser is
//! written from that description. Of the `extended` code types only the
//! plain writes (0, 1, 2) are taken; the rest are reported and skipped.
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

/// One write.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cheat {
    pub target: Target,
    pub addr: u32,
    /// Bytes to store at `addr`, in memory order (little-endian for the
    /// sized types, as given for `bytes`).
    pub data: Vec<u8>,
    /// `place` 0 or 3: applied once, at start. Otherwise every frame.
    pub once: bool,
}

/// Parse a pnach file. Every line the parser rejects becomes a warning
/// naming its line number; the rest of the file is still used.
pub fn parse(text: &str) -> (Vec<Cheat>, Vec<String>) {
    let mut cheats = Vec::new();
    let mut warnings = Vec::new();
    for (i, raw) in text.trim_start_matches('\u{FEFF}').lines().enumerate() {
        let line = raw.split("//").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('[') {
            continue;
        }
        let Some((key, rest)) = line.split_once('=') else {
            warnings.push(format!("line {}: not a key=value line", i + 1));
            continue;
        };
        if !key.trim().eq_ignore_ascii_case("patch") {
            // gametitle=, author=, comment=, description=, gs*=: metadata.
            continue;
        }
        match parse_patch(rest) {
            Ok(c) => cheats.push(c),
            Err(e) => warnings.push(format!("line {}: {e}", i + 1)),
        }
    }
    (cheats, warnings)
}

/// `place,cpu,address,type,data`. Note that `place` 0 is "once at
/// start", not "off": a pnach has no way to switch a line off.
fn parse_patch(rest: &str) -> Result<Cheat, String> {
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
    let kind = f[3].to_ascii_lowercase();
    let (addr, data) = match kind.as_str() {
        "byte" | "short" | "word" | "double" | "beshort" | "beword" | "bedouble" => {
            let width = match kind.trim_start_matches("be") {
                "byte" => 1,
                "short" => 2,
                "word" => 4,
                _ => 8,
            };
            let v = hex(f[4]).map_err(|e| format!("data: {e}"))?;
            let mut bytes = v.to_le_bytes()[..width].to_vec();
            if kind.starts_with("be") {
                bytes.reverse();
            }
            (addr as u32, bytes)
        }
        "bytes" => {
            let s = f[4].trim_start_matches("0x");
            if s.is_empty() || !s.len().is_multiple_of(2) {
                return Err("bytes: need an even number of hex digits".into());
            }
            let bytes = (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| "bytes: not hex".to_string()))
                .collect::<Result<Vec<u8>, _>>()?;
            (addr as u32, bytes)
        }
        "extended" => {
            let width = match addr >> 28 {
                0 => 1,
                1 => 2,
                2 => 4,
                t => return Err(format!("extended code type {t:X} is not supported")),
            };
            let v = hex(f[4]).map_err(|e| format!("data: {e}"))?;
            ((addr & 0x0FFF_FFFF) as u32, v.to_le_bytes()[..width].to_vec())
        }
        t => return Err(format!("unknown type '{t}'")),
    };
    Ok(Cheat { target, addr, data, once })
}

fn hex(s: &str) -> Result<u64, String> {
    let s = s.trim_start_matches("0x").trim_start_matches("0X");
    u64::from_str_radix(s, 16).map_err(|_| format!("'{s}' is not hex"))
}

/// The installed table and what has happened to it.
#[derive(Default)]
pub struct Table {
    cheats: Vec<Cheat>,
    /// Per cheat: a one-shot has fired, or a write was refused and warned.
    done: Vec<bool>,
    pub enabled: bool,
}

impl Table {
    pub fn len(&self) -> usize {
        self.cheats.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cheats.is_empty()
    }
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
        let t = &mut self.cheats;
        for (c, done) in t.cheats.iter().zip(t.done.iter_mut()) {
            if *done {
                continue;
            }
            if !write(&mut self.bus, c) {
                warn!(target = ?c.target, addr = format_args!("{:#010x}", c.addr), "cheat write refused; dropping it");
                *done = true;
            } else if c.once {
                *done = true;
            }
        }
    }
}

/// Store the cheat's bytes; false when any of them landed outside RAM.
fn write(bus: &mut Bus, c: &Cheat) -> bool {
    c.data.iter().enumerate().all(|(i, &b)| {
        let a = c.addr.wrapping_add(i as u32);
        match c.target {
            Target::Ee => bus.poke8(a, b),
            Target::Iop => bus.iop_poke8(a, b),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let ee = |addr, data: &[u8], once| Cheat { target: Target::Ee, addr, data: data.to_vec(), once };
        assert_eq!(c, vec![
            ee(0x0012_3456, &[0xCD, 0xAB, 0, 0], false),
            ee(0x0010_0000, &[0x34, 0x12], true),
            Cheat { target: Target::Iop, addr: 0x1000, data: vec![0x7F], once: false },
            ee(0x0020_0000, &[8, 7, 6, 5, 4, 3, 2, 1], false),
            ee(0x0020_0008, &[0x11, 0x22, 0x33, 0x44], false),
            ee(0x0020_0010, &[0xDE, 0xAD, 0xBE, 0xEF], false),
            ee(0x003E_5320, &[0x70, 0x43, 0, 0], false),
            ee(0x0030_0000, &[0x34, 0x12], false),
            ee(0x0040_0000, &[0xAB], true),
        ]);
    }

    #[test]
    fn bad_lines_are_reported_by_number_and_the_rest_kept() {
        let text = "patch=1,EE,00100000,word\n\
                    patch=9,EE,00100000,word,1\n\
                    patch=1,VU0,00100000,word,1\n\
                    patch=1,EE,zz,word,1\n\
                    patch=1,EE,00100000,float,1\n\
                    patch=1,EE,30100000,extended,1\n\
                    patch=1,EE,00100000,bytes,abc\n\
                    what is this\n\
                    patch=1,EE,00100000,byte,5\n";
        let (c, w) = parse(text);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].data, [5]);
        let lines: Vec<u32> = w.iter().map(|m| m.split(&[' ', ':']).nth(1).unwrap().parse().unwrap()).collect();
        assert_eq!(lines, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(w[5].contains("type 3"), "{}", w[5]);
        assert!(w[4].contains("float"));
    }
}
