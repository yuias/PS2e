//! The texture unit's CLUT buffer: 1 KB that a TEX0/TEX2 write with CLD
//! set fills from local memory, and that palette lookups read. A texture
//! drawn with CLD=0 uses whatever the buffer holds, even after the memory
//! it was loaded from has been overwritten.
//!
//! Filling it eagerly would read local memory hundreds of millions of
//! times per gameplay minute (most CLD=1 writes name a 256-entry
//! palette), so a load is only *recorded* here. Its memory is read when
//! something could tell the difference: local memory about to change
//! under it, or a lookup that needs bytes from more than one load.

use std::ops::Range;

use serde::{Deserialize, Serialize};

/// Bytes in the buffer.
pub const BYTES: usize = 1024;
/// Recorded loads before the buffer is materialised to make room.
const MAX_LOADS: usize = 16;

/// One CLD-triggered load: where it reads from and which entries it
/// fills. Local memory has not changed since it was recorded.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct Load {
    pub cbp: u16,
    pub cpsm: u8,
    pub csm: u8,
    /// First entry, counted in entries of `cpsm`'s width (CSA * 16).
    pub first: u16,
    /// 16 for a 4-bit index, 256 for 8-bit.
    pub count: u16,
}

impl Load {
    /// Entry width in bytes: 32-bit palettes or 16-bit ones.
    pub fn width(&self) -> usize {
        if self.cpsm == 0 { 4 } else { 2 }
    }

    /// The entries this load fills, clipped to the buffer.
    pub fn entries(&self) -> Range<usize> {
        let end = (self.first as usize + self.count as usize).min(BYTES / self.width());
        (self.first as usize).min(end)..end
    }

    fn bytes(&self) -> Range<usize> {
        let e = self.entries();
        e.start * self.width()..e.end * self.width()
    }

    /// Local memory blocks the palette image occupies (16x16x32-bit).
    pub fn blocks(&self) -> Range<u32> {
        self.cbp as u32..self.cbp as u32 + 4
    }

    /// Bits that identify this load in a decoded-palette key.
    pub fn key_bits(&self) -> u64 {
        (self.cbp as u64)
            | ((self.cpsm as u64) << 14)
            | ((self.csm as u64) << 18)
            | ((self.first as u64) << 19)
            | (((self.count == 256) as u64) << 28)
    }
}

/// The entries a lookup reads: `first..first + count` in `cpsm`'s width.
#[derive(Clone, Copy, Debug)]
pub struct View {
    pub cpsm: u8,
    pub first: u16,
    pub count: u16,
}

impl View {
    fn width(&self) -> usize {
        if self.cpsm == 0 { 4 } else { 2 }
    }

    fn bytes(&self) -> Range<usize> {
        let w = self.width();
        let s = self.first as usize * w;
        s..(s + self.count as usize * w).min(BYTES)
    }
}

/// Where a lookup's entries come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Every entry from this one recorded load: read them from its memory.
    Load(Load),
    /// From the materialised bytes alone.
    Buffer,
    /// From the bytes and one or more loads: materialise first.
    Mixed,
}

/// What a TEX0 write asks of the unit, after CLD and CBP0/CBP1 are applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    None,
    Load(Load),
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Unit {
    /// The buffer as of the last materialisation, as little-endian words.
    buf: Vec<u32>,
    /// Loads recorded since then, oldest first; later ones overwrite.
    loads: Vec<Load>,
    /// Bumped whenever `buf` changes, so decoded palettes can key on it.
    epoch: u32,
    /// CLD 2-5's remembered buffer pointers. A 14-bit CBP never equals the
    /// initial value, so the first CLD=4/5 write always loads.
    cbp0: u16,
    cbp1: u16,
}

impl Default for Unit {
    fn default() -> Self {
        Self { buf: vec![0; BYTES / 4], loads: Vec::new(), epoch: 0, cbp0: u16::MAX, cbp1: u16::MAX }
    }
}

impl Unit {
    /// Apply a TEX0 (or TEX2-merged TEX0) write's CLD field. Returns the
    /// load the caller must [`record`](Self::record) — split so the caller
    /// can settle queued drawing into the palette's memory first.
    pub fn request(&mut self, tex0: u64) -> Request {
        let count = match (tex0 >> 20) & 0x3F {
            0x13 | 0x1B => 256,
            0x14 | 0x24 | 0x2C => 16,
            _ => return Request::None,
        };
        let cbp = ((tex0 >> 37) & 0x3FFF) as u16;
        let load = match (tex0 >> 61) & 7 {
            1 => true,
            2 => {
                self.cbp0 = cbp;
                true
            }
            3 => {
                self.cbp1 = cbp;
                true
            }
            4 if self.cbp0 != cbp => {
                self.cbp0 = cbp;
                true
            }
            5 if self.cbp1 != cbp => {
                self.cbp1 = cbp;
                true
            }
            _ => false,
        };
        if !load {
            return Request::None;
        }
        let csa = ((tex0 >> 56) & 0x1F) as u16;
        Request::Load(Load {
            cbp,
            cpsm: ((tex0 >> 51) & 0xF) as u8,
            csm: ((tex0 >> 55) & 1) as u8,
            // An 8-bit index reads the whole buffer; CSA is meaningless.
            first: if count == 256 { 0 } else { csa * 16 },
            count,
        })
    }

    /// Whether `record` would first have to materialise.
    pub fn full(&self) -> bool {
        self.loads.len() >= MAX_LOADS
    }

    /// Record a load. Loads it overwrites completely are dropped, so a
    /// program alternating palettes keeps the list at their number.
    pub fn record(&mut self, load: Load) {
        debug_assert!(!self.full());
        if self.loads.last() == Some(&load) {
            return;
        }
        let r = load.bytes();
        self.loads.retain(|o| {
            let ob = o.bytes();
            !(r.start <= ob.start && ob.end <= r.end)
        });
        self.loads.push(load);
    }

    /// Recorded loads whose memory lies in `blocks`.
    pub fn pending_in(&self, blocks: &Range<u32>) -> bool {
        self.loads.iter().any(|l| {
            let b = l.blocks();
            blocks.start < b.end && b.start < blocks.end
        })
    }

    pub fn pending(&self) -> bool {
        !self.loads.is_empty()
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    /// Where the entries of `view` come from.
    pub fn source(&self, view: &View) -> Source {
        let r = view.bytes();
        for l in self.loads.iter().rev() {
            let lb = l.bytes();
            if r.start < lb.end && lb.start < r.end {
                let covers = lb.start <= r.start && r.end <= lb.end;
                return if covers && l.width() == view.width() { Source::Load(*l) } else { Source::Mixed };
            }
        }
        Source::Buffer
    }

    /// Read every recorded load's memory into the buffer, oldest first.
    /// `read(load, e)` returns the raw entry `e` (of that load's width)
    /// of the palette image in local memory.
    pub fn materialise(&mut self, mut read: impl FnMut(&Load, u32) -> u32) {
        if self.loads.is_empty() {
            return;
        }
        for l in std::mem::take(&mut self.loads) {
            // `entries` is clipped to the buffer; the indexing below is
            // the check that it stays so.
            for e in l.entries() {
                let v = read(&l, e as u32);
                if l.width() == 4 {
                    self.buf[e] = v;
                } else {
                    let shift = (e & 1) * 16;
                    let w = &mut self.buf[e / 2];
                    *w = (*w & !(0xFFFF << shift)) | ((v & 0xFFFF) << shift);
                }
            }
        }
        self.epoch = self.epoch.wrapping_add(1);
    }

    /// Raw entry `e` of the materialised bytes in `cpsm`'s width; zero
    /// past the end of the buffer.
    pub fn entry(&self, cpsm: u8, e: usize) -> u32 {
        if cpsm == 0 {
            self.buf.get(e).copied().unwrap_or(0)
        } else {
            self.buf.get(e / 2).map_or(0, |w| (w >> ((e & 1) * 16)) & 0xFFFF)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tex0(psm: u64, cbp: u64, cpsm: u64, csa: u64, cld: u64) -> u64 {
        (psm << 20) | (cbp << 37) | (cpsm << 51) | (csa << 56) | (cld << 61)
    }

    fn load(u: &mut Unit, t: u64) -> Option<Load> {
        match u.request(t) {
            Request::Load(l) => {
                u.record(l);
                Some(l)
            }
            Request::None => None,
        }
    }

    #[test]
    fn cld_decides_whether_a_write_loads() {
        let mut u = Unit::default();
        assert!(load(&mut u, tex0(0x13, 100, 0, 0, 0)).is_none(), "CLD=0 leaves the buffer alone");
        assert!(load(&mut u, tex0(0x00, 100, 0, 0, 1)).is_none(), "a direct-colour texture has no palette");
        assert!(load(&mut u, tex0(0x13, 100, 0, 0, 1)).is_some());
        // 4: load only when CBP differs from CBP0, which starts unmatchable.
        assert!(load(&mut u, tex0(0x13, 200, 0, 0, 4)).is_some());
        assert!(load(&mut u, tex0(0x13, 200, 0, 0, 4)).is_none());
        assert!(load(&mut u, tex0(0x13, 201, 0, 0, 4)).is_some());
        // 2 loads and sets CBP0, so a following 4 with that CBP is quiet.
        assert!(load(&mut u, tex0(0x13, 300, 0, 0, 2)).is_some());
        assert!(load(&mut u, tex0(0x13, 300, 0, 0, 4)).is_none());
        // 5 and 3 pair the same way on CBP1, independent of CBP0.
        assert!(load(&mut u, tex0(0x13, 300, 0, 0, 5)).is_some());
        assert!(load(&mut u, tex0(0x13, 300, 0, 0, 5)).is_none());
        assert!(load(&mut u, tex0(0x13, 400, 0, 0, 3)).is_some());
        assert!(load(&mut u, tex0(0x13, 400, 0, 0, 5)).is_none());
        assert!(load(&mut u, tex0(0x13, 400, 0, 0, 4)).is_some());
        // Reserved values change nothing.
        assert!(load(&mut u, tex0(0x13, 500, 0, 0, 6)).is_none());
        assert!(load(&mut u, tex0(0x13, 500, 0, 0, 7)).is_none());
    }

    #[test]
    fn a_load_names_the_entries_it_fills() {
        let mut u = Unit::default();
        let l = load(&mut u, tex0(0x14, 100, 0, 3, 1)).unwrap();
        assert_eq!((l.first, l.count, l.width()), (48, 16, 4));
        assert_eq!(l.bytes(), 192..256);
        let l = load(&mut u, tex0(0x13, 100, 0x2, 5, 1)).unwrap();
        assert_eq!((l.first, l.count, l.width()), (0, 256, 2), "8-bit ignores CSA");
        assert_eq!(l.bytes(), 0..512);
        // A 4-bit window past the buffer's end is clipped, not wrapped.
        let l = load(&mut u, tex0(0x14, 100, 0, 31, 1)).unwrap();
        assert!(l.entries().is_empty());
    }

    #[test]
    fn repeated_and_overwritten_loads_do_not_accumulate() {
        let mut u = Unit::default();
        let a = tex0(0x13, 100, 0, 0, 1);
        let b = tex0(0x13, 200, 0, 0, 1);
        for _ in 0..40 {
            load(&mut u, a);
            load(&mut u, b);
        }
        assert_eq!(u.loads.len(), 1, "a full-buffer load erases what came before");
        let mut u = Unit::default();
        let c = tex0(0x14, 300, 0, 2, 1);
        let d = tex0(0x14, 300, 0, 3, 1);
        for _ in 0..40 {
            load(&mut u, c);
            load(&mut u, d);
        }
        assert_eq!(u.loads.len(), 2, "two windows alternate without growing");
        assert!(!u.full());
    }

    #[test]
    fn a_lookup_names_its_source() {
        let mut u = Unit::default();
        let v8 = View { cpsm: 0, first: 0, count: 256 };
        let v4 = |csa: u16| View { cpsm: 0, first: csa * 16, count: 16 };
        assert_eq!(u.source(&v8), Source::Buffer);
        let a = load(&mut u, tex0(0x13, 100, 0, 0, 1)).unwrap();
        assert_eq!(u.source(&v8), Source::Load(a));
        assert_eq!(u.source(&v4(3)), Source::Load(a), "a window inside the load reads it");
        let w = load(&mut u, tex0(0x14, 200, 0, 3, 1)).unwrap();
        assert_eq!(u.source(&v4(3)), Source::Load(w));
        assert_eq!(u.source(&v4(2)), Source::Load(a));
        assert_eq!(u.source(&v8), Source::Mixed, "the whole palette now straddles two loads");
        // A 16-bit view over a 32-bit load reads bytes, not entries.
        assert_eq!(u.source(&View { cpsm: 2, first: 0, count: 16 }), Source::Mixed);
        u.materialise(|_, _| 0);
        assert_eq!(u.source(&v8), Source::Buffer);
        assert!(!u.pending());
    }

    #[test]
    fn materialising_applies_loads_in_order_and_bumps_the_epoch() {
        let mut u = Unit::default();
        load(&mut u, tex0(0x13, 100, 0, 0, 1)); // 32-bit, whole buffer
        load(&mut u, tex0(0x14, 200, 2, 1, 1)); // 16-bit, entries 16..32
        let e0 = u.epoch();
        u.materialise(|l, e| if l.cbp == 100 { 0x1000_0000 + e } else { 0x8000 + e });
        assert_eq!(u.epoch(), e0 + 1);
        assert_eq!(u.entry(0, 5), 0x1000_0005);
        // The 16-bit window overwrote bytes 32..64 = words 8..16.
        assert_eq!(u.entry(2, 16), 0x8010);
        assert_eq!(u.entry(2, 31), 0x801F);
        assert_eq!(u.entry(0, 8), 0x8011_8010);
        assert_eq!(u.entry(0, 16), 0x1000_0010);
        assert_eq!(u.entry(0, 999), 0, "past the end reads zero");
        u.materialise(|_, _| 7);
        assert_eq!(u.epoch(), e0 + 1, "nothing recorded, nothing changes");
    }

    #[test]
    fn pending_loads_are_found_by_their_memory() {
        let mut u = Unit::default();
        load(&mut u, tex0(0x13, 100, 0, 0, 1));
        assert!(u.pending_in(&(102..110)));
        assert!(!u.pending_in(&(104..110)));
        assert!(!u.pending_in(&(90..100)));
    }
}
