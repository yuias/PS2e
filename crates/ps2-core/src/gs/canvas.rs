//! VRAM as a shareable canvas.
//!
//! The rasterizer may split a primitive's scanlines across worker threads,
//! each writing its own rows; the buffer is therefore accessed through raw
//! pointers behind `&self`. Soundness rests on the discipline in
//! `raster.rs`: concurrent writers touch disjoint pixels, and reads of a
//! region being written come only from the thread writing it. Everything
//! else (uploads, scanout, dumps) runs on the GS thread alone.

use super::{VRAM_SIZE, layout};

pub struct Canvas {
    ptr: *mut u8,
    /// Buffer size in bytes; a power of two, so `len - 1` wraps addresses.
    len: usize,
}

// SAFETY: see the module docs — the raster code keeps writers disjoint.
unsafe impl Send for Canvas {}
unsafe impl Sync for Canvas {}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self::with_size(VRAM_SIZE)
    }

    /// A canvas of `len` bytes (power of two): the real 4 MiB local memory,
    /// or the scaled shadow the internal-2x overlay draws into.
    pub fn with_size(len: usize) -> Self {
        debug_assert!(len.is_power_of_two());
        let buf: Box<[u8]> = vec![0u8; len].into_boxed_slice();
        Self { ptr: Box::into_raw(buf) as *mut u8, len }
    }

    /// Address wrap mask (`len - 1`).
    #[inline(always)]
    pub fn mask(&self) -> usize {
        self.len - 1
    }

    #[inline(always)]
    pub fn size(&self) -> usize {
        self.len
    }

    /// The whole buffer (only while no worker is writing).
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: ptr..ptr+len is our allocation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn to_vec(&self) -> Box<[u8]> {
        self.bytes().to_vec().into_boxed_slice()
    }

    // Offsets are always produced by `layout`, which wraps them into the
    // 4 MiB buffer at their own alignment, so the accesses stay in bounds.
    #[inline(always)]
    pub fn rd32(&self, o: usize) -> u32 {
        debug_assert!(o + 4 <= self.len);
        // SAFETY: in-bounds, unaligned read of our buffer.
        unsafe { (self.ptr.add(o) as *const u32).read_unaligned() }
    }
    #[inline(always)]
    pub fn wr32(&self, o: usize, v: u32) {
        debug_assert!(o + 4 <= self.len);
        // SAFETY: in-bounds, unaligned write of our buffer.
        unsafe { (self.ptr.add(o) as *mut u32).write_unaligned(v) }
    }
    #[inline(always)]
    pub fn rd64(&self, o: usize) -> u64 {
        debug_assert!(o + 8 <= self.len);
        // SAFETY: in-bounds, unaligned read of our buffer.
        unsafe { (self.ptr.add(o) as *const u64).read_unaligned() }
    }
    #[inline(always)]
    pub fn wr64(&self, o: usize, v: u64) {
        debug_assert!(o + 8 <= self.len);
        // SAFETY: in-bounds, unaligned write of our buffer.
        unsafe { (self.ptr.add(o) as *mut u64).write_unaligned(v) }
    }
    #[inline(always)]
    pub fn rd16(&self, o: usize) -> u16 {
        debug_assert!(o + 2 <= self.len);
        // SAFETY: in-bounds, unaligned read of our buffer.
        unsafe { (self.ptr.add(o) as *const u16).read_unaligned() }
    }
    #[inline(always)]
    pub fn wr16(&self, o: usize, v: u16) {
        debug_assert!(o + 2 <= self.len);
        // SAFETY: in-bounds, unaligned write of our buffer.
        unsafe { (self.ptr.add(o) as *mut u16).write_unaligned(v) }
    }
    #[inline(always)]
    pub fn rd8(&self, o: usize) -> u8 {
        debug_assert!(o < self.len);
        // SAFETY: in-bounds read of our buffer.
        unsafe { *self.ptr.add(o) }
    }
    #[inline(always)]
    pub fn wr8(&self, o: usize, v: u8) {
        debug_assert!(o < self.len);
        // SAFETY: in-bounds write of our buffer.
        unsafe { *self.ptr.add(o) = v }
    }

    // --- pixel accessors (block pointer, buffer width, x, y) --------------

    #[inline]
    pub fn write_psmct32(&self, bp: u32, bw: u32, x: u32, y: u32, v: u32) {
        self.wr32(layout::addr32(bp, bw, x, y, false) & self.mask(), v);
    }
    /// Replace only the `mask` bits of a 32-bit pixel.
    #[inline]
    pub fn write_psmct32_bits(&self, bp: u32, bw: u32, x: u32, y: u32, v: u32, mask: u32) {
        let o = layout::addr32(bp, bw, x, y, false) & self.mask();
        let cur = self.rd32(o);
        self.wr32(o, (cur & !mask) | (v & mask));
    }
    #[inline]
    pub fn read_psmct32(&self, bp: u32, bw: u32, x: u32, y: u32) -> u32 {
        self.rd32(layout::addr32(bp, bw, x, y, false) & self.mask())
    }
    /// PSMZ32/PSMZ24 word (Z buffers use their own block order).
    #[inline]
    pub fn write_psmz32(&self, bp: u32, bw: u32, x: u32, y: u32, v: u32) {
        self.wr32(layout::addr32(bp, bw, x, y, true) & self.mask(), v);
    }
    #[inline]
    pub fn read_psmz32(&self, bp: u32, bw: u32, x: u32, y: u32) -> u32 {
        self.rd32(layout::addr32(bp, bw, x, y, true) & self.mask())
    }
    /// 16-bit pixel in any of PSMCT16/16S/PSMZ16/16S (`psm` picks the block order).
    #[inline]
    pub fn write_psmct16(&self, bp: u32, bw: u32, x: u32, y: u32, psm: u32, v: u16) {
        self.wr16(layout::addr16(bp, bw, x, y, psm & 8 != 0, psm & 0x30 != 0) & self.mask(), v);
    }
    #[inline]
    pub fn read_psmct16(&self, bp: u32, bw: u32, x: u32, y: u32, psm: u32) -> u16 {
        self.rd16(layout::addr16(bp, bw, x, y, psm & 8 != 0, psm & 0x30 != 0) & self.mask())
    }
    #[inline]
    pub fn write_psmt8(&self, bp: u32, bw: u32, x: u32, y: u32, v: u8) {
        self.wr8(layout::addr8(bp, bw, x, y) & self.mask(), v);
    }
    #[inline]
    pub fn read_psmt8(&self, bp: u32, bw: u32, x: u32, y: u32) -> u8 {
        self.rd8(layout::addr8(bp, bw, x, y) & self.mask())
    }
    #[inline]
    pub fn write_psmt4(&self, bp: u32, bw: u32, x: u32, y: u32, v: u8) {
        let idx = layout::addr4(bp, bw, x, y);
        let o = (idx >> 1) & self.mask();
        let cur = self.rd8(o);
        if idx & 1 == 0 {
            self.wr8(o, (cur & 0xF0) | (v & 0xF));
        } else {
            self.wr8(o, (cur & 0x0F) | (v << 4));
        }
    }
    #[inline]
    pub fn read_psmt4(&self, bp: u32, bw: u32, x: u32, y: u32) -> u8 {
        let idx = layout::addr4(bp, bw, x, y);
        let b = self.rd8((idx >> 1) & self.mask());
        if idx & 1 == 0 { b & 0xF } else { b >> 4 }
    }
}

impl Drop for Canvas {
    fn drop(&mut self) {
        // SAFETY: reconstitutes the Box from `new`.
        unsafe { drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.ptr, self.len))) };
    }
}
