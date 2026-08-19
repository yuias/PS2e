//! GS (Graphics Synthesizer): software rasterizer.
//!
//! VRAM addressing is LINEAR for every pixel format, not the real page/
//! block/column swizzle. Draws, IMAGE transfers, texture sampling, CLUT
//! reads and scanout all share the same mapping, so rendering is self-
//! consistent; it only breaks if software aliases the same memory through
//! two formats. Revisit with real swizzle tables when that bites
//! (docs/ARCHITECTURE.md).

mod raster;

use tracing::{debug, trace, warn};

mod canvas;
pub mod front;
mod layout;

pub use canvas::Canvas;

pub use front::{Frame, GsFront, Stats};

pub const VRAM_SIZE: usize = 4 * 1024 * 1024;

// Pixel storage formats.
pub const PSMCT32: u32 = 0x00;
pub const PSMCT24: u32 = 0x01;
pub const PSMCT16: u32 = 0x02;
pub const PSMCT16S: u32 = 0x0A;
pub const PSMT8: u32 = 0x13;
pub const PSMT4: u32 = 0x14;
pub const PSMT8H: u32 = 0x1B;
pub const PSMT4HL: u32 = 0x24;
pub const PSMT4HH: u32 = 0x2C;
pub const PSMZ32: u32 = 0x30;
pub const PSMZ24: u32 = 0x31;
pub const PSMZ16: u32 = 0x32;
pub const PSMZ16S: u32 = 0x3A;

/// Per-pixel write callback used by IMAGE transfers.

/// One vertex as accumulated from register writes.
#[derive(Clone, Copy, Default)]
pub struct Vertex {
    /// 12.4 fixed point, already relative to XYOFFSET.
    pub x: i32,
    pub y: i32,
    pub z: u32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
    pub q: f32,
    pub s: f32,
    pub t: f32,
    /// 12.4 fixed texel coords (UV addressing).
    pub u: i32,
    pub v: i32,
}

#[derive(Default, Clone, Copy)]
pub struct Context {
    pub xyoffset: u64,
    pub scissor: u64,
    pub frame: u64,
    pub zbuf: u64,
    pub test: u64,
    pub alpha: u64,
    pub tex0: u64,
    pub tex1: u64,
    pub clamp: u64,
}

pub struct Gs {
    /// Local memory, shareable with rasterizer worker threads.
    pub canvas: Canvas,
    // Privileged registers.
    pub pmode: u64,
    pub smode1: u64,
    pub smode2: u64,
    pub dispfb1: u64,
    pub display1: u64,
    pub dispfb2: u64,
    pub display2: u64,
    pub bgcolor: u64,
    // General registers.
    pub prim: u64,
    pub prmode: u64,
    pub prmodecont: u64,
    pub rgbaq: u64,
    pub st: u64,
    pub uv: u64,
    pub texa: u64,
    pub fogcol: u64,
    pub colclamp: u64,
    pub dthe: u64,
    pub pabe: u64,
    pub ctx: [Context; 2],
    // Vertex queue.
    vq: [Vertex; 4],
    vq_len: usize,
    /// Total vertices kicked since PRIM (for strips/fans).
    vq_total: usize,
    // HOST -> LOCAL transfer state.
    pub bitbltbuf: u64,
    pub trxpos: u64,
    pub trxreg: u64,
    pub trxdir: u64,
    trx_x: u32,
    trx_y: u32,
    /// Partial pixel carried between 64-bit chunks of a packed 24-bit
    /// IMAGE stream (3 bytes per pixel, no chunk alignment).
    trx24: [u8; 3],
    trx24_len: u8,
    /// Statistics for bring-up logging.
    pub prims_drawn: u64,
    pub prims_textured: u64,
    /// Pixels that entered the shading pipeline (after scissor/coverage).
    pub pixels_shaded: u64,
    /// Primitives rasterized on the worker pool.
    pub prims_split: u64,
    /// Texture samples per TEX0 PSM, for bring-up logging.
    pub tex_psm_hist: [u64; 64],
    /// IMAGE transfer formats already reported as unhandled (bit per PSM).
    warned_trx_psm: u64,
    /// Distinct TEX0 values already logged (bring-up aid; capped).
    seen_tex0: std::collections::HashSet<u64>,
    /// Render-target setups already logged, with how many prims were shown.
    seen_targets: std::collections::HashMap<u64, u32>,
    /// Registers already reported as unhandled (warn once, not per write).
    warned_regs: [u64; 4],
    /// Rasterizer scratch for the GS thread and for the worker-pool bands
    /// (see raster.rs; empty pool = never split).
    scratch: raster::Scratch,
    pool: Vec<raster::Scratch>,
    /// Woven interlaced display for [`Gs::framebuffer_woven`], and its size.
    woven: Vec<u8>,
    /// Per-pixel motion flags of the last composited frame (see
    /// [`Deinterlace::Adaptive`]), one byte per pixel of `woven`.
    motion: Vec<u8>,
    woven_dims: (u32, u32),
    /// Decoded CLUT (RGBA8 per entry) for the last palette setup; entries
    /// beyond 256 serve 4-bit textures with a CSA offset into a 16-bit CLUT.
    clut: Box<[u32; 512]>,
    /// Palette setup (`TexInfo::clut_key`) the cache was decoded from.
    clut_key: u64,
    /// VRAM changed by a transfer since the CLUT was decoded.
    clut_dirty: bool,
}

impl Default for Gs {
    fn default() -> Self {
        Self::new()
    }
}

impl Gs {
    pub fn new() -> Self {
        Self {
            canvas: Canvas::new(),
            pmode: 0,
            smode1: 0,
            smode2: 0,
            dispfb1: 0,
            display1: 0,
            dispfb2: 0,
            display2: 0,
            bgcolor: 0,
            prim: 0,
            prmode: 0,
            prmodecont: 1,
            rgbaq: 0,
            st: 0,
            uv: 0,
            texa: 0,
            fogcol: 0,
            colclamp: 1,
            dthe: 0,
            pabe: 0,
            ctx: [Context::default(); 2],
            vq: [Vertex::default(); 4],
            vq_len: 0,
            vq_total: 0,
            bitbltbuf: 0,
            trxpos: 0,
            trxreg: 0,
            trxdir: 0,
            trx_x: 0,
            trx_y: 0,
            trx24: [0; 3],
            trx24_len: 0,
            prims_drawn: 0,
            prims_textured: 0,
            pixels_shaded: 0,
            prims_split: 0,
            tex_psm_hist: [0; 64],
            warned_trx_psm: 0,
            seen_tex0: std::collections::HashSet::new(),
            seen_targets: std::collections::HashMap::new(),
            warned_regs: [0; 4],
            scratch: raster::Scratch::default(),
            pool: if cfg!(feature = "threads") {
                (0..4).map(|_| raster::Scratch::default()).collect()
            } else {
                Vec::new()
            },
            woven: Vec::new(),
            motion: Vec::new(),
            woven_dims: (0, 0),
            clut: Box::new([0; 512]),
            clut_key: u64::MAX,
            clut_dirty: true,
        }
    }

    // --- display registers ----------------------------------------------

    /// Display-side privileged registers (PMODE, SMODE, DISPFB, DISPLAY,
    /// BGCOLOR); CSR/IMR live in [`GsFront`].
    pub fn priv_write(&mut self, addr: u32, v: u64) {
        match addr & 0x1FF0 {
            0x0000 => self.pmode = v,
            0x0010 => self.smode1 = v,
            0x0020 => self.smode2 = v,
            0x0070 => self.dispfb1 = v,
            0x0080 => self.display1 = v,
            0x0090 => self.dispfb2 = v,
            0x00A0 => self.display2 = v,
            0x00E0 => self.bgcolor = v,
            _ => {}
        }
    }

    // --- general registers ----------------------------------------------

    pub fn write_reg(&mut self, reg: u8, v: u64) {
        trace!(target: "ps2_core::gs::reg", reg = format_args!("{reg:#04x}"), value = format_args!("{v:#018x}"), "write");
        match reg {
            0x00 => {
                self.prim = v;
                self.vq_len = 0;
                self.vq_total = 0;
            }
            0x01 => self.rgbaq = v,
            0x02 => self.st = v,
            0x03 => self.uv = v,
            0x04 | 0x0C => {
                // XYZF2/XYZF3: F in bits 56-63, Z in bits 32-55.
                let z = ((v >> 32) & 0xFF_FFFF) as u32;
                self.vertex_kick(v, z, reg == 0x04);
            }
            0x05 | 0x0D => {
                let z = (v >> 32) as u32;
                self.vertex_kick(v, z, reg == 0x05);
            }
            0x06 | 0x07 => {
                self.ctx[(reg - 0x06) as usize].tex0 = v;
                self.log_tex0(v);
            }
            0x08 => self.ctx[0].clamp = v,
            0x09 => self.ctx[1].clamp = v,
            0x0A => {} // FOG
            0x14 => self.ctx[0].tex1 = v,
            0x15 => self.ctx[1].tex1 = v,
            0x16 | 0x17 => {
                // TEX2: partial TEX0 update (PSM + CLUT fields).
                let i = (reg - 0x16) as usize;
                const MASK: u64 = 0xFFFF_FFE0_03F0_0000;
                self.ctx[i].tex0 = (self.ctx[i].tex0 & !MASK) | (v & MASK);
                self.log_tex0(self.ctx[i].tex0);
            }
            0x18 => self.ctx[0].xyoffset = v,
            0x19 => self.ctx[1].xyoffset = v,
            0x1A => self.prmodecont = v,
            0x1B => self.prmode = v,
            0x1C => {}        // TEXCLUT
            0x22 => {}        // SCANMSK
            0x34..=0x37 => {} // MIPTBP
            0x3B => self.texa = v,
            0x3D => self.fogcol = v,
            0x3F => {} // TEXFLUSH
            0x40 => self.ctx[0].scissor = v,
            0x41 => self.ctx[1].scissor = v,
            0x42 => self.ctx[0].alpha = v,
            0x43 => self.ctx[1].alpha = v,
            0x44 => {} // DIMX
            0x45 => self.dthe = v,
            0x46 => self.colclamp = v,
            0x47 => self.ctx[0].test = v,
            0x48 => self.ctx[1].test = v,
            0x49 => self.pabe = v,
            0x4A => {} // FBA_1
            0x4B => {} // FBA_2
            0x4C => self.ctx[0].frame = v,
            0x4D => self.ctx[1].frame = v,
            0x4E => self.ctx[0].zbuf = v,
            0x4F => self.ctx[1].zbuf = v,
            0x50 => self.bitbltbuf = v,
            0x51 => self.trxpos = v,
            0x52 => self.trxreg = v,
            0x53 => {
                self.trxdir = v & 3;
                self.trx_x = 0;
                self.trx_y = 0;
                self.trx24_len = 0;
                debug!(target: "ps2_core::gs",
                    dir = v & 3,
                    sbp = self.bitbltbuf & 0x3FFF,
                    spsm = format_args!("{:#04x}", (self.bitbltbuf >> 24) & 0x3F),
                    dbp = (self.bitbltbuf >> 32) & 0x3FFF,
                    dbw = (self.bitbltbuf >> 48) & 0x3F,
                    dpsm = format_args!("{:#04x}", (self.bitbltbuf >> 56) & 0x3F),
                    dsax = (self.trxpos >> 32) & 0x7FF,
                    dsay = (self.trxpos >> 48) & 0x7FF,
                    rrw = self.trxreg & 0xFFF,
                    rrh = (self.trxreg >> 32) & 0xFFF,
                    prims = self.prims_drawn,
                    "TRXDIR");
                if v & 3 == 2 {
                    self.local_copy();
                }
            }
            0x54 => self.hwreg(v),
            _ => {
                let (slot, bit) = ((reg >> 6) as usize, reg & 63);
                if self.warned_regs[slot] & (1 << bit) == 0 {
                    self.warned_regs[slot] |= 1 << bit;
                    warn!(target: "ps2_core::gs", reg = format_args!("{reg:#04x}"), "unhandled GS register (reported once)");
                }
            }
        }
    }

    /// Attribute source: PRIM or PRMODE per PRMODECONT.
    #[inline]
    pub fn attrs(&self) -> u64 {
        if self.prmodecont & 1 != 0 {
            self.prim
        } else {
            (self.prmode & !0x7) | (self.prim & 0x7)
        }
    }

    #[inline]
    fn ctx_index(&self) -> usize {
        ((self.attrs() >> 9) & 1) as usize
    }

    fn vertex_kick(&mut self, xy: u64, z: u32, draw: bool) {
        let ctxi = self.ctx_index();
        let off = self.ctx[ctxi].xyoffset;
        let v = Vertex {
            x: (xy & 0xFFFF) as i32 - (off & 0xFFFF) as i32,
            y: ((xy >> 16) & 0xFFFF) as i32 - ((off >> 32) & 0xFFFF) as i32,
            z,
            r: self.rgbaq as u8,
            g: (self.rgbaq >> 8) as u8,
            b: (self.rgbaq >> 16) as u8,
            a: (self.rgbaq >> 24) as u8,
            q: f32::from_bits((self.rgbaq >> 32) as u32),
            s: f32::from_bits(self.st as u32),
            t: f32::from_bits((self.st >> 32) as u32),
            u: (self.uv & 0x3FFF) as i32,
            v: ((self.uv >> 16) & 0x3FFF) as i32,
        };
        if self.vq_len < self.vq.len() {
            self.vq[self.vq_len] = v;
            self.vq_len += 1;
        }
        self.vq_total += 1;

        let kind = (self.prim & 7) as u32;
        match kind {
            0 => {
                // Point.
                if draw {
                    self.draw_point();
                }
                self.vq_len = 0;
            }
            1 | 2 => {
                // Line / line strip: not needed for the boot screen yet.
                if self.vq_len == 2 {
                    if draw {
                        trace!(target: "ps2_core::gs", "line prim (not rasterized)");
                    }
                    if kind == 1 {
                        self.vq_len = 0;
                    } else {
                        self.vq[0] = self.vq[1];
                        self.vq_len = 1;
                    }
                }
            }
            3 => {
                if self.vq_len == 3 {
                    if draw {
                        self.draw_triangle(0, 1, 2);
                    }
                    self.vq_len = 0;
                }
            }
            4 => {
                // Triangle strip.
                if self.vq_len == 3 {
                    if draw {
                        self.draw_triangle(0, 1, 2);
                    }
                    self.vq[0] = self.vq[1];
                    self.vq[1] = self.vq[2];
                    self.vq_len = 2;
                }
            }
            5 => {
                // Triangle fan.
                if self.vq_len == 3 {
                    if draw {
                        self.draw_triangle(0, 1, 2);
                    }
                    self.vq[1] = self.vq[2];
                    self.vq_len = 2;
                }
            }
            6 => {
                if self.vq_len == 2 {
                    if draw {
                        self.draw_sprite();
                    }
                    self.vq_len = 0;
                }
            }
            _ => {
                self.vq_len = 0;
            }
        }
    }

    /// Log each distinct TEX0 once (decoded), to see what textures a
    /// program samples without tracing every primitive.
    fn log_tex0(&mut self, v: u64) {
        if self.seen_tex0.len() >= 4096 || !self.seen_tex0.insert(v) {
            return;
        }
        debug!(target: "ps2_core::gs::tex",
            tbp = v & 0x3FFF,
            tbw = (v >> 14) & 0x3F,
            psm = format_args!("{:#04x}", (v >> 20) & 0x3F),
            tw = 1u32 << ((v >> 26) & 0xF),
            th = 1u32 << ((v >> 30) & 0xF),
            tcc = (v >> 34) & 1,
            tfx = (v >> 35) & 3,
            cbp = (v >> 37) & 0x3FFF,
            cpsm = format_args!("{:#04x}", (v >> 51) & 0xF),
            csm = (v >> 55) & 1,
            csa = (v >> 56) & 0x1F,
            "TEX0");
    }

    // --- transfers -------------------------------------------------------

    /// A run of HWREG words. PSMT8 and PSMCT32 destinations (textures and
    /// frame-sized uploads) keep a per-row base address and step through
    /// the column table, the rest go word by word through [`Gs::hwreg`].
    pub fn image(&mut self, data: &[u64]) {
        if self.trxdir != 0 {
            return;
        }
        let dpsm = ((self.bitbltbuf >> 56) & 0x3F) as u32;
        if dpsm != PSMT8 && dpsm != PSMCT32 {
            for &v in data {
                self.hwreg(v);
            }
            return;
        }
        let _p = crate::prof::scope(crate::prof::Slot::GsXfer);
        self.clut_dirty = true;
        let dbp = ((self.bitbltbuf >> 32) & 0x3FFF) as u32;
        let dbw = ((self.bitbltbuf >> 48) & 0x3F) as u32;
        let dsax = ((self.trxpos >> 32) & 0x7FF) as u32;
        let dsay = ((self.trxpos >> 48) & 0x7FF) as u32;
        let rrw = (self.trxreg & 0xFFF) as u32;
        let rrh = ((self.trxreg >> 32) & 0xFFF) as u32;
        if rrw == 0 {
            return;
        }
        let (mut x, mut y) = (self.trx_x, self.trx_y);
        let canvas = &self.canvas;
        match dpsm {
            PSMT8 => {
                let mut base = layout::row_base8(dbp, dbw, dsay + y);
                'words: for &w in data {
                    for i in 0..8 {
                        if y >= rrh {
                            break 'words;
                        }
                        let ay = dsay + y;
                        canvas.wr8((base + layout::col_off8(ay, dsax + x)) & (VRAM_SIZE - 1), (w >> (i * 8)) as u8);
                        x += 1;
                        if x >= rrw {
                            x = 0;
                            y += 1;
                            base = layout::row_base8(dbp, dbw, dsay + y);
                        }
                    }
                }
            }
            _ => {
                let mut base = layout::row_base32(dbp, dbw, dsay + y, false);
                'words: for &w in data {
                    for i in 0..2 {
                        if y >= rrh {
                            break 'words;
                        }
                        let ay = dsay + y;
                        canvas.wr32(
                            (base + layout::col_off32(ay, dsax + x, false)) & (VRAM_SIZE - 1),
                            (w >> (i * 32)) as u32,
                        );
                        x += 1;
                        if x >= rrw {
                            x = 0;
                            y += 1;
                            base = layout::row_base32(dbp, dbw, dsay + y, false);
                        }
                    }
                }
            }
        }
        self.trx_x = x;
        self.trx_y = y;
    }

    /// HWREG: one 64-bit chunk of a HOST->LOCAL image transfer.
    fn hwreg(&mut self, v: u64) {
        if self.trxdir != 0 {
            return;
        }
        let _p = crate::prof::scope(crate::prof::Slot::GsXfer);
        self.clut_dirty = true;
        let dbp = ((self.bitbltbuf >> 32) & 0x3FFF) as u32;
        let dbw = ((self.bitbltbuf >> 48) & 0x3F) as u32;
        let dpsm = ((self.bitbltbuf >> 56) & 0x3F) as u32;
        let dsax = ((self.trxpos >> 32) & 0x7FF) as u32;
        let dsay = ((self.trxpos >> 48) & 0x7FF) as u32;
        let rrw = (self.trxreg & 0xFFF) as u32;
        let rrh = ((self.trxreg >> 32) & 0xFFF) as u32;
        if rrw == 0 {
            return;
        }
        // Consume the 64 bits as pixels in raster order.
        fn push(gs: &mut Gs, count: u32, mut write: impl FnMut(&mut Gs, u32, u32, u32), data: u64, bits: u32) {
            let dsax = ((gs.trxpos >> 32) & 0x7FF) as u32;
            let dsay = ((gs.trxpos >> 48) & 0x7FF) as u32;
            let rrw = (gs.trxreg & 0xFFF) as u32;
            let rrh = ((gs.trxreg >> 32) & 0xFFF) as u32;
            for i in 0..count {
                if gs.trx_y >= rrh {
                    return;
                }
                let px = (data >> (i * bits)) as u32 & (((1u64 << bits) - 1) as u32);
                write(gs, dsax + gs.trx_x, dsay + gs.trx_y, px);
                gs.trx_x += 1;
                if gs.trx_x >= rrw {
                    gs.trx_x = 0;
                    gs.trx_y += 1;
                }
            }
        }
        match dpsm {
            PSMCT32 => push(
                self,
                2,
                move |gs: &mut Gs, x, y, px| gs.write_psmct32(dbp, dbw, x, y, px),
                v,
                32,
            ),
            PSMZ32 => push(
                self,
                2,
                move |gs: &mut Gs, x, y, px| gs.write_psmz32(dbp, dbw, x, y, px),
                v,
                32,
            ),
            PSMCT24 => {
                // Packed stream, 3 bytes per pixel with no 64-bit alignment:
                // carry the partial pixel across chunks.
                for &b in &v.to_le_bytes() {
                    self.trx24[self.trx24_len as usize] = b;
                    self.trx24_len += 1;
                    if self.trx24_len < 3 {
                        continue;
                    }
                    self.trx24_len = 0;
                    if self.trx_y >= rrh {
                        continue;
                    }
                    let px = u32::from(self.trx24[0])
                        | u32::from(self.trx24[1]) << 8
                        | u32::from(self.trx24[2]) << 16;
                    self.write_psmct32(dbp, dbw, dsax + self.trx_x, dsay + self.trx_y, px);
                    self.trx_x += 1;
                    if self.trx_x >= rrw {
                        self.trx_x = 0;
                        self.trx_y += 1;
                    }
                }
            }
            PSMCT16 | PSMCT16S | PSMZ16 | PSMZ16S => push(
                self,
                4,
                move |gs: &mut Gs, x, y, px| gs.write_psmct16(dbp, dbw, x, y, dpsm, px as u16),
                v,
                16,
            ),
            PSMT8 => push(
                self,
                8,
                move |gs: &mut Gs, x, y, px| gs.write_psmt8(dbp, dbw, x, y, px as u8),
                v,
                8,
            ),
            PSMT4 => push(
                self,
                16,
                move |gs: &mut Gs, x, y, px| gs.write_psmt4(dbp, dbw, x, y, px as u8),
                v,
                4,
            ),
            // Index-in-upper-bits formats share the 32-bit pixel's storage
            // and leave its colour bits alone.
            PSMT8H => push(
                self,
                8,
                move |gs: &mut Gs, x, y, px| gs.write_psmct32_bits(dbp, dbw, x, y, px << 24, 0xFF00_0000),
                v,
                8,
            ),
            PSMT4HL => push(
                self,
                16,
                move |gs: &mut Gs, x, y, px| gs.write_psmct32_bits(dbp, dbw, x, y, px << 24, 0x0F00_0000),
                v,
                4,
            ),
            PSMT4HH => push(
                self,
                16,
                move |gs: &mut Gs, x, y, px| gs.write_psmct32_bits(dbp, dbw, x, y, px << 28, 0xF000_0000),
                v,
                4,
            ),
            _ => {
                if self.warned_trx_psm & (1 << dpsm) == 0 {
                    self.warned_trx_psm |= 1 << dpsm;
                    warn!(target: "ps2_core::gs", dpsm = format_args!("{dpsm:#x}"), "unhandled IMAGE transfer format (reported once)");
                }
            }
        }
    }

    /// LOCAL->LOCAL copy, used by the kernel to move fonts around.
    fn local_copy(&mut self) {
        let _p = crate::prof::scope(crate::prof::Slot::GsXfer);
        self.clut_dirty = true;
        let sbp = (self.bitbltbuf & 0x3FFF) as u32;
        let sbw = ((self.bitbltbuf >> 16) & 0x3F) as u32;
        let spsm = ((self.bitbltbuf >> 24) & 0x3F) as u32;
        let dbp = ((self.bitbltbuf >> 32) & 0x3FFF) as u32;
        let dbw = ((self.bitbltbuf >> 48) & 0x3F) as u32;
        let dpsm = ((self.bitbltbuf >> 56) & 0x3F) as u32;
        let ssax = (self.trxpos & 0x7FF) as u32;
        let ssay = ((self.trxpos >> 16) & 0x7FF) as u32;
        let dsax = ((self.trxpos >> 32) & 0x7FF) as u32;
        let dsay = ((self.trxpos >> 48) & 0x7FF) as u32;
        let rrw = (self.trxreg & 0xFFF) as u32;
        let rrh = ((self.trxreg >> 32) & 0xFFF) as u32;
        if spsm != dpsm {
            warn!(target: "ps2_core::gs", spsm, dpsm, "local copy with format conversion (unhandled)");
            return;
        }
        debug!(target: "ps2_core::gs", rrw, rrh, "local->local copy");
        for y in 0..rrh {
            for x in 0..rrw {
                match spsm {
                    PSMCT32 | PSMCT24 => {
                        let px = self.read_psmct32(sbp, sbw, ssax + x, ssay + y);
                        self.write_psmct32(dbp, dbw, dsax + x, dsay + y, px);
                    }
                    PSMZ32 | PSMZ24 => {
                        let px = self.read_psmz32(sbp, sbw, ssax + x, ssay + y);
                        self.write_psmz32(dbp, dbw, dsax + x, dsay + y, px);
                    }
                    PSMCT16 | PSMCT16S | PSMZ16 | PSMZ16S => {
                        let px = self.read_psmct16(sbp, sbw, ssax + x, ssay + y, spsm);
                        self.write_psmct16(dbp, dbw, dsax + x, dsay + y, spsm, px);
                    }
                    PSMT8 => {
                        let px = self.read_psmt8(sbp, sbw, ssax + x, ssay + y);
                        self.write_psmt8(dbp, dbw, dsax + x, dsay + y, px);
                    }
                    PSMT4 => {
                        let px = self.read_psmt4(sbp, sbw, ssax + x, ssay + y);
                        self.write_psmt4(dbp, dbw, dsax + x, dsay + y, px);
                    }
                    _ => return,
                }
            }
        }
    }

    // --- VRAM accessors (see `Canvas`) -----------------------------------

    #[inline]
    pub fn write_psmct32(&mut self, bp: u32, bw: u32, x: u32, y: u32, v: u32) {
        self.canvas.write_psmct32(bp, bw, x, y, v);
    }
    #[inline]
    pub fn write_psmct32_bits(&mut self, bp: u32, bw: u32, x: u32, y: u32, v: u32, mask: u32) {
        self.canvas.write_psmct32_bits(bp, bw, x, y, v, mask);
    }
    #[inline]
    pub fn read_psmct32(&self, bp: u32, bw: u32, x: u32, y: u32) -> u32 {
        self.canvas.read_psmct32(bp, bw, x, y)
    }
    #[inline]
    pub fn write_psmz32(&mut self, bp: u32, bw: u32, x: u32, y: u32, v: u32) {
        self.canvas.write_psmz32(bp, bw, x, y, v);
    }
    #[inline]
    pub fn read_psmz32(&self, bp: u32, bw: u32, x: u32, y: u32) -> u32 {
        self.canvas.read_psmz32(bp, bw, x, y)
    }
    #[inline]
    pub fn write_psmct16(&mut self, bp: u32, bw: u32, x: u32, y: u32, psm: u32, v: u16) {
        self.canvas.write_psmct16(bp, bw, x, y, psm, v);
    }
    #[inline]
    pub fn read_psmct16(&self, bp: u32, bw: u32, x: u32, y: u32, psm: u32) -> u16 {
        self.canvas.read_psmct16(bp, bw, x, y, psm)
    }
    #[inline]
    pub fn write_psmt8(&mut self, bp: u32, bw: u32, x: u32, y: u32, v: u8) {
        self.canvas.write_psmt8(bp, bw, x, y, v);
    }
    #[inline]
    pub fn read_psmt8(&self, bp: u32, bw: u32, x: u32, y: u32) -> u8 {
        self.canvas.read_psmt8(bp, bw, x, y)
    }
    #[inline]
    pub fn write_psmt4(&mut self, bp: u32, bw: u32, x: u32, y: u32, v: u8) {
        self.canvas.write_psmt4(bp, bw, x, y, v);
    }
    #[inline]
    pub fn read_psmt4(&self, bp: u32, bw: u32, x: u32, y: u32) -> u8 {
        self.canvas.read_psmt4(bp, bw, x, y)
    }

    // --- scanout ---------------------------------------------------------

    /// Compose the currently displayed frame as RGBA8. Returns (w, h, data).
    /// Interlaced field buffers (SMODE2 INT+FFMD) are line-doubled.
    pub fn framebuffer(&self) -> (u32, u32, Vec<u8>) {
        let (w, h, view) = self.display_view();
        let mut out = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            let sy = if view.field_buffer { y / 2 } else { y };
            self.scan_line(&view, sy, &mut out[(y * w * 4) as usize..][..(w * 4) as usize]);
        }
        (w, h, out)
    }

    /// Like [`Gs::framebuffer`], but interlaced field buffers are woven:
    /// this field's lines land on rows of parity `field`, the other rows
    /// keep the previous field. Bobbing each field alone would show the
    /// game's half-line field offset as a 30 Hz shake.
    pub fn framebuffer_woven(&mut self, field: bool, mode: Deinterlace) -> (u32, u32, Vec<u8>) {
        let (w, h, view) = self.display_view();
        if !view.field_buffer {
            return self.framebuffer();
        }
        if self.woven_dims != (w, h) {
            self.woven = vec![0u8; (w * h * 4) as usize];
            self.motion = vec![0u8; (w * h) as usize];
            self.woven_dims = (w, h);
        }
        let mut woven = std::mem::take(&mut self.woven);
        let mut motion = std::mem::take(&mut self.motion);
        let stride = (w * 4) as usize;
        // The new field lands on its rows; the other rows keep the previous
        // field. Motion is the change against the same field two vblanks
        // ago, which is exactly what those rows held until now.
        let mut line = vec![0u8; stride];
        for sy in 0..h / 2 {
            let y = sy * 2 + field as u32;
            self.scan_line(&view, sy, &mut line);
            let row = &mut woven[y as usize * stride..][..stride];
            if mode == Deinterlace::Adaptive {
                let flags = &mut motion[(y * w) as usize..][..w as usize];
                for (x, f) in flags.iter_mut().enumerate() {
                    let (a, b) = (&row[x * 4..x * 4 + 3], &line[x * 4..x * 4 + 3]);
                    let diff: u32 = a.iter().zip(b).map(|(&p, &q)| p.abs_diff(q) as u32).sum();
                    *f = u8::from(diff > MOTION_THRESHOLD);
                }
            }
            row.copy_from_slice(&line);
        }
        let mut out = woven.clone();
        match mode {
            Deinterlace::Weave => {}
            Deinterlace::Bob => {
                // Only the new field is real: rebuild the other rows from it.
                for y in (0..h).filter(|y| (y & 1 != 0) != field) {
                    Self::interpolate_row(&woven, &mut out, w, h, y, None);
                }
            }
            Deinterlace::Adaptive => {
                // Keep the old field where nothing moved, interpolate the new
                // one where either neighbouring new row changed.
                for y in (0..h).filter(|y| (y & 1 != 0) != field) {
                    Self::interpolate_row(&woven, &mut out, w, h, y, Some(&motion));
                }
            }
        }
        self.woven = woven;
        self.motion = motion;
        (w, h, out)
    }

    /// Fill row `y` of `out` from the rows above and below in `woven` (the
    /// current field), everywhere or only where `motion` flags either of
    /// them.
    fn interpolate_row(woven: &[u8], out: &mut [u8], w: u32, h: u32, y: u32, motion: Option<&[u8]>) {
        let stride = (w * 4) as usize;
        let above = y.saturating_sub(1);
        let below = (y + 1).min(h - 1);
        // Edge rows have one real neighbour only.
        let (above, below) = if y == 0 { (below, below) } else if y == h - 1 { (above, above) } else { (above, below) };
        let (ra, rb) = (&woven[above as usize * stride..][..stride], &woven[below as usize * stride..][..stride]);
        let row = &mut out[y as usize * stride..][..stride];
        for x in 0..w as usize {
            if let Some(m) = motion {
                let moved = m[(above * w) as usize + x] | m[(below * w) as usize + x];
                if moved == 0 {
                    continue;
                }
            }
            for c in 0..4 {
                row[x * 4 + c] = ((ra[x * 4 + c] as u16 + rb[x * 4 + c] as u16 + 1) >> 1) as u8;
            }
        }
    }

    /// Decode PMODE/DISPFB/DISPLAY into what the read circuit shows.
    fn display_view(&self) -> (u32, u32, DisplayView) {
        // Prefer an enabled read circuit; fall back to circuit 1.
        let (dispfb, display) = if self.pmode & 1 != 0 {
            (self.dispfb1, self.display1)
        } else if self.pmode & 2 != 0 {
            (self.dispfb2, self.display2)
        } else {
            (self.dispfb1, self.display1)
        };
        let magh = ((display >> 23) & 0xF) as u32 + 1;
        let magv = ((display >> 27) & 3) as u32 + 1;
        let mut w = (((display >> 32) & 0xFFF) as u32 + 1) / magh;
        let mut h = (((display >> 44) & 0x7FF) as u32 + 1) / magv;
        if w == 0 || w > 1024 {
            w = 640;
        }
        if h == 0 || h > 1024 {
            h = 448;
        }
        let view = DisplayView {
            fbp: ((dispfb & 0x1FF) * 32) as u32, // pages -> blocks
            fbw: ((dispfb >> 9) & 0x3F) as u32,
            psm: ((dispfb >> 15) & 0x1F) as u32,
            dbx: ((dispfb >> 32) & 0x7FF) as u32,
            dby: ((dispfb >> 43) & 0x7FF) as u32,
            // INT+FFMD: each field is a half-height buffer.
            field_buffer: self.smode2 & 3 == 3,
        };
        (w, h & !1, view)
    }

    /// Read one displayed line (`sy` in buffer lines) as RGBA8 into `out`.
    fn scan_line(&self, v: &DisplayView, sy: u32, out: &mut [u8]) {
        let w = out.len() as u32 / 4;
        for x in 0..w {
            let (r, g, b) = match v.psm {
                PSMCT32 | PSMCT24 => {
                    let px = self.read_psmct32(v.fbp, v.fbw, v.dbx + x, v.dby + sy);
                    (px as u8, (px >> 8) as u8, (px >> 16) as u8)
                }
                PSMCT16 | PSMCT16S => {
                    let px = self.read_psmct16(v.fbp, v.fbw, v.dbx + x, v.dby + sy, v.psm);
                    (
                        ((px & 0x1F) << 3) as u8,
                        (((px >> 5) & 0x1F) << 3) as u8,
                        (((px >> 10) & 0x1F) << 3) as u8,
                    )
                }
                _ => (255, 0, 255),
            };
            let o = (x * 4) as usize;
            out[o] = r;
            out[o + 1] = g;
            out[o + 2] = b;
            out[o + 3] = 255;
        }
    }
}

/// One decoded texture row: RGBA8 texels `u_lo..` of row `key.1` of the
/// texture set up by TEX0 `key.0`. Valid within one primitive only.
#[derive(Default)]
pub(super) struct TexRow {
    pub key: (u64, i32),
    pub u_lo: i32,
    pub data: Vec<u32>,
}

/// How interlaced field buffers are shown (see [`Gs::framebuffer_woven`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Deinterlace {
    /// Latest two fields interleaved: full detail, combs on motion.
    #[default]
    Weave,
    /// Latest field only, the other rows interpolated from it: no combing,
    /// half the vertical detail.
    Bob,
    /// Weave where nothing moved, bob where it did.
    Adaptive,
}

/// Summed RGB difference above which a pixel counts as moving.
const MOTION_THRESHOLD: u32 = 24;

/// Read-circuit parameters for scanout.
struct DisplayView {
    fbp: u32,
    fbw: u32,
    psm: u32,
    dbx: u32,
    dby: u32,
    field_buffer: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interlaced field buffers: weave keeps both fields, bob rebuilds the
    /// other rows from the new field, adaptive does so only where the new
    /// field differs from the one two vblanks ago.
    #[test]
    fn deinterlace_modes() {
        let mut gs = Gs::new();
        // 64x8 display (4 lines per field), field buffer at bp 0, PSMCT32,
        // PMODE circuit 1, SMODE2 INT+FFMD.
        gs.priv_write(0x0000, 1);
        gs.priv_write(0x0020, 3);
        gs.priv_write(0x0070, 1 << 9); // fbw 1
        gs.priv_write(0x0080, (63u64 << 32) | (7u64 << 44)); // 64x8
        let fill = |gs: &mut Gs, v: u32| {
            for y in 0..4 {
                for x in 0..64 {
                    gs.write_psmct32(0, 1, x, y, v);
                }
            }
        };
        let px = |f: &[u8], w: u32, x: u32, y: u32| f[((y * w + x) * 4) as usize];
        // Field 0 = 0x10, field 1 = 0x20, then field 0 again = 0x30 (moved).
        fill(&mut gs, 0x10);
        let (_, _, f) = gs.framebuffer_woven(false, Deinterlace::Weave);
        assert_eq!(px(&f, 64, 5, 0), 0x10);
        fill(&mut gs, 0x20);
        let (_, _, f) = gs.framebuffer_woven(true, Deinterlace::Weave);
        assert_eq!((px(&f, 64, 5, 0), px(&f, 64, 5, 1)), (0x10, 0x20));
        fill(&mut gs, 0x30);
        let (_, _, f) = gs.framebuffer_woven(false, Deinterlace::Weave);
        assert_eq!((px(&f, 64, 5, 0), px(&f, 64, 5, 1)), (0x30, 0x20));
        // Bob on field 1 (0x40): even rows become the average of odd rows.
        fill(&mut gs, 0x40);
        let (_, _, f) = gs.framebuffer_woven(true, Deinterlace::Bob);
        assert_eq!((px(&f, 64, 5, 2), px(&f, 64, 5, 3)), (0x40, 0x40));
        // Adaptive: field 0 moves (0x30 -> 0x50) so the odd rows (0x40) are
        // replaced by interpolation of the new even rows; then a still
        // field 1 (0x40 again) keeps the woven even rows.
        fill(&mut gs, 0x50);
        let (_, _, f) = gs.framebuffer_woven(false, Deinterlace::Adaptive);
        assert_eq!((px(&f, 64, 5, 2), px(&f, 64, 5, 3)), (0x50, 0x50));
        fill(&mut gs, 0x40);
        let (_, _, f) = gs.framebuffer_woven(true, Deinterlace::Adaptive);
        assert_eq!((px(&f, 64, 5, 2), px(&f, 64, 5, 3)), (0x50, 0x40));
    }

    /// Full-screen textured sprite copying one 640x224 buffer into another
    /// (Amagami's OP does this to build a refraction source).
    #[test]
    fn textured_sprite_copies_between_frame_buffers() {
        let mut gs = Gs::new();
        for y in 0..224 {
            for x in 0..640 {
                gs.write_psmct32(2240, 10, x, y, 0x8060_7080);
            }
        }
        gs.write_reg(0x1A, 1); // PRMODECONT: use PRIM
        gs.write_reg(0x4C, 210 | (10 << 16)); // FRAME_1: 6720, fbw 10, PSMCT32
        gs.write_reg(0x4E, 140 | (1 << 24) | (1 << 32)); // ZBUF_1: 4480, Z24, masked
        gs.write_reg(0x47, 0x30000); // TEST_1: ZTE, ALWAYS
        gs.write_reg(0x40, 639 << 16 | 223 << 48); // SCISSOR_1
        gs.write_reg(0x18, (1728 * 16) | ((1936 * 16) << 32)); // XYOFFSET_1
        gs.write_reg(0x06, 0x6_2812_88c0); // TEX0_1: 2240, tbw 10, PSMCT24, 1024x256
        gs.write_reg(0x00, 0x116); // sprite, TME, FST
        gs.write_reg(0x01, 0x8080_8080); // RGBAQ: unity modulate
        gs.write_reg(0x03, 0);
        gs.write_reg(0x05, (1728 * 16) | ((1936 * 16) << 16));
        gs.write_reg(0x03, (639 * 16 + 8) | ((223 * 16 + 8) << 16));
        gs.write_reg(0x05, ((1728 + 639) * 16 + 8) | (((1936 + 223) * 16 + 8) << 16));
        assert_eq!(gs.prims_drawn, 1);
        assert_eq!(gs.read_psmct32(6720, 10, 320, 100) & 0xFF_FFFF, 0x60_7080);
        assert_eq!(gs.read_psmct32(6720, 10, 0, 0) & 0xFF_FFFF, 0x60_7080);

        // Same copy Z-tested (GEQUAL) at z = 0xFFFFFF against a Z24 buffer
        // full of garbage upper bytes: a 24-bit compare must pass.
        for y in 0..224 {
            for x in 0..640 {
                gs.write_psmct32(4480, 10, x, y, 0xFF12_3456);
            }
        }
        gs.write_reg(0x4C, 280 | (10 << 16)); // FRAME_1: 8960
        gs.write_reg(0x47, 0x50000); // TEST_1: ZTE, GEQUAL
        gs.write_reg(0x03, 0);
        gs.write_reg(0x05, (1728 * 16) | ((1936 * 16) << 16) | (0xFF_FFFF << 32));
        gs.write_reg(0x03, (639 * 16 + 8) | ((223 * 16 + 8) << 16));
        gs.write_reg(0x05, ((1728 + 639) * 16 + 8) | (((1936 + 223) * 16 + 8) << 16) | (0xFF_FFFF << 32));
        assert_eq!(gs.prims_drawn, 2);
        assert_eq!(gs.read_psmct32(8960, 10, 320, 100) & 0xFF_FFFF, 0x60_7080);
    }
}
