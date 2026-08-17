//! Pixel pipeline: triangle/sprite rasterization, texture sampling,
//! alpha blending and the frame-buffer write path.
//!
//! [`Gs`] decodes a primitive into geometry plus a [`PixelPipe`] and hands
//! scanlines to a [`Painter`], which owns nothing but references: the
//! shared [`Canvas`] and CLUT, the pipe, and a per-thread [`Scratch`]. Large
//! primitives are split across worker threads in two-row bands (rows
//! `2k, 2k+1` share the 32-bit column layout's cache lines), each band with
//! its own scratch, so no two threads write the same pixel; the split is
//! skipped when the texture may alias the render target, since a primitive
//! that samples what it draws would otherwise depend on thread timing.

use super::*;

/// Interpolated per-pixel attributes.
#[derive(Clone, Copy)]
struct Frag {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
    z: u32,
    s: f32,
    t: f32,
    q: f32,
    u: f32,
    v: f32,
}

/// Per-thread rasterization state.
pub(super) struct Scratch {
    /// Decoded texture rows (see `Painter::fill_tex_row`).
    tex_rows: [TexRow; 2],
    pixels: u64,
    tex_samples: [u64; 64],
}

impl Default for Scratch {
    fn default() -> Self {
        Self { tex_rows: Default::default(), pixels: 0, tex_samples: [0; 64] }
    }
}

/// Rows a task handles: `py` in `start..end` with `(py >> 1) % lanes ==
/// lane` — two-row bands so a 32-bit column pair never straddles tasks.
#[derive(Clone, Copy)]
struct Rows {
    start: i32,
    end: i32,
    lane: usize,
    lanes: usize,
}

impl Rows {
    fn all(start: i32, end: i32) -> Self {
        Self { start, end, lane: 0, lanes: 1 }
    }
    fn iter(self) -> impl Iterator<Item = i32> {
        (self.start..self.end).filter(move |py| self.lanes == 1 || ((py >> 1) as usize) % self.lanes == self.lane)
    }
}

/// Pixels a primitive must cover before it is split across threads.
const PARALLEL_MIN_PIXELS: i64 = 16 * 1024;
/// Bands (tasks) a split primitive is cut into.
const PARALLEL_LANES: usize = 4;

/// Sprite geometry needed per scanline.
#[derive(Clone, Copy)]
struct SpriteGeom {
    x0: i32,
    y0: i32,
    inv_wid: f32,
    inv_hei: f32,
    inv_q: f32,
    u0: i32,
    u1: i32,
    s0: f32,
    s1: f32,
    tv0: i32,
    tv1: i32,
    t0: f32,
    t1: f32,
    pxa: i32,
    pxb: i32,
    v1: Vertex,
}

/// Triangle geometry needed per scanline.
#[derive(Clone, Copy)]
struct TriGeom {
    a: Vertex,
    b: Vertex,
    c: Vertex,
    inv_area: f32,
    minx: i32,
    maxx: i32,
    miny: i32,
    /// Edge functions at the top-left sample and their per-pixel steps.
    w0: i64,
    w1: i64,
    w2: i64,
    dx: [i64; 3],
    dy: [i64; 3],
    ca: [f32; 4],
    cb: [f32; 4],
    cc: [f32; 4],
    sa: [f32; 4],
    sb: [f32; 4],
    sc: [f32; 4],
}

impl Gs {
    pub(super) fn draw_point(&mut self) {
        let _p = crate::prof::scope(crate::prof::Slot::GsDraw);
        let v = self.vq[0];
        let pipe = self.pixel_pipe();
        let frag = Frag {
            r: v.r as f32,
            g: v.g as f32,
            b: v.b as f32,
            a: v.a as f32,
            z: v.z,
            s: v.s,
            t: v.t,
            q: v.q,
            u: v.u as f32 / 16.0,
            v: v.v as f32 / 16.0,
        };
        let (px, py) = (v.x >> 4, v.y >> 4);
        if px >= pipe.scx0 && px <= pipe.scx1 && py >= pipe.scy0 && py <= pipe.scy1 {
            let mut p = Painter { canvas: &self.canvas, clut: &self.clut, pipe: &pipe, scratch: &mut self.scratch };
            let texel = if pipe.tme { p.sample(&frag) } else { 0 };
            let row = Row::new(&pipe, py as u32);
            p.shade_row_px(&row, px as u32, frag, texel);
        }
        self.prims_drawn += 1;
        self.merge_scratch();
    }

    /// Bring-up aid: describe each distinct render-target setup once.
    fn log_target(&mut self, attrs: u64, kind: &str, n: usize) {
        let ctx = self.ctx[((attrs >> 9) & 1) as usize];
        let key = ctx.frame ^ ctx.zbuf.rotate_left(13) ^ ctx.test.rotate_left(29) ^ ctx.scissor.rotate_left(43) ^ 0x5A5A;
        if self.seen_targets.len() >= 4096 && !self.seen_targets.contains_key(&key) {
            return;
        }
        let count = self.seen_targets.entry(key).or_insert(0);
        if *count >= 4 {
            return;
        }
        *count += 1;
        let xy = |v: &Vertex| format!("({},{},z={:#x})", v.x as f32 / 16.0, v.y as f32 / 16.0, v.z);
        let verts: Vec<String> = self.vq[..n].iter().map(xy).collect();
        tracing::debug!(target: "ps2_core::gs::target",
            kind,
            verts = verts.join(" "),
            prim = format_args!("{:#x}", attrs),
            tex0 = format_args!("{:#x}", ctx.tex0),
            fbp = (ctx.frame & 0x1FF) * 32,
            fbw = (ctx.frame >> 16) & 0x3F,
            fpsm = format_args!("{:#04x}", (ctx.frame >> 24) & 0x3F),
            fbmsk = format_args!("{:#010x}", ctx.frame >> 32),
            zbp = (ctx.zbuf & 0x1FF) * 32,
            zpsm = format_args!("{:#04x}", (ctx.zbuf >> 24) & 0xF),
            zmsk = (ctx.zbuf >> 32) & 1,
            test = format_args!("{:#x}", ctx.test),
            scissor = format_args!("{}..{} x {}..{}", ctx.scissor & 0x7FF, (ctx.scissor >> 16) & 0x7FF, (ctx.scissor >> 32) & 0x7FF, (ctx.scissor >> 48) & 0x7FF),
            xyoff = format_args!("{},{}", (ctx.xyoffset & 0xFFFF) >> 4, ((ctx.xyoffset >> 32) & 0xFFFF) >> 4),
            "render target");
    }

    /// Bring-up aid: describe small primitives (particles, glyphs) once per
    /// distinct texture/blend setup.
    fn log_small_prim(&mut self, kind: &str, attrs: u64, w: i32, h: i32, v0: &Vertex, v1: &Vertex) {
        let ctx = self.ctx[((attrs >> 9) & 1) as usize];
        let key = ctx.tex0 ^ (attrs << 1) ^ ctx.alpha.rotate_left(17) ^ ctx.test.rotate_left(40);
        if self.seen_tex0.len() >= 4096 || !self.seen_tex0.insert(key) {
            return;
        }
        tracing::debug!(target: "ps2_core::gs::small",
            kind, w, h,
            tme = (attrs >> 4) & 1,
            abe = (attrs >> 6) & 1,
            fst = (attrs >> 8) & 1,
            tex0 = format_args!("{:#018x}", ctx.tex0),
            alpha = format_args!("{:#x}", ctx.alpha),
            test = format_args!("{:#x}", ctx.test),
            rgba0 = format_args!("{},{},{},{}", v0.r, v0.g, v0.b, v0.a),
            rgba1 = format_args!("{},{},{},{}", v1.r, v1.g, v1.b, v1.a),
            uv0 = format_args!("{},{}", v0.u as f32 / 16.0, v0.v as f32 / 16.0),
            uv1 = format_args!("{},{}", v1.u as f32 / 16.0, v1.v as f32 / 16.0),
            st0 = format_args!("{},{},{}", v0.s, v0.t, v0.q),
            "small prim");
    }

    /// Fold the main scratch's counters into the statistics.
    fn merge_scratch(&mut self) {
        self.pixels_shaded += std::mem::take(&mut self.scratch.pixels);
        for (h, s) in self.tex_psm_hist.iter_mut().zip(self.scratch.tex_samples.iter_mut()) {
            *h += std::mem::take(s);
        }
    }

    /// Whether the texture may alias the frame or Z buffer within the rows
    /// drawn (conservative block-range test): such primitives read what
    /// they write and must stay on one thread.
    fn texture_aliases_target(pipe: &PixelPipe, rows: i32) -> bool {
        if !pipe.tme {
            return false;
        }
        let ti = &pipe.tex;
        // Row height per page: 32 (32-bit), 64 (16/8-bit), 128 (4-bit).
        let tex_pages = (ti.th / 32 + 1) * ti.tbw.max(1);
        let tex = ti.tbp..ti.tbp + tex_pages * 32;
        let target_pages = ((rows as u32) / 32 + 2) * pipe.fbw.max(1);
        let fb = pipe.fbp..pipe.fbp + target_pages * 32;
        let zb = pipe.zbp..pipe.zbp + target_pages * 32;
        let overlaps = |a: &std::ops::Range<u32>, b: &std::ops::Range<u32>| a.start < b.end && b.start < a.end;
        overlaps(&tex, &fb) || (pipe.zte && overlaps(&tex, &zb))
    }

    pub(super) fn draw_sprite(&mut self) {
        let _p = crate::prof::scope(crate::prof::Slot::GsDraw);
        let v0 = self.vq[0];
        let v1 = self.vq[1];
        let (x0, x1) = (v0.x.min(v1.x), v0.x.max(v1.x));
        let (y0, y1) = (v0.y.min(v1.y), v0.y.max(v1.y));
        // Pixel centers in 12.4: draw [x0, x1) rounding up from the left.
        let px0 = (x0 + 15) >> 4;
        let px1 = (x1 + 15) >> 4;
        let py0 = (y0 + 15) >> 4;
        let py1 = (y1 + 15) >> 4;
        let attrs = self.attrs();
        let tme = attrs & (1 << 4) != 0;
        self.prims_drawn += 1;
        if tme {
            self.prims_textured += 1;
        }
        self.log_target(attrs, "sprite", 2);
        if px1 - px0 <= 24 && py1 - py0 <= 24 {
            self.log_small_prim("sprite", attrs, px1 - px0, py1 - py0, &v0, &v1);
        }
        // Texture coords run left-to-right / top-to-bottom regardless of
        // vertex order; color is flat from the second vertex.
        let (u0, u1, s0, s1) = if v0.x <= v1.x {
            (v0.u, v1.u, v0.s, v1.s)
        } else {
            (v1.u, v0.u, v1.s, v0.s)
        };
        let (tv0, tv1, t0, t1) = if v0.y <= v1.y {
            (v0.v, v1.v, v0.t, v1.t)
        } else {
            (v1.v, v0.v, v1.t, v0.t)
        };
        let pipe = self.pixel_pipe();
        let geom = SpriteGeom {
            x0,
            y0,
            inv_wid: 1.0 / (x1 - x0).max(1) as f32,
            inv_hei: 1.0 / (y1 - y0).max(1) as f32,
            inv_q: {
                let q = if v1.q.abs() < 1e-9 { 1.0 } else { v1.q };
                1.0 / q
            },
            u0,
            u1,
            s0,
            s1,
            tv0,
            tv1,
            t0,
            t1,
            pxa: px0.max(pipe.scx0),
            pxb: px_clip(px1).min(pipe.scx1 + 1),
            v1,
        };
        let (rya, ryb) = (py0.max(pipe.scy0), px_clip(py1).min(pipe.scy1 + 1));
        self.run_rows(&pipe, rya, ryb, geom.pxb - geom.pxa, |p, rows| p.sprite_rows(&geom, rows));
    }

    pub(super) fn draw_triangle(&mut self, i0: usize, i1: usize, i2: usize) {
        let _p = crate::prof::scope(crate::prof::Slot::GsDraw);
        let (v0, v1, v2) = (self.vq[i0], self.vq[i1], self.vq[i2]);
        self.prims_drawn += 1;
        let attrs = self.attrs();
        if attrs & (1 << 4) != 0 {
            self.prims_textured += 1;
        }
        self.log_target(attrs, "triangle", 3);
        // 12.4 edge functions; area in 8.8.
        let area = edge(v0.x, v0.y, v1.x, v1.y, v2.x, v2.y);
        if area == 0 {
            return;
        }
        let (a, b, c, area) = if area < 0 {
            (v0, v2, v1, -area)
        } else {
            (v0, v1, v2, area)
        };
        let minx = (a.x.min(b.x).min(c.x) >> 4).max(0);
        let maxx = ((a.x.max(b.x).max(c.x) + 15) >> 4).min(2047);
        let miny = (a.y.min(b.y).min(c.y) >> 4).max(0);
        let maxy = ((a.y.max(b.y).max(c.y) + 15) >> 4).min(2047);
        if maxx - minx <= 24 && maxy - miny <= 24 {
            self.log_small_prim("triangle", attrs, maxx - minx, maxy - miny, &a, &b);
        }
        let pipe = self.pixel_pipe();
        let minx = minx.max(pipe.scx0);
        let maxx = maxx.min(pipe.scx1);
        let miny = miny.max(pipe.scy0);
        let maxy = maxy.min(pipe.scy1);
        if minx > maxx || miny > maxy {
            return;
        }
        // Edge functions are affine in (sx, sy): evaluate at the top-left
        // sample once and step by whole pixels (16 units) — exact in i64.
        let (sx0, sy0) = ((minx << 4) + 8, (miny << 4) + 8);
        let col = |v: &Vertex| [v.r as f32, v.g as f32, v.b as f32, v.a as f32];
        let stq = |v: &Vertex| [v.s, v.t, v.q, v.u as f32];
        let geom = TriGeom {
            a,
            b,
            c,
            inv_area: 1.0 / area as f32,
            minx,
            maxx,
            miny,
            w0: edge(b.x, b.y, c.x, c.y, sx0, sy0),
            w1: edge(c.x, c.y, a.x, a.y, sx0, sy0),
            w2: edge(a.x, a.y, b.x, b.y, sx0, sy0),
            dx: [-(c.y - b.y) as i64 * 16, -(a.y - c.y) as i64 * 16, -(b.y - a.y) as i64 * 16],
            dy: [(c.x - b.x) as i64 * 16, (a.x - c.x) as i64 * 16, (b.x - a.x) as i64 * 16],
            ca: col(&a),
            cb: col(&b),
            cc: col(&c),
            sa: stq(&a),
            sb: stq(&b),
            sc: stq(&c),
        };
        self.run_rows(&pipe, miny, maxy + 1, maxx - minx + 1, |p, rows| p.tri_rows(&geom, rows));
    }

    /// Rasterize rows `start..end`: on the worker pool in bands when the
    /// primitive is large and cannot sample its own target, else inline.
    fn run_rows(
        &mut self,
        pipe: &PixelPipe,
        start: i32,
        end: i32,
        width: i32,
        f: impl Fn(&mut Painter, Rows) + Sync,
    ) {
        let pixels = (end - start).max(0) as i64 * width.max(0) as i64;
        let split = pixels >= PARALLEL_MIN_PIXELS
            && self.pool.len() >= PARALLEL_LANES
            && !Self::texture_aliases_target(pipe, end);
        if split {
            #[cfg(feature = "threads")]
            {
                let canvas = &self.canvas;
                let clut = &self.clut;
                let (rows, f) = (Rows { start, end, lane: 0, lanes: PARALLEL_LANES }, &f);
                rayon::scope(|s| {
                    for (lane, scratch) in self.pool.iter_mut().take(PARALLEL_LANES).enumerate() {
                        s.spawn(move |_| {
                            let mut p = Painter { canvas, clut, pipe, scratch };
                            f(&mut p, Rows { lane, ..rows });
                        });
                    }
                });
                for s in self.pool.iter_mut() {
                    self.pixels_shaded += std::mem::take(&mut s.pixels);
                    for (h, n) in self.tex_psm_hist.iter_mut().zip(s.tex_samples.iter_mut()) {
                        *h += std::mem::take(n);
                    }
                }
                return;
            }
        }
        let mut p = Painter { canvas: &self.canvas, clut: &self.clut, pipe, scratch: &mut self.scratch };
        f(&mut p, Rows::all(start, end));
        self.merge_scratch();
    }

    /// Decode the drawing environment for the current context once per
    /// primitive; the per-pixel path only reads it.
    fn pixel_pipe(&mut self) -> PixelPipe {
        let attrs = self.attrs();
        let ctx = self.ctx[((attrs >> 9) & 1) as usize];
        let test = ctx.test;
        let tex = TexInfo::new(&ctx, self.texa);
        if attrs & (1 << 4) != 0 && tex.clut_bits != 0 {
            self.refresh_clut(&tex);
        }
        PixelPipe {
            kind: (self.prim & 7) as u8,
            scx0: (ctx.scissor & 0x7FF) as i32,
            scx1: ((ctx.scissor >> 16) & 0x7FF) as i32,
            scy0: ((ctx.scissor >> 32) & 0x7FF) as i32,
            scy1: ((ctx.scissor >> 48) & 0x7FF) as i32,
            tme: attrs & (1 << 4) != 0,
            fst: attrs & (1 << 8) != 0,
            abe: attrs & (1 << 6) != 0,
            tfx: ((ctx.tex0 >> 35) & 3) as u8,
            tcc: ctx.tex0 & (1 << 34) != 0,
            bilinear: (ctx.tex1 >> 5) & 1 != 0,
            tex,
            ate: test & 1 != 0,
            atst: ((test >> 1) & 7) as u8,
            aref: ((test >> 4) & 0xFF) as u32,
            afail: ((test >> 12) & 3) as u8,
            zte: test & (1 << 16) != 0,
            ztst: ((test >> 17) & 3) as u8,
            zbp: ((ctx.zbuf & 0x1FF) * 32) as u32,
            zmsk: ctx.zbuf & (1 << 32) != 0,
            // Z buffer depth: PSMZ32 keeps 32 bits, PSMZ24 24, PSMZ16(S) 16;
            // the upper bits of the stored word belong to whatever else
            // shares the memory (Amagami parks 8-bit textures over its Z24
            // buffer).
            zmask: match (ctx.zbuf >> 24) & 0xF {
                0x0 => u32::MAX,
                0x1 => 0x00FF_FFFF,
                _ => 0xFFFF,
            },
            fbp: ((ctx.frame & 0x1FF) * 32) as u32,
            fbw: ((ctx.frame >> 16) & 0x3F) as u32,
            fb24: ((ctx.frame >> 24) & 0x3F) as u32 == PSMCT24,
            fbmsk: (ctx.frame >> 32) as u32,
            blend_a: (ctx.alpha & 3) as u8,
            blend_b: ((ctx.alpha >> 2) & 3) as u8,
            blend_c: ((ctx.alpha >> 4) & 3) as u8,
            blend_d: ((ctx.alpha >> 6) & 3) as u8,
            blend_fix: ((ctx.alpha >> 32) & 0xFF) as u32,
        }
    }

    /// Re-decode the CLUT cache when the palette setup changed or a
    /// transfer touched VRAM. Real hardware only reloads on TEX0 writes
    /// with CLD set; keying on the setup instead is a superset of that
    /// (drawing primitives into CLUT memory is not tracked).
    fn refresh_clut(&mut self, ti: &TexInfo) {
        let key = ti.clut_key();
        if key == self.clut_key && !self.clut_dirty {
            return;
        }
        let entries = if ti.clut_bits == 8 { 256 } else { 16 };
        for e in ti.clut_base..ti.clut_base + entries {
            self.clut[e] = self.clut_lookup(ti.tex0, e as u32);
        }
        self.clut_key = key;
        self.clut_dirty = false;
    }

    /// Read palette entry `e` (index plus CSA offset) from VRAM.
    fn clut_lookup(&self, tex0: u64, e: u32) -> u32 {
        let cbp = ((tex0 >> 37) & 0x3FFF) as u32;
        let cpsm = ((tex0 >> 51) & 0xF) as u32;
        let csm = (tex0 >> 55) & 1;
        let (x, y) = if csm == 0 {
            // CSM1 packs the CLUT as a 16x16 image whose entries sit in
            // 8x2-entry tiles — equivalently, a linear 16x16 layout with
            // bits 3 and 4 of the entry number swapped.
            let e = (e & 0xE7) | ((e & 0x08) << 1) | ((e & 0x10) >> 1);
            (e & 0xF, e >> 4)
        } else {
            // CSM2: linear row (TEXCLUT offset/width not modelled).
            (e & 0xFF, e >> 8)
        };
        if cpsm == 0 {
            self.canvas.read_psmct32(cbp, 1, x, y)
        } else {
            expand16(self.canvas.read_psmct16(cbp, 1, x, y, if cpsm == 0xA { PSMCT16S } else { PSMCT16 }), self.texa)
        }
    }
}

/// Scanline rasterizer over shared VRAM with per-thread scratch.
struct Painter<'a> {
    canvas: &'a Canvas,
    clut: &'a [u32; 512],
    pipe: &'a PixelPipe,
    scratch: &'a mut Scratch,
}

impl Painter<'_> {
    fn sprite_rows(&mut self, g: &SpriteGeom, rows: Rows) {
        let pipe = self.pipe;
        // Rows decoded for an earlier primitive may have been drawn over
        // since; only reuse within this sprite (a sprite that samples what
        // its own earlier rows wrote is the accepted deviation).
        self.scratch.tex_rows[0].key.0 = u64::MAX;
        self.scratch.tex_rows[1].key.0 = u64::MAX;
        let (v1, x0, y0) = (g.v1, g.x0, g.y0);
        let (pxa, pxb) = (g.pxa, g.pxb);
        let tw = pipe.tex.tw as f32;
        let th = pipe.tex.th as f32;
        // Texel-space u for a pixel column, exactly as `sample` derives it
        // from the fragment (same operations, same rounding).
        let fu_at = |px: i32| -> f32 {
            let fx = ((px << 4) as f32 + 8.0 - x0 as f32) * g.inv_wid;
            if pipe.fst {
                (g.u0 as f32 + (g.u1 - g.u0) as f32 * fx) / 16.0
            } else {
                (g.s0 + (g.s1 - g.s0) * fx) * g.inv_q * tw
            }
        };
        for py in rows.iter() {
            let fy = ((py << 4) as f32 + 8.0 - y0 as f32) * g.inv_hei;
            let frag = Frag {
                r: v1.r as f32,
                g: v1.g as f32,
                b: v1.b as f32,
                a: v1.a as f32,
                z: v1.z,
                s: 0.0,
                t: g.t0 + (g.t1 - g.t0) * fy,
                q: v1.q,
                u: 0.0,
                v: (g.tv0 as f32 + (g.tv1 - g.tv0) as f32 * fy) / 16.0,
            };
            let row = Row::new(pipe, py as u32);
            if pipe.tme && pxb > pxa {
                // v is constant along the row: decode the one or two
                // texture rows the row samples once, then blend from them.
                let fv = if pipe.fst { frag.v } else { frag.t * g.inv_q * th };
                let (fu_a, fu_b) = (fu_at(pxa), fu_at(pxb - 1));
                let (fu_lo, fu_hi) = (fu_a.min(fu_b), fu_a.max(fu_b));
                let (y_row, wy, u_lo, u_hi) = if pipe.bilinear {
                    let y = fv - 0.5;
                    let y_row = floor_i32(y);
                    let wy = ((y - y_row as f32) * 256.0) as u32;
                    (y_row, wy, floor_i32(fu_lo - 0.5), floor_i32(fu_hi - 0.5) + 1)
                } else {
                    (floor_i32(fv), 0, floor_i32(fu_lo), floor_i32(fu_hi))
                };
                if u_hi - u_lo < 4096 {
                    self.fill_tex_row(0, y_row, u_lo, u_hi);
                    if pipe.bilinear {
                        self.fill_tex_row(1, y_row + 1, u_lo, u_hi);
                    }
                    let [row0, row1] = std::mem::take(&mut self.scratch.tex_rows);
                    for px in pxa..pxb {
                        let fu = fu_at(px);
                        let texel = if pipe.bilinear {
                            let x = fu - 0.5;
                            let x0 = floor_i32(x);
                            let wx = ((x - x0 as f32) * 256.0) as u32;
                            let i = (x0 - u_lo) as usize;
                            if wx | wy == 0 {
                                row0.data[i]
                            } else {
                                bilerp_rgba(row0.data[i], row0.data[i + 1], row1.data[i], row1.data[i + 1], wx, wy)
                            }
                        } else {
                            row0.data[(floor_i32(fu) - u_lo) as usize]
                        };
                        self.shade_row_px(&row, px as u32, frag, texel);
                    }
                    self.scratch.tex_rows = [row0, row1];
                    continue;
                }
            }
            for px in pxa..pxb {
                let fx = ((px << 4) as f32 + 8.0 - x0 as f32) * g.inv_wid;
                let frag = Frag {
                    s: g.s0 + (g.s1 - g.s0) * fx,
                    u: (g.u0 as f32 + (g.u1 - g.u0) as f32 * fx) / 16.0,
                    ..frag
                };
                let texel = if pipe.tme { self.sample(&frag) } else { 0 };
                self.shade_row_px(&row, px as u32, frag, texel);
            }
        }
    }

    fn tri_rows(&mut self, g: &TriGeom, rows: Rows) {
        let pipe = self.pipe;
        self.scratch.tex_rows[0].key.0 = u64::MAX;
        self.scratch.tex_rows[1].key.0 = u64::MAX;
        let (a, b, c) = (g.a, g.b, g.c);
        for py in rows.iter() {
            let k = (py - g.miny) as i64;
            let (mut w0, mut w1, mut w2) = (g.w0 + g.dy[0] * k, g.w1 + g.dy[1] * k, g.w2 + g.dy[2] * k);
            let row = Row::new(pipe, py as u32);
            for px in g.minx..=g.maxx {
                let (cw0, cw1, cw2) = (w0, w1, w2);
                w0 += g.dx[0];
                w1 += g.dx[1];
                w2 += g.dx[2];
                if cw0 < 0 || cw1 < 0 || cw2 < 0 {
                    continue;
                }
                let (w0, w1, w2) = (cw0, cw1, cw2);
                let l0 = w0 as f32 * g.inv_area;
                let l1 = w1 as f32 * g.inv_area;
                let l2 = w2 as f32 * g.inv_area;
                let rgba = interp3(&g.ca, &g.cb, &g.cc, l0, l1, l2);
                let stqu = interp3(&g.sa, &g.sb, &g.sc, l0, l1, l2);
                let frag = Frag {
                    r: rgba[0],
                    g: rgba[1],
                    b: rgba[2],
                    a: rgba[3],
                    z: (a.z as f64 * l0 as f64 + b.z as f64 * l1 as f64 + c.z as f64 * l2 as f64)
                        as u32,
                    s: stqu[0],
                    t: stqu[1],
                    q: stqu[2],
                    u: stqu[3] / 16.0,
                    v: (a.v as f32 * l0 + b.v as f32 * l1 + c.v as f32 * l2) / 16.0,
                };
                let texel = if pipe.tme { self.sample_cached(&frag) } else { 0 };
                self.shade_row_px(&row, px as u32, frag, texel);
            }
        }
    }

    /// Make `tex_rows[slot]` hold texels `u_lo..=u_hi` of texture row `y`
    /// (wrapped/clamped like any sample); reuses the previous contents when
    /// they already cover the request.
    fn fill_tex_row(&mut self, slot: usize, y: i32, u_lo: i32, u_hi: i32) {
        let ti = &self.pipe.tex;
        let key = (ti.tex0, y);
        let covers = |row: &TexRow| {
            row.key == key && row.u_lo == u_lo && row.u_lo + row.data.len() as i32 > u_hi
        };
        if covers(&self.scratch.tex_rows[slot]) {
            return;
        }
        // The other slot may hold this very row (the previous output row's
        // second tap row becomes this row's first).
        if covers(&self.scratch.tex_rows[slot ^ 1]) {
            self.scratch.tex_rows.swap(0, 1);
            return;
        }
        let mut row = std::mem::take(&mut self.scratch.tex_rows[slot]);
        row.key = key;
        row.u_lo = u_lo;
        row.data.clear();
        row.data.reserve((u_hi - u_lo + 1) as usize);
        let v = wrap(y, ti.wmt, ti.th as i32, ti.minv, ti.maxv) as u32;
        // The row's texture line is fixed: address it as base + column
        // table for the formats the fast paths matter for.
        match ti.psm {
            PSMT8 => {
                let base = layout::row_base8(ti.tbp, ti.tbw, v);
                for u in u_lo..=u_hi {
                    let u = wrap(u, ti.wms, ti.tw as i32, ti.minu, ti.maxu) as u32;
                    let idx = self.canvas.rd8((base + layout::col_off8(v, u)) & (VRAM_SIZE - 1));
                    row.data.push(self.clut[idx as usize]);
                }
            }
            PSMCT32 | PSMCT24 | PSMT8H | PSMT4HL | PSMT4HH => {
                let base = layout::row_base32(ti.tbp, ti.tbw, v, false);
                for u in u_lo..=u_hi {
                    let u = wrap(u, ti.wms, ti.tw as i32, ti.minu, ti.maxu) as u32;
                    let px = self.canvas.rd32((base + layout::col_off32(v, u, false)) & (VRAM_SIZE - 1));
                    row.data.push(match ti.psm {
                        PSMCT32 => px,
                        PSMCT24 => (px & 0xFF_FFFF) | (((ti.texa & 0xFF) as u32) << 24),
                        PSMT8H => self.clut[(px >> 24) as usize],
                        PSMT4HL => self.clut[((px >> 24) & 0xF) as usize + ti.clut_base],
                        _ => self.clut[(px >> 28) as usize + ti.clut_base],
                    });
                }
            }
            _ => {
                for u in u_lo..=u_hi {
                    row.data.push(self.texel(u, y));
                }
            }
        }
        self.scratch.tex_rows[slot] = row;
    }

    /// [`Painter::sample`] through the decoded-row cache: texels come from
    /// `tex_rows`, refilled in 32-texel chunks around a miss. Same texels
    /// and weights as the direct path.
    fn sample_cached(&mut self, frag: &Frag) -> u32 {
        let pipe = self.pipe;
        let ti = &pipe.tex;
        let (fu, fv) = if pipe.fst {
            (frag.u, frag.v)
        } else {
            let q = if frag.q.abs() < 1e-9 { 1.0 } else { frag.q };
            let inv_q = 1.0 / q;
            (frag.s * inv_q * ti.tw as f32, frag.t * inv_q * ti.th as f32)
        };
        if !pipe.bilinear {
            return self.cached_texel(0, floor_i32(fu), floor_i32(fv));
        }
        let x = fu - 0.5;
        let y = fv - 0.5;
        let (x0, y0) = (floor_i32(x), floor_i32(y));
        let fx = ((x - x0 as f32) * 256.0) as u32;
        let fy = ((y - y0 as f32) * 256.0) as u32;
        if fx | fy == 0 {
            return self.cached_texel(0, x0, y0);
        }
        // Common case: both rows cached and the 2x2 footprint inside them.
        let (r0, r1) = (&self.scratch.tex_rows[0], &self.scratch.tex_rows[1]);
        if r0.key == (ti.tex0, y0)
            && r1.key == (ti.tex0, y0 + 1)
            && x0 >= r0.u_lo
            && x0 + 1 < r0.u_lo + r0.data.len() as i32
            && x0 >= r1.u_lo
            && x0 + 1 < r1.u_lo + r1.data.len() as i32
        {
            let (i0, i1) = ((x0 - r0.u_lo) as usize, (x0 - r1.u_lo) as usize);
            return bilerp_rgba(r0.data[i0], r0.data[i0 + 1], r1.data[i1], r1.data[i1 + 1], fx, fy);
        }
        let t00 = self.cached_texel(0, x0, y0);
        let t10 = self.cached_texel(0, x0 + 1, y0);
        let t01 = self.cached_texel(1, x0, y0 + 1);
        let t11 = self.cached_texel(1, x0 + 1, y0 + 1);
        bilerp_rgba(t00, t10, t01, t11, fx, fy)
    }

    #[inline(always)]
    fn cached_texel(&mut self, slot: usize, u: i32, y: i32) -> u32 {
        let row = &self.scratch.tex_rows[slot];
        if row.key == (self.pipe.tex.tex0, y) && u >= row.u_lo && u < row.u_lo + row.data.len() as i32 {
            return row.data[(u - row.u_lo) as usize];
        }
        self.fill_tex_row(slot, y, u - 8, u + 23);
        self.scratch.tex_rows[slot].data[8]
    }

    /// The pixel pipeline proper: `row` carries the scanline's frame/Z
    /// buffer bases so per-pixel addressing is a table lookup. The caller
    /// has already clipped to the scissor box. Color math is integer, as
    /// on hardware: interpolated colors truncate to 8 bits first.
    #[inline(always)]
    fn shade_row_px(&mut self, row: &Row, x: u32, frag: Frag, texel: u32) {
        let pipe = self.pipe;
        let y = row.y;
        self.scratch.pixels += 1;
        #[cfg(feature = "profile")]
        {
            let k = (pipe.kind as usize) | ((pipe.tme as usize) << 2) | ((pipe.bilinear as usize) << 3)
                | ((pipe.abe as usize) << 4) | ((pipe.tex.psm as usize & 0x3F) << 5);
            crate::prof::count_pixel(k);
        }

        // Source color: vertex color, optionally combined with a texel.
        let mut r = frag.r as u32;
        let mut g = frag.g as u32;
        let mut b = frag.b as u32;
        let mut a = frag.a as u32;
        if pipe.tme {
            self.scratch.tex_samples[pipe.tex.psm as usize] += 1;
            let (tr, tg, tb, ta) = (texel & 0xFF, (texel >> 8) & 0xFF, (texel >> 16) & 0xFF, texel >> 24);
            match pipe.tfx {
                0 => {
                    // MODULATE
                    r = ((tr * r) >> 7).min(255);
                    g = ((tg * g) >> 7).min(255);
                    b = ((tb * b) >> 7).min(255);
                    if pipe.tcc {
                        a = ((ta * a) >> 7).min(255);
                    }
                }
                1 => {
                    // DECAL
                    r = tr;
                    g = tg;
                    b = tb;
                    if pipe.tcc {
                        a = ta;
                    }
                }
                _ => {
                    // HIGHLIGHT/HIGHLIGHT2: approximate.
                    r = (((tr * r) >> 7) + a).min(255);
                    g = (((tg * g) >> 7) + a).min(255);
                    b = (((tb * b) >> 7) + a).min(255);
                    if pipe.tcc {
                        a = ta;
                    }
                }
            }
        }

        // Alpha test.
        if pipe.ate {
            let aref = pipe.aref;
            let pass = match pipe.atst {
                0 => false,
                1 => true,
                2 => a < aref,
                3 => a <= aref,
                4 => a == aref,
                5 => a >= aref,
                6 => a > aref,
                _ => a != aref,
            };
            if !pass {
                match pipe.afail {
                    0 => return, // KEEP
                    1 => {}      // FB_ONLY: continue without z write
                    2 => return, // ZB_ONLY: no color -> nothing visible
                    _ => {}      // RGB_ONLY
                }
            }
        }

        // Depth test (linear z buffer, PSMZ32-style storage).
        let zmask = pipe.zmask;
        let z_off = (row.z_base + layout::col_off32(y, x, true)) & (VRAM_SIZE - 1);
        let fb_off = (row.fb_base + layout::col_off32(y, x, false)) & (VRAM_SIZE - 1);
        if pipe.zte {
            let zcur = self.canvas.rd32(z_off);
            let z = frag.z & zmask;
            let pass = match pipe.ztst {
                0 => false,
                1 => true,
                2 => z >= (zcur & zmask),
                _ => z > (zcur & zmask),
            };
            if !pass {
                return;
            }
            if !pipe.zmsk {
                self.canvas.wr32(z_off, (zcur & !zmask) | z);
            }
        }

        // Destination blend.
        let dst = self.canvas.rd32(fb_off);

        if pipe.abe {
            // ALPHA: Cv = ((A - B) * C >> 7) + D, on three 21-bit lanes of a
            // u64 (R at 0, G at 21, B at 42). With the identity
            // floor(((A-B)*C + 128*D) / 128) = ((A-B)*C >> 7) + D and a
            // per-lane bias of 2^16 the lanes stay positive and disjoint;
            // each lane then decodes to (lane >> 7) - 512, clamped to 8 bits.
            let src = spread21(r | (g << 8) | (b << 16));
            let dstc = spread21(dst & 0xFF_FFFF);
            let pick = |k: u8| -> u64 {
                match k {
                    0 => src,
                    1 => dstc,
                    _ => 0,
                }
            };
            let alpha = match pipe.blend_c {
                0 => a as u64,
                1 => (dst >> 24) as u64,
                _ => pipe.blend_fix as u64,
            };
            const BIAS: u64 = (1 << 16) | (1 << (16 + 21)) | (1 << (16 + 42));
            let x = pick(pipe.blend_a) * alpha + pick(pipe.blend_d) * 128 + BIAS
                - pick(pipe.blend_b) * alpha;
            let lane = |sh: u32| -> u32 {
                let v = (((x >> sh) & 0x1F_FFFF) >> 7) as i32 - 512;
                v.clamp(0, 255) as u32
            };
            r = lane(0);
            g = lane(21);
            b = lane(42);
        }

        let out = r | (g << 8) | (b << 16) | (a.min(255) << 24);
        let mut merged = (out & !pipe.fbmsk) | (dst & pipe.fbmsk);
        if pipe.fb24 {
            merged = (merged & 0xFF_FFFF) | (dst & 0xFF00_0000);
        }
        self.canvas.wr32(fb_off, merged);
    }

    /// Texture sample as RGBA8 (nearest or bilinear per TEX1 MMAG).
    #[inline(always)]
    fn sample(&self, frag: &Frag) -> u32 {
        let pipe = self.pipe;
        let ti = &pipe.tex;

        // FST: UV addressing vs STQ. Texel-space coordinates, fractional.
        let (fu, fv) = if pipe.fst {
            (frag.u, frag.v)
        } else {
            let q = if frag.q.abs() < 1e-9 { 1.0 } else { frag.q };
            let inv_q = 1.0 / q;
            (frag.s * inv_q * ti.tw as f32, frag.t * inv_q * ti.th as f32)
        };

        // TEX1 MMAG selects the magnification filter; minification and
        // mipmaps are not modelled, so it decides for every sample.
        if !pipe.bilinear {
            return self.texel(floor_i32(fu), floor_i32(fv));
        }
        let x = fu - 0.5;
        let y = fv - 0.5;
        let (x0, y0) = (floor_i32(x), floor_i32(y));
        // Weights in 1/256; exact texel centres skip the blend.
        let fx = ((x - x0 as f32) * 256.0) as u32;
        let fy = ((y - y0 as f32) * 256.0) as u32;
        if fx | fy == 0 {
            return self.texel(x0, y0);
        }
        let t00 = self.texel(x0, y0);
        let t10 = self.texel(x0 + 1, y0);
        let t01 = self.texel(x0, y0 + 1);
        let t11 = self.texel(x0 + 1, y0 + 1);
        bilerp_rgba(t00, t10, t01, t11, fx, fy)
    }

    /// One RGBA8 texel at integer texel coordinates, after CLAMP wrapping.
    #[inline(always)]
    fn texel(&self, u: i32, v: i32) -> u32 {
        let ti = &self.pipe.tex;
        let cv = self.canvas;
        let u = wrap(u, ti.wms, ti.tw as i32, ti.minu, ti.maxu) as u32;
        let v = wrap(v, ti.wmt, ti.th as i32, ti.minv, ti.maxv) as u32;
        let (tbp, tbw) = (ti.tbp, ti.tbw);
        match ti.psm {
            PSMCT32 => cv.read_psmct32(tbp, tbw, u, v),
            PSMCT24 => (cv.read_psmct32(tbp, tbw, u, v) & 0xFF_FFFF) | (((ti.texa & 0xFF) as u32) << 24),
            PSMCT16 | PSMCT16S | PSMZ16 | PSMZ16S => {
                expand16(cv.read_psmct16(tbp, tbw, u, v, ti.psm), ti.texa)
            }
            PSMZ32 => cv.read_psmz32(tbp, tbw, u, v),
            PSMZ24 => (cv.read_psmz32(tbp, tbw, u, v) & 0xFF_FFFF) | (((ti.texa & 0xFF) as u32) << 24),
            PSMT8 => self.clut[cv.read_psmt8(tbp, tbw, u, v) as usize],
            PSMT4 => self.clut[cv.read_psmt4(tbp, tbw, u, v) as usize + ti.clut_base],
            PSMT8H => self.clut[(cv.read_psmct32(tbp, tbw, u, v) >> 24) as usize],
            PSMT4HL => {
                self.clut[((cv.read_psmct32(tbp, tbw, u, v) >> 24) & 0xF) as usize + ti.clut_base]
            }
            PSMT4HH => {
                self.clut[(cv.read_psmct32(tbp, tbw, u, v) >> 28) as usize + ti.clut_base]
            }
            _ => 0xFF00_FFFF,
        }
    }
}

/// Per-scanline frame/Z buffer bases (see `layout::row_base32`).
struct Row {
    y: u32,
    fb_base: usize,
    z_base: usize,
}

impl Row {
    #[inline(always)]
    fn new(pipe: &PixelPipe, y: u32) -> Self {
        Self {
            y,
            fb_base: layout::row_base32(pipe.fbp, pipe.fbw, y, false),
            z_base: layout::row_base32(pipe.zbp, pipe.fbw, y, true),
        }
    }
}

/// Drawing environment decoded once per primitive (see `pixel_pipe`).
struct PixelPipe {
    /// Primitive kind being drawn (PRIM bits 0-2), for the pixel histogram.
    #[cfg_attr(not(feature = "profile"), allow(dead_code))]
    kind: u8,
    // Scissor, inclusive pixel bounds.
    scx0: i32,
    scx1: i32,
    scy0: i32,
    scy1: i32,
    tme: bool,
    fst: bool,
    abe: bool,
    tfx: u8,
    tcc: bool,
    bilinear: bool,
    tex: TexInfo,
    ate: bool,
    atst: u8,
    aref: u32,
    afail: u8,
    zte: bool,
    ztst: u8,
    zbp: u32,
    zmsk: bool,
    zmask: u32,
    fbp: u32,
    fbw: u32,
    fb24: bool,
    fbmsk: u32,
    /// ALPHA register selectors (0 = source, 1 = destination, 2 = zero /
    /// FIX for `blend_c`) and the FIX value.
    blend_a: u8,
    blend_b: u8,
    blend_c: u8,
    blend_d: u8,
    blend_fix: u32,
}

/// TEX0/CLAMP fields decoded once per primitive.
struct TexInfo {
    tex0: u64,
    tbp: u32,
    tbw: u32,
    psm: u32,
    tw: u32,
    th: u32,
    wms: u64,
    wmt: u64,
    minu: i32,
    maxu: i32,
    minv: i32,
    maxv: i32,
    texa: u64,
    /// Palette index width (0 for direct-color formats).
    clut_bits: u8,
    /// First CLUT cache entry: CSA offset in 16-entry slots (4-bit only).
    clut_base: usize,
}

impl TexInfo {
    fn new(ctx: &Context, texa: u64) -> Self {
        let tex0 = ctx.tex0;
        let psm = ((tex0 >> 20) & 0x3F) as u32;
        let clut_bits = match psm {
            PSMT8 | PSMT8H => 8,
            PSMT4 | PSMT4HL | PSMT4HH => 4,
            _ => 0,
        };
        // CLAMP register: 0 repeat, 1 clamp, 2 region clamp, 3 region repeat.
        Self {
            tex0,
            tbp: (tex0 & 0x3FFF) as u32,
            tbw: ((tex0 >> 14) & 0x3F) as u32,
            psm,
            tw: 1u32 << ((tex0 >> 26) & 0xF).min(10),
            th: 1u32 << ((tex0 >> 30) & 0xF).min(10),
            wms: ctx.clamp & 3,
            wmt: (ctx.clamp >> 2) & 3,
            minu: ((ctx.clamp >> 4) & 0x3FF) as i32,
            maxu: ((ctx.clamp >> 14) & 0x3FF) as i32,
            minv: ((ctx.clamp >> 24) & 0x3FF) as i32,
            maxv: ((ctx.clamp >> 34) & 0x3FF) as i32,
            texa,
            clut_bits,
            clut_base: if clut_bits == 4 { ((tex0 >> 56) & 0x1F) as usize * 16 } else { 0 },
        }
    }

    /// Everything the decoded CLUT depends on: CBP/CPSM/CSM/CSA, the index
    /// width, and the TEXA fields used to expand 16-bit entries.
    fn clut_key(&self) -> u64 {
        ((self.tex0 >> 37) & 0xFF_FFFF)
            | ((self.clut_bits as u64) << 24)
            | ((self.texa & 0xFF) << 32)
            | (((self.texa >> 15) & 1) << 40)
            | (((self.texa >> 32) & 0xFF) << 41)
    }
}

#[inline]
fn edge(x0: i32, y0: i32, x1: i32, y1: i32, x: i32, y: i32) -> i64 {
    (x1 - x0) as i64 * (y - y0) as i64 - (y1 - y0) as i64 * (x - x0) as i64
}

/// Bilinear blend of a 2x2 texel footprint (`t00 t10` top, `t01 t11`
/// bottom) with weights `wx`, `wy` in 1/256: horizontal lerps first, then
/// vertical, each truncating to 8 bits like [`lerp_rgba`].
#[inline(always)]
fn bilerp_rgba(t00: u32, t10: u32, t01: u32, t11: u32, wx: u32, wy: u32) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: SSE2 is part of the x86-64 baseline.
        unsafe { bilerp_sse2(t00, t10, t01, t11, wx, wy) }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        lerp_rgba(lerp_rgba(t00, t10, wx), lerp_rgba(t01, t11, wx), wy)
    }
}

/// Both rows' four channels in eight 16-bit lanes: 255 * 256 fits, and
/// `mullo` keeps the low 16 bits either way, so the sums are exact.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bilerp_sse2(t00: u32, t10: u32, t01: u32, t11: u32, wx: u32, wy: u32) -> u32 {
    use core::arch::x86_64::*;
    // SAFETY: SSE2 baseline; pure register arithmetic.
    unsafe {
        let zero = _mm_setzero_si128();
        let left = _mm_unpacklo_epi8(_mm_set_epi64x(0, (t00 as i64) | ((t01 as i64) << 32)), zero);
        let right = _mm_unpacklo_epi8(_mm_set_epi64x(0, (t10 as i64) | ((t11 as i64) << 32)), zero);
        let x = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(left, _mm_set1_epi16((256 - wx) as i16)),
                _mm_mullo_epi16(right, _mm_set1_epi16(wx as i16)),
            ),
            8,
        );
        let top = x;
        let bottom = _mm_srli_si128(x, 8);
        let y = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(top, _mm_set1_epi16((256 - wy) as i16)),
                _mm_mullo_epi16(bottom, _mm_set1_epi16(wy as i16)),
            ),
            8,
        );
        _mm_cvtsi128_si32(_mm_packus_epi16(y, y)) as u32
    }
}

/// Blend two RGBA8 pixels with weight `w` (0..=256) for `b`, all four
/// channels at once: the R/B and G/A pairs each get 16 bits of headroom.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn lerp_rgba(a: u32, b: u32, w: u32) -> u32 {
    let inv = 256 - w;
    let rb = ((a & 0x00FF_00FF) * inv + (b & 0x00FF_00FF) * w) >> 8;
    let ga = ((a >> 8) & 0x00FF_00FF) * inv + ((b >> 8) & 0x00FF_00FF) * w;
    (rb & 0x00FF_00FF) | (ga & 0xFF00_FF00)
}

/// `a*l0 + b*l1 + c*l2` on four lanes, evaluated as the scalar form
/// `(a*l0 + b*l1) + c*l2` per lane so results match it bit for bit.
#[inline(always)]
fn interp3(a: &[f32; 4], b: &[f32; 4], c: &[f32; 4], l0: f32, l1: f32, l2: f32) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        use core::arch::x86_64::*;
        // SAFETY: SSE baseline; unaligned loads/stores of the arrays.
        unsafe {
            let va = _mm_loadu_ps(a.as_ptr());
            let vb = _mm_loadu_ps(b.as_ptr());
            let vc = _mm_loadu_ps(c.as_ptr());
            let v = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(va, _mm_set1_ps(l0)), _mm_mul_ps(vb, _mm_set1_ps(l1))),
                _mm_mul_ps(vc, _mm_set1_ps(l2)),
            );
            let mut out = [0f32; 4];
            _mm_storeu_ps(out.as_mut_ptr(), v);
            out
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        [
            a[0] * l0 + b[0] * l1 + c[0] * l2,
            a[1] * l0 + b[1] * l1 + c[1] * l2,
            a[2] * l0 + b[2] * l1 + c[2] * l2,
            a[3] * l0 + b[3] * l1 + c[3] * l2,
        ]
    }
}

/// RGB8 (R low) into three 21-bit lanes of a u64.
#[inline(always)]
fn spread21(c: u32) -> u64 {
    (c & 0xFF) as u64 | (((c >> 8) & 0xFF) as u64) << 21 | (((c >> 16) & 0xFF) as u64) << 42
}

/// `f32::floor` as an integer, without the libm call the SSE2 baseline
/// needs for the intrinsic.
#[inline(always)]
fn floor_i32(x: f32) -> i32 {
    let i = x as i32;
    if (i as f32) > x { i - 1 } else { i }
}

#[inline]
fn px_clip(v: i32) -> i32 {
    v.clamp(0, 2048)
}

#[inline(always)]
fn wrap(c: i32, mode: u64, size: i32, min: i32, max: i32) -> i32 {
    match mode {
        // Texture sizes are powers of two, so REPEAT is a mask.
        0 => c & (size - 1),
        1 => c.clamp(0, size - 1),
        2 => c.clamp(min, max.max(min)),
        _ => (c & min) | max,
    }
}

/// 16-bit -> 32-bit expansion using TEXA.
#[inline]
fn expand16(px: u16, texa: u64) -> u32 {
    let r = ((px & 0x1F) << 3) as u32;
    let g = (((px >> 5) & 0x1F) << 3) as u32;
    let b = (((px >> 10) & 0x1F) << 3) as u32;
    let ta0 = (texa & 0xFF) as u32;
    let ta1 = ((texa >> 32) & 0xFF) as u32;
    let aem = texa & (1 << 15) != 0;
    let a = if px & 0x8000 != 0 {
        ta1
    } else if aem && px & 0x7FFF == 0 {
        0
    } else {
        ta0
    };
    r | (g << 8) | (b << 16) | (a << 24)
}
