//! Pixel pipeline: triangle/sprite rasterization, texture sampling,
//! alpha blending and the frame-buffer write path.
//!
//! [`Gs`] decodes a primitive into geometry plus a [`PixelPipe`] and queues
//! it in a [`Batch`]; a flush hands scanlines to [`Painter`]s, which own
//! nothing but references: the shared [`Canvas`] and CLUT, the pipe, and a
//! per-thread [`Scratch`]. A flush runs the whole queue on the worker pool
//! in two-row bands (rows `2k, 2k+1` share the 32-bit column layout's cache
//! lines): every lane walks all queued primitives in order but only touches
//! its own rows, so no two threads write the same pixel and each pixel sees
//! its primitives in program order — results are bit-identical to a serial
//! pass. Batching amortizes the pool dispatch over many primitives, which
//! is what makes the small-triangle rushes (the OSD boot towers) parallel.
//! The batch is flushed before anything else reads or writes VRAM (IMAGE
//! transfers, local copies, CLUT decodes, scanout) and a primitive whose
//! texture may alias what the queue wrote flushes first; one that samples
//! its *own* target runs inline, since it would otherwise depend on thread
//! timing.

use super::*;

/// Sprite-row bilinear weight classes, pixels each: nearest, copy (both
/// weights 0), constant-wx row, constant 2-tap vertical, constant 4-tap,
/// varying wx. Temporary tuning aid for the profile report.
#[cfg(feature = "profile")]
pub static BIL_CLASSES: [core::sync::atomic::AtomicU64; 6] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 6];

/// Texels decoded into the row cache, by texture PSM: how much decode
/// traffic the fills cause on top of `tex_samples`.
#[cfg(feature = "profile")]
pub static FILL_TEXELS: [core::sync::atomic::AtomicU64; 64] =
    [const { core::sync::atomic::AtomicU64::new(0) }; 64];

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
    /// Prefiltered texel run for constant-weight bilinear sprite rows.
    filtered: Vec<u32>,
    pixels: u64,
    tex_samples: [u64; 64],
}

impl Default for Scratch {
    fn default() -> Self {
        Self { tex_rows: Default::default(), filtered: Vec::new(), pixels: 0, tex_samples: [0; 64] }
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

/// Decoded palettes for the primitives a batch holds.
///
/// A queued primitive names a block here rather than carrying its own copy,
/// so switching palette mid-batch costs a block instead of a flush. Blocks
/// stay live until the batch is drawn; `reset` then frees them all at once.
pub(super) struct ClutPool {
    /// `SLOTS` blocks of 512 entries. 512 because a 4-bit texture's CSA
    /// selects a 16-entry window anywhere in the CLUT.
    data: Vec<u32>,
    /// Lookup key per block, or `u64::MAX` for one that may not be reused
    /// (its palette's memory has changed since it was decoded).
    keys: Vec<u64>,
    used: usize,
}

/// Blocks a single batch may use before it has to be drawn. A mission frame
/// alternates a few dozen palettes; this is well clear of that at 512 KiB.
const CLUT_SLOTS: usize = 128;
const CLUT_ENTRIES: usize = 512;

impl Default for ClutPool {
    fn default() -> Self {
        Self {
            data: vec![0; CLUT_SLOTS * CLUT_ENTRIES],
            keys: vec![u64::MAX; CLUT_SLOTS],
            used: 0,
        }
    }
}

impl ClutPool {
    /// The block starting at `off`, as the pixel paths index it.
    #[inline(always)]
    fn block(&self, off: usize) -> &[u32] {
        &self.data[off..off + CLUT_ENTRIES]
    }

    /// Forget every block. Called once the batch referencing them is drawn.
    fn reset(&mut self) {
        self.used = 0;
        self.keys.fill(u64::MAX);
    }

    /// Stop handing out matches without disturbing blocks in use: their
    /// palettes' memory has changed, so a later primitive wanting the same
    /// setup has to decode it again into a block of its own.
    fn forget_keys(&mut self) {
        self.keys[..self.used].fill(u64::MAX);
    }

    fn find(&self, key: u64) -> Option<usize> {
        self.keys[..self.used].iter().position(|&k| k == key).map(|i| i * CLUT_ENTRIES)
    }

    fn alloc(&mut self, key: u64) -> Option<usize> {
        if self.used == CLUT_SLOTS {
            return None;
        }
        let slot = self.used;
        self.used += 1;
        self.keys[slot] = key;
        Some(slot * CLUT_ENTRIES)
    }
}

/// Pixels a batch must cover before its flush is split across threads.
pub(super) const PARALLEL_MIN_PIXELS: i64 = 4096;
/// Bands (tasks) a flush is cut into.
pub(super) const PARALLEL_LANES: usize = 14;
/// Pixel estimate that triggers a batch flush on its own.
const BATCH_MAX_PIXELS: i64 = 1 << 20;

/// Queued primitive geometry (decoded, self-contained).
enum Prim {
    Tri(TriGeom),
    Sprite(SpriteGeom),
    Line(LineGeom),
}

struct Queued {
    pipe: PixelPipe,
    prim: Prim,
    start: i32,
    end: i32,
    /// Internal-2x twin: drawn into the overlay canvas in 2x coordinates.
    hi: bool,
}

/// Primitives decoded and queued for one parallel rasterization pass.
#[derive(Default)]
pub(super) struct Batch {
    queued: Vec<Queued>,
    /// Bounding-box pixel estimate of the queue.
    px: i64,
    /// Merged block ranges the queue writes (frame and Z), for the
    /// read-after-write flush test, each tagged with the buffer that wrote
    /// it (see [`Write`]).
    writes: Vec<Write>,
    /// Merged block ranges the queue samples as texture, for the
    /// write-after-read flush test.
    reads: Vec<(u32, u32)>,
}

/// A block range the queue writes, and the buffer whose addressing produced
/// it. A parallel flush bands rows by `py` alone, so two buffers that share
/// memory at different bases put one pixel in two different bands: the
/// lanes then write it in an order nobody controls. Ranges that overlap
/// under different `(base, bw)` therefore cannot share a batch.
struct Write {
    start: u32,
    end: u32,
    base: u32,
    bw: u32,
}

impl Batch {
    fn note_write(&mut self, r: std::ops::Range<u32>, base: u32, bw: u32) {
        for w in self.writes.iter_mut() {
            if w.base == base && w.bw == bw && r.start <= w.end && w.start <= r.end {
                w.start = w.start.min(r.start);
                w.end = w.end.max(r.end);
                return;
            }
        }
        self.writes.push(Write { start: r.start, end: r.end, base, bw });
    }

    fn note_read(&mut self, r: std::ops::Range<u32>) {
        for t in self.reads.iter_mut() {
            if r.start <= t.1 && t.0 <= r.end {
                t.0 = t.0.min(r.start);
                t.1 = t.1.max(r.end);
                return;
            }
        }
        self.reads.push((r.start, r.end));
    }

    /// Would this read see memory the queue already wrote?
    fn after_write(&self, r: &std::ops::Range<u32>) -> bool {
        self.writes.iter().any(|w| r.start < w.end && w.start < r.end)
    }

    /// Would this write disturb memory the queue already sampled? A lane
    /// replays the whole queue for its own rows, so a later primitive's
    /// write reaches an earlier primitive's read as soon as the two land in
    /// different bands — the batch has to be split between them.
    fn before_read(&self, r: &std::ops::Range<u32>) -> bool {
        self.reads.iter().any(|t| r.start < t.1 && t.0 < r.end)
    }

    /// Would this write land on memory the queue already writes through a
    /// different row mapping?
    fn aliases(&self, r: &std::ops::Range<u32>, base: u32, bw: u32) -> bool {
        self.writes
            .iter()
            .any(|w| (w.base != base || w.bw != bw) && r.start < w.end && w.start < r.end)
    }
}

/// Line geometry: the DDA re-derives everything else per step.
struct LineGeom {
    a: Vertex,
    b: Vertex,
    gouraud: bool,
    steps: i32,
}

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
    /// Per-pixel steps of the colour and STQU attributes, v and z along a
    /// row, for the incremental fast loop.
    d_rgba: [f32; 4],
    d_stqu: [f32; 4],
    d_v: f32,
    d_z: f64,
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
            // Points are rare: draw in place, keeping order with the queue.
            self.flush_batch();
            let mut p = Painter {
                canvas: &self.canvas,
                tex: &self.canvas,
                clut: self.clut.block(pipe.clut_off as usize),
                pipe: &pipe,
                scratch: &mut self.scratch,
            };
            let texel = if pipe.tme { p.sample(&frag) } else { 0 };
            let row = Row::new(&pipe, py as u32);
            p.shade_row_px(&row, px as u32, frag, texel);
            self.mirror_rect(&pipe, px, px, py, py + 1);
        }
        self.prims_drawn += 1;
        self.merge_scratch();
    }

    /// Rasterize a line with a pixel-step DDA over the major axis. As on
    /// hardware the end point's pixel is not drawn, so the joints of a
    /// strip land exactly once — the PS2 logo's additive wireframe counts
    /// on that. Attributes interpolate (or stick to the second vertex's
    /// colour when shading is flat), and pixels go through the generic
    /// per-pixel pipeline.
    pub(super) fn draw_line(&mut self) {
        let _p = crate::prof::scope(crate::prof::Slot::GsDraw);
        self.log_target(self.prim, "line", 2);
        let (a, b) = (self.vq[0], self.vq[1]);
        let pipe = self.pixel_pipe();
        tracing::trace!(target: "ps2_core::gs::line",
            v0 = format_args!("({},{},z={:#x})", a.x as f32 / 16.0, a.y as f32 / 16.0, a.z),
            v1 = format_args!("({},{})", b.x as f32 / 16.0, b.y as f32 / 16.0),
            rgba = format_args!("{},{},{},{}", b.r, b.g, b.b, b.a),
            prim = format_args!("{:#x}", self.prim),
            fbp = (self.ctx[((self.prim >> 9) & 1) as usize].frame & 0x1FF) * 32,
            "line");
        let gouraud = self.prim & 8 != 0;
        let Some((geom, start, end, px)) = Self::line_prim(&pipe, a, b, gouraud) else {
            return;
        };
        self.prims_drawn += 1;
        let queued = self.enqueue(pipe.clone(), Prim::Line(geom), start, end, px, None);
        if let Some(p2) = self.scaled_pipe(&pipe) {
            if !queued {
                self.mirror_rect(&pipe, pipe.scx0, pipe.scx1, start, end);
            } else if let Some((g2, s2, e2, px2)) =
                Self::line_prim(&p2, Self::scale_vertex(a), Self::scale_vertex(b), gouraud)
            {
                self.enqueue_hi(p2, Prim::Line(g2), s2, e2, px2);
            }
        }
    }

    /// Line geometry and its row/pixel extent for [`Gs::enqueue`].
    fn line_prim(pipe: &PixelPipe, a: Vertex, b: Vertex, gouraud: bool) -> Option<(LineGeom, i32, i32, i64)> {
        let (fy0, fy1) = (a.y as f32 / 16.0, b.y as f32 / 16.0);
        let steps = ((b.x - a.x) as f32 / 16.0)
            .abs()
            .max((fy1 - fy0).abs())
            .round() as i32;
        if steps <= 0 {
            return None;
        }
        let start = (fy0.min(fy1).floor() as i32).max(pipe.scy0);
        let end = ((fy0.max(fy1).ceil() as i32 + 1).min(pipe.scy1 + 1)).max(start);
        Some((LineGeom { a, b, gouraud, steps }, start, end, steps as i64))
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

    /// Texel rows one page of `psm` holds, and the block distance between
    /// consecutive rows of pages. Mirrors the arithmetic in `layout`'s
    /// `addr*`: getting this wrong under-states where a texture lives and
    /// lets a read-after-write hazard through.
    fn tex_page_geom(psm: u32, tbw: u32) -> (u32, u32) {
        let bw = tbw.max(1);
        match psm {
            // 128x64 texel pages, addressed with half the declared width.
            PSMT8 => (64, (bw >> 1) * 32),
            // 128x128 texel pages, likewise.
            PSMT4 => (128, (bw >> 1) * 32),
            PSMCT16 | PSMCT16S | PSMZ16 | PSMZ16S => (64, bw * 32),
            // 32-bit pages, including the ones packed into their alpha.
            _ => (32, bw * 32),
        }
    }

    /// Block range the texture may be sampled from (conservative): `tex_v`
    /// is the texel row range the primitive samples (`None` = unknown,
    /// assume the whole declared height). `None` result = untextured.
    fn tex_blocks(pipe: &PixelPipe, tex_v: Option<(f32, f32)>) -> Option<std::ops::Range<u32>> {
        if !pipe.tme {
            return None;
        }
        let ti = &pipe.tex;
        // REGION_CLAMP and REGION_REPEAT map a sample to a row `wrap` picks
        // from MINV/MAXV, which can sit past the declared height — so they
        // neither trust the sampled range nor stop at `th`.
        let (v_lo, v_hi) = match (ti.wmt, tex_v) {
            // Sampled range plus one for the bilinear tap, when it cannot wrap.
            (0 | 1, Some((lo, hi))) if lo >= 0.0 && hi + 1.0 < ti.th as f32 => {
                (lo as u32, hi as u32 + 1)
            }
            (2, _) => (0, ti.th.max(ti.maxv.max(ti.minv).max(0) as u32 + 1)),
            (3, _) => (0, ti.th.max((ti.minv | ti.maxv).max(0) as u32 + 1)),
            _ => (0, ti.th),
        };
        let (page_rows, stride) = Self::tex_page_geom(ti.psm, ti.tbw);
        // A single-page-wide buffer has a zero stride; it still spans a page.
        let span = stride.max(32);
        Some(ti.tbp + (v_lo / page_rows) * stride..ti.tbp + (v_hi / page_rows) * stride + span)
    }

    /// Block ranges a primitive covering rows `0..rows` writes: frame and Z
    /// buffer, `None` where fully masked. Only written buffers can feed
    /// back into a texture.
    fn written_blocks(pipe: &PixelPipe, rows: i32) -> (Option<std::ops::Range<u32>>, Option<std::ops::Range<u32>>) {
        let target_pages = ((rows.max(0) as u32).div_ceil(32)) * pipe.fbw.max(1);
        let fb = (pipe.fbmsk != u32::MAX).then(|| pipe.fbp..pipe.fbp + target_pages * 32);
        let zb = (pipe.zte && !pipe.zmsk).then(|| pipe.zbp..pipe.zbp + target_pages * 32);
        (fb, zb)
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
        let pipe = self.pixel_pipe();
        let (geom, rya, ryb, px, tex_v) = Self::sprite_prim(&pipe, v0, v1);
        let queued = self.enqueue(pipe.clone(), Prim::Sprite(geom), rya, ryb, px, Some(tex_v));
        if let Some(p2) = self.scaled_pipe(&pipe) {
            if !queued {
                self.mirror_rect(&pipe, pipe.scx0, pipe.scx1, rya, ryb);
            } else {
                let (g2, s2, e2, px2, _) =
                    Self::sprite_prim(&p2, Self::scale_vertex(v0), Self::scale_vertex(v1));
                self.enqueue_hi(p2, Prim::Sprite(g2), s2, e2, px2);
            }
        }
    }

    /// Sprite geometry and its row/pixel extent for [`Gs::enqueue`].
    fn sprite_prim(pipe: &PixelPipe, v0: Vertex, v1: Vertex) -> (SpriteGeom, i32, i32, i64, (f32, f32)) {
        let (x0, x1) = (v0.x.min(v1.x), v0.x.max(v1.x));
        let (y0, y1) = (v0.y.min(v1.y), v0.y.max(v1.y));
        // Pixel centers in 12.4: draw [x0, x1) rounding up from the left.
        let px0 = (x0 + 15) >> 4;
        let px1 = (x1 + 15) >> 4;
        let py0 = (y0 + 15) >> 4;
        let py1 = (y1 + 15) >> 4;
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
        let th = pipe.tex.th as f32;
        let tex_v = if pipe.fst {
            (tv0.min(tv1) as f32 / 16.0, tv0.max(tv1) as f32 / 16.0)
        } else {
            let (a, b) = (t0 * geom.inv_q * th, t1 * geom.inv_q * th);
            (a.min(b), a.max(b))
        };
        let px = (ryb - rya).max(0) as i64 * (geom.pxb - geom.pxa).max(0) as i64;
        (geom, rya, ryb, px, tex_v)
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
        let (a, b, c) = if area < 0 {
            (v0, v2, v1)
        } else {
            (v0, v1, v2)
        };
        let minx = (a.x.min(b.x).min(c.x) >> 4).max(0);
        let maxx = ((a.x.max(b.x).max(c.x) + 15) >> 4).min(4095);
        let miny = (a.y.min(b.y).min(c.y) >> 4).max(0);
        let maxy = ((a.y.max(b.y).max(c.y) + 15) >> 4).min(4095);
        if maxx - minx <= 24 && maxy - miny <= 24 {
            self.log_small_prim("triangle", attrs, maxx - minx, maxy - miny, &a, &b);
        }
        let pipe = self.pixel_pipe();
        let Some((geom, s, e, px, tex_v)) = Self::tri_prim(&pipe, a, b, c) else {
            return;
        };
        let queued = self.enqueue(pipe.clone(), Prim::Tri(geom), s, e, px, Some(tex_v));
        if let Some(p2) = self.scaled_pipe(&pipe) {
            if !queued {
                self.mirror_rect(&pipe, pipe.scx0, pipe.scx1, s, e);
            } else if let Some((g2, s2, e2, px2, _)) =
                Self::tri_prim(&p2, Self::scale_vertex(a), Self::scale_vertex(b), Self::scale_vertex(c))
            {
                self.enqueue_hi(p2, Prim::Tri(g2), s2, e2, px2);
            }
        }
    }

    /// Triangle geometry and its row/pixel extent for [`Gs::enqueue`];
    /// `a, b, c` are already wound positive. `None` = clipped out.
    fn tri_prim(pipe: &PixelPipe, a: Vertex, b: Vertex, c: Vertex) -> Option<(TriGeom, i32, i32, i64, (f32, f32))> {
        let area = edge(a.x, a.y, b.x, b.y, c.x, c.y);
        let minx = (a.x.min(b.x).min(c.x) >> 4).max(0).max(pipe.scx0);
        let maxx = (((a.x.max(b.x).max(c.x)) + 15) >> 4).min(4095).min(pipe.scx1);
        let miny = (a.y.min(b.y).min(c.y) >> 4).max(0).max(pipe.scy0);
        let maxy = (((a.y.max(b.y).max(c.y)) + 15) >> 4).min(4095).min(pipe.scy1);
        if minx > maxx || miny > maxy {
            return None;
        }
        // Edge functions are affine in (sx, sy): evaluate at the top-left
        // sample once and step by whole pixels (16 units) — exact in i64.
        let (sx0, sy0) = ((minx << 4) + 8, (miny << 4) + 8);
        let col = |v: &Vertex| [v.r as f32, v.g as f32, v.b as f32, v.a as f32];
        let stq = |v: &Vertex| [v.s, v.t, v.q, v.u as f32];
        let inv_area = 1.0 / area as f32;
        // Barycentric weights change by dx[i] * inv_area per pixel, so any
        // attribute changes by the weighted sum of its vertex values.
        let dl = [
            -(c.y - b.y) as f32 * 16.0 * inv_area,
            -(a.y - c.y) as f32 * 16.0 * inv_area,
            -(b.y - a.y) as f32 * 16.0 * inv_area,
        ];
        let step = |pa: [f32; 4], pb: [f32; 4], pc: [f32; 4]| -> [f32; 4] {
            let mut d = [0f32; 4];
            for i in 0..4 {
                d[i] = pa[i] * dl[0] + pb[i] * dl[1] + pc[i] * dl[2];
            }
            d
        };
        let geom = TriGeom {
            a,
            b,
            c,
            inv_area,
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
            d_rgba: step(col(&a), col(&b), col(&c)),
            d_stqu: step(stq(&a), stq(&b), stq(&c)),
            d_v: a.v as f32 * dl[0] + b.v as f32 * dl[1] + c.v as f32 * dl[2],
            d_z: a.z as f64 * dl[0] as f64 + b.z as f64 * dl[1] as f64 + c.z as f64 * dl[2] as f64,
        };
        let th = pipe.tex.th as f32;
        let tex_v = if pipe.fst {
            let vs = [a.v, b.v, c.v];
            (*vs.iter().min().unwrap() as f32 / 16.0, *vs.iter().max().unwrap() as f32 / 16.0)
        } else {
            let tv = |v: &Vertex| v.t / if v.q.abs() < 1e-9 { 1.0 } else { v.q } * th;
            let (ta, tb, tc) = (tv(&a), tv(&b), tv(&c));
            (ta.min(tb).min(tc), ta.max(tb).max(tc))
        };
        let px = (maxy - miny + 1) as i64 * (maxx - minx + 1) as i64;
        Some((geom, miny, maxy + 1, px, tex_v))
    }

    /// Queue a decoded primitive, flushing first when it would read what
    /// the queue wrote, when it would write what the queue read, or when it
    /// would write memory the queue already writes through a different
    /// buffer base. The lanes of a parallel flush are cut by row, so each
    /// of the three would otherwise resolve in whatever order the lanes
    /// happened to run.
    ///
    /// A primitive sampling its own target runs inline — its result depends
    /// on draw order within itself — and `false` is returned so the caller
    /// mirrors the 1x result into the overlay instead of queueing a 2x twin.
    fn enqueue(&mut self, pipe: PixelPipe, prim: Prim, start: i32, end: i32, px: i64, tex_v: Option<(f32, f32)>) -> bool {
        let tex = Self::tex_blocks(&pipe, tex_v);
        let (fb, zb) = Self::written_blocks(&pipe, end);
        let targets = [(&fb, pipe.fbp), (&zb, pipe.zbp)];
        if targets.iter().any(|(w, base)| {
            w.as_ref().is_some_and(|w| self.batch.aliases(w, *base, pipe.fbw))
        }) {
            self.flush_batch();
        }
        if let Some(t) = &tex {
            if self.batch.after_write(t) {
                self.flush_batch();
            }
            let overlaps = |w: &Option<std::ops::Range<u32>>| {
                w.as_ref().is_some_and(|w| t.start < w.end && w.start < t.end)
            };
            if overlaps(&fb) || overlaps(&zb) {
                self.flush_batch();
                let mut p = Painter {
                    canvas: &self.canvas,
                    tex: &self.canvas,
                    clut: self.clut.block(pipe.clut_off as usize),
                    pipe: &pipe,
                    scratch: &mut self.scratch,
                };
                Self::run_prim(&mut p, &prim, Rows::all(start, end));
                self.merge_scratch();
                return false;
            }
        }
        if targets.iter().any(|(w, _)| w.as_ref().is_some_and(|w| self.batch.before_read(w))) {
            self.flush_batch();
        }
        if let Some(t) = tex {
            self.batch.note_read(t);
        }
        for (r, base) in [(fb, pipe.fbp), (zb, pipe.zbp)] {
            if let Some(r) = r {
                self.batch.note_write(r, base, pipe.fbw);
            }
        }
        self.batch.px += px;
        self.batch.queued.push(Queued { pipe, prim, start, end, hi: false });
        if self.batch.px >= BATCH_MAX_PIXELS {
            self.flush_batch();
        }
        true
    }

    /// Queue the internal-2x twin of a primitive just queued at 1x. It
    /// shares the queue (order among twins matches the 1x order) but skips
    /// the dependency checks — those were decided by its 1x sibling.
    fn enqueue_hi(&mut self, pipe: PixelPipe, prim: Prim, start: i32, end: i32, px: i64) {
        self.batch.px += px;
        self.batch.queued.push(Queued { pipe, prim, start, end, hi: true });
        if self.batch.px >= BATCH_MAX_PIXELS {
            self.flush_batch();
        }
    }

    /// Internal-2x: the drawing environment re-addressed in the overlay's
    /// doubled coordinate space (`None` when the overlay is off). Texture
    /// fields stay in 1x space — sampling reads local memory.
    fn scaled_pipe(&self, pipe: &PixelPipe) -> Option<PixelPipe> {
        self.overlay.as_ref()?;
        let mut p = pipe.clone();
        p.scx0 *= 2;
        p.scy0 *= 2;
        p.scx1 = p.scx1 * 2 + 1;
        p.scy1 = p.scy1 * 2 + 1;
        p.fbp *= 4;
        p.zbp *= 4;
        p.fbw *= 2;
        Some(p)
    }

    /// A vertex in the overlay's doubled 12.4 coordinate space.
    fn scale_vertex(v: Vertex) -> Vertex {
        Vertex { x: v.x << 1, y: v.y << 1, ..v }
    }

    /// Copy pixels `x0..=x1` of rows `y0..y1` of the pipe's frame (and
    /// written Z) from local memory into the overlay as 2x2 duplicates —
    /// for primitives whose 2x twin cannot be rasterized because they
    /// sample their own target.
    fn mirror_rect(&mut self, pipe: &PixelPipe, x0: i32, x1: i32, y0: i32, y1: i32) {
        let Some(ov) = &self.overlay else { return };
        let write_z = pipe.zte && !pipe.zmsk;
        for y in y0.max(0)..y1 {
            for x in x0.max(0)..=x1 {
                let (x, y) = (x as u32, y as u32);
                if pipe.fbmsk != u32::MAX {
                    let v = self.canvas.read_psmct32(pipe.fbp, pipe.fbw, x, y);
                    for d in 0..4u32 {
                        ov.write_psmct32(pipe.fbp * 4, pipe.fbw * 2, 2 * x + (d & 1), 2 * y + (d >> 1), v);
                    }
                }
                if write_z {
                    let z = self.canvas.read_psmz32(pipe.zbp, pipe.fbw, x, y);
                    for d in 0..4u32 {
                        ov.write_psmz32(pipe.zbp * 4, pipe.fbw * 2, 2 * x + (d & 1), 2 * y + (d >> 1), z);
                    }
                }
            }
        }
    }

    fn run_prim(p: &mut Painter, prim: &Prim, rows: Rows) {
        match prim {
            Prim::Tri(g) => p.tri_rows(g, rows),
            Prim::Sprite(g) => p.sprite_rows(g, rows),
            Prim::Line(g) => p.line_rows(g, rows),
        }
    }

    /// Rasterize the queued primitives: across the worker pool in two-row
    /// bands when there is enough work, else serially. Each lane replays
    /// the whole queue in order restricted to its own rows. That matches
    /// the serial answer only because `enqueue` refuses to queue a
    /// primitive whose reads or writes cross another queued primitive's: a
    /// lane reaches a later primitive's write long before another lane
    /// reaches an earlier primitive's read of the same memory.
    pub(super) fn flush_batch(&mut self) {
        if self.batch.queued.is_empty() {
            return;
        }
        let queued = std::mem::take(&mut self.batch.queued);
        let px = std::mem::take(&mut self.batch.px);
        self.batch.writes.clear();
        self.batch.reads.clear();
        #[cfg(feature = "threads")]
        if px >= PARALLEL_MIN_PIXELS && self.pool.len() >= PARALLEL_LANES {
            self.prims_split += queued.len() as u64;
            let canvas = &self.canvas;
            let overlay = self.overlay.as_ref();
            let clut = &self.clut;
            let q = &queued;
            let _p = crate::prof::scope(crate::prof::Slot::GsJoin);
            rayon::scope(|s| {
                for (lane, scratch) in self.pool.iter_mut().take(PARALLEL_LANES).enumerate() {
                    s.spawn(move |_| {
                        for item in q {
                            let fb = if item.hi { overlay.unwrap_or(canvas) } else { canvas };
                            let mut p = Painter { canvas: fb, tex: canvas, clut: clut.block(item.pipe.clut_off as usize), pipe: &item.pipe, scratch };
                            let rows = Rows { start: item.start, end: item.end, lane, lanes: PARALLEL_LANES };
                            Self::run_prim(&mut p, &item.prim, rows);
                        }
                    });
                }
            });
            for s in self.pool.iter_mut() {
                self.pixels_shaded += std::mem::take(&mut s.pixels);
                for (h, n) in self.tex_psm_hist.iter_mut().zip(s.tex_samples.iter_mut()) {
                    *h += std::mem::take(n);
                }
            }
            self.clut.reset();
            self.batch.queued = {
                let mut v = queued;
                v.clear();
                v
            };
            return;
        }
        let _ = px;
        for item in &queued {
            let fb = if item.hi { self.overlay.as_ref().unwrap_or(&self.canvas) } else { &self.canvas };
            let mut p = Painter {
                canvas: fb,
                tex: &self.canvas,
                clut: self.clut.block(item.pipe.clut_off as usize),
                pipe: &item.pipe,
                scratch: &mut self.scratch,
            };
            Self::run_prim(&mut p, &item.prim, Rows::all(item.start, item.end));
        }
        self.merge_scratch();
        self.clut.reset();
        self.batch.queued = {
            let mut v = queued;
            v.clear();
            v
        };
    }

    /// Decode the drawing environment for the current context once per
    /// primitive; the per-pixel path only reads it.
    fn pixel_pipe(&mut self) -> PixelPipe {
        let attrs = self.attrs();
        let ctx = self.ctx[((attrs >> 9) & 1) as usize];
        let test = ctx.test;
        let tex = TexInfo::new(&ctx, self.texa);
        let clut_off =
            if attrs & (1 << 4) != 0 && tex.clut_bits != 0 { self.refresh_clut(&tex) } else { 0 };

        let mut pipe = PixelPipe {
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
            z_touched: false,
            clut_off,
            fast: false,
            fast_decal: false,
            flat_fill: false,
        };
        let z_touched = pipe.zte && !(pipe.ztst == 1 && pipe.zmsk);
        pipe.z_touched = z_touched;
        pipe.fast = pipe.tme && pipe.tfx == 0 && pipe.fbmsk == 0;
        pipe.fast_decal = pipe.tme && pipe.tfx == 1 && pipe.fbmsk == 0;
        // Z may be written (ALWAYS) but never tested; the alpha test on a
        // flat colour is decided once per row by the loop itself.
        pipe.flat_fill = !pipe.tme && (!z_touched || pipe.ztst == 1) && pipe.fbmsk == 0;
        pipe
    }

    /// Re-decode the CLUT cache when the palette setup changed or a
    /// transfer touched VRAM. Real hardware only reloads on TEX0 writes
    /// with CLD set; keying on the setup instead is a superset of that
    /// (drawing primitives into CLUT memory is not tracked).
    /// Make sure a decoded block exists for this palette setup and return
    /// its offset. The CSA window is part of the key, so a block only ever
    /// holds entries one texture actually reads.
    fn refresh_clut(&mut self, ti: &TexInfo) -> u32 {
        let key = ti.clut_key() | ((ti.clut_base as u64) << 49);
        if self.clut_dirty {
            // A transfer has been through VRAM; nothing decoded before it
            // may be matched again, though blocks already queued stay.
            self.clut.forget_keys();
            self.clut_dirty = false;
        }
        if let Some(off) = self.clut.find(key) {
            return off as u32;
        }
        // Decoding reads local memory, so a queue that writes this palette's
        // own memory has to be drawn first.
        let cbp = ((ti.tex0 >> 37) & 0x3FFF) as u32;
        if self.batch.after_write(&(cbp..cbp + 4)) {
            self.flush_batch();
        }
        let off = match self.clut.alloc(key) {
            Some(off) => off,
            None => {
                // Out of blocks: drawing the batch frees every one of them.
                self.flush_batch();
                self.clut.alloc(key).expect("a drawn batch frees the pool")
            }
        };
        let entries = if ti.clut_bits == 8 { 256 } else { 16 };
        for e in ti.clut_base..ti.clut_base + entries {
            self.clut.data[off + e] = self.clut_lookup(ti.tex0, e as u32);
        }
        off as u32
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
    /// Frame/Z target: local memory, or the internal-2x overlay.
    canvas: &'a Canvas,
    /// Texture source: always the real local memory.
    tex: &'a Canvas,
    clut: &'a [u32],
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
                    if (pipe.fast || pipe.fast_decal) && pxb - pxa >= 2 {
                        // Fixed-point u (32.32 texels) stepped across the row
                        // between the same endpoints the float path uses.
                        let fa = f64::from(fu_at(pxa));
                        let fb = f64::from(fu_at(pxb - 1));
                        let du = ((fb - fa) / f64::from(pxb - 1 - pxa) * 4294967296.0) as i64;
                        let ua = (fa * 4294967296.0).floor() as i64;
                        let args = FastRow { row: &row, pxa, pxb, frag: &frag, row0: &row0.data, row1: &row1.data, u_lo, wy, ua, du, z: frag.z };
                        match (pipe.fast_decal, pipe.bilinear, pipe.abe, pipe.ate) {
                            (false, false, false, false) => self.fast_sprite_row::<false, false, false, false>(&args),
                            (false, false, false, true) => self.fast_sprite_row::<false, false, false, true>(&args),
                            (false, false, true, false) => self.fast_sprite_row::<false, false, true, false>(&args),
                            (false, false, true, true) => self.fast_sprite_row::<false, false, true, true>(&args),
                            (false, true, false, false) => self.fast_sprite_row::<false, true, false, false>(&args),
                            (false, true, false, true) => self.fast_sprite_row::<false, true, false, true>(&args),
                            (false, true, true, false) => self.fast_sprite_row::<false, true, true, false>(&args),
                            (false, true, true, true) => self.fast_sprite_row::<false, true, true, true>(&args),
                            (true, false, false, false) => self.fast_sprite_row::<true, false, false, false>(&args),
                            (true, false, false, true) => self.fast_sprite_row::<true, false, false, true>(&args),
                            (true, false, true, false) => self.fast_sprite_row::<true, false, true, false>(&args),
                            (true, false, true, true) => self.fast_sprite_row::<true, false, true, true>(&args),
                            (true, true, false, false) => self.fast_sprite_row::<true, true, false, false>(&args),
                            (true, true, false, true) => self.fast_sprite_row::<true, true, false, true>(&args),
                            (true, true, true, false) => self.fast_sprite_row::<true, true, true, false>(&args),
                            (true, true, true, true) => self.fast_sprite_row::<true, true, true, true>(&args),
                        }
                        self.scratch.tex_rows = [row0, row1];
                        continue;
                    }
                    let _p = crate::prof::scope(crate::prof::Slot::GsGeneric);
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
            if pipe.flat_fill && pxb > pxa {
                self.flat_sprite_row(&row, pxa, pxb, &frag);
                continue;
            }
            let _p = crate::prof::scope(crate::prof::Slot::GsGeneric);
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

    /// Pixel-step DDA over the major axis; as on hardware the end point's
    /// pixel is not drawn, so the joints of a strip land exactly once — the
    /// PS2 logo's additive wireframe counts on that. Attributes interpolate
    /// (or stick to the second vertex's colour when shading is flat) and
    /// pixels go through the generic per-pixel pipeline. Every lane walks
    /// the full DDA and plots only its own rows, keeping values identical
    /// to a serial pass.
    fn line_rows(&mut self, g: &LineGeom, rows: Rows) {
        let pipe = self.pipe;
        let (a, b) = (g.a, g.b);
        let (fx0, fy0) = (a.x as f32 / 16.0, a.y as f32 / 16.0);
        let (fx1, fy1) = (b.x as f32 / 16.0, b.y as f32 / 16.0);
        let (dx, dy) = (fx1 - fx0, fy1 - fy0);
        let inv = 1.0 / g.steps as f32;
        let lerp = |p: f32, q: f32, t: f32| p + (q - p) * t;
        for i in 0..g.steps {
            let t = i as f32 * inv;
            let px = (fx0 + dx * t).round() as i32;
            let py = (fy0 + dy * t).round() as i32;
            if px < pipe.scx0 || px > pipe.scx1 || py < pipe.scy0 || py > pipe.scy1 {
                continue;
            }
            if rows.lanes > 1 && ((py >> 1) as usize) % rows.lanes != rows.lane {
                continue;
            }
            let frag = Frag {
                r: if g.gouraud { lerp(a.r as f32, b.r as f32, t) } else { b.r as f32 },
                g: if g.gouraud { lerp(a.g as f32, b.g as f32, t) } else { b.g as f32 },
                b: if g.gouraud { lerp(a.b as f32, b.b as f32, t) } else { b.b as f32 },
                a: if g.gouraud { lerp(a.a as f32, b.a as f32, t) } else { b.a as f32 },
                z: lerp(a.z as f32, b.z as f32, t) as u32,
                s: lerp(a.s, b.s, t),
                t: lerp(a.t, b.t, t),
                q: lerp(a.q, b.q, t),
                u: lerp(a.u as f32, b.u as f32, t) / 16.0,
                v: lerp(a.v as f32, b.v as f32, t) / 16.0,
            };
            let texel = if pipe.tme { self.sample(&frag) } else { 0 };
            let row = Row::new(pipe, py as u32);
            self.shade_row_px(&row, px as u32, frag, texel);
        }
    }

    fn tri_rows(&mut self, g: &TriGeom, rows: Rows) {
        let pipe = self.pipe;
        self.scratch.tex_rows[0].key.0 = u64::MAX;
        self.scratch.tex_rows[1].key.0 = u64::MAX;
        let (a, b, c) = (g.a, g.b, g.c);
        for py in rows.iter() {
            let k = (py - g.miny) as i64;
            let w = [g.w0 + g.dy[0] * k, g.w1 + g.dy[1] * k, g.w2 + g.dy[2] * k];
            // Covered span: each edge function is affine in x, so the
            // pixels with all three >= 0 form one interval (the same set
            // the per-pixel test would accept).
            let (mut xs, mut xe) = (g.minx, g.maxx);
            for i in 0..3 {
                let d = g.dx[i];
                if d > 0 {
                    // w + d*n >= 0  <=>  n >= ceil(-w / d)
                    let span = (g.maxx - g.minx + 1) as i64;
                    let n = ((-w[i]).div_euclid(d) + ((-w[i]).rem_euclid(d) != 0) as i64).clamp(0, span);
                    xs = xs.max(g.minx + n as i32);
                } else if d < 0 {
                    // w + d*n >= 0  <=>  n <= floor(w / -d)
                    if w[i] < 0 {
                        xe = g.minx - 1;
                    } else {
                        xe = xe.min(g.minx + (w[i] / -d).min((g.maxx - g.minx + 1) as i64) as i32);
                    }
                } else if w[i] < 0 {
                    xe = g.minx - 1;
                }
            }
            if xs > xe {
                continue;
            }
            let row = Row::new(pipe, py as u32);
            if pipe.fast {
                let n = (xs - g.minx) as i64;
                let (w0, w1, w2) = (w[0] + g.dx[0] * n, w[1] + g.dx[1] * n, w[2] + g.dx[2] * n);
                let l0 = w0 as f32 * g.inv_area;
                let l1 = w1 as f32 * g.inv_area;
                let l2 = w2 as f32 * g.inv_area;
                let args = FastTri {
                    row: &row,
                    xs,
                    xe,
                    rgba: interp3(&g.ca, &g.cb, &g.cc, l0, l1, l2),
                    stqu: interp3(&g.sa, &g.sb, &g.sc, l0, l1, l2),
                    v: a.v as f32 * l0 + b.v as f32 * l1 + c.v as f32 * l2,
                    z: a.z as f64 * l0 as f64 + b.z as f64 * l1 as f64 + c.z as f64 * l2 as f64,
                    d_rgba: g.d_rgba,
                    d_stqu: g.d_stqu,
                    d_v: g.d_v,
                    d_z: g.d_z,
                };
                if self.sprite_like_tri_row(&args) {
                    continue;
                }
                match (pipe.bilinear, pipe.abe, pipe.ate) {
                    (false, false, false) => self.fast_tri_row::<false, false, false>(&args),
                    (false, false, true) => self.fast_tri_row::<false, false, true>(&args),
                    (false, true, false) => self.fast_tri_row::<false, true, false>(&args),
                    (false, true, true) => self.fast_tri_row::<false, true, true>(&args),
                    (true, false, false) => self.fast_tri_row::<true, false, false>(&args),
                    (true, false, true) => self.fast_tri_row::<true, false, true>(&args),
                    (true, true, false) => self.fast_tri_row::<true, true, false>(&args),
                    (true, true, true) => self.fast_tri_row::<true, true, true>(&args),
                }
                continue;
            }
            let _p = crate::prof::scope(crate::prof::Slot::GsGeneric);
            let n = (xs - g.minx) as i64;
            let (mut w0, mut w1, mut w2) = (w[0] + g.dx[0] * n, w[1] + g.dx[1] * n, w[2] + g.dx[2] * n);
            for px in xs..=xe {
                let l0 = w0 as f32 * g.inv_area;
                let l1 = w1 as f32 * g.inv_area;
                let l2 = w2 as f32 * g.inv_area;
                w0 += g.dx[0];
                w1 += g.dx[1];
                w2 += g.dx[2];
                let rgba = interp3(&g.ca, &g.cb, &g.cc, l0, l1, l2);
                // Texture coordinates cost four lanes and two divides per
                // pixel; an untextured span never looks at them.
                let (stqu, tv) = if pipe.tme {
                    (
                        interp3(&g.sa, &g.sb, &g.sc, l0, l1, l2),
                        (a.v as f32 * l0 + b.v as f32 * l1 + c.v as f32 * l2) / 16.0,
                    )
                } else {
                    ([0.0; 4], 0.0)
                };
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
                    v: tv,
                };
                let texel = if pipe.tme { self.sample_cached(&frag) } else { 0 };
                self.shade_row_px(&row, px as u32, frag, texel);
            }
        }
    }

    /// A triangle span with a flat colour, constant texture row and
    /// linear u — an axis-aligned quad drawn as triangles, the common 2D
    /// case — is a sprite row: decode its texture row(s) once and run
    /// [`Painter::fast_sprite_row`]. Returns false when the span does not
    /// qualify.
    fn sprite_like_tri_row(&mut self, a: &FastTri) -> bool {
        let pipe = self.pipe;
        if a.d_rgba != [0.0; 4] || (pipe.z_touched && a.d_z != 0.0) || (!pipe.fst && a.d_stqu[2] != 0.0) {
            return false;
        }
        let span = (a.xe - a.xs) as f32;
        let uv = |k: f32| -> (f32, f32) {
            if pipe.fst {
                ((a.stqu[3] + a.d_stqu[3] * k) / 16.0, (a.v + a.d_v * k) / 16.0)
            } else {
                let q = if a.stqu[2].abs() < 1e-9 { 1.0 } else { a.stqu[2] };
                let inv_q = 1.0 / q;
                (
                    (a.stqu[0] + a.d_stqu[0] * k) * inv_q * pipe.tex.tw as f32,
                    (a.stqu[1] + a.d_stqu[1] * k) * inv_q * pipe.tex.th as f32,
                )
            }
        };
        let (fu_a, fv_a) = uv(0.0);
        let (fu_b, fv_b) = uv(span);
        let (fu_lo, fu_hi) = (fu_a.min(fu_b), fu_a.max(fu_b));
        // The texture row(s) must be the same at both ends (v is monotonic
        // along the span, so then everywhere between).
        let (y_row, wy, u_lo, u_hi) = if pipe.bilinear {
            let (ya, yb) = (fv_a - 0.5, fv_b - 0.5);
            let (ra, rb) = (floor_i32(ya), floor_i32(yb));
            let (wa, wb) = (((ya - ra as f32) * 256.0) as u32, ((yb - rb as f32) * 256.0) as u32);
            if ra != rb || wa != wb {
                return false;
            }
            (ra, wa, floor_i32(fu_lo - 0.5), floor_i32(fu_hi - 0.5) + 1)
        } else {
            let (ra, rb) = (floor_i32(fv_a), floor_i32(fv_b));
            if ra != rb {
                return false;
            }
            (ra, 0, floor_i32(fu_lo), floor_i32(fu_hi))
        };
        if u_hi - u_lo >= 4096 {
            return false;
        }
        self.fill_tex_row(0, y_row, u_lo, u_hi);
        if pipe.bilinear {
            self.fill_tex_row(1, y_row + 1, u_lo, u_hi);
        }
        let [row0, row1] = std::mem::take(&mut self.scratch.tex_rows);
        let du = if span > 0.0 { ((f64::from(fu_b) - f64::from(fu_a)) / f64::from(span) * 4294967296.0) as i64 } else { 0 };
        let ua = (f64::from(fu_a) * 4294967296.0).floor() as i64;
        let frag = Frag {
            r: a.rgba[0],
            g: a.rgba[1],
            b: a.rgba[2],
            a: a.rgba[3],
            z: a.z as u32,
            s: 0.0,
            t: 0.0,
            q: 1.0,
            u: 0.0,
            v: 0.0,
        };
        let args = FastRow {
            row: a.row,
            pxa: a.xs,
            pxb: a.xe + 1,
            frag: &frag,
            row0: &row0.data,
            row1: &row1.data,
            u_lo,
            wy,
            ua,
            du,
            z: frag.z,
        };
        match (pipe.bilinear, pipe.abe, pipe.ate) {
            (false, false, false) => self.fast_sprite_row::<false, false, false, false>(&args),
            (false, false, true) => self.fast_sprite_row::<false, false, false, true>(&args),
            (false, true, false) => self.fast_sprite_row::<false, false, true, false>(&args),
            (false, true, true) => self.fast_sprite_row::<false, false, true, true>(&args),
            (true, false, false) => self.fast_sprite_row::<false, true, false, false>(&args),
            (true, false, true) => self.fast_sprite_row::<false, true, false, true>(&args),
            (true, true, false) => self.fast_sprite_row::<false, true, true, false>(&args),
            (true, true, true) => self.fast_sprite_row::<false, true, true, true>(&args),
        }
        self.scratch.tex_rows = [row0, row1];
        true
    }

    /// Textured MODULATE triangle span for `PixelPipe::fast` setups: the
    /// attributes step incrementally along the row, texels come through the
    /// row cache, and the filter / alpha test / blend are compile-time
    /// choices (see [`Painter::fast_sprite_row`]).
    #[inline(never)]
    fn fast_tri_row<const BIL: bool, const ABE: bool, const ATE: bool>(&mut self, a: &FastTri) {
        let _p = crate::prof::scope(crate::prof::Slot::GsFastTri);
        let pipe = self.pipe;
        let row = a.row;
        let y = row.y;
        let n = (a.xe - a.xs + 1) as u64;
        self.scratch.pixels += n;
        self.scratch.tex_samples[pipe.tex.psm as usize] += n;
        #[cfg(feature = "profile")]
        {
            let k = pipe.profile_key(false);
            crate::prof::count_pixels(k, n);
        }
        let tcc = pipe.tcc;
        let fst = pipe.fst;
        let (tw, th) = (pipe.tex.tw as f32, pipe.tex.th as f32);
        let (atst, aref, afail) = (pipe.atst, pipe.aref, pipe.afail);
        let blend = Blend::new(pipe);
        let canvas = self.canvas;
        let fb24 = pipe.fb24;
        let ztest = ZTest::new(pipe);
        let mut rgba = a.rgba;
        let mut stqu = a.stqu;
        let mut v = a.v;
        let mut zf = a.z;
        let tf = TexFetch::new(&pipe.tex);
        let uv_at = |stqu: &[f32; 4], v: f32| -> (f32, f32) {
            if fst {
                clamp_uv(stqu[3] / 16.0, v / 16.0)
            } else {
                let q = if stqu[2].abs() < 1e-9 { 1.0 } else { stqu[2] };
                let inv_q = 1.0 / q;
                clamp_uv(stqu[0] * inv_q * tw, stqu[1] * inv_q * th)
            }
        };
        // The decoded-row cache pays off when the span stays on one or two
        // texture rows and is magnified (it decodes each texel once); a
        // mapping that walks v along the row, or a minified one that skips
        // texels, would refill the 32-texel chunks every few pixels, so
        // those sample VRAM directly.
        let direct = {
            let span = (a.xe - a.xs) as f32;
            let mut stqu_e = a.stqu;
            for i in 0..4 {
                stqu_e[i] += a.d_stqu[i] * span;
            }
            let (fu_s, fv_s) = uv_at(&a.stqu, a.v);
            let (fu_e, fv_e) = uv_at(&stqu_e, a.v + a.d_v * span);
            (fv_e - fv_s).abs() >= 1.0 || (fu_e - fu_s).abs() > 2.0 * span + 8.0
        };
        for px in a.xs..=a.xe {
            let (fu, fv) = uv_at(&stqu, v);
            let texel = if direct {
                if BIL { self.sample_bilinear_direct(tf, fu, fv) } else { self.texel(floor_i32(fu), floor_i32(fv)) }
            } else if BIL {
                self.sample_bilinear_cached(fu, fv)
            } else {
                self.cached_texel(0, floor_i32(fu), floor_i32(fv))
            };
            let (cr, cg, cb, ca) = (rgba[0] as u32, rgba[1] as u32, rgba[2] as u32, rgba[3] as u32);
            let z = (zf as u32) & pipe.zmask;
            for i in 0..4 {
                rgba[i] += a.d_rgba[i];
                stqu[i] += a.d_stqu[i];
            }
            v += a.d_v;
            zf += a.d_z;
            let ta = texel >> 24;
            let a8 = if tcc { (ta * ca) >> 7 } else { ca };
            let (mut write_z, mut keep_dst_alpha) = (true, false);
            if ATE {
                let pass = match atst {
                    0 => false,
                    1 => true,
                    2 => a8 < aref,
                    3 => a8 <= aref,
                    4 => a8 == aref,
                    5 => a8 >= aref,
                    6 => a8 > aref,
                    _ => a8 != aref,
                };
                if !pass {
                    match afail {
                        0 => continue, // KEEP
                        2 => {
                            // ZB_ONLY: the pixel still updates Z.
                            ztest.pass(canvas, row, px as u32, z);
                            continue;
                        }
                        1 => write_z = false,
                        _ => {
                            write_z = false;
                            keep_dst_alpha = true;
                        }
                    }
                }
            }
            if ztest.on {
                let zok = if write_z {
                    ztest.pass(canvas, row, px as u32, z)
                } else {
                    ztest.test(canvas, row, px as u32, z)
                };
                if !zok {
                    continue;
                }
            }
            let mut out = Modulate::new(cr, cg, cb, ca, tcc).apply(texel);
            let fb_off = (row.fb_base + layout::col_off32(y, px as u32, false)) & canvas.mask();
            let dst = if ABE || fb24 || keep_dst_alpha { canvas.rd32(fb_off) } else { 0 };
            if ABE {
                out = blend.apply(out, dst, a8);
            }
            if fb24 || keep_dst_alpha {
                // PSMCT24 frame, or RGB_ONLY: the alpha byte belongs to
                // whatever shares the word.
                out = (out & 0xFF_FFFF) | (dst & 0xFF00_0000);
            }
            canvas.wr32(fb_off, out);
        }
    }

    /// Bilinear texel at texel-space `(fu, fv)` through the row cache
    /// (the bilinear half of [`Painter::sample_cached`]).
    #[inline(always)]
    fn sample_bilinear_cached(&mut self, fu: f32, fv: f32) -> u32 {
        let pipe = self.pipe;
        let ti = &pipe.tex;
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

    /// Make `tex_rows[slot]` hold texels `u_lo..=u_hi` of texture row `y`
    /// (wrapped/clamped like any sample); reuses the previous contents when
    /// they already cover the request.
    fn fill_tex_row(&mut self, slot: usize, y: i32, u_lo: i32, u_hi: i32) {
        let _p = crate::prof::scope(crate::prof::Slot::GsTexFill);
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
        let (tex, tm) = (self.tex, self.tex.mask());
        // The row's texture line is fixed: address it as base + column
        // table for the formats the fast paths matter for.
        match ti.psm {
            PSMT8 => {
                let base = layout::row_base8(ti.tbp, ti.tbw, v);
                for u in u_lo..=u_hi {
                    let u = wrap(u, ti.wms, ti.tw as i32, ti.minu, ti.maxu) as u32;
                    let idx = tex.rd8((base + layout::col_off8(v, u)) & tm);
                    row.data.push(self.clut[idx as usize]);
                }
            }
            PSMCT32 | PSMCT24 | PSMT8H | PSMT4HL | PSMT4HH => {
                let base = layout::row_base32(ti.tbp, ti.tbw, v, false);
                if matches!(ti.psm, PSMCT32 | PSMCT24) && wrap_identity(u_lo, u_hi, ti) {
                    // No wrapping in the span: an even x and its neighbour
                    // share one aligned 8-byte column pair, so most of the
                    // row moves as u64 loads with the mask/alpha applied
                    // branchlessly (CT32: keep all, CT24: substitute TA0).
                    let (m, or) = if ti.psm == PSMCT32 {
                        (u32::MAX, 0)
                    } else {
                        (0xFF_FFFF, ((ti.texa & 0xFF) as u32) << 24)
                    };
                    let at = |u: u32| (base + layout::col_off32(v, u, false)) & tm;
                    let mut u = u_lo as u32;
                    if u & 1 != 0 {
                        row.data.push((tex.rd32(at(u)) & m) | or);
                        u += 1;
                    }
                    while (u + 1) as i32 <= u_hi {
                        let pair = tex.rd64(at(u));
                        row.data.push((pair as u32 & m) | or);
                        row.data.push(((pair >> 32) as u32 & m) | or);
                        u += 2;
                    }
                    if u as i32 <= u_hi {
                        row.data.push((tex.rd32(at(u)) & m) | or);
                    }
                } else {
                    for u in u_lo..=u_hi {
                        let u = wrap(u, ti.wms, ti.tw as i32, ti.minu, ti.maxu) as u32;
                        let px = tex.rd32((base + layout::col_off32(v, u, false)) & tm);
                        row.data.push(match ti.psm {
                            PSMCT32 => px,
                            PSMCT24 => (px & 0xFF_FFFF) | (((ti.texa & 0xFF) as u32) << 24),
                            PSMT8H => self.clut[(px >> 24) as usize],
                            PSMT4HL => self.clut[((px >> 24) & 0xF) as usize + ti.clut_base],
                            _ => self.clut[(px >> 28) as usize + ti.clut_base],
                        });
                    }
                }
            }
            PSMCT16 | PSMCT16S => {
                let s = ti.psm == PSMCT16S;
                let base = layout::row_base16(ti.tbp, ti.tbw, v);
                let at =
                    |u: u32| (base + layout::col_off16(v, u, s, false)) & tm;
                if wrap_identity(u_lo, u_hi, ti) {
                    for u in u_lo..=u_hi {
                        row.data.push(expand16(tex.rd16(at(u as u32)), ti.texa));
                    }
                } else {
                    for u in u_lo..=u_hi {
                        let u = wrap(u, ti.wms, ti.tw as i32, ti.minu, ti.maxu) as u32;
                        row.data.push(expand16(tex.rd16(at(u)), ti.texa));
                    }
                }
            }
            _ => {
                for u in u_lo..=u_hi {
                    row.data.push(self.texel(u, y));
                }
            }
        }
        #[cfg(feature = "profile")]
        FILL_TEXELS[(ti.psm & 63) as usize]
            .fetch_add(row.data.len() as u64, core::sync::atomic::Ordering::Relaxed);
        self.scratch.tex_rows[slot] = row;
    }

    /// [`Painter::sample`] through the decoded-row cache: texels come from
    /// `tex_rows`, refilled in 32-texel chunks around a miss. Same texels
    /// and weights as the direct path.
    fn sample_cached(&mut self, frag: &Frag) -> u32 {
        let pipe = self.pipe;
        let ti = &pipe.tex;
        let (fu, fv) = if pipe.fst {
            clamp_uv(frag.u, frag.v)
        } else {
            let q = if frag.q.abs() < 1e-9 { 1.0 } else { frag.q };
            let inv_q = 1.0 / q;
            clamp_uv(frag.s * inv_q * ti.tw as f32, frag.t * inv_q * ti.th as f32)
        };
        if !pipe.bilinear {
            return self.cached_texel(0, floor_i32(fu), floor_i32(fv));
        }
        self.sample_bilinear_cached(fu, fv)
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

    /// Constant-colour sprite row (`PixelPipe::flat_fill`): the alpha test
    /// is evaluated once, then the colour (and Z when ZTE ALWAYS writes it)
    /// is stored two pixels at a time — a 32-bit column holds pixel pairs
    /// (x even, x+1) in adjacent words. Blending or a 24-bit frame read
    /// the destination per pixel instead. Same results as
    /// [`Painter::shade_row_px`].
    #[inline(never)]
    fn flat_sprite_row(&mut self, row: &Row, pxa: i32, pxb: i32, frag: &Frag) {
        let _p = crate::prof::scope(crate::prof::Slot::GsFlat);
        let pipe = self.pipe;
        let y = row.y;
        let n = (pxb - pxa) as u64;
        self.scratch.pixels += n;
        #[cfg(feature = "profile")]
        {
            let k = pipe.profile_key(false);
            crate::prof::count_pixels(k, n);
        }
        let (r, g, b, a) = (frag.r as u32, frag.g as u32, frag.b as u32, frag.a as u32);
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
            if !pass && (pipe.afail == 0 || pipe.afail == 2) {
                return;
            }
        }
        let out = r | (g << 8) | (b << 16) | (a.min(255) << 24);
        let write_z = pipe.zte && !pipe.zmsk;
        let z = frag.z & pipe.zmask;
        let z_merge = pipe.zmask != u32::MAX;
        let canvas = self.canvas;
        let m = canvas.mask();
        let fb_at = |x: i32| (row.fb_base + layout::col_off32(y, x as u32, false)) & m;
        let z_at = |x: i32| (row.z_base + layout::col_off32(y, x as u32, true)) & m;
        let put_z = |x: i32| {
            let o = z_at(x);
            if z_merge {
                canvas.wr32(o, (canvas.rd32(o) & !pipe.zmask) | z);
            } else {
                canvas.wr32(o, z);
            }
        };
        if pipe.abe || pipe.fb24 {
            // Destination-dependent: blend the constant colour per pixel.
            let blend = Blend::new(pipe);
            for px in pxa..pxb {
                let o = fb_at(px);
                let dst = canvas.rd32(o);
                let mut v = out;
                if pipe.abe {
                    v = blend.apply(out, dst, a);
                }
                if pipe.fb24 {
                    v = (v & 0xFF_FFFF) | (dst & 0xFF00_0000);
                }
                canvas.wr32(o, v);
                if write_z {
                    put_z(px);
                }
            }
            return;
        }
        let mut px = pxa;
        if px & 1 != 0 {
            canvas.wr32(fb_at(px), out);
            if write_z {
                put_z(px);
            }
            px += 1;
        }
        let pair = u64::from(out) | (u64::from(out) << 32);
        let zpair = u64::from(z) | (u64::from(z) << 32);
        while px + 1 < pxb {
            // Even x: its pair partner sits in the next word (unless the
            // pair straddles the end of VRAM).
            let o = fb_at(px);
            if o + 8 <= canvas.size() {
                canvas.wr64(o, pair);
            } else {
                canvas.wr32(o, out);
                canvas.wr32(fb_at(px + 1), out);
            }
            if write_z {
                let zo = z_at(px);
                if z_merge || zo + 8 > canvas.size() {
                    put_z(px);
                    put_z(px + 1);
                } else {
                    canvas.wr64(zo, zpair);
                }
            }
            px += 2;
        }
        if px < pxb {
            canvas.wr32(fb_at(px), out);
            if write_z {
                put_z(px);
            }
        }
    }

    /// The textured sprite row loop for `PixelPipe::fast` setups: u is
    /// stepped in fixed point, the Z / frame-mask / 24-bit paths are gone,
    /// the filter, alpha test and blend are compile-time choices and the
    /// colour modulation runs on SSE2 lanes. Same results as
    /// [`Painter::shade_row_px`] apart from the u rounding at the 2^-32
    /// level.
    #[inline(never)]
    fn fast_sprite_row<const DEC: bool, const BIL: bool, const ABE: bool, const ATE: bool>(&mut self, a: &FastRow) {
        let _p = crate::prof::scope(crate::prof::Slot::GsFastSprite);
        let pipe = self.pipe;
        let n = (a.pxb - a.pxa) as u64;
        self.scratch.pixels += n;
        self.scratch.tex_samples[pipe.tex.psm as usize] += n;
        #[cfg(feature = "profile")]
        {
            let frag = a.frag;
            let neutral = frag.r == 128.0 && frag.g == 128.0 && frag.b == 128.0 && frag.a == 128.0;
            let k = pipe.profile_key(neutral);
            crate::prof::count_pixels(k, n);
            let cls = if !BIL {
                0
            } else {
                let wx0 = (((a.ua - (1i64 << 31)) >> 24) & 0xFF) as u32;
                if a.du & 0xFFFF_FFFF != 0 {
                    5
                } else if wx0 == 0 && a.wy == 0 {
                    1
                } else if a.wy == 0 {
                    2
                } else if wx0 == 0 {
                    3
                } else {
                    4
                }
            };
            BIL_CLASSES[cls].fetch_add(n, core::sync::atomic::Ordering::Relaxed);
        }
        if BIL && a.du & 0xFFFF_FFFF == 0 {
            // An integral texel step keeps the filter weights constant
            // across the row (the OSD's half-texel blur copies): filter
            // the texel run once, then the nearest loop reads it 1:1.
            let wx = (((a.ua - (1i64 << 31)) >> 24) & 0xFF) as u32;
            let mut filtered = core::mem::take(&mut self.scratch.filtered);
            Self::prefilter_const(&mut filtered, a, wx);
            let fa = FastRow { row0: &filtered, row1: &filtered, u_lo: 0, ua: 0, du: 1i64 << 32, wy: 0, ..*a };
            self.sprite_row_px::<DEC, false, ABE, ATE>(&fa);
            self.scratch.filtered = filtered;
            return;
        }
        self.sprite_row_px::<DEC, BIL, ABE, ATE>(a);
    }

    /// Fill `dst` with the row's filtered texels for a constant-weight
    /// bilinear sprite row: per pixel the taps move by the integral `du`
    /// while `wx`/`wy` stay fixed. The in-bounds run with a step of one is
    /// done two pixels per SSE2 vector with exactly [`bilerp_sse2`]'s
    /// operation order, so the results stay bit-identical to the per-pixel
    /// path.
    fn prefilter_const(dst: &mut Vec<u32>, a: &FastRow, wx: u32) {
        let count = (a.pxb - a.pxa).max(0) as usize;
        dst.clear();
        dst.reserve(count);
        let last0 = a.row0.len().saturating_sub(2) as i64;
        let step = a.du >> 32;
        let i0 = ((a.ua - (1i64 << 31)) >> 32) - a.u_lo as i64;
        let clamp = |k: usize| (i0 + step * k as i64).clamp(0, last0) as usize;
        if wx | a.wy == 0 {
            // Both weights zero: the filter is a plain texel fetch.
            for k in 0..count {
                dst.push(a.row0[clamp(k)]);
            }
            return;
        }
        if step != 1 {
            for k in 0..count {
                let i = clamp(k);
                dst.push(bilerp_rgba(a.row0[i], a.row0[i + 1], a.row1[i], a.row1[i + 1], wx, a.wy));
            }
            return;
        }
        // Step one: clamping only trims the ends, the middle indexes are
        // `i0 + k` and contiguous.
        let k_lo = (-i0).clamp(0, count as i64) as usize;
        let k_hi = (last0 - i0 + 1).clamp(k_lo as i64, count as i64) as usize;
        for k in 0..k_lo {
            let i = clamp(k);
            dst.push(bilerp_rgba(a.row0[i], a.row0[i + 1], a.row1[i], a.row1[i + 1], wx, a.wy));
        }
        let mut k = k_lo;
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;
            let (wxl, wxr) = (256 - wx, wx);
            let (wyl, wyr) = (256 - a.wy, a.wy);
            // SAFETY: SSE2 baseline; all loads stay in bounds (pixel k+1
            // reads texels up to i0+k+2 <= last0+1).
            unsafe {
                let zero = _mm_setzero_si128();
                let (wxl, wxr) = (_mm_set1_epi16(wxl as i16), _mm_set1_epi16(wxr as i16));
                let (wyl, wyr) = (_mm_set1_epi16(wyl as i16), _mm_set1_epi16(wyr as i16));
                let lerp = |l: __m128i, r: __m128i, wl: __m128i, wr: __m128i| {
                    _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(l, wl), _mm_mullo_epi16(r, wr)), 8)
                };
                while k + 1 < k_hi {
                    let i = (i0 + k as i64) as usize;
                    let two = |r: &[u32], at: usize| {
                        _mm_unpacklo_epi8(_mm_loadl_epi64(r.as_ptr().add(at) as *const __m128i), zero)
                    };
                    let top = lerp(two(a.row0, i), two(a.row0, i + 1), wxl, wxr);
                    let bot = lerp(two(a.row1, i), two(a.row1, i + 1), wxl, wxr);
                    let out = _mm_packus_epi16(lerp(top, bot, wyl, wyr), zero);
                    let mut pair = [0u32; 2];
                    _mm_storel_epi64(pair.as_mut_ptr() as *mut __m128i, out);
                    dst.extend_from_slice(&pair);
                    k += 2;
                }
            }
        }
        for k in k..k_hi {
            let i = (i0 + k as i64) as usize;
            dst.push(bilerp_rgba(a.row0[i], a.row0[i + 1], a.row1[i], a.row1[i + 1], wx, a.wy));
        }
        for k in k_hi..count {
            let i = clamp(k);
            dst.push(bilerp_rgba(a.row0[i], a.row0[i + 1], a.row1[i], a.row1[i + 1], wx, a.wy));
        }
    }

    /// The pixel half of [`Painter::fast_sprite_row`], after the row's
    /// texel addressing has been decided.
    #[inline(never)]
    fn sprite_row_px<const DEC: bool, const BIL: bool, const ABE: bool, const ATE: bool>(&mut self, a: &FastRow) {
        let pipe = self.pipe;
        let (row, frag) = (a.row, a.frag);
        let y = row.y;
        let (cr, cg, cb, ca) = (frag.r as u32, frag.g as u32, frag.b as u32, frag.a as u32);
        let tcc = pipe.tcc;
        let (atst, aref, afail) = (pipe.atst, pipe.aref, pipe.afail);
        let blend = Blend::new(pipe);
        let canvas = self.canvas;
        let fb24 = pipe.fb24;
        let ztest = ZTest::new(pipe);
        let z = a.z & pipe.zmask;
        let (row0, row1) = (a.row0, a.row1);
        let last0 = row0.len().saturating_sub(if BIL { 2 } else { 1 });
        let mut u = a.ua - if BIL { 1i64 << 31 } else { 0 };
        let mod_lanes = Modulate::new(cr, cg, cb, ca, tcc);
        let _ = &mod_lanes;
        for px in a.pxa..a.pxb {
            // Texel: floor(u) (nearest) or the four taps around u - 0.5.
            let ix = (u >> 32) as i32;
            let i = ((ix - a.u_lo).max(0) as usize).min(last0);
            let texel = if BIL {
                let wx = ((u >> 24) & 0xFF) as u32;
                if wx | a.wy == 0 {
                    row0[i]
                } else {
                    bilerp_rgba(row0[i], row0[i + 1], row1[i], row1[i + 1], wx, a.wy)
                }
            } else {
                row0[i]
            };
            u += a.du;
            let ta = texel >> 24;
            // DECAL takes the texel's alpha as it stands; MODULATE scales
            // the vertex alpha by it.
            let a8 = if DEC {
                if tcc { ta } else { ca }
            } else if tcc {
                (ta * ca) >> 7
            } else {
                ca
            };
            let (mut write_z, mut keep_dst_alpha) = (true, false);
            if ATE {
                let pass = match atst {
                    0 => false,
                    1 => true,
                    2 => a8 < aref,
                    3 => a8 <= aref,
                    4 => a8 == aref,
                    5 => a8 >= aref,
                    6 => a8 > aref,
                    _ => a8 != aref,
                };
                if !pass {
                    match afail {
                        0 => continue, // KEEP
                        2 => {
                            // ZB_ONLY: the pixel still updates Z.
                            ztest.pass(canvas, row, px as u32, z);
                            continue;
                        }
                        1 => write_z = false,
                        _ => {
                            write_z = false;
                            keep_dst_alpha = true;
                        }
                    }
                }
            }
            if ztest.on {
                let zok = if write_z {
                    ztest.pass(canvas, row, px as u32, z)
                } else {
                    ztest.test(canvas, row, px as u32, z)
                };
                if !zok {
                    continue;
                }
            }
            let mut out = if DEC {
                (texel & 0x00FF_FFFF) | (a8 << 24)
            } else {
                mod_lanes.apply(texel)
            };
            let fb_off = (row.fb_base + layout::col_off32(y, px as u32, false)) & canvas.mask();
            let dst = if ABE || fb24 || keep_dst_alpha { canvas.rd32(fb_off) } else { 0 };
            if ABE {
                out = blend.apply(out, dst, a8);
            }
            if fb24 || keep_dst_alpha {
                // PSMCT24 frame, or RGB_ONLY: the alpha byte belongs to
                // whatever shares the word.
                out = (out & 0xFF_FFFF) | (dst & 0xFF00_0000);
            }
            canvas.wr32(fb_off, out);
        }
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
            let neutral = frag.r == 128.0 && frag.g == 128.0 && frag.b == 128.0 && frag.a == 128.0;
            crate::prof::count_pixel(pipe.profile_key(neutral));
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

        // Alpha test. AFAIL decides what a failing pixel still updates:
        // FB_ONLY leaves Z alone, ZB_ONLY updates only Z, RGB_ONLY keeps
        // the destination alpha as well as Z.
        let (mut write_z, mut write_fb, mut keep_dst_alpha) = (true, true, false);
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
                    1 => write_z = false,
                    2 => write_fb = false,
                    _ => {
                        write_z = false;
                        keep_dst_alpha = true;
                    }
                }
            }
        }

        // Depth test (linear z buffer, PSMZ32-style storage).
        let zmask = pipe.zmask;
        let z_off = (row.z_base + layout::col_off32(y, x, true)) & self.canvas.mask();
        let fb_off = (row.fb_base + layout::col_off32(y, x, false)) & self.canvas.mask();
        // ZTST ALWAYS with the Z write masked touches nothing: skip the Z
        // read (every in-game sprite of Amagami draws that way).
        if pipe.zte && !(pipe.ztst == 1 && pipe.zmsk) {
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
            if !pipe.zmsk && write_z {
                self.canvas.wr32(z_off, (zcur & !zmask) | z);
            }
        }
        if !write_fb {
            return;
        }

        // Destination pixel, only when something depends on it.
        let dst = if pipe.abe || pipe.fbmsk != 0 || pipe.fb24 || keep_dst_alpha {
            self.canvas.rd32(fb_off)
        } else {
            0
        };

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
        if keep_dst_alpha {
            merged = (merged & 0xFF_FFFF) | (dst & 0xFF00_0000);
        }
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
            clamp_uv(frag.u, frag.v)
        } else {
            let q = if frag.q.abs() < 1e-9 { 1.0 } else { frag.q };
            let inv_q = 1.0 / q;
            clamp_uv(frag.s * inv_q * ti.tw as f32, frag.t * inv_q * ti.th as f32)
        };

        // TEX1 MMAG selects the magnification filter; minification and
        // mipmaps are not modelled, so it decides for every sample.
        if !pipe.bilinear {
            return self.texel(floor_i32(fu), floor_i32(fv));
        }
        self.sample_bilinear_direct(TexFetch::new(&pipe.tex), fu, fv)
    }

    /// Bilinear texel at texel-space `(fu, fv)` straight from VRAM: the
    /// four taps share their wrapped coordinates and row bases, and the
    /// common formats address through the column tables.
    #[inline(always)]
    fn sample_bilinear_direct(&self, ti: TexFetch, fu: f32, fv: f32) -> u32 {
        let x = fu - 0.5;
        let y = fv - 0.5;
        let (x0, y0) = (floor_i32(x), floor_i32(y));
        // Weights in 1/256; exact texel centres skip the blend.
        let fx = ((x - x0 as f32) * 256.0) as u32;
        let fy = ((y - y0 as f32) * 256.0) as u32;
        if fx | fy == 0 {
            return self.texel(x0, y0);
        }
        let cv = self.tex;
        let u0 = wrap(x0, ti.wms, ti.tw, ti.minu, ti.maxu) as u32;
        let u1 = wrap(x0 + 1, ti.wms, ti.tw, ti.minu, ti.maxu) as u32;
        let v0 = wrap(y0, ti.wmt, ti.th, ti.minv, ti.maxv) as u32;
        let v1 = wrap(y0 + 1, ti.wmt, ti.th, ti.minv, ti.maxv) as u32;
        let (tbp, tbw) = (ti.tbp, ti.tbw);
        let m = cv.mask();
        let [t00, t10, t01, t11] = match ti.psm {
            PSMCT16 | PSMCT16S | PSMZ16 | PSMZ16S => {
                let (s, z) = (ti.psm & 8 != 0, ti.psm & 0x30 != 0);
                let (r0, r1) = (layout::row_base16(tbp, tbw, v0), layout::row_base16(tbp, tbw, v1));
                let p = [
                    cv.rd16((r0 + layout::col_off16(v0, u0, s, z)) & m),
                    cv.rd16((r0 + layout::col_off16(v0, u1, s, z)) & m),
                    cv.rd16((r1 + layout::col_off16(v1, u0, s, z)) & m),
                    cv.rd16((r1 + layout::col_off16(v1, u1, s, z)) & m),
                ];
                #[cfg(target_arch = "x86_64")]
                {
                    // SAFETY: SSE2 baseline.
                    return unsafe { bilerp16_sse2(p, ti.texa, fx, fy) };
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    [expand16(p[0], ti.texa), expand16(p[1], ti.texa), expand16(p[2], ti.texa), expand16(p[3], ti.texa)]
                }
            }
            PSMT8 => {
                let (r0, r1) = (layout::row_base8(tbp, tbw, v0), layout::row_base8(tbp, tbw, v1));
                let clut = self.clut;
                [
                    clut[cv.rd8((r0 + layout::col_off8(v0, u0)) & m) as usize],
                    clut[cv.rd8((r0 + layout::col_off8(v0, u1)) & m) as usize],
                    clut[cv.rd8((r1 + layout::col_off8(v1, u0)) & m) as usize],
                    clut[cv.rd8((r1 + layout::col_off8(v1, u1)) & m) as usize],
                ]
            }
            PSMCT32 => {
                let (r0, r1) = (layout::row_base32(tbp, tbw, v0, false), layout::row_base32(tbp, tbw, v1, false));
                [
                    cv.rd32((r0 + layout::col_off32(v0, u0, false)) & m),
                    cv.rd32((r0 + layout::col_off32(v0, u1, false)) & m),
                    cv.rd32((r1 + layout::col_off32(v1, u0, false)) & m),
                    cv.rd32((r1 + layout::col_off32(v1, u1, false)) & m),
                ]
            }
            _ => [self.texel(x0, y0), self.texel(x0 + 1, y0), self.texel(x0, y0 + 1), self.texel(x0 + 1, y0 + 1)],
        };
        bilerp_rgba(t00, t10, t01, t11, fx, fy)
    }

    /// One RGBA8 texel at integer texel coordinates, after CLAMP wrapping.
    #[inline(always)]
    fn texel(&self, u: i32, v: i32) -> u32 {
        let ti = &self.pipe.tex;
        let cv = self.tex;
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

impl PixelPipe {
    /// Histogram key for the profile report: kind (3 bits), tme, bilinear,
    /// abe, texture psm (6), tfx (2), ate, Z read, Z write, frame mask,
    /// 24-bit frame, neutral vertex colour.
    #[cfg(feature = "profile")]
    fn profile_key(&self, neutral: bool) -> usize {
        let zread = self.zte && !(self.ztst == 1 && self.zmsk);
        let zwrite = self.zte && !self.zmsk;
        (self.kind as usize & 7)
            | ((self.tme as usize) << 3)
            | ((self.bilinear as usize) << 4)
            | ((self.abe as usize) << 5)
            | ((self.tex.psm as usize & 0x3F) << 6)
            | ((self.tfx as usize & 3) << 12)
            | ((self.ate as usize) << 14)
            | ((zread as usize) << 15)
            | ((zwrite as usize) << 16)
            | (((self.fbmsk != 0) as usize) << 17)
            | ((self.fb24 as usize) << 18)
            | ((neutral as usize) << 19)
    }
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

/// Arguments of [`Painter::fast_sprite_row`].
#[derive(Clone, Copy)]
struct FastRow<'a> {
    row: &'a Row,
    pxa: i32,
    pxb: i32,
    frag: &'a Frag,
    /// Decoded texture rows (the second only for bilinear) and the texel
    /// index their first entry holds.
    row0: &'a [u32],
    row1: &'a [u32],
    u_lo: i32,
    /// Bilinear row weight (1/256).
    wy: u32,
    /// Texel u of the first pixel and its per-pixel step, 32.32 fixed.
    ua: i64,
    du: i64,
    /// The sprite's (constant) Z.
    z: u32,
}

/// Arguments of [`Painter::fast_tri_row`]: the span and the attributes at
/// its first pixel plus their per-pixel steps.
struct FastTri<'a> {
    row: &'a Row,
    xs: i32,
    xe: i32,
    rgba: [f32; 4],
    stqu: [f32; 4],
    v: f32,
    z: f64,
    d_rgba: [f32; 4],
    d_stqu: [f32; 4],
    d_v: f32,
    d_z: f64,
}

/// Texture addressing parameters copied out of the pipe for the fast
/// loops, so they live in registers rather than behind the pipe pointer.
#[derive(Clone, Copy)]
struct TexFetch {
    psm: u32,
    tbp: u32,
    tbw: u32,
    tw: i32,
    th: i32,
    wms: u64,
    wmt: u64,
    minu: i32,
    maxu: i32,
    minv: i32,
    maxv: i32,
    texa: u64,
}

impl TexFetch {
    #[inline(always)]
    fn new(ti: &TexInfo) -> Self {
        Self {
            psm: ti.psm,
            tbp: ti.tbp,
            tbw: ti.tbw,
            tw: ti.tw as i32,
            th: ti.th as i32,
            wms: ti.wms,
            wmt: ti.wmt,
            minu: ti.minu,
            maxu: ti.maxu,
            minv: ti.minv,
            maxv: ti.maxv,
            texa: ti.texa,
        }
    }
}

/// ALPHA blending for the fast loops: `Cv = ((A - B) * C >> 7) + D` with
/// the A/B/D sources and the C source decoded once. The SSE2 form pairs
/// each channel's `A - B` with 128 and multiplies against `C` and `D` in
/// one `pmaddwd`, so the 32-bit sum needs no lane bias; `packs`/`packus`
/// clamp to 0..255 like the scalar path.
#[derive(Clone, Copy)]
struct Blend {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
    fix: u32,
}

impl Blend {
    #[inline(always)]
    fn new(pipe: &PixelPipe) -> Self {
        Self { a: pipe.blend_a, b: pipe.blend_b, c: pipe.blend_c, d: pipe.blend_d, fix: pipe.blend_fix }
    }

    /// Blended RGB of `src` over `dst` (both RGBA8); the result keeps the
    /// alpha byte of `src`. `src_a` is the (unsaturated) source alpha.
    #[inline(always)]
    fn apply(self, src: u32, dst: u32, src_a: u32) -> u32 {
        let pick = |k: u8| -> u32 {
            match k {
                0 => src,
                1 => dst,
                _ => 0,
            }
        };
        let alpha = match self.c {
            0 => src_a,
            1 => dst >> 24,
            _ => self.fix,
        };
        let (a, b, d) = (pick(self.a), pick(self.b), pick(self.d));
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;
            // SAFETY: SSE2 baseline; pure register arithmetic.
            unsafe {
                let zero = _mm_setzero_si128();
                let a16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(a as i32), zero);
                let b16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(b as i32), zero);
                let d16 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(d as i32), zero);
                let diff = _mm_sub_epi16(a16, b16);
                // [diff_r, 128, diff_g, 128, ...] . [C, d_r, C, d_g, ...]
                let x = _mm_unpacklo_epi16(diff, _mm_set1_epi16(128));
                let y = _mm_unpacklo_epi16(_mm_set1_epi16(alpha as i16), d16);
                let acc = _mm_srai_epi32(_mm_madd_epi16(x, y), 7);
                let packed = _mm_packus_epi16(_mm_packs_epi32(acc, acc), zero);
                (_mm_cvtsi128_si32(packed) as u32 & 0xFF_FFFF) | (src & 0xFF00_0000)
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let lane = |sh: u32| -> u32 {
                let (av, bv, dv) = (((a >> sh) & 0xFF) as i32, ((b >> sh) & 0xFF) as i32, ((d >> sh) & 0xFF) as i32);
                ((((av - bv) * alpha as i32) >> 7) + dv).clamp(0, 255) as u32
            };
            lane(0) | (lane(8) << 8) | (lane(16) << 16) | (src & 0xFF00_0000)
        }
    }
}

/// Per-pixel depth test and write for the fast loops, decoded once.
#[derive(Clone, Copy)]
struct ZTest {
    /// Anything to do at all (`PixelPipe::z_touched`).
    on: bool,
    ztst: u8,
    write: bool,
    zmask: u32,
}

impl ZTest {
    #[inline(always)]
    fn new(pipe: &PixelPipe) -> Self {
        Self { on: pipe.z_touched, ztst: pipe.ztst, write: pipe.zte && !pipe.zmsk, zmask: pipe.zmask }
    }

    /// Test `z` without updating the buffer: an alpha test that failed
    /// with FB_ONLY or RGB_ONLY still gates on Z but must not write it.
    #[inline(always)]
    fn test(self, canvas: &Canvas, row: &Row, x: u32, z: u32) -> bool {
        let z_off = (row.z_base + layout::col_off32(row.y, x, true)) & canvas.mask();
        let zcur = canvas.rd32(z_off);
        match self.ztst {
            0 => false,
            1 => true,
            2 => z >= (zcur & self.zmask),
            _ => z > (zcur & self.zmask),
        }
    }

    /// Test `z` (already masked) against the buffer; writes it when the
    /// pixel passes and writes are enabled. Same as the generic path.
    #[inline(always)]
    fn pass(self, canvas: &Canvas, row: &Row, x: u32, z: u32) -> bool {
        let z_off = (row.z_base + layout::col_off32(row.y, x, true)) & canvas.mask();
        let zcur = canvas.rd32(z_off);
        let pass = match self.ztst {
            0 => false,
            1 => true,
            2 => z >= (zcur & self.zmask),
            _ => z > (zcur & self.zmask),
        };
        if pass && self.write {
            canvas.wr32(z_off, (zcur & !self.zmask) | z);
        }
        pass
    }
}

/// MODULATE colour lanes: `(texel * colour) >> 7` per channel saturated
/// to 255, alpha from the texel (TCC) or the vertex.
#[derive(Clone, Copy)]
struct Modulate {
    #[cfg(target_arch = "x86_64")]
    lanes: core::arch::x86_64::__m128i,
    #[cfg(not(target_arch = "x86_64"))]
    c: [u32; 4],
    /// Texel alpha stands in for 128 when TCC is off, so one multiply
    /// yields the vertex alpha.
    tex_mask: u32,
    tex_or: u32,
    /// All factors are 128: the multiply is the identity, skip it. Very
    /// common (2D draws leave the vertex colour neutral).
    neutral: bool,
}

impl Modulate {
    #[inline(always)]
    fn new(cr: u32, cg: u32, cb: u32, ca: u32, tcc: bool) -> Self {
        let (tex_mask, tex_or) = if tcc { (u32::MAX, 0) } else { (0x00FF_FFFF, 0x8000_0000) };
        Self {
            #[cfg(target_arch = "x86_64")]
            // SAFETY: SSE2 is x86-64 baseline.
            lanes: unsafe { core::arch::x86_64::_mm_set_epi16(0, 0, 0, 0, ca as i16, cb as i16, cg as i16, cr as i16) },
            #[cfg(not(target_arch = "x86_64"))]
            c: [cr, cg, cb, ca],
            tex_mask,
            tex_or,
            neutral: cr == 128 && cg == 128 && cb == 128 && ca == 128,
        }
    }

    #[inline(always)]
    fn apply(self, texel: u32) -> u32 {
        let t = (texel & self.tex_mask) | self.tex_or;
        if self.neutral {
            // t * 128 >> 7 == t for every lane, including the substituted
            // 0x80 alpha when TCC is off.
            return t;
        }
        #[cfg(target_arch = "x86_64")]
        {
            use core::arch::x86_64::*;
            // SAFETY: SSE2 baseline; 255*255 fits the 16-bit lanes and
            // packus saturates like `.min(255)`.
            unsafe {
                let zero = _mm_setzero_si128();
                let lanes = _mm_unpacklo_epi8(_mm_cvtsi32_si128(t as i32), zero);
                let m = _mm_srli_epi16(_mm_mullo_epi16(lanes, self.lanes), 7);
                _mm_cvtsi128_si32(_mm_packus_epi16(m, m)) as u32
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let ch = |sh: u32, c: u32| ((((t >> sh) & 0xFF) * c) >> 7).min(255) << sh;
            ch(0, self.c[0]) | ch(8, self.c[1]) | ch(16, self.c[2]) | ch(24, self.c[3])
        }
    }
}

/// Drawing environment decoded once per primitive (see `pixel_pipe`).
#[derive(Clone)]
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
    /// The Z buffer is read or written per pixel (ZTE with a real test,
    /// or unmasked writes).
    z_touched: bool,
    /// Textured MODULATE drawing without a frame mask: eligible for the
    /// specialised row loops (`Painter::fast_sprite_row`,
    /// `Painter::fast_tri_row`).
    /// Offset of this primitive's palette block in [`ClutPool`].
    clut_off: u32,
    fast: bool,
    /// As `fast`, but the texture function is DECAL: the texel replaces the
    /// vertex colour instead of scaling it. Sprites only — the triangle
    /// loops still take MODULATE alone.
    fast_decal: bool,
    /// Untextured sprite with a constant colour per row (blended or not)
    /// and no Z test: `Painter::flat_sprite_row`.
    flat_fill: bool,
}

/// TEX0/CLAMP fields decoded once per primitive.
#[derive(Clone)]
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

/// Bilinear blend of four 16-bit texels (`p[0] p[1]` top, `p[2] p[3]`
/// bottom), expanding them under TEXA in 16-bit lanes on the way: 5-bit
/// channels scale by 8, alpha is TA1 for set MSBs, TA0 otherwise, and 0
/// for all-zero texels when AEM is on. Same output as `expand16` +
/// [`bilerp_sse2`].
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn bilerp16_sse2(p: [u16; 4], texa: u64, wx: u32, wy: u32) -> u32 {
    use core::arch::x86_64::*;
    // SAFETY: SSE2 baseline; pure register arithmetic.
    unsafe {
        let t = _mm_set_epi16(0, 0, 0, 0, p[3] as i16, p[2] as i16, p[1] as i16, p[0] as i16);
        let m5 = _mm_set1_epi16(0x1F);
        let r = _mm_slli_epi16(_mm_and_si128(t, m5), 3);
        let g = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(t, 5), m5), 3);
        let b = _mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(t, 10), m5), 3);
        let ta0 = _mm_set1_epi16((texa & 0xFF) as i16);
        let ta1 = _mm_set1_epi16(((texa >> 32) & 0xFF) as i16);
        let msb = _mm_srai_epi16(t, 15);
        let mut a = _mm_or_si128(_mm_and_si128(msb, ta1), _mm_andnot_si128(msb, ta0));
        if texa & (1 << 15) != 0 {
            let zero = _mm_cmpeq_epi16(_mm_and_si128(t, _mm_set1_epi16(0x7FFF)), _mm_setzero_si128());
            a = _mm_andnot_si128(zero, a);
        }
        // Channel-major -> pixel-major: [r g b a] per texel, texels 0,1 in
        // `top`, 2,3 in `bottom`.
        let rg = _mm_unpacklo_epi16(r, g);
        let ba = _mm_unpacklo_epi16(b, a);
        let top = _mm_unpacklo_epi32(rg, ba);
        let bottom = _mm_unpackhi_epi32(rg, ba);
        let wxl = _mm_set1_epi16((256 - wx) as i16);
        let wxr = _mm_set1_epi16(wx as i16);
        let hx = |row: __m128i| -> __m128i {
            let left = row;
            let right = _mm_srli_si128(row, 8);
            _mm_srli_epi16(_mm_add_epi16(_mm_mullo_epi16(left, wxl), _mm_mullo_epi16(right, wxr)), 8)
        };
        let (ht, hb) = (hx(top), hx(bottom));
        let y = _mm_srli_epi16(
            _mm_add_epi16(
                _mm_mullo_epi16(ht, _mm_set1_epi16((256 - wy) as i16)),
                _mm_mullo_epi16(hb, _mm_set1_epi16(wy as i16)),
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

/// Texel coordinates are clamped to this before anything integer touches
/// them. A near-zero Q or a runaway ST sends S/Q * TW past what f32 can
/// floor into an i32, and the tap and cache-window arithmetic downstream
/// then wraps around i32 and reads an empty row. The limit is a power of
/// two well above any texture, so REPEAT masks and CLAMP saturate to the
/// same texel they would have without it.
const TEX_COORD_LIMIT: f32 = (1 << 24) as f32;

/// Both texel coordinates through [`TEX_COORD_LIMIT`]. NaN survives as
/// NaN, which `floor_i32` turns into 0 rather than an out-of-range index.
#[inline(always)]
fn clamp_uv(fu: f32, fv: f32) -> (f32, f32) {
    (
        fu.clamp(-TEX_COORD_LIMIT, TEX_COORD_LIMIT),
        fv.clamp(-TEX_COORD_LIMIT, TEX_COORD_LIMIT),
    )
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
    v.clamp(0, 4096)
}

/// Whether [`wrap`] is the identity over the whole span `u_lo..=u_hi`,
/// letting a row fill skip the per-texel wrap.
#[inline]
fn wrap_identity(u_lo: i32, u_hi: i32, ti: &TexInfo) -> bool {
    match ti.wms {
        0 | 1 => u_lo >= 0 && u_hi < ti.tw as i32,
        2 => ti.minu <= ti.maxu && u_lo >= ti.minu && u_hi <= ti.maxu,
        _ => false,
    }
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
