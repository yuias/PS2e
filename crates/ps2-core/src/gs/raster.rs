//! Pixel pipeline: triangle/sprite rasterization, texture sampling,
//! alpha blending and the frame-buffer write path.

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

impl Gs {
    pub(super) fn draw_point(&mut self) {
        let v = self.vq[0];
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
        self.shade_pixel(v.x >> 4, v.y >> 4, frag);
        self.prims_drawn += 1;
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

    pub(super) fn draw_sprite(&mut self) {
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
        let wid = (x1 - x0).max(1) as f32;
        let hei = (y1 - y0).max(1) as f32;
        if px1 - px0 <= 24 && py1 - py0 <= 24 {
            self.log_small_prim("sprite", attrs, px1 - px0, py1 - py0, &v0, &v1);
        }
        for py in py0..px_clip(py1) {
            for px in px0..px_clip(px1) {
                // Interpolate texture coords across the rectangle; color is
                // flat from the second vertex.
                let fx = ((px << 4) as f32 + 8.0 - x0 as f32) / wid;
                let fy = ((py << 4) as f32 + 8.0 - y0 as f32) / hei;
                let (u0, u1) = if v0.x <= v1.x {
                    (v0.u, v1.u)
                } else {
                    (v1.u, v0.u)
                };
                let (tv0, tv1) = if v0.y <= v1.y {
                    (v0.v, v1.v)
                } else {
                    (v1.v, v0.v)
                };
                let (s0, s1) = if v0.x <= v1.x {
                    (v0.s, v1.s)
                } else {
                    (v1.s, v0.s)
                };
                let (t0, t1) = if v0.y <= v1.y {
                    (v0.t, v1.t)
                } else {
                    (v1.t, v0.t)
                };
                let frag = Frag {
                    r: v1.r as f32,
                    g: v1.g as f32,
                    b: v1.b as f32,
                    a: v1.a as f32,
                    z: v1.z,
                    s: s0 + (s1 - s0) * fx,
                    t: t0 + (t1 - t0) * fy,
                    q: v1.q,
                    u: (u0 as f32 + (u1 - u0) as f32 * fx) / 16.0,
                    v: (tv0 as f32 + (tv1 - tv0) as f32 * fy) / 16.0,
                };
                self.shade_pixel(px, py, frag);
            }
        }
    }

    pub(super) fn draw_triangle(&mut self, i0: usize, i1: usize, i2: usize) {
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
        let inv_area = 1.0 / area as f32;
        for py in miny..=maxy {
            for px in minx..=maxx {
                let sx = (px << 4) + 8;
                let sy = (py << 4) + 8;
                let w0 = edge(b.x, b.y, c.x, c.y, sx, sy);
                let w1 = edge(c.x, c.y, a.x, a.y, sx, sy);
                let w2 = edge(a.x, a.y, b.x, b.y, sx, sy);
                if w0 < 0 || w1 < 0 || w2 < 0 {
                    continue;
                }
                let l0 = w0 as f32 * inv_area;
                let l1 = w1 as f32 * inv_area;
                let l2 = w2 as f32 * inv_area;
                let frag = Frag {
                    r: a.r as f32 * l0 + b.r as f32 * l1 + c.r as f32 * l2,
                    g: a.g as f32 * l0 + b.g as f32 * l1 + c.g as f32 * l2,
                    b: a.b as f32 * l0 + b.b as f32 * l1 + c.b as f32 * l2,
                    a: a.a as f32 * l0 + b.a as f32 * l1 + c.a as f32 * l2,
                    z: (a.z as f64 * l0 as f64 + b.z as f64 * l1 as f64 + c.z as f64 * l2 as f64)
                        as u32,
                    s: a.s * l0 + b.s * l1 + c.s * l2,
                    t: a.t * l0 + b.t * l1 + c.t * l2,
                    q: a.q * l0 + b.q * l1 + c.q * l2,
                    u: (a.u as f32 * l0 + b.u as f32 * l1 + c.u as f32 * l2) / 16.0,
                    v: (a.v as f32 * l0 + b.v as f32 * l1 + c.v as f32 * l2) / 16.0,
                };
                self.shade_pixel(px, py, frag);
            }
        }
    }

    /// Full per-pixel pipeline: scissor, texture, tests, blend, write.
    fn shade_pixel(&mut self, x: i32, y: i32, frag: Frag) {
        let ctxi = self.ctx_index();
        let ctx = self.ctx[ctxi];
        let attrs = self.attrs();

        // Scissor (pixel units, inclusive).
        let scx0 = (ctx.scissor & 0x7FF) as i32;
        let scx1 = ((ctx.scissor >> 16) & 0x7FF) as i32;
        let scy0 = ((ctx.scissor >> 32) & 0x7FF) as i32;
        let scy1 = ((ctx.scissor >> 48) & 0x7FF) as i32;
        if x < scx0 || x > scx1 || y < scy0 || y > scy1 {
            return;
        }
        let (x, y) = (x as u32, y as u32);

        // Source color: vertex color, optionally combined with a texel.
        let mut r = frag.r;
        let mut g = frag.g;
        let mut b = frag.b;
        let mut a = frag.a;
        if attrs & (1 << 4) != 0 {
            self.tex_psm_hist[((ctx.tex0 >> 20) & 0x3F) as usize] += 1;
            let (tr, tg, tb, ta) = self.sample(ctx, attrs, &frag);
            let tfx = (ctx.tex0 >> 35) & 3;
            let tcc = ctx.tex0 & (1 << 34) != 0;
            match tfx {
                0 => {
                    // MODULATE
                    r = (tr * r / 128.0).min(255.0);
                    g = (tg * g / 128.0).min(255.0);
                    b = (tb * b / 128.0).min(255.0);
                    if tcc {
                        a = (ta * a / 128.0).min(255.0);
                    }
                }
                1 => {
                    // DECAL
                    r = tr;
                    g = tg;
                    b = tb;
                    if tcc {
                        a = ta;
                    }
                }
                _ => {
                    // HIGHLIGHT/HIGHLIGHT2: approximate.
                    r = (tr * r / 128.0 + a).min(255.0);
                    g = (tg * g / 128.0 + a).min(255.0);
                    b = (tb * b / 128.0 + a).min(255.0);
                    if tcc {
                        a = ta;
                    }
                }
            }
        }

        // Alpha test.
        let test = ctx.test;
        if test & 1 != 0 {
            let atst = (test >> 1) & 7;
            let aref = ((test >> 4) & 0xFF) as f32;
            let pass = match atst {
                0 => false,
                1 => true,
                2 => a < aref,
                3 => a <= aref,
                4 => (a as u32) == aref as u32,
                5 => a >= aref,
                6 => a > aref,
                _ => (a as u32) != aref as u32,
            };
            if !pass {
                let afail = (test >> 12) & 3;
                match afail {
                    0 => return, // KEEP
                    1 => {}      // FB_ONLY: continue without z write
                    2 => return, // ZB_ONLY: no color -> nothing visible
                    _ => {}      // RGB_ONLY
                }
            }
        }

        // Depth test (linear z buffer, PSMZ32-style storage).
        let zte = test & (1 << 16) != 0;
        let ztst = (test >> 17) & 3;
        let zbp = ((ctx.zbuf & 0x1FF) * 32) as u32;
        let zmsk = ctx.zbuf & (1 << 32) != 0;
        let fbw = ((ctx.frame >> 16) & 0x3F) as u32;
        // Z buffer depth: PSMZ32 keeps 32 bits, PSMZ24 24, PSMZ16(S) 16;
        // the upper bits of the stored word belong to whatever else shares
        // the memory (Amagami parks 8-bit textures over its Z24 buffer).
        let zmask = match (ctx.zbuf >> 24) & 0xF {
            0x0 => u32::MAX,
            0x1 => 0x00FF_FFFF,
            _ => 0xFFFF,
        };
        if zte && ztst != 1 {
            let zcur = self.read_psmct32(zbp, fbw, x, y) & zmask;
            let z = frag.z & zmask;
            let pass = match ztst {
                0 => false,
                2 => z >= zcur,
                _ => z > zcur,
            };
            if !pass {
                return;
            }
        }
        if zte && !zmsk {
            let cur = self.read_psmct32(zbp, fbw, x, y);
            self.write_psmct32(zbp, fbw, x, y, (cur & !zmask) | (frag.z & zmask));
        }

        // Destination blend.
        let fbp = ((ctx.frame & 0x1FF) * 32) as u32;
        let psm = ((ctx.frame >> 24) & 0x3F) as u32;
        let fbmsk = (ctx.frame >> 32) as u32;
        let dst = self.read_psmct32(fbp, fbw, x, y);

        let (dr, dg, db, da) = (
            (dst & 0xFF) as f32,
            ((dst >> 8) & 0xFF) as f32,
            ((dst >> 16) & 0xFF) as f32,
            ((dst >> 24) & 0xFF) as f32,
        );
        if attrs & (1 << 6) != 0 {
            // ALPHA: Cv = ((A - B) * C >> 7) + D.
            let al = ctx.alpha;
            let sel = |k: u64, s: f32, d: f32| -> f32 {
                match k & 3 {
                    0 => s,
                    1 => d,
                    _ => 0.0,
                }
            };
            let ca = (sel(al, r, dr), sel(al, g, dg), sel(al, b, db));
            let cb = (
                sel(al >> 2, r, dr),
                sel(al >> 2, g, dg),
                sel(al >> 2, b, db),
            );
            let alpha = match (al >> 4) & 3 {
                0 => a,
                1 => da,
                _ => ((al >> 32) & 0xFF) as f32,
            };
            let cd = (
                sel(al >> 6, r, dr),
                sel(al >> 6, g, dg),
                sel(al >> 6, b, db),
            );
            r = ((ca.0 - cb.0) * alpha / 128.0 + cd.0).clamp(0.0, 255.0);
            g = ((ca.1 - cb.1) * alpha / 128.0 + cd.1).clamp(0.0, 255.0);
            b = ((ca.2 - cb.2) * alpha / 128.0 + cd.2).clamp(0.0, 255.0);
        }

        let out =
            (r as u32) | ((g as u32) << 8) | ((b as u32) << 16) | ((a.min(255.0) as u32) << 24);
        let cur = dst;
        let mut merged = (out & !fbmsk) | (cur & fbmsk);
        if psm == PSMCT24 {
            merged = (merged & 0xFF_FFFF) | (cur & 0xFF00_0000);
        }
        self.write_psmct32(fbp, fbw, x, y, merged);
    }

    /// Nearest-neighbour texture sample.
    fn sample(&self, ctx: Context, attrs: u64, frag: &Frag) -> (f32, f32, f32, f32) {
        let tex0 = ctx.tex0;
        let tbp = (tex0 & 0x3FFF) as u32;
        let tbw = ((tex0 >> 14) & 0x3F) as u32;
        let psm = ((tex0 >> 20) & 0x3F) as u32;
        let tw = 1u32 << ((tex0 >> 26) & 0xF).min(10);
        let th = 1u32 << ((tex0 >> 30) & 0xF).min(10);

        // FST: UV addressing vs STQ.
        let (mut u, mut v) = if attrs & (1 << 8) != 0 {
            (frag.u as i32, frag.v as i32)
        } else {
            let q = if frag.q.abs() < 1e-9 { 1.0 } else { frag.q };
            (
                (frag.s / q * tw as f32) as i32,
                (frag.t / q * th as f32) as i32,
            )
        };

        // CLAMP register: 0 repeat, 1 clamp, 2 region clamp, 3 region repeat.
        let wms = ctx.clamp & 3;
        let wmt = (ctx.clamp >> 2) & 3;
        let minu = ((ctx.clamp >> 4) & 0x3FF) as i32;
        let maxu = ((ctx.clamp >> 14) & 0x3FF) as i32;
        let minv = ((ctx.clamp >> 24) & 0x3FF) as i32;
        let maxv = ((ctx.clamp >> 34) & 0x3FF) as i32;
        u = wrap(u, wms, tw as i32, minu, maxu);
        v = wrap(v, wmt, th as i32, minv, maxv);
        let (u, v) = (u as u32, v as u32);

        let texel = match psm {
            PSMCT32 | PSMCT24 => self.read_psmct32(tbp, tbw, u, v),
            PSMCT16 | PSMCT16S => expand16(self.read_psmct16(tbp, tbw, u, v), self.texa),
            PSMT8 => {
                let idx = self.read_psmt8(tbp, tbw, u, v);
                self.clut_lookup(tex0, idx as u32, true)
            }
            PSMT4 => {
                let idx = self.read_psmt4(tbp, tbw, u, v);
                self.clut_lookup(tex0, idx as u32, false)
            }
            PSMT8H => {
                let idx = self.read_psmct32(tbp, tbw, u, v) >> 24;
                self.clut_lookup(tex0, idx, true)
            }
            PSMT4HL => {
                let idx = (self.read_psmct32(tbp, tbw, u, v) >> 24) & 0xF;
                self.clut_lookup(tex0, idx, false)
            }
            PSMT4HH => {
                let idx = self.read_psmct32(tbp, tbw, u, v) >> 28;
                self.clut_lookup(tex0, idx, false)
            }
            _ => 0xFF00_FFFF,
        };
        let mut ta = ((texel >> 24) & 0xFF) as f32;
        if psm == PSMCT24 {
            ta = (self.texa & 0xFF) as f32;
        }
        (
            (texel & 0xFF) as f32,
            ((texel >> 8) & 0xFF) as f32,
            ((texel >> 16) & 0xFF) as f32,
            ta,
        )
    }

    /// Look a palette index up through the CLUT buffer.
    fn clut_lookup(&self, tex0: u64, index: u32, eight_bit: bool) -> u32 {
        let cbp = ((tex0 >> 37) & 0x3FFF) as u32;
        let cpsm = ((tex0 >> 51) & 0xF) as u32;
        let csm = (tex0 >> 55) & 1;
        let csa = ((tex0 >> 56) & 0x1F) as u32;
        // CSA offsets in 16-entry slots (must be 0 for 8-bit CLUTs).
        let e = if eight_bit { index } else { index + csa * 16 };
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
            self.read_psmct32(cbp, 1, x, y)
        } else {
            expand16(self.read_psmct16(cbp, 1, x, y), self.texa)
        }
    }
}

#[inline]
fn edge(x0: i32, y0: i32, x1: i32, y1: i32, x: i32, y: i32) -> i64 {
    (x1 - x0) as i64 * (y - y0) as i64 - (y1 - y0) as i64 * (x - x0) as i64
}

#[inline]
fn px_clip(v: i32) -> i32 {
    v.clamp(0, 2048)
}

#[inline]
fn wrap(c: i32, mode: u64, size: i32, min: i32, max: i32) -> i32 {
    match mode {
        0 => c.rem_euclid(size.max(1)),
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
