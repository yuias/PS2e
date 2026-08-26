//! SPU2 reverb: the PS1 algorithm (nocash "SPU Reverb Formula") run at
//! half the sample rate over a work area in sound RAM between ESA and EEA.
//! Offsets are halfword addresses relative to ESA and rotate with the
//! buffer position, wrapping inside the area.

use std::sync::LazyLock;
use serde::{Deserialize, Serialize};

/// Reverb register snapshot for one core, as halfword offsets and 16-bit
/// signed coefficients.
#[derive(Clone, Copy, Default)]
pub struct ReverbRegs {
    /// Work area start and inclusive end (halfword addresses).
    pub esa: u32,
    pub eea: u32,
    /// The 22 address registers in register order: dAPF1, dAPF2, mLSAME,
    /// mRSAME, mLCOMB1, mRCOMB1, mLCOMB2, mRCOMB2, dLSAME, dRSAME, mLDIFF,
    /// mRDIFF, mLCOMB3, mRCOMB3, mLCOMB4, mRCOMB4, dLDIFF, dRDIFF, mLAPF1,
    /// mRAPF1, mLAPF2, mRAPF2.
    pub addr: [u32; 22],
    /// vIIR, vCOMB1..4, vWALL, vAPF1, vAPF2, vLIN, vRIN.
    pub vol: [i32; 10],
}

const D_APF1: usize = 0;
const D_APF2: usize = 1;
const M_LSAME: usize = 2;
const M_RSAME: usize = 3;
const M_LCOMB1: usize = 4;
const M_RCOMB1: usize = 5;
const M_LCOMB2: usize = 6;
const M_RCOMB2: usize = 7;
const D_LSAME: usize = 8;
const D_RSAME: usize = 9;
const M_LDIFF: usize = 10;
const M_RDIFF: usize = 11;
const M_LCOMB3: usize = 12;
const M_RCOMB3: usize = 13;
const M_LCOMB4: usize = 14;
const M_RCOMB4: usize = 15;
const D_LDIFF: usize = 16;
const D_RDIFF: usize = 17;
const M_LAPF1: usize = 18;
const M_RAPF1: usize = 19;
const M_LAPF2: usize = 20;
const M_RAPF2: usize = 21;

const V_IIR: usize = 0;
const V_COMB1: usize = 1;
const V_WALL: usize = 5;
const V_APF1: usize = 6;
const V_APF2: usize = 7;
const V_LIN: usize = 8;
const V_RIN: usize = 9;

/// Length of the resampling filter between the output rate and the
/// reverb's own half rate, one in each direction.
const FIR_TAPS: usize = 39;
/// Ring capacity for the filter histories (a power of two >= FIR_TAPS,
/// and >= FIR_TAPS / 2 for the half-rate side).
const HIST: usize = 64;

/// Resampling filter for the reverb path. The hardware runs the reverb at
/// half the output rate behind a 39-tap FIR in each direction (19 samples
/// of group delay each way); the coefficients themselves are not
/// published, so this is a Blackman-windowed sinc of the same length,
/// cutting off at the half-rate Nyquist. What matters audibly is that the
/// input is band-limited before it is decimated and that the output is
/// interpolated rather than held: a plain two-tap average and a
/// sample-and-hold leave audible aliasing and images behind.
static FIR: LazyLock<[i32; FIR_TAPS]> = LazyLock::new(|| {
    use std::f64::consts::PI;
    let mut h = [0f64; FIR_TAPS];
    let mid = (FIR_TAPS / 2) as f64;
    for (i, v) in h.iter_mut().enumerate() {
        let n = i as f64 - mid;
        let sinc = if n == 0.0 { 0.5 } else { (PI * n * 0.5).sin() / (PI * n) };
        let x = 2.0 * PI * i as f64 / (FIR_TAPS - 1) as f64;
        *v = sinc * (0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos());
    }
    // 15-bit weights with unity DC gain, so decimation preserves level.
    let scale = 32768.0 / h.iter().sum::<f64>();
    h.map(|v| (v * scale).round() as i32)
});

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Reverb {
    /// Rotating offset into the work area.
    pos: u32,
    /// The algorithm runs every other sample.
    phase: bool,
    /// Wet input at the output rate, for the decimating filter.
    #[serde(with = "rings")]
    input: [[i32; HIST]; 2],
    /// Reverb output at the half rate, for the interpolating filter.
    #[serde(with = "rings")]
    half: [[i32; HIST]; 2],
    /// Write positions in the two rings.
    in_pos: usize,
    half_pos: usize,
}

impl Default for Reverb {
    fn default() -> Self {
        Self {
            pos: 0,
            phase: false,
            input: [[0; HIST]; 2],
            half: [[0; HIST]; 2],
            in_pos: 0,
            half_pos: 0,
        }
    }
}

fn mul(a: i32, v: i32) -> i32 {
    (a * v) >> 15
}

fn sat(v: i32) -> i32 {
    v.clamp(-0x8000, 0x7FFF)
}

impl Reverb {
    /// Feed one sample pair of wet input; returns the reverb output for
    /// this sample (before EVOL).
    pub fn sample(&mut self, ram: &mut [u8], regs: &ReverbRegs, lin: i32, rin: i32) -> [i32; 2] {
        self.in_pos = (self.in_pos + 1) % HIST;
        self.input[0][self.in_pos] = lin;
        self.input[1][self.in_pos] = rin;
        // A half-rate step lands on every other output sample; that sample
        // is the one the interpolator sees at zero phase.
        let aligned = self.phase;
        if aligned {
            let l = self.decimate(0);
            let r = self.decimate(1);
            let out = self.step(ram, regs, l, r);
            self.half_pos = (self.half_pos + 1) % HIST;
            self.half[0][self.half_pos] = out[0];
            self.half[1][self.half_pos] = out[1];
        }
        self.phase = !self.phase;
        [self.interpolate(0, !aligned), self.interpolate(1, !aligned)]
    }

    /// Band-limit the wet input and take it down to the reverb's rate.
    fn decimate(&self, c: usize) -> i32 {
        let mut acc = 0i64;
        for (k, &h) in FIR.iter().enumerate() {
            acc += i64::from(h) * i64::from(self.input[c][(self.in_pos + HIST - k) % HIST]);
        }
        sat((acc >> 15) as i32)
    }

    /// Interpolate the half-rate output back up. Only every other tap sees
    /// a sample (the rest of the zero-stuffed stream is zero), and the
    /// extra factor of two makes up for those zeros.
    fn interpolate(&self, c: usize, odd: bool) -> i32 {
        let mut acc = 0i64;
        let mut k = usize::from(odd);
        let mut j = 0;
        while k < FIR_TAPS {
            acc += i64::from(FIR[k]) * i64::from(self.half[c][(self.half_pos + HIST - j) % HIST]);
            k += 2;
            j += 1;
        }
        sat((acc >> 14) as i32)
    }

    fn step(&mut self, ram: &mut [u8], regs: &ReverbRegs, lin: i32, rin: i32) -> [i32; 2] {
        if regs.eea < regs.esa || regs.eea >= (ram.len() / 2) as u32 {
            return [0; 2];
        }
        let size = i64::from(regs.eea - regs.esa + 1);
        let pos = i64::from(self.pos);
        let at = |off: i64| -> usize { (regs.esa as usize + (pos + off).rem_euclid(size) as usize) * 2 };
        let rd = |ram: &[u8], off: i64| -> i32 {
            let a = at(off);
            i32::from(i16::from_le_bytes([ram[a], ram[a + 1]]))
        };
        let wr = |ram: &mut [u8], off: i64, v: i32| {
            let a = at(off);
            ram[a..a + 2].copy_from_slice(&(sat(v) as i16).to_le_bytes());
        };
        let a = |i: usize| i64::from(regs.addr[i]);
        let v = &regs.vol;

        let lin = mul(lin, v[V_LIN]);
        let rin = mul(rin, v[V_RIN]);

        // Same-side and cross-side reflections (one-pole IIR into the
        // buffer, fed by the wall-reflected delayed sample).
        let reflect = |ram: &mut [u8], input: i32, m: usize, d: usize| {
            let prev = rd(ram, a(m) - 1);
            let val = mul(input + mul(rd(ram, a(d)), v[V_WALL]) - prev, v[V_IIR]) + prev;
            wr(ram, a(m), val);
        };
        reflect(ram, lin, M_LSAME, D_LSAME);
        reflect(ram, rin, M_RSAME, D_RSAME);
        reflect(ram, lin, M_LDIFF, D_RDIFF);
        reflect(ram, rin, M_RDIFF, D_LDIFF);

        // Early echo: four comb taps.
        let comb = |ram: &[u8], m: [usize; 4]| -> i32 {
            sat(mul(rd(ram, a(m[0])), v[V_COMB1])
                + mul(rd(ram, a(m[1])), v[V_COMB1 + 1])
                + mul(rd(ram, a(m[2])), v[V_COMB1 + 2])
                + mul(rd(ram, a(m[3])), v[V_COMB1 + 3]))
        };
        let mut lout = comb(ram, [M_LCOMB1, M_LCOMB2, M_LCOMB3, M_LCOMB4]);
        let mut rout = comb(ram, [M_RCOMB1, M_RCOMB2, M_RCOMB3, M_RCOMB4]);

        // Late reverb: two all-pass stages.
        let apf = |ram: &mut [u8], out: i32, m: usize, d: usize, vol: i32| -> i32 {
            let delayed = rd(ram, a(m) - a(d));
            let fed = sat(out - mul(delayed, vol));
            wr(ram, a(m), fed);
            sat(mul(fed, vol) + delayed)
        };
        lout = apf(ram, lout, M_LAPF1, D_APF1, v[V_APF1]);
        rout = apf(ram, rout, M_RAPF1, D_APF1, v[V_APF1]);
        lout = apf(ram, lout, M_LAPF2, D_APF2, v[V_APF2]);
        rout = apf(ram, rout, M_RAPF2, D_APF2, v[V_APF2]);

        self.pos = ((pos + 1) % size) as u32;
        [lout, rout]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn impulse_decays_through_the_comb_and_allpass_taps() {
        // A small work area with unit-ish gains: an impulse must come out
        // delayed by the comb tap and then fade, never blow up.
        let mut ram = vec![0u8; 0x4000];
        let mut regs = ReverbRegs { esa: 0x100, eea: 0x4FF, ..Default::default() };
        regs.addr[M_LSAME] = 0x200;
        regs.addr[M_RSAME] = 0x210;
        regs.addr[D_LSAME] = 0x100;
        regs.addr[D_RSAME] = 0x110;
        regs.addr[M_LDIFF] = 0x220;
        regs.addr[M_RDIFF] = 0x230;
        regs.addr[D_LDIFF] = 0x120;
        regs.addr[D_RDIFF] = 0x130;
        regs.addr[M_LCOMB1] = 0x1F0;
        regs.addr[M_RCOMB1] = 0x1F8;
        regs.addr[M_LAPF1] = 0x300;
        regs.addr[M_RAPF1] = 0x320;
        regs.addr[D_APF1] = 0x10;
        regs.addr[M_LAPF2] = 0x340;
        regs.addr[M_RAPF2] = 0x360;
        regs.addr[D_APF2] = 0x8;
        regs.vol = [0x7000, 0x6000, 0, 0, 0, 0x5000, 0x4000, 0x4000, 0x7FFF, 0x7FFF];
        let mut rv = Reverb::default();
        let mut peak = 0;
        let mut first_nonzero = None;
        for i in 0..4000 {
            let (l, r) = if i == 0 { (0x4000, 0x4000) } else { (0, 0) };
            let out = rv.sample(&mut ram, &regs, l, r);
            if out[0] != 0 && first_nonzero.is_none() {
                first_nonzero = Some(i);
            }
            peak = peak.max(out[0].abs());
        }
        // mLSAME is written at offset 0x200 and read back by the comb at
        // 0x1F0 (16 halfwords earlier): 16 reverb steps = 32 samples, plus
        // the resampling filters' group delay (19 samples each way, and
        // their leading skirt reaches the output before the centre does).
        assert!(matches!(first_nonzero, Some(50..=60)), "{first_nonzero:?}");
        assert!(peak > 0x100 && peak < 0x8000, "{peak:#x}");
        let tail = rv.sample(&mut ram, &regs, 0, 0);
        assert!(tail[0].abs() < 0x100, "{tail:?}");
    }
}

/// The two filter histories, flattened: `serde`'s array impls stop at 32
/// elements and these rings are longer.
mod rings {
    use super::HIST;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(v: &[[i32; HIST]; 2], s: S) -> Result<S::Ok, S::Error> {
        v.concat().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[[i32; HIST]; 2], D::Error> {
        let flat = Vec::<i32>::deserialize(d)?;
        if flat.len() != HIST * 2 {
            return Err(serde::de::Error::custom("bad reverb history length"));
        }
        let mut out = [[0i32; HIST]; 2];
        for (c, chunk) in flat.chunks_exact(HIST).enumerate() {
            out[c].copy_from_slice(chunk);
        }
        Ok(out)
    }
}
