//! SPU2 reverb: the PS1 algorithm (nocash "SPU Reverb Formula") run at
//! half the sample rate over a work area in sound RAM between ESA and EEA.
//! Offsets are halfword addresses relative to ESA and rotate with the
//! buffer position, wrapping inside the area.

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

#[derive(Clone, Copy, Default)]
pub struct Reverb {
    /// Rotating offset into the work area.
    pos: u32,
    /// The algorithm runs every other sample; the even sample's input is
    /// held here and averaged with the odd one.
    phase: bool,
    held: [i32; 2],
    /// Last computed output, repeated for the in-between sample.
    out: [i32; 2],
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
        if self.phase {
            let l = (self.held[0] + lin) >> 1;
            let r = (self.held[1] + rin) >> 1;
            self.out = self.step(ram, regs, l, r);
        } else {
            self.held = [lin, rin];
        }
        self.phase = !self.phase;
        self.out
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
        // 0x1F0 (16 halfwords earlier): 16 reverb steps = 32 samples,
        // plus the 2-sample decimation.
        assert!(matches!(first_nonzero, Some(30..=36)), "{first_nonzero:?}");
        assert!(peak > 0x100 && peak < 0x8000, "{peak:#x}");
        let tail = rv.sample(&mut ram, &regs, 0, 0);
        assert!(tail[0].abs() < 0x100, "{tail:?}");
    }
}
