//! VU1 microprogram interpreter.
//!
//! Executes the micro memory filled by VIF1 MPG when MSCAL/MSCNT fires.
//! One instruction pair per step, upper (FMAC) then lower — the true
//! parallel read semantics are approximated by committing the upper result
//! before the lower executes, which the OSD's programs tolerate. XGKICK
//! streams GIF packets from VU1 data memory (PATH1). Unimplemented
//! opcodes panic loudly with pc and instruction, same as the CPU cores:
//! during bring-up a silent wrong result is worse than a stop.

use crate::gif::Gif;
use crate::gs::GsFront;
use tracing::{trace, warn};
use serde::{Deserialize, Serialize};

/// Safety net for a microprogram that never reaches its E bit: stop after
/// this many instruction pairs rather than hang the machine.
pub(crate) const PAIR_LIMIT: u32 = 1_000_000;

const MICRO_SIZE: usize = 16 * 1024;
const DATA_SIZE: usize = 16 * 1024;
/// VU0's memories are a quarter the size of VU1's.
const VU0_MEM_SIZE: usize = 4 * 1024;

#[derive(Serialize, Deserialize)]
pub struct Vu1 {
    /// Micro memory (code, filled by VIF MPG or EE stores).
    pub micro: Box<[u8]>,
    /// Data memory (filled by VIF UNPACK; XGKICK reads from here).
    pub data: Box<[u8]>,
    /// Float registers as raw bits; vf00 = (0,0,0,1).
    pub(crate) vf: [[u32; 4]; 32],
    /// 16-bit integer registers; vi00 = 0.
    pub(crate) vi: [u16; 16],
    pub(crate) acc: [u32; 4],
    /// Address masks. VU1 has 16 KB of each memory, VU0 a quarter of that:
    /// `micro_mask` is in bytes, `data_qw_mask` in quadwords, `pc_mask` in
    /// instruction pairs.
    pub(crate) micro_mask: usize,
    pub(crate) data_qw_mask: u32,
    pub(crate) pc_mask: u16,
    /// Opcodes already reported as unimplemented, to keep the log readable.
    #[serde(skip)]
    warned_ops: std::collections::HashSet<(u8, u8)>,
    pub(crate) q: f32,
    pub(crate) i: f32,
    pub(crate) r: u32,
    /// P register (EFU result, VU1 only). Stub: whatever was last set.
    pub(crate) p: f32,
    /// MAC/status as an FMAC just computed them. Readers see the copy the
    /// pipeline has aged for four cycles, not this one.
    pub(crate) mac: u16,
    pub(crate) status: u16,
    /// Four cycles of flag history: what the pipeline hands to FMAND and
    /// friends, oldest first from `flag_cycle`. Not saved — it is in
    /// flight for four pairs at most, and keeping it out of the state
    /// leaves old saves loadable.
    #[serde(skip)]
    pub(crate) flag_pipe: [[u16; 2]; 4],
    #[serde(skip)]
    pub(crate) flag_cycle: u32,
    /// The aged values the flag-reading instructions actually see.
    #[serde(skip)]
    pub(crate) mac_seen: u16,
    #[serde(skip)]
    pub(crate) status_seen: u16,
    pub(crate) clip: u32,
    /// TOP/ITOP as latched by VIF at MSCAL/MSCNT (XTOP/XITOP).
    pub top: u16,
    pub itop: u16,
    /// Resume address for MSCNT, in instruction pairs.
    pub(crate) next_pc: u16,
    /// Bumped whenever micro memory actually changes. The recompiler keys
    /// its whole cache on this: microcode is uploaded wholesale, so there
    /// is nothing to gain from finer-grained invalidation.
    #[serde(skip)]
    pub(crate) micro_gen: u32,
    /// Target a compiled JR/JALR left for the exit after its delay pair.
    #[serde(skip)]
    pub(crate) jit_target: u32,
    /// The integer branch hazard: a branch reads a VI register the lower
    /// instruction immediately before it wrote as it was *before* that
    /// write. Loads (ILW, ILWR) and the flag readers are exempt: a branch
    /// straight after those sees what they wrote. `vi_written_*` is what this pair's lower wrote (register and
    /// the value it replaced, register 0 for none); at the end of the pair
    /// it becomes `vi_hazard_*`, which the next pair's branch consults.
    /// The recompiler keeps the same fields by hand, so its fallbacks and
    /// the interpreter agree. In flight for one pair only, so not saved.
    #[serde(skip)]
    pub(crate) vi_written_reg: u8,
    #[serde(skip)]
    pub(crate) vi_written_val: u16,
    #[serde(skip)]
    pub(crate) vi_hazard_reg: u8,
    #[serde(skip)]
    pub(crate) vi_hazard_val: u16,
    /// Data memory's base address, so translated loads and stores reach it
    /// without unpacking a slice. Refreshed at every program entry, since
    /// deserializing moves the allocation.
    #[serde(skip)]
    pub(crate) data_ptr: usize,
    /// VU1's recompiler; `None` runs the interpreter. Translated code is
    /// not state, so a load starts from an empty cache.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    #[serde(skip)]
    pub(crate) jit: Option<Box<crate::vu1_jit::Vu1Jit>>,
}

impl Default for Vu1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Vu1 {
    pub fn new() -> Self {
        let mut vf = [[0; 4]; 32];
        vf[0] = [0, 0, 0, f32::to_bits(1.0)];
        Self {
            micro: vec![0u8; MICRO_SIZE].into_boxed_slice(),
            data: vec![0u8; DATA_SIZE].into_boxed_slice(),
            vf,
            vi: [0; 16],
            acc: [0; 4],
            micro_mask: MICRO_SIZE - 1,
            data_qw_mask: (DATA_SIZE / 16 - 1) as u32,
            pc_mask: (MICRO_SIZE / 8 - 1) as u16,
            warned_ops: std::collections::HashSet::new(),
            q: 0.0,
            i: 0.0,
            r: 0,
            p: 0.0,
            mac: 0,
            status: 0,
            flag_pipe: [[0; 2]; 4],
            flag_cycle: 0,
            mac_seen: 0,
            status_seen: 0,
            clip: 0,
            top: 0,
            itop: 0,
            next_pc: 0,
            micro_gen: 0,
            jit_target: 0,
            vi_written_reg: 0,
            vi_written_val: 0,
            vi_hazard_reg: 0,
            vi_hazard_val: 0,
            data_ptr: 0,
            #[cfg(all(feature = "jit", target_arch = "x86_64"))]
            jit: None,
        }
    }

    /// The same core wired up as VU0: a quarter of the memory, and the
    /// EE's COP2 macro mode drives it as well as its own microprograms.
    pub fn new_vu0() -> Self {
        Self {
            micro: vec![0u8; VU0_MEM_SIZE].into_boxed_slice(),
            data: vec![0u8; VU0_MEM_SIZE].into_boxed_slice(),
            micro_mask: VU0_MEM_SIZE - 1,
            data_qw_mask: (VU0_MEM_SIZE / 16 - 1) as u32,
            pc_mask: (VU0_MEM_SIZE / 8 - 1) as u16,
            ..Self::new()
        }
    }

    /// Write `bytes` into micro memory at byte offset `off`, masked into
    /// range the way the SRAM decode is; a write running off the end is
    /// clipped rather than wrapped, which no aligned caller can produce.
    ///
    /// Identical content is not a change. Games re-upload the same
    /// microprogram every frame, and counting that as a write would throw
    /// the recompiler's cache away each time.
    pub(crate) fn write_micro(&mut self, off: usize, bytes: &[u8]) {
        let a = off & self.micro_mask;
        let n = bytes.len().min(self.micro.len() - a);
        if self.micro[a..a + n] != bytes[..n] {
            self.micro[a..a + n].copy_from_slice(&bytes[..n]);
            self.micro_gen = self.micro_gen.wrapping_add(1);
        }
    }

    /// Turn the recompiler on or off. VU0 keeps the interpreter: it runs
    /// far less code, and COP2 macro mode never enters a microprogram.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    pub(crate) fn set_jit(&mut self, on: bool) -> Result<(), String> {
        if on && self.jit.is_none() {
            let jit = crate::vu1_jit::Vu1Jit::new()
                .map_err(|e| format!("cannot allocate VU1 JIT arena: {e}"))?;
            self.jit = Some(Box::new(jit));
        } else if !on {
            self.jit = None;
        }
        Ok(())
    }

    /// Drop everything the recompiler translated. Micro memory has been
    /// replaced wholesale from outside the normal write paths.
    #[cfg(all(feature = "jit", target_arch = "x86_64"))]
    pub(crate) fn flush_jit(&mut self) {
        let generation = self.micro_gen;
        if let Some(j) = &mut self.jit {
            j.flush_for(generation);
        }
    }

    /// MSCAL/MSCALF: run from `start` (in instruction pairs).
    pub fn start(&mut self, gs: &mut GsFront, gif: &mut Gif, start: u16) {
        self.run(gs, gif, start);
    }

    /// MSCNT: continue after the previously executed program.
    pub fn continue_run(&mut self, gs: &mut GsFront, gif: &mut Gif) {
        let pc = self.next_pc;
        self.run(gs, gif, pc);
    }

    /// One COP2 macro-mode instruction (this instance acting as VU0).
    /// Macro ops share the microcode field layout: special2 op2 >= 0x30
    /// selects the lower-pipeline set (DIV, MOVE, MTIR, LQI, ...), all
    /// other encodings are the upper FMAC set.
    pub fn exec_macro(&mut self, gs: &mut GsFront, gif: &mut Gif, instr: u32) {
        let op = instr & 0x3F;
        match op {
            // Integer ops (VIADD..VIOR) only exist in the lower pipeline.
            0x30..=0x35 => self.exec_lower_special(gs, gif, 0, instr),
            0x3C..=0x3F => {
                let op2 = (instr & 3) | ((instr >> 4) & 0x7C);
                if op2 >= 0x30 {
                    self.exec_lower_special(gs, gif, 0, instr);
                } else {
                    self.exec_upper(0, instr);
                }
            }
            _ => self.exec_upper(0, instr),
        }
    }

    // --- register helpers ------------------------------------------------

    #[inline]
    fn vf_read(&self, r: usize) -> [f32; 4] {
        let raw = self.vf[r];
        [
            f32::from_bits(raw[0]),
            f32::from_bits(raw[1]),
            f32::from_bits(raw[2]),
            f32::from_bits(raw[3]),
        ]
    }

    /// Round an FMAC result into the range a VU register can hold: the
    /// PS2 has no NaN or infinity, so an overflow saturates, and a
    /// denormal result is truncated to zero. Without this the host's
    /// infinities propagate into later arithmetic as NaNs, which the
    /// hardware never produces.
    fn vu_num(v: f32) -> f32 {
        let b = v.to_bits();
        match b & 0x7F80_0000 {
            0x7F80_0000 => f32::from_bits((b & 0x8000_0000) | 0x7F7F_FFFF),
            0 => f32::from_bits(b & 0x8000_0000),
            _ => v,
        }
    }

    /// Write masked fields and update MAC/status flags for them.
    fn vf_write(&mut self, r: usize, dest: u32, vals: [f32; 4]) {
        let vals = vals.map(Self::vu_num);
        if r != 0 {
            for f in 0..4 {
                if dest & (8 >> f) != 0 {
                    self.vf[r][f] = vals[f].to_bits();
                }
            }
        }
        self.update_flags(dest, vals);
    }

    /// MAX/MINI, ABS and the integer conversions write a float result
    /// without touching the flags.
    fn vf_write_noflag(&mut self, r: usize, dest: u32, vals: [f32; 4]) {
        self.vf_write_raw(r, dest, vals.map(f32::to_bits));
    }

    /// The ACC-writing FMAC ops set the flags the same way their
    /// register-writing counterparts do.
    fn acc_write(&mut self, dest: u32, vals: [f32; 4]) {
        let vals = vals.map(Self::vu_num);
        for f in 0..4 {
            if dest & (8 >> f) != 0 {
                self.acc[f] = vals[f].to_bits();
            }
        }
        self.update_flags(dest, vals);
    }

    /// MAC holds one nibble per flag with x in the high bit; status keeps
    /// the aggregates in bits 0..5 and their sticky copies above.
    fn update_flags(&mut self, dest: u32, vals: [f32; 4]) {
        let mut mac = 0u16;
        for f in 0..4 {
            if dest & (8 >> f) != 0 {
                let v = vals[f];
                if v == 0.0 {
                    mac |= 8 >> f; // zero flags, x at bit 3
                }
                if v.is_sign_negative() {
                    mac |= (8 >> f) << 4; // sign flags, x at bit 7
                }
            }
        }
        self.mac = mac;
        let mut st = self.status & !0x3;
        if mac & 0x0F != 0 {
            st |= 1; // Z
        }
        if mac & 0xF0 != 0 {
            st |= 2; // S
        }
        // Sticky copies.
        st |= (st & 0x3F) << 6;
        self.status = st;
    }

    fn vf_write_raw(&mut self, r: usize, dest: u32, vals: [u32; 4]) {
        if r == 0 {
            return;
        }
        for f in 0..4 {
            if dest & (8 >> f) != 0 {
                self.vf[r][f] = vals[f];
            }
        }
    }

    #[inline]
    fn vi_write(&mut self, r: usize, v: u16) {
        let r = r & 0xF;
        if r != 0 {
            self.vi_written_reg = r as u8;
            self.vi_written_val = self.vi[r];
            self.vi[r] = v;
        }
    }

    /// A VI write a branch straight after it does see: the loads (ILW,
    /// ILWR), which take long enough that the branch waits for the value,
    /// and the flag readers (FSAND, FMAND, FCAND ...), for which the BIOS's
    /// own `FMAND vi13 / IBNE vi13` shows the new value is what is read.
    fn vi_write_seen(&mut self, r: usize, v: u16) {
        let r = r & 0xF;
        if r != 0 {
            self.vi[r] = v;
        }
    }

    /// A VI register as a branch sees it: the value from before the
    /// previous lower instruction's write, if that is what it wrote.
    fn vi_for_branch(&self, r: usize) -> u16 {
        let r = r & 0xF;
        // Register 0 in the record means "nothing in flight", so vi00
        // itself must never match it.
        if r != 0 && usize::from(self.vi_hazard_reg) == r { self.vi_hazard_val } else { self.vi[r] }
    }

    fn data_qword(&mut self, qw: u32) -> [u32; 4] {
        self.note_vu1_register_window(qw);
        let a = ((qw & self.data_qw_mask) as usize) * 16;
        let w = |o: usize| u32::from_le_bytes(self.data[a + o..a + o + 4].try_into().unwrap());
        [w(0), w(4), w(8), w(12)]
    }

    fn set_data_qword(&mut self, qw: u32, dest: u32, vals: [u32; 4]) {
        self.note_vu1_register_window(qw);
        let a = ((qw & self.data_qw_mask) as usize) * 16;
        for f in 0..4 {
            if dest & (8 >> f) != 0 {
                self.data[a + f * 4..a + f * 4 + 4].copy_from_slice(&vals[f].to_le_bytes());
            }
        }
    }

    /// A data address past the end of a VU's own memory is not a fault:
    /// the decode keeps only as many address bits as the SRAM has — 14 for
    /// VU1's 16 KB, 12 for VU0's 4 KB — so it wraps, and microprograms do
    /// lean on that. `data_qw_mask` already reproduces it.
    ///
    /// VU0 is the one exception. Byte address bit 0x4000, quadword 0x400,
    /// selects VU1's register file instead of VU0's memory, the low bits
    /// picking a register. That needs the other unit in hand, which this
    /// one does not have, so it says so once rather than wrapping silently
    /// into VU0's own memory.
    fn note_vu1_register_window(&mut self, qw: u32) {
        let is_vu0 = self.data.len() == VU0_MEM_SIZE;
        if is_vu0 && qw & 0x400 != 0 && self.warned_ops.insert((0xFF, 0)) {
            warn!(
                target: "ps2_core::vu1",
                qw = format_args!("{qw:#x}"),
                "VU0 reached VU1's register file window, reading its own memory instead (reported once)"
            );
        }
    }

    /// An opcode with nothing behind it. Killing the machine over one loses
    /// everything running around it, so the slot behaves as a nop and the
    /// gap is named once per opcode.
    fn unimplemented(&mut self, kind: &str, pc: u16, instr: u32) {
        let op = instr & 0x3F;
        if self.warned_ops.insert((kind.as_bytes()[0], op as u8)) {
            warn!(
                target: "ps2_core::vu1",
                pc,
                instr = format_args!("{instr:#010x}"),
                op = format_args!("{op:#04x}"),
                "unimplemented VU1 {kind}, running it as a nop (reported once)"
            );

        }
    }

    // --- main loop -------------------------------------------------------

    fn run(&mut self, gs: &mut GsFront, gif: &mut Gif, start: u16) {
        let _p = crate::prof::scope(crate::prof::Slot::Vu1);
        // vf00/vi00 are architectural constants.
        self.vf[0] = [0, 0, 0, f32::to_bits(1.0)];
        self.vi[0] = 0;
        self.vi_written_reg = 0;
        self.vi_hazard_reg = 0;
        #[cfg(all(feature = "jit", target_arch = "x86_64"))]
        if self.jit.is_some() {
            crate::vu1_jit::run(self, gs, gif, start);
            return;
        }
        self.run_interp(gs, gif, start, PAIR_LIMIT);
    }

    /// The interpreter loop: at most `budget` instruction pairs from
    /// `start`. A program that never reaches its E bit stops at the budget
    /// with a warning rather than hanging the machine.
    pub(crate) fn run_interp(
        &mut self,
        gs: &mut GsFront,
        gif: &mut Gif,
        start: u16,
        budget: u32,
    ) {
        let mut pc = start & self.pc_mask;
        let mut end_after: i32 = -1; // pairs still to run after E bit
        let mut branch: Option<u16> = None;
        for _ in 0..budget {
            // Flags reach the readers four cycles after the FMAC that set
            // them, so a program may put unrelated FMACs in between.
            let slot = (self.flag_cycle & 3) as usize;
            [self.mac_seen, self.status_seen] = self.flag_pipe[slot];
            let a = (pc & self.pc_mask) as usize * 8;
            let lower = u32::from_le_bytes(self.micro[a..a + 4].try_into().unwrap());
            let upper = u32::from_le_bytes(self.micro[a + 4..a + 8].try_into().unwrap());
            let next = branch.take();
            if upper & (1 << 31) != 0 {
                // I bit: the lower slot holds a 32-bit float constant that
                // this very pair's upper instruction may use.
                self.i = f32::from_bits(lower);
            }
            self.exec_upper(pc, upper);
            if upper & (1 << 31) == 0 {
                self.exec_lower(gs, gif, pc, lower, &mut branch);
            }
            self.flag_pipe[slot] = [self.mac, self.status];
            self.flag_cycle = self.flag_cycle.wrapping_add(1);
            self.vi_hazard_reg = self.vi_written_reg;
            self.vi_hazard_val = self.vi_written_val;
            self.vi_written_reg = 0;
            pc = match next {
                Some(t) => t & self.pc_mask,
                None => (pc + 1) & self.pc_mask,
            };
            if end_after >= 0 {
                end_after -= 1;
                if end_after < 0 {
                    self.next_pc = pc;
                    return;
                }
            }
            if upper & (1 << 30) != 0 {
                end_after = 0; // E bit: one more pair (delay slot), then stop
            }
        }
        warn!(target: "ps2_core::vu1", start, "VU1 program hit its iteration limit");
        self.next_pc = pc;
    }

    // --- upper pipeline --------------------------------------------------

    pub(crate) fn exec_upper(&mut self, pc: u16, instr: u32) {
        let dest = (instr >> 21) & 0xF;
        let ft = ((instr >> 16) & 0x1F) as usize;
        let fs = ((instr >> 11) & 0x1F) as usize;
        let fd = ((instr >> 6) & 0x1F) as usize;
        let s = self.vf_read(fs);
        let t = self.vf_read(ft);
        let bc = t[(instr & 3) as usize];
        let acc = |me: &Self| -> [f32; 4] {
            [
                f32::from_bits(me.acc[0]),
                f32::from_bits(me.acc[1]),
                f32::from_bits(me.acc[2]),
                f32::from_bits(me.acc[3]),
            ]
        };
        let map =
            |f: &dyn Fn(usize) -> f32| -> [f32; 4] { [f(0), f(1), f(2), f(3)] };
        let op = instr & 0x3F;
        match op {
            0x00..=0x03 => self.vf_write(fd, dest, map(&|f| s[f] + bc)),
            0x04..=0x07 => self.vf_write(fd, dest, map(&|f| s[f] - bc)),
            0x08..=0x0B => {
                let a = acc(self);
                self.vf_write(fd, dest, map(&|f| a[f] + s[f] * bc));
            }
            0x0C..=0x0F => {
                let a = acc(self);
                self.vf_write(fd, dest, map(&|f| a[f] - s[f] * bc));
            }
            0x10..=0x13 => self.vf_write_noflag(fd, dest, map(&|f| s[f].max(bc))),
            0x14..=0x17 => self.vf_write_noflag(fd, dest, map(&|f| s[f].min(bc))),
            0x18..=0x1B => self.vf_write(fd, dest, map(&|f| s[f] * bc)),
            0x1C => {
                let q = self.q;
                self.vf_write(fd, dest, map(&|f| s[f] * q));
            }
            0x1D => {
                let i = self.i;
                self.vf_write_noflag(fd, dest, map(&|f| s[f].max(i)));
            }
            0x1E => {
                let i = self.i;
                self.vf_write(fd, dest, map(&|f| s[f] * i));
            }
            0x1F => {
                let i = self.i;
                self.vf_write_noflag(fd, dest, map(&|f| s[f].min(i)));
            }
            0x20 => {
                let q = self.q;
                self.vf_write(fd, dest, map(&|f| s[f] + q));
            }
            0x21 => {
                let (a, q) = (acc(self), self.q);
                self.vf_write(fd, dest, map(&|f| a[f] + s[f] * q));
            }
            0x22 => {
                let i = self.i;
                self.vf_write(fd, dest, map(&|f| s[f] + i));
            }
            0x23 => {
                let (a, i) = (acc(self), self.i);
                self.vf_write(fd, dest, map(&|f| a[f] + s[f] * i));
            }
            0x24 => {
                let q = self.q;
                self.vf_write(fd, dest, map(&|f| s[f] - q));
            }
            0x25 => {
                let (a, q) = (acc(self), self.q);
                self.vf_write(fd, dest, map(&|f| a[f] - s[f] * q));
            }
            0x26 => {
                let i = self.i;
                self.vf_write(fd, dest, map(&|f| s[f] - i));
            }
            0x27 => {
                let (a, i) = (acc(self), self.i);
                self.vf_write(fd, dest, map(&|f| a[f] - s[f] * i));
            }
            0x28 => self.vf_write(fd, dest, map(&|f| s[f] + t[f])),
            0x29 => {
                let a = acc(self);
                self.vf_write(fd, dest, map(&|f| a[f] + s[f] * t[f]));
            }
            0x2A => self.vf_write(fd, dest, map(&|f| s[f] * t[f])),
            0x2B => self.vf_write_noflag(fd, dest, map(&|f| s[f].max(t[f]))),
            0x2C => self.vf_write(fd, dest, map(&|f| s[f] - t[f])),
            0x2D => {
                let a = acc(self);
                self.vf_write(fd, dest, map(&|f| a[f] - s[f] * t[f]));
            }
            0x2E => {
                // OPMSUB: outer product stage 2 (xyz).
                let a = acc(self);
                let v = [
                    a[0] - s[1] * t[2],
                    a[1] - s[2] * t[0],
                    a[2] - s[0] * t[1],
                    0.0,
                ];
                self.vf_write(fd, dest & 0xE, v);
            }
            0x2F => self.vf_write_noflag(fd, dest, map(&|f| s[f].min(t[f]))),
            0x3C..=0x3F => {
                let op2 = (instr & 3) | ((instr >> 4) & 0x7C);
                self.exec_upper2(pc, instr, op2, dest, ft, fs, s, t, bc);
            }
            _ => self.unimplemented("upper", pc, instr),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_upper2(
        &mut self,
        pc: u16,
        instr: u32,
        op2: u32,
        dest: u32,
        ft: usize,
        fs: usize,
        s: [f32; 4],
        t: [f32; 4],
        bc: f32,
    ) {
        let map =
            |f: &dyn Fn(usize) -> f32| -> [f32; 4] { [f(0), f(1), f(2), f(3)] };
        let acc = |me: &Self| -> [f32; 4] {
            [
                f32::from_bits(me.acc[0]),
                f32::from_bits(me.acc[1]),
                f32::from_bits(me.acc[2]),
                f32::from_bits(me.acc[3]),
            ]
        };
        let set_acc = |me: &mut Self, dest: u32, vals: [f32; 4]| me.acc_write(dest, vals);
        match op2 {
            0x00..=0x03 => set_acc(self, dest, map(&|f| s[f] + bc)), // ADDAbc
            0x04..=0x07 => set_acc(self, dest, map(&|f| s[f] - bc)), // SUBAbc
            0x08..=0x0B => {
                let a = acc(self);
                set_acc(self, dest, map(&|f| a[f] + s[f] * bc)); // MADDAbc
            }
            0x0C..=0x0F => {
                let a = acc(self);
                set_acc(self, dest, map(&|f| a[f] - s[f] * bc)); // MSUBAbc
            }
            0x10 => {
                // ITOF0
                let raw = self.vf[fs];
                self.vf_write_raw(
                    ft,
                    dest,
                    raw.map(|v| (v as i32 as f32).to_bits()),
                );
            }
            0x11 => {
                let raw = self.vf[fs];
                self.vf_write_raw(
                    ft,
                    dest,
                    raw.map(|v| ((v as i32 as f32) / 16.0).to_bits()),
                );
            }
            0x12 => {
                let raw = self.vf[fs];
                self.vf_write_raw(
                    ft,
                    dest,
                    raw.map(|v| ((v as i32 as f32) / 4096.0).to_bits()),
                );
            }
            0x13 => {
                let raw = self.vf[fs];
                self.vf_write_raw(
                    ft,
                    dest,
                    raw.map(|v| ((v as i32 as f32) / 32768.0).to_bits()),
                );
            }
            0x14 => self.vf_write_raw(ft, dest, s.map(|v| clamp_i32(v) as u32)), // FTOI0
            0x15 => self.vf_write_raw(ft, dest, s.map(|v| clamp_i32(v * 16.0) as u32)),
            0x16 => self.vf_write_raw(ft, dest, s.map(|v| clamp_i32(v * 4096.0) as u32)),
            0x17 => self.vf_write_raw(ft, dest, s.map(|v| clamp_i32(v * 32768.0) as u32)),
            0x18..=0x1B => set_acc(self, dest, map(&|f| s[f] * bc)), // MULAbc
            0x1C => {
                let q = self.q;
                set_acc(self, dest, map(&|f| s[f] * q)); // MULAq
            }
            0x1D => {
                // ABS
                self.vf_write_noflag(ft, dest, s.map(|v| v.abs()));
            }
            0x1E => {
                let i = self.i;
                set_acc(self, dest, map(&|f| s[f] * i)); // MULAi
            }
            0x1F => {
                // CLIP: judge fs.xyz against |ft.w|, shift into the flag.
                let w = t[3].abs();
                let mut j = 0u32;
                j |= ((s[0] > w) as u32) << 0;
                j |= ((s[0] < -w) as u32) << 1;
                j |= ((s[1] > w) as u32) << 2;
                j |= ((s[1] < -w) as u32) << 3;
                j |= ((s[2] > w) as u32) << 4;
                j |= ((s[2] < -w) as u32) << 5;
                self.clip = ((self.clip << 6) | j) & 0xFF_FFFF;
            }
            0x20 => {
                let q = self.q;
                set_acc(self, dest, map(&|f| s[f] + q)); // ADDAq
            }
            0x21 => {
                let (a, q) = (acc(self), self.q);
                set_acc(self, dest, map(&|f| a[f] + s[f] * q)); // MADDAq
            }
            0x22 => {
                let i = self.i;
                set_acc(self, dest, map(&|f| s[f] + i)); // ADDAi
            }
            0x23 => {
                let (a, i) = (acc(self), self.i);
                set_acc(self, dest, map(&|f| a[f] + s[f] * i)); // MADDAi
            }
            0x24 => {
                let q = self.q;
                set_acc(self, dest, map(&|f| s[f] - q)); // SUBAq
            }
            0x25 => {
                let (a, q) = (acc(self), self.q);
                set_acc(self, dest, map(&|f| a[f] - s[f] * q)); // MSUBAq
            }
            0x26 => {
                let i = self.i;
                set_acc(self, dest, map(&|f| s[f] - i)); // SUBAi
            }
            0x27 => {
                let (a, i) = (acc(self), self.i);
                set_acc(self, dest, map(&|f| a[f] - s[f] * i)); // MSUBAi
            }
            0x28 => set_acc(self, dest, map(&|f| s[f] + t[f])), // ADDA
            0x29 => {
                let a = acc(self);
                set_acc(self, dest, map(&|f| a[f] + s[f] * t[f])); // MADDA
            }
            0x2A => set_acc(self, dest, map(&|f| s[f] * t[f])), // MULA
            0x2C => set_acc(self, dest, map(&|f| s[f] - t[f])), // SUBA
            0x2D => {
                let a = acc(self);
                set_acc(self, dest, map(&|f| a[f] - s[f] * t[f])); // MSUBA
            }
            0x2E => {
                // OPMULA: outer product stage 1 (xyz into ACC).
                let v = [s[1] * t[2], s[2] * t[0], s[0] * t[1], 0.0];
                set_acc(self, dest & 0xE, v);
            }
            0x2F => {} // NOP
            _ => self.unimplemented("upper", pc, instr),
        }
    }

    // --- lower pipeline --------------------------------------------------

    pub(crate) fn exec_lower(
        &mut self,
        gs: &mut GsFront,
        gif: &mut Gif,
        pc: u16,
        instr: u32,
        branch: &mut Option<u16>,
    ) {
        let opcode = instr >> 25;
        let dest = (instr >> 21) & 0xF;
        let it = ((instr >> 16) & 0x1F) as usize;
        let is = ((instr >> 11) & 0x1F) as usize;
        let imm11 = ((instr & 0x7FF) as i32) << 21 >> 21; // sign-extended
        let vi_s = self.vi[is & 0xF] as i16;
        let vi_t = self.vi[it & 0xF] as i16;
        let take = |branch: &mut Option<u16>, cond: bool| {
            if cond {
                *branch = Some((pc as i32 + 1 + imm11) as u16);
            }
        };
        match opcode {
            0x00 => {
                // LQ
                let qw = (vi_s as i32 + imm11) as u32;
                let v = self.data_qword(qw);
                self.vf_write_raw(it, dest, v);
            }
            0x01 => {
                // SQ
                let qw = (vi_t as i32 + imm11) as u32;
                let v = self.vf[is & 0x1F];
                self.set_data_qword(qw, dest, v);
            }
            0x04 => {
                // ILW: 16-bit int from the selected field's 32-bit slot.
                let qw = (vi_s as i32 + imm11) as u32;
                let v = self.data_qword(qw);
                let f = field_index(dest);
                self.vi_write_seen(it, v[f] as u16);
            }
            0x05 => {
                // ISW: like ILW, the address comes from `is` and the
                // integer register named by `it` is the value stored.
                let qw = (vi_s as i32 + imm11) as u32;
                let f = field_index(dest);
                let mut v = [0u32; 4];
                v[f] = self.vi[it & 0xF] as u32;
                self.set_data_qword(qw, 8 >> f, v);
            }
            0x08 => {
                // IADDIU
                let imm15 = (instr & 0x7FF) | ((instr >> 10) & 0x7800);
                self.vi_write(it, (vi_s as u16).wrapping_add(imm15 as u16));
            }
            0x09 => {
                let imm15 = (instr & 0x7FF) | ((instr >> 10) & 0x7800);
                self.vi_write(it, (vi_s as u16).wrapping_sub(imm15 as u16));
            }
            0x11 => self.clip = instr & 0xFF_FFFF, // FCSET
            0x12 => {
                // FCAND
                self.vi_write_seen(1, (self.clip & (instr & 0xFF_FFFF) != 0) as u16);
            }
            0x13 => {
                // FCOR
                let all = (self.clip | (instr & 0xFF_FFFF)) == 0xFF_FFFF;
                self.vi_write_seen(1, all as u16);
            }
            0x16 => {
                // FSAND
                // FSAND's 12-bit immediate keeps its top bit in bit 21.
                let imm12 = (instr & 0x7FF) | ((instr >> 10) & 0x800);
                self.vi_write_seen(it, self.status_seen & imm12 as u16);
            }
            0x1A => {
                // FMAND
                self.vi_write_seen(it, self.mac_seen & self.vi[is & 0xF]);
            }
            0x1C => {
                // FCGET
                self.vi_write_seen(it, (self.clip & 0xFFF) as u16);
            }
            0x20 => take(branch, true), // B
            0x21 => {
                // BAL: link points past the delay slot.
                self.vi_write(it, pc + 2);
                take(branch, true);
            }
            0x24 => *branch = Some(self.vi_for_branch(is)), // JR
            0x25 => {
                let target = self.vi_for_branch(is);
                self.vi_write(it, pc + 2);
                *branch = Some(target); // JALR
            }
            0x28 => take(branch, self.vi_for_branch(it) == self.vi_for_branch(is)),
            0x29 => take(branch, self.vi_for_branch(it) != self.vi_for_branch(is)),
            0x2C => take(branch, (self.vi_for_branch(is) as i16) < 0),
            0x2D => take(branch, (self.vi_for_branch(is) as i16) > 0),
            0x2E => take(branch, (self.vi_for_branch(is) as i16) <= 0),
            0x2F => take(branch, (self.vi_for_branch(is) as i16) >= 0),
            0x40 => self.exec_lower_special(gs, gif, pc, instr),
            _ => self.unimplemented("lower", pc, instr),
        }
    }

    fn exec_lower_special(&mut self, gs: &mut GsFront, gif: &mut Gif, pc: u16, instr: u32) {
        let dest = (instr >> 21) & 0xF;
        let it = ((instr >> 16) & 0x1F) as usize;
        let is = ((instr >> 11) & 0x1F) as usize;
        let id = ((instr >> 6) & 0x1F) as usize;
        let funct = instr & 0x3F;
        match funct {
            0x30 => {
                let v = self.vi[is & 0xF].wrapping_add(self.vi[it & 0xF]);
                self.vi_write(id, v); // IADD
            }
            0x31 => {
                let v = self.vi[is & 0xF].wrapping_sub(self.vi[it & 0xF]);
                self.vi_write(id, v); // ISUB
            }
            0x32 => {
                // IADDI: 5-bit signed immediate in the id slot.
                let imm5 = ((id as i32) << 27 >> 27) as i16;
                self.vi_write(it, (self.vi[is & 0xF] as i16).wrapping_add(imm5) as u16);
            }
            0x34 => {
                let v = self.vi[is & 0xF] & self.vi[it & 0xF];
                self.vi_write(id, v); // IAND
            }
            0x35 => {
                let v = self.vi[is & 0xF] | self.vi[it & 0xF];
                self.vi_write(id, v); // IOR
            }
            0x3C..=0x3F => {
                let id2 = (instr & 3) | ((instr >> 4) & 0x7C);
                self.exec_lower2(gs, gif, pc, instr, id2, dest, it, is);
            }
            _ => self.unimplemented("lower", pc, instr),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn exec_lower2(
        &mut self,
        gs: &mut GsFront,
        gif: &mut Gif,
        pc: u16,
        instr: u32,
        id2: u32,
        dest: u32,
        it: usize,
        is: usize,
    ) {
        let fsf = ((instr >> 21) & 3) as usize;
        let ftf = ((instr >> 23) & 3) as usize;
        match id2 {
            0x30 => {
                // MOVE
                let v = self.vf[is & 0x1F];
                self.vf_write_raw(it, dest, v);
            }
            0x31 => {
                // MR32: x<-y, y<-z, z<-w, w<-x.
                let v = self.vf[is & 0x1F];
                self.vf_write_raw(it, dest, [v[1], v[2], v[3], v[0]]);
            }
            0x34 => {
                // LQI
                let qw = self.vi[is & 0xF] as u32;
                let v = self.data_qword(qw);
                self.vf_write_raw(it, dest, v);
                self.vi_write(is, self.vi[is & 0xF].wrapping_add(1));
            }
            0x35 => {
                // SQI
                let qw = self.vi[it & 0xF] as u32;
                let v = self.vf[is & 0x1F];
                self.set_data_qword(qw, dest, v);
                self.vi_write(it, self.vi[it & 0xF].wrapping_add(1));
            }
            0x36 => {
                // LQD
                let qw = self.vi[is & 0xF].wrapping_sub(1);
                self.vi_write(is, qw);
                let v = self.data_qword(qw as u32);
                self.vf_write_raw(it, dest, v);
            }
            0x37 => {
                // SQD
                let qw = self.vi[it & 0xF].wrapping_sub(1);
                self.vi_write(it, qw);
                let v = self.vf[is & 0x1F];
                self.set_data_qword(qw as u32, dest, v);
            }
            0x38 => {
                // DIV: division by zero saturates instead of producing inf.
                let n = f32::from_bits(self.vf[is & 0x1F][fsf]);
                let d = f32::from_bits(self.vf[it & 0x1F][ftf]);
                self.q = Self::vu_num(if d == 0.0 {
                    if n.is_sign_negative() != d.is_sign_negative() {
                        -f32::MAX
                    } else {
                        f32::MAX
                    }
                } else {
                    n / d
                });
            }
            0x39 => {
                // SQRT
                let d = f32::from_bits(self.vf[it & 0x1F][ftf]);
                self.q = Self::vu_num(d.abs().sqrt());
            }
            0x3A => {
                // RSQRT
                let n = f32::from_bits(self.vf[is & 0x1F][fsf]);
                let d = f32::from_bits(self.vf[it & 0x1F][ftf]);
                let r = d.abs().sqrt();
                self.q = Self::vu_num(if r == 0.0 { f32::MAX.copysign(n) } else { n / r });
            }
            0x3B => {} // WAITQ: Q has no latency here
            0x3C => {
                // MTIR
                self.vi_write(it, self.vf[is & 0x1F][fsf] as u16);
            }
            0x3D => {
                // MFIR: sign-extended 16-bit int into fields.
                let v = self.vi[is & 0xF] as i16 as i32 as u32;
                self.vf_write_raw(it, dest, [v; 4]);
            }
            0x3E => {
                // ILWR
                let v = self.data_qword(self.vi[is & 0xF] as u32);
                let f = field_index(dest);
                self.vi_write_seen(it, v[f] as u16);
            }
            0x3F => {
                // ISWR
                let f = field_index(dest);
                let mut v = [0u32; 4];
                v[f] = self.vi[it & 0xF] as u32;
                self.set_data_qword(self.vi[is & 0xF] as u32, 8 >> f, v);
            }
            0x40 => {
                // RNEXT: advance the LFSR, then read.
                let x = (self.r >> 4) & 1;
                let y = (self.r >> 22) & 1;
                self.r = (((self.r << 1) ^ x ^ y) & 0x7F_FFFF) | 0x3F80_0000;
                self.vf_write_raw(it, dest, [self.r; 4]);
            }
            0x41 => self.vf_write_raw(it, dest, [self.r; 4]), // RGET
            0x42 => {
                // RINIT
                self.r = 0x3F80_0000 | (self.vf[is & 0x1F][fsf] & 0x7F_FFFF);
            }
            0x43 => {
                // RXOR
                self.r = 0x3F80_0000 | ((self.r ^ self.vf[is & 0x1F][fsf]) & 0x7F_FFFF);
            }
            0x64 => {
                // MFP
                let p = self.p.to_bits();
                self.vf_write_raw(it, dest, [p; 4]);
            }
            0x68 => self.vi_write(it, self.top),  // XTOP
            0x69 => self.vi_write(it, self.itop), // XITOP
            0x6C => self.xgkick(gs, gif, self.vi[is & 0xF]),
            0x70..=0x7A | 0x7C..=0x7E => {
                let v = self.vf_read(is);
                self.p = Self::vu_num(Self::efu(id2, v, v[fsf]));
            }
            0x7B => {} // WAITP: P has no latency here
            _ => self.unimplemented("lower", pc, instr),
        }
    }

    /// The elementary function unit, VU1 only: one input, one result in P.
    /// `v` is VF[fs] and `f` its fsf field, already selected.
    ///
    /// The three transcendentals evaluate the polynomials the VU User's
    /// Manual publishes for them rather than the host's `sin`/`exp`/`atan`,
    /// so a microprogram sees the coefficients it was tuned against. Each
    /// is only valid over the range the manual gives; outside it the
    /// hardware returns the polynomial's answer too, so nothing is
    /// clamped here.
    fn efu(op: u32, v: [f32; 4], f: f32) -> f32 {
        // sin(x) over -pi/2..pi/2.
        const S: [u32; 5] = [0x3F80_0000, 0xBE2A_AAA4, 0x3C08_873E, 0xB94F_B21F, 0x362E_9C14];
        // exp(-x) over 0..MAX, as the reciprocal of a sextic raised to 4.
        const E: [u32; 6] =
            [0x3E7F_FFA8, 0x3D00_07F4, 0x3B29_D3FF, 0x3933_E553, 0x36B6_3510, 0x3539_61AC];
        // arctan over 0..1, in t = (x-1)/(x+1), plus pi/4.
        const T: [u32; 8] = [
            0x3F7F_FFF5, 0xBEAA_A61C, 0x3E4C_40A6, 0xBE0E_6C63, 0x3DC5_77DF, 0xBD65_01C4,
            0x3CB3_1652, 0xBB84_D7E7,
        ];
        let poly = |c: &[u32], t: f32, step: u32| {
            let (mut acc, mut x) = (0.0f32, t);
            for &k in c {
                acc += f32::from_bits(k) * x;
                for _ in 0..step {
                    x *= t;
                }
            }
            acc
        };
        // Reciprocals saturate rather than produce an infinity, the way
        // DIV does: VU floats have no representation for one.
        let recip = |d: f32| if d == 0.0 { f32::MAX } else { 1.0 / d };
        let sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let atan = |t: f32| poly(&T, t, 2) + std::f32::consts::FRAC_PI_4;
        match op {
            0x70 => sq,                                    // ESADD
            0x71 => recip(sq),                             // ERSADD
            0x72 => sq.sqrt(),                             // ELENG
            0x73 => recip(sq.sqrt()),                      // ERLENG
            0x74 => atan((v[1] - v[0]) / (v[1] + v[0])),   // EATANxy
            0x75 => atan((v[2] - v[0]) / (v[2] + v[0])),   // EATANxz
            0x76 => v[0] + v[1] + v[2] + v[3],             // ESUM
            0x78 => f.abs().sqrt(),                        // ESQRT
            0x79 => recip(f.abs().sqrt()),                 // ERSQRT
            0x7A => recip(f),                              // ERCPR
            0x7C => poly(&S, f, 2),                        // ESIN
            0x7D => atan((f - 1.0) / (f + 1.0)),           // EATAN
            // EEXP: the manual's formula is 1 / (1 + E1 x + ... )^4.
            _ => recip((1.0 + poly(&E, f, 1)).powi(4)),
        }
    }

    /// XGKICK: stream GIF packets from data memory (PATH1) until a tag
    /// with EOP finishes.
    fn xgkick(&mut self, gs: &mut GsFront, gif: &mut Gif, start: u16) {
        if !gif.idle() {
            let st = gif.debug_state();
            warn!(target: "ps2_core::vu1", start, state = format_args!("{st:?}"), "XGKICK with GIF mid-packet");
        }
        let mut qw = start as u32;
        for _ in 0..1024 {
            let v = self.data_qword(qw);
            qw += 1;
            let lo = v[0] as u64 | ((v[1] as u64) << 32);
            let hi = v[2] as u64 | ((v[3] as u64) << 32);
            gif.process(gs, lo, hi);
            if gif.end_of_packet() {
                return;
            }
        }
        // No EOP within data memory: the packet's continuation isn't
        // written yet (the OSD kicks split packets). Abandon it so the
        // next kick starts from a clean between-packets state.
        trace!(target: "ps2_core::vu1", start, "XGKICK reached the scan limit without EOP; resetting");
        gif.reset_path();
    }
}

/// FTOI with the hardware's saturating conversion.
fn clamp_i32(v: f32) -> i32 {
    if v >= 2147483647.0 {
        i32::MAX
    } else if v <= -2147483648.0 {
        i32::MIN
    } else if v.is_nan() {
        i32::MAX
    } else {
        v as i32
    }
}

/// Map a single-field dest mask to its field index (x=0..w=3).
fn field_index(dest: u32) -> usize {
    match dest {
        8 => 0,
        4 => 1,
        2 => 2,
        _ => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(upper: u32, lower: u32) -> [u8; 8] {
        let mut out = [0u8; 8];
        out[..4].copy_from_slice(&lower.to_le_bytes());
        out[4..].copy_from_slice(&upper.to_le_bytes());
        out
    }

    #[test]
    fn vu0_is_a_quarter_of_vu1() {
        let vu1 = Vu1::new();
        let vu0 = Vu1::new_vu0();
        assert_eq!(vu1.micro.len(), 4 * vu0.micro.len());
        assert_eq!(vu1.data.len(), 4 * vu0.data.len());
        // 4 KB is 512 instruction pairs and 256 quadwords.
        assert_eq!(vu0.pc_mask, 0x1FF);
        assert_eq!(vu0.data_qw_mask, 0xFF);
        assert_eq!(vu0.micro_mask, 0xFFF);
        assert_eq!(vu1.pc_mask, 0x7FF);
        assert_eq!(vu1.data_qw_mask, 0x3FF);
    }

    #[test]
    fn vu0_data_wraps_inside_its_own_memory() {
        let mut vu = Vu1::new_vu0();
        vu.set_data_qword(0, 0xF, [1, 2, 3, 4]);
        // Quadword 0x100 is the first past VU0's memory.
        assert_eq!(vu.data_qword(0x100), [1, 2, 3, 4]);
        // ...and VU1, with four times the memory, does not wrap there.
        let mut vu1 = Vu1::new();
        vu1.set_data_qword(0, 0xF, [1, 2, 3, 4]);
        assert_eq!(vu1.data_qword(0x100), [0, 0, 0, 0]);
    }

    /// VU1 keeps 14 address bits, so quadword 0x5EE — which Ace Combat 5's
    /// microprograms reach — is 0x1EE of its own memory, not a fault and
    /// not VU0's register window.
    #[test]
    fn vu1_data_wraps_inside_its_sixteen_kilobytes() {
        let mut vu = Vu1::new();
        vu.set_data_qword(0x1EE, 0xF, [5, 6, 7, 8]);
        assert_eq!(vu.data_qword(0x5EE), [5, 6, 7, 8]);
        // 0x400 is VU0's register window; for VU1 it is just memory.
        vu.set_data_qword(0x400, 0xF, [9, 10, 11, 12]);
        assert_eq!(vu.data_qword(0x400), [9, 10, 11, 12]);
    }

    /// The EFU's algebraic results are exact, and its three polynomials
    /// track the functions they approximate across the ranges the VU
    /// User's Manual declares them valid over.
    #[test]
    fn efu_matches_the_functions_it_approximates() {
        let v = [3.0f32, 4.0, 12.0, 1.0];
        assert_eq!(Vu1::efu(0x70, v, 0.0), 169.0); // ESADD
        assert_eq!(Vu1::efu(0x72, v, 0.0), 13.0); // ELENG
        assert!((Vu1::efu(0x73, v, 0.0) - 1.0 / 13.0).abs() < 1e-6); // ERLENG
        assert_eq!(Vu1::efu(0x76, v, 0.0), 20.0); // ESUM
        assert_eq!(Vu1::efu(0x78, v, 9.0), 3.0); // ESQRT
        assert_eq!(Vu1::efu(0x7A, v, 4.0), 0.25); // ERCPR
        // A reciprocal of zero saturates instead of reaching infinity.
        assert_eq!(Vu1::efu(0x7A, v, 0.0), f32::MAX);

        for i in 0..=20 {
            let x = i as f32 / 20.0;
            // ESIN over -pi/2..pi/2, EATAN over 0..1, EEXP over 0..MAX.
            let a = x * std::f32::consts::FRAC_PI_2;
            assert!((Vu1::efu(0x7C, v, a) - a.sin()).abs() < 1e-4, "sin {a}");
            assert!((Vu1::efu(0x7C, v, -a) + a.sin()).abs() < 1e-4, "sin {}", -a);
            assert!((Vu1::efu(0x7D, v, x) - x.atan()).abs() < 1e-4, "atan {x}");
            let e = x * 8.0;
            assert!((Vu1::efu(0x7E, v, e) - (-e).exp()).abs() < 1e-4, "exp {e}");
        }
        // EATANxy takes the ratio of two fields, over 0 <= y <= x.
        let q = [2.0f32, 1.0, 0.5, 0.0];
        assert!((Vu1::efu(0x74, q, 0.0) - 0.5f32.atan()).abs() < 1e-4);
        assert!((Vu1::efu(0x75, q, 0.0) - 0.25f32.atan()).abs() < 1e-4);
    }

    /// ELENG as Ace Combat 5 encodes it, then MFP to read P back. Pins
    /// the LowerOP field-type-3 decode as much as the arithmetic.
    #[test]
    fn eleng_then_mfp_moves_a_length_into_a_register() {
        let mut vu = Vu1::new();
        vu.vf[25] = [f32::to_bits(3.0), f32::to_bits(4.0), f32::to_bits(12.0), 0];
        // 0x81c0cf3e: ELENG P, VF25 — dest xyz, bits 10:6 = 0x1c, funct 0x3e.
        let eleng = 0x8000_0000 | (0xE << 21) | (25 << 11) | (0x1C << 6) | 0x3E;
        assert_eq!(eleng, 0x81c0_cf3e);
        let mfp = 0x8000_0000 | (0xF << 21) | (2 << 16) | (0x19 << 6) | 0x3C;
        run_prog(&mut vu, &[(0, eleng), (1 << 30, mfp), (0, 0)]);
        assert_eq!(vu.vf[2], [f32::to_bits(13.0); 4]);
    }

    /// ISW names the stored register in `it` and the address base in
    /// `is`, the same way ILW and ISWR do. Swapping them silently
    /// scribbles over the wrong quadword and never saves the value the
    /// program reloads later.
    #[test]
    fn isw_stores_it_at_the_is_address() {
        let mut vu = Vu1::new();
        vu.vi[3] = 40; // address base
        vu.vi[5] = 0x1234; // value
        // ISW.y vi05, 2(vi03): opcode 0x05, dest y, it = 5, is = 3.
        let isw = (0x05 << 25) | (4 << 21) | (5 << 16) | (3 << 11) | 2;
        run_prog(&mut vu, &[(1 << 30, isw), (0, 0x8000_033C)]);
        let qw = &vu.data[42 * 16..42 * 16 + 16];
        assert_eq!(&qw[4..8], &0x1234u32.to_le_bytes());
        assert_eq!(&qw[0..4], &[0; 4]); // only the named field is written

        // ILW.y vi06, 2(vi03) reads the same slot back.
        let ilw = (0x04 << 25) | (4 << 21) | (6 << 16) | (3 << 11) | 2;
        run_prog(&mut vu, &[(1 << 30, ilw), (0, 0x8000_033C)]);
        assert_eq!(vu.vi[6], 0x1234);
    }

    fn run_prog(vu: &mut Vu1, pairs: &[(u32, u32)]) {
        for (i, &(u, l)) in pairs.iter().enumerate() {
            vu.micro[i * 8..i * 8 + 8].copy_from_slice(&pair(u, l));
        }
        let (mut gs, mut gif) = (GsFront::inline(), Gif::new());
        vu.start(&mut gs, &mut gif, 0);
    }

    #[test]
    fn add_and_e_bit_stop() {
        let mut vu = Vu1::new();
        vu.vf[1] = [f32::to_bits(1.5); 4];
        vu.vf[2] = [f32::to_bits(2.0); 4];
        // ADD.xyzw vf03, vf01, vf02 with E bit; delay-slot NOP pair.
        let add = (0xF << 21) | (2 << 16) | (1 << 11) | (3 << 6) | 0x28;
        run_prog(&mut vu, &[(add | (1 << 30), 0x8000_033C), (0, 0x8000_033C)]);
        assert_eq!(f32::from_bits(vu.vf[3][0]), 3.5);
    }

    #[test]
    fn i_bit_loads_float_constant() {
        let mut vu = Vu1::new();
        vu.vf[1] = [f32::to_bits(2.0); 4];
        // MULi.xyzw vf03, vf01, I with I=0.5 (I bit), then E-bit pair.
        let muli = (1 << 31) | (0xF << 21) | (1 << 11) | (3 << 6) | 0x1E;
        run_prog(
            &mut vu,
            &[
                (muli, f32::to_bits(0.5)),
                (1 << 30, 0x8000_033C),
                (0, 0x8000_033C),
            ],
        );
        assert_eq!(f32::from_bits(vu.vf[3][2]), 1.0);
    }

    #[test]
    fn lq_sq_round_trip() {
        let mut vu = Vu1::new();
        vu.data[16..20].copy_from_slice(&f32::to_bits(7.0).to_le_bytes());
        // LQ.xyzw vf04, 1(vi00); SQ.xyzw vf04, 2(vi00); E.
        let lq = (0xF << 21) | (4 << 16) | 1;
        let sq = (0x1 << 25) | (0xF << 21) | (4 << 11) | 2;
        run_prog(
            &mut vu,
            &[
                (0, lq),
                (1 << 30, sq),
                (0, 0x8000_033C),
            ],
        );
        assert_eq!(&vu.data[32..36], &f32::to_bits(7.0).to_le_bytes());
    }

    #[test]
    fn xgkick_stops_at_eop() {
        let mut vu = Vu1::new();
        let put = |vu: &mut Vu1, qw: usize, v: [u32; 4]| {
            for (i, w) in v.iter().enumerate() {
                vu.data[qw * 16 + i * 4..qw * 16 + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        };
        // The OSD's runaway shape: A+D tag (EOP=0), then a PACKED
        // ST/RGBAQ/XYZF2 tag with EOP=1.
        put(&mut vu, 18, [0x0000_0002, 0x1000_0000, 0x0000_000E, 0]);
        put(&mut vu, 19, [0, 0, 0x47, 0]);
        put(&mut vu, 20, [0, 0, 0x42, 0]);
        put(&mut vu, 21, [0x0000_8004, 0x302E_6000, 0x0000_0412, 0]);
        // 4 loops x 3 regs = 12 data qwords at 22..34 (zeros are fine).
        let (mut gs, mut gif) = (GsFront::inline(), Gif::new());
        vu.xgkick(&mut gs, &mut gif, 18);
        assert!(gif.end_of_packet());
    }

    #[test]
    fn xtop_and_iaddiu() {
        let mut vu = Vu1::new();
        vu.top = 0x200;
        // XTOP vi01; IADDIU vi02, vi01, 4; E.
        // id2 = (instr & 3) | ((instr >> 4) & 0x7C) must equal 0x68:
        // funct = 0x3C, bits 6-10 = 0x68 >> 2.
        let xtop = (0x40 << 25) | (1 << 16) | ((0x68 >> 2) << 6) | 0x3C;
        let iaddiu = (0x08 << 25) | (2 << 16) | (1 << 11) | 4;
        run_prog(&mut vu, &[(0, xtop), (1 << 30, iaddiu), (0, 0x8000_033C)]);
        assert_eq!(vu.vi[2], 0x204);
    }
}
