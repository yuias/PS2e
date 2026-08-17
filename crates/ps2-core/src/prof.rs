//! Coarse subsystem time accounting for the `profile` feature.
//!
//! Sampling profilers need elevation on Windows, so the bring-up loop uses
//! cheap TSC scopes instead: `let _g = prof::scope(Slot::X);` charges the
//! time until the guard drops. Scopes nest, and time is always charged to
//! the innermost open scope, so every bucket is *self* time. Without the
//! feature the guard is a ZST and everything compiles to nothing.

#[cfg(feature = "profile")]
use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Accounting buckets. Order matters only for the report.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Slot {
    /// Outside every scope (front-end, screenshots, ...).
    Other,
    /// EE interpreter (fetch/decode/execute, non-DMA bus traffic).
    Ee,
    /// IOP interpreter and IOP-side DMA.
    Iop,
    /// Periodic tick: timers, deferred DMA IRQs.
    Timers,
    /// EE ch1 DMA: VIF1 parsing/unpack.
    Vif1,
    /// EE ch2 DMA: GIF packet decode, GS register writes.
    Gif,
    /// SIF0/SIF1 pumps.
    Sif,
    /// VU1 microprogram execution.
    Vu1,
    /// GS primitive rasterization.
    GsDraw,
    /// GS IMAGE / local-copy transfers.
    GsXfer,
    /// SPU2 voice mixing.
    Spu2,
}

#[cfg(feature = "profile")]
const N: usize = 11;
#[cfg(feature = "profile")]
const NAMES: [&str; N] = [
    "other", "EE", "IOP", "timers", "VIF1", "GIF", "SIF", "VU1", "GS draw", "GS xfer", "SPU2",
];

#[cfg(feature = "profile")]
static TICKS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
#[cfg(feature = "profile")]
static COUNTS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
#[cfg(feature = "profile")]
static CURRENT: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "profile")]
static LAST: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "profile")]
#[inline(always)]
fn tsc() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// Charge the time since the last switch to the current slot and make
/// `next` current.
#[cfg(feature = "profile")]
#[inline(always)]
fn switch(next: usize) -> usize {
    let now = tsc();
    let last = LAST.swap(now, Ordering::Relaxed);
    let cur = CURRENT.swap(next, Ordering::Relaxed);
    if last != 0 {
        TICKS[cur].fetch_add(now.wrapping_sub(last), Ordering::Relaxed);
    }
    cur
}

/// Open scope; drops back to the enclosing slot.
#[must_use]
pub struct Guard {
    #[cfg(feature = "profile")]
    prev: usize,
}

impl Drop for Guard {
    #[inline(always)]
    fn drop(&mut self) {
        #[cfg(feature = "profile")]
        switch(self.prev);
    }
}

/// Enter `slot` until the returned guard drops.
#[inline(always)]
pub fn scope(slot: Slot) -> Guard {
    #[cfg(feature = "profile")]
    {
        COUNTS[slot as usize].fetch_add(1, Ordering::Relaxed);
        Guard { prev: switch(slot as usize) }
    }
    #[cfg(not(feature = "profile"))]
    {
        let _ = slot;
        Guard {}
    }
}

/// EE instruction mix (major opcode, plus function for SPECIAL/MMI) and a
/// per-word PC histogram over the 32 MiB of RAM to find hot loops.
#[cfg(feature = "profile")]
static OPS: [AtomicU64; 4096] = [const { AtomicU64::new(0) }; 4096];
#[cfg(feature = "profile")]
static PCS: std::sync::OnceLock<Vec<AtomicU64>> = std::sync::OnceLock::new();
#[cfg(feature = "profile")]
fn pcs() -> &'static [AtomicU64] {
    PCS.get_or_init(|| (0..(32 << 20) / 4).map(|_| AtomicU64::new(0)).collect())
}

#[cfg(feature = "profile")]
static IOP_PCS: std::sync::OnceLock<Vec<AtomicU64>> = std::sync::OnceLock::new();
#[cfg(feature = "profile")]
fn iop_pcs() -> &'static [AtomicU64] {
    IOP_PCS.get_or_init(|| (0..(2 << 20) / 4).map(|_| AtomicU64::new(0)).collect())
}

/// Record one IOP instruction (RAM addresses only) for the profile report.
#[inline(always)]
pub fn count_iop(pc: u32) {
    #[cfg(feature = "profile")]
    {
        if pc & 0x1FE0_0000 == 0 {
            iop_pcs()[((pc & 0x1F_FFFF) >> 2) as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
    #[cfg(not(feature = "profile"))]
    {
        let _ = pc;
    }
}

/// Record one EE instruction for the profile report.
#[inline(always)]
pub fn count_ee(pc: u32, instr: u32) {
    #[cfg(feature = "profile")]
    {
        let op = instr >> 26;
        let key = if op == 0 || op == 0x1C { (op << 6) | (instr & 0x3F) } else { op << 6 };
        OPS[key as usize].fetch_add(1, Ordering::Relaxed);
        if pc & 0x1E00_0000 == 0 {
            pcs()[((pc & 0x1FF_FFFF) >> 2) as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
    #[cfg(not(feature = "profile"))]
    {
        let _ = (pc, instr);
    }
}

/// Human-readable breakdown, or `None` when profiling is compiled out.
pub fn report() -> Option<String> {
    #[cfg(not(feature = "profile"))]
    {
        None
    }
    #[cfg(feature = "profile")]
    {
        switch(CURRENT.load(Ordering::Relaxed)); // flush the open scope
        let t: Vec<u64> = TICKS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        let c: Vec<u64> = COUNTS.iter().map(|a| a.load(Ordering::Relaxed)).collect();
        let total: u64 = t.iter().sum();
        if total == 0 {
            return Some("profile: no samples".into());
        }
        let mut out = String::from("profile (self time):\n");
        for i in 0..N {
            out.push_str(&format!(
                "  {:<8} {:5.1}%  {:>13} scopes  {:>9.1} ticks/scope\n",
                NAMES[i],
                t[i] as f64 * 100.0 / total as f64,
                c[i],
                t[i] as f64 / c[i].max(1) as f64,
            ));
        }
        let mut ops: Vec<(usize, u64)> =
            OPS.iter().enumerate().map(|(k, a)| (k, a.load(Ordering::Relaxed))).filter(|&(_, n)| n > 0).collect();
        let total_ops: u64 = ops.iter().map(|&(_, n)| n).sum::<u64>().max(1);
        ops.sort_by(|a, b| b.1.cmp(&a.1));
        out.push_str(&format!(
            "EE instructions: {} ({:.1} ticks each of EE self time)
",
            total_ops,
            t[Slot::Ee as usize] as f64 / total_ops as f64
        ));
        out.push_str("EE instruction mix (op<<6|funct for SPECIAL/MMI):
");
        for (k, n) in ops.iter().take(24) {
            out.push_str(&format!("  {:#05x} {:5.1}%
", k, *n as f64 * 100.0 / total_ops as f64));
        }
        let mut pcs: Vec<(usize, u64)> =
            pcs().iter().enumerate().map(|(k, a)| (k, a.load(Ordering::Relaxed))).filter(|&(_, n)| n > 0).collect();
        pcs.sort_by(|a, b| b.1.cmp(&a.1));
        out.push_str("EE hot words (physical RAM address):
");
        for (k, n) in pcs.iter().take(40) {
            out.push_str(&format!("  {:#09x} {:5.1}%
", k << 2, *n as f64 * 100.0 / total_ops as f64));
        }
        let mut ipcs: Vec<(usize, u64)> =
            iop_pcs().iter().enumerate().map(|(k, a)| (k, a.load(Ordering::Relaxed))).filter(|&(_, n)| n > 0).collect();
        let iop_total: u64 = ipcs.iter().map(|&(_, n)| n).sum::<u64>().max(1);
        ipcs.sort_by(|a, b| b.1.cmp(&a.1));
        out.push_str(&format!("IOP instructions in RAM: {iop_total}; hot words:
"));
        for (k, n) in ipcs.iter().take(24) {
            out.push_str(&format!("  {:#09x} {:5.1}%
", k << 2, *n as f64 * 100.0 / iop_total as f64));
        }
        Some(out)
    }
}
