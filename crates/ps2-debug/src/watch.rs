//! Write watching shared by the gdb stub and the headless `--watch` log:
//! which RAM offset a target address names, and which DMA engine wrote it.

use crate::{Target, Watchpoint, canonical, peek};
use ps2_core::Ps2System;
use ps2_core::bus::DmaWatch;
use tracing::info;

/// The RAM offset a watched address names, if it is in a RAM the DMA
/// engines write: EE RAM (32 MiB) or IOP RAM (2 MiB).
pub(crate) fn ram_offset(t: Target, addr: u32) -> Option<u32> {
    let phys = canonical(t, addr) & 0x1FFF_FFFF;
    match t {
        Target::Ee if phys < 0x0200_0000 => Some(phys),
        Target::Iop if phys < 0x0080_0000 => Some(phys & 0x1F_FFFF),
        _ => None,
    }
}

/// Register `addr..addr+len` on the target's RAM with the bus so DMA
/// writes to it are recorded.
pub(crate) fn arm_dma(sys: &mut Ps2System, t: Target, addr: u32, len: u32) {
    if let Some(start) = ram_offset(t, addr) {
        sys.bus.dma_watch.push(DmaWatch { iop: t == Target::Iop, start, len });
    }
}

pub(crate) fn disarm_dma(sys: &mut Ps2System, t: Target, addr: u32, len: u32) {
    if let Some(start) = ram_offset(t, addr) {
        let w = DmaWatch { iop: t == Target::Iop, start, len };
        sys.bus.dma_watch.retain(|x| *x != w);
    }
}

/// One line per DMA run recorded since the hits were last cleared that
/// overlaps the watched bytes; empty when the core itself wrote them.
pub(crate) fn dma_writers(sys: &Ps2System, t: Target, addr: u32, len: u32) -> Vec<String> {
    let Some(start) = ram_offset(t, addr) else { return Vec::new() };
    let (s, e) = (u64::from(start), u64::from(start) + u64::from(len));
    sys.bus
        .dma_hits
        .iter()
        .filter(|h| {
            h.iop == (t == Target::Iop)
                && u64::from(h.start) < e
                && s < u64::from(h.start) + u64::from(h.bytes)
        })
        .map(|h| {
            format!(
                "watchpoint at {addr:#x} hit by {} DMA, dest={:#x} size={:#x}, cycle {}",
                h.channel, h.start, h.bytes, h.cycle
            )
        })
        .collect()
}

/// Headless writer log. Single-steps the machine (interpreted, so pair it
/// with a save state close to the event) and reports every change to the
/// watched bytes on the `ps2_debug::watch` log target: the cycle, both
/// cores' pcs, and the DMA engine when one wrote them.
#[derive(Default)]
pub struct WatchLog {
    watches: Vec<(Target, Watchpoint)>,
}

impl WatchLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.watches.is_empty()
    }

    /// Watch `len` bytes at `addr` on `target`, snapshotting them now.
    pub fn add(&mut self, sys: &mut Ps2System, target: Target, addr: u32, len: u32) {
        let old = (0..len)
            .map(|i| peek(sys, target, addr.wrapping_add(i)).unwrap_or(0))
            .collect();
        arm_dma(sys, target, addr, len);
        self.watches.push((target, Watchpoint { addr, len, old }));
    }

    /// Run `cycles` EE cycles one instruction at a time, logging changes.
    pub fn run(&mut self, sys: &mut Ps2System, cycles: u64) {
        let end = sys.cycles + cycles;
        while sys.cycles < end {
            sys.step();
            self.poll(sys);
            sys.bus.dma_hits.clear();
        }
    }

    fn poll(&mut self, sys: &mut Ps2System) {
        for (t, wp) in &mut self.watches {
            let now: Vec<u8> = (0..wp.len)
                .map(|i| peek(sys, *t, wp.addr.wrapping_add(i)).unwrap_or(0))
                .collect();
            if now == wp.old {
                continue;
            }
            let writers = dma_writers(sys, *t, wp.addr, wp.len);
            info!(
                target: "ps2_debug::watch",
                target_core = t.name(),
                addr = format_args!("{:#x}", wp.addr),
                old = format_args!("{:02x?}", wp.old),
                new = format_args!("{:02x?}", now),
                cycle = sys.cycles,
                ee_pc = format_args!("{:#010x}", sys.ee.pc),
                iop_pc = format_args!("{:#010x}", sys.iop.pc),
                by = if writers.is_empty() { "cpu" } else { "dma" },
                "watched bytes changed"
            );
            for line in writers {
                info!(target: "ps2_debug::watch", "{line}");
            }
            wp.old = now;
        }
    }
}
