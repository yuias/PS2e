# PS2e Architecture

PS2 (PlayStation 2) emulator in Rust. Follows the structure and philosophy of
the sibling project PS1e (`N:\PS1e`): lightweight, component-oriented like the
real hardware, and designed as a bring-up/verification tool for the custom
BIOS reimplementation project (`PS2BiosRebuild`).

## Goals

- Lightweight and fast native app (Windows/macOS/Linux).
- Components mirror the real hardware and own their own state.
- Rich debug logging in debug builds — the emulator doubles as the primary
  debug environment for the custom BIOS (no existing PS2 emulator has strong
  enough debug tooling for that).
- Future: LLDB attach for game debugging (gdb-remote stub crate), and a wasm
  build with an HTML5 front-end driving the platform-independent core.

## Workspace layout

| Crate       | Role                                                                  |
| ----------- | --------------------------------------------------------------------- |
| `ps2-core`  | Platform-independent emulator core. No windowing, graphics API or I/O dependencies; wasm-safe. |
| `ps2-app`   | Native front-end. Headless CLI first; egui + wgpu UI once the GS renders. |
| `ps2-debug` | gdb-remote debug stub (LLDB first-class), EE and IOP targets on separate TCP ports. `--debug-ee <port>` / `--debug-iop <port>`; `--wait-debugger` holds at the reset vector until attach. |

Planned crates:

| Crate       | Role                                                        |
| ----------- | ----------------------------------------------------------- |
| `ps2-wasm`  | wasm bindings for a browser front-end.                       |

## Decisions

| Decision | Choice | Why |
| -------- | ------ | --- |
| CPU execution | Interpreter (`match`-based, no JIT) | Bring-up and BIOS debugging first; deterministic, wasm-safe. Cached interpreter / JIT is a later performance option. |
| GS rendering | Software rasterizer | Accuracy and debuggability first, same as PS1e. wgpu only uploads the final framebuffer. A hardware renderer can be added later behind the same command interface. |
| BIOS | LLE only, SCPH-50000 as the reference image | The emulator must faithfully run the original BIOS so it can validate the reimplemented one. No HLE hooks in the execution path. |
| TTY observation | Watch writes to the EE SIO TXFIFO (0x1000F180) | The kernel's debug output channel. Pure observation, no effect on execution — safe for BIOS bring-up. |
| Bus design | Concrete fields + address `match` dispatch, no traits | Simplicity and speed; avoids generics. Same as PS1e. |
| EE↔IOP timing | Alternating slices at the 8:1 clock ratio (EE 294.912 MHz : IOP 36.864 MHz) | Simple and deterministic; refine granularity when SIF timing demands it. |
| Address translation | Direct segment fold (`addr & 0x1FFF_FFFF`), scratchpad at 0x70000000, uncached-accelerated RAM mirror at 0x30100000 | The kernel's TLB mappings are essentially identity; TLB instructions record entries but do not remap yet. Revisit if software relies on real TLB behavior. |
| Cycle counting | 1 cycle per EE instruction for now | Good enough for bring-up; add memory wait states and dual-issue approximation later (PS1e-style penalty accounting). |
| Logging | `tracing` with per-component targets (`ps2_core::ee::cpu`, `ps2_core::tty`, …) | Fine-grained runtime filtering; static max-level features strip verbose logs from release builds. |
| Unimplemented ops/MMIO | Log at `error` and panic (ops) / log and shadow (MMIO) | During bring-up, silently continuing past an unknown instruction corrupts state; a loud stop with context is the iteration loop. |
| Save states | Not yet; will use `serde` + `postcard` like PS1e, excluding external assets (BIOS by fingerprint) | Deferred until the component set stabilizes. |
| Debugger | PS1e `psx-debug` design: polled TCP, no threads, `pump(&mut sys, budget)`; one gdb-remote port per core instead of gdb multiprocess extensions | The cores are lock-stepped 8:1, so halting either target halts the whole machine and per-core ports keep the single-thread protocol LLDB already speaks. EE registers are presented as 64-bit GPR/LO/HI (low halves; MMI upper halves, LO1/HI1, SA not exposed) under a `mips64el` triple so doubleword ops disassemble. Write watchpoints (`Z2`) are polled per instruction against a byte snapshot — slow but exact, and they catch DMA writes too. Debugger memory access goes through side-effect-free `peek8`/`poke8` (MMIO refused). |

## Milestones

1. **EE bring-up** *(done)* — R5900 interpreter + bus + BIOS load; runs
   SCPH-50000 through full kernel init to the DECI2 manager banner.
2. **IOP + SIF** *(done)* — R3000A core, SIF mailboxes/flags + SIF0/SIF1
   DMA, EE/IOP interrupts and timers. Cooperative boot completes: IOP
   modules load, SIF RPC works, EELOAD restarts the kernel into OSDSYS.
   Findings recorded in `docs/hw-notes.md`.
3. **DMAC + GIF + GS** *(paused, resumes after the debugger)* — software
   rasterizer, boot screen visible, egui UI. Done so far: GS core
   (linear-addressed VRAM, raster pipeline, scanout), GIF, DMA ch2,
   real EE TLB, EE timer EQUF semantics, CDVD S-command NVRAM/RTC
   model, SIO2 no-device stub, `--screenshot`. Boot reaches OSDSYS with
   working RPC and 4 kernel restarts, but OSDSYS does not draw yet.
   Known blockers: a stale kernel T3-callback dispatch fires with a
   cleared handler table (crashes via a null exec ~4 s in), cdvdman
   polls N-status 0x1F402005 for a value other than 0x40, and VU0
   macro ops are still nops (kernel context save/restore only so far).

   **`ps2-debug` is done** (pulled forward from milestone 6): the
   remaining blockers are kernel-internal timing/state bugs that static
   disassembly of RAM dumps was too slow to chase. The stub (PS1e's
   `psx-debug` as the template) gives both cores attach/halt, register
   and memory access, address breakpoints, single-step and polled write
   watchpoints; validated end-to-end at the wire-protocol level and
   against the real BIOS. Milestone 3 resumes here, debugger in hand.
4. **CDVD + ELF loading** — homebrew boot.
5. **VU/VIF/IPU/SPU2/pads** — commercial game boot.
6. **Platform reach** — wasm front-end (`ps2-debug` was pulled forward
   and completed during milestone 3).

## Component map (ps2-core)

```
ps2-core/src/
├── lib.rs        # Ps2System: EE + IOP interleave (8:1), vblank scheduling
├── bus.rs        # Both memory maps, MMIO dispatch, SIF DMA pump, INTC/DMAC/IOP-DMA state
├── sif.rs        # SIF mailboxes/flags/control + SIF0/SIF1 FIFOs
├── timers.rs     # EE timers (lazy counts, compare interrupts)
├── ee/
│   ├── mod.rs    # R5900 interpreter (128-bit GPRs, MMI, branch delay slots)
│   ├── cop0.rs   # Status/Cause/EPC, exceptions, ERET, interrupt gating
│   └── fpu.rs    # COP1 (non-IEEE single-precision; host f32 approximation for now)
└── iop/
    └── mod.rs    # R3000A interpreter (load delay slots, PS1-style COP0)
```

Planned: `gs/`, `vu/`, `dmac/` (full 10-channel), `gif/`, `vif/`, `ipu/`,
`cdvd/`, `spu2/`, `scheduler` (event-driven for VBlank-class events only;
everything else catch-up ticks, as in PS1e).
