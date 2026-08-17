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
| `ps2-app`   | Native front-end: headless CLI for bring-up (`--cycles`, screenshots, `--press` scripting, dumps) and an eframe/wgpu window with keyboard pad input and cpal audio (default without `--cycles`). |
| `ps2-debug` | gdb-remote debug stub (LLDB first-class), EE and IOP targets on separate TCP ports. `--debug-ee <port>` / `--debug-iop <port>`; `--wait-debugger` holds at the reset vector until attach. |

Planned crates:

| Crate       | Role                                                        |
| ----------- | ----------------------------------------------------------- |
| `ps2-wasm`  | wasm bindings for a browser front-end.                       |

## Decisions

| Decision | Choice | Why |
| -------- | ------ | --- |
| CPU execution | Interpreter (`match`-based, no JIT) | Bring-up and BIOS debugging first; deterministic, wasm-safe. Cached interpreter / JIT is a later performance option. Both cores skip stepping while spinning in their kernel idle loops (only an interrupt can move them), which keeps emulated timing exact. |
| GS rendering | Software rasterizer over VRAM in the hardware page/block/column layout (`gs/layout.rs`) | Accuracy and debuggability first, same as PS1e. The real layout matters: games pack CLUTs and overlay formats in ways a linear model breaks. wgpu only uploads the final framebuffer. A hardware renderer can be added later behind the same command interface. |
| BIOS | LLE only, SCPH-50000 as the reference image | The emulator must faithfully run the original BIOS so it can validate the reimplemented one. No HLE hooks in the execution path. |
| TTY observation | Watch writes to the EE SIO TXFIFO (0x1000F180) | The kernel's debug output channel. Pure observation, no effect on execution — safe for BIOS bring-up. |
| Bus design | Concrete fields + address `match` dispatch, no traits | Simplicity and speed; avoids generics. Same as PS1e. |
| EE↔IOP timing | Alternating slices at the 8:1 clock ratio (EE 294.912 MHz : IOP 36.864 MHz) | Simple and deterministic; refine granularity when SIF timing demands it. |
| Address translation | KSEG0/1 fold, real EE TLB for mapped segments with a 1024-entry translation cache and a one-page instruction-fetch cache | Games map their own pages; the caches keep the common case at a compare and a load. |
| Cycle counting | 1 cycle per EE instruction for now | Good enough for bring-up; add memory wait states and dual-issue approximation later (PS1e-style penalty accounting). |
| Profiling | `--features profile`: rdtsc scope accounting per subsystem plus EE/IOP instruction and PC histograms; `examples/gs_bench.rs`, `examples/ee_bench.rs` | Sampling profilers need elevation on the development box; the scopes answer "which subsystem" and the histograms "which loop" cheaply. |
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
3. **DMAC + GIF + GS** *(done)* — software rasterizer (hardware VRAM
   layout, bilinear filtering, CLUT cache, integer pixel pipeline), GIF
   paths 1-3, VIF1 with real UNPACK, VU1 interpreter, VU0 macro mode,
   real EE TLB, scanout with field doubling. The OSD renders end to end
   (browser, system configuration, version screens).
   `ps2-debug` was pulled forward from milestone 6 and is done: attach,
   halt, registers, memory, breakpoints, single-step, polled write
   watchpoints on both cores.
4. **CDVD + ELF loading** *(done)* — osdconfig/NVRAM/RTC, memory card
   (MagicGate replies), 2048-byte-sector ISO streaming, disc key,
   DEC-SET, PS2LOGO → LoadExecPS2 → EELOAD → game ELF.
5. **VU/VIF/IPU/SPU2/pads** *(mostly done)* — SIO2 pads and cards, SPU2
   voices/ADMA/mixer with WAV capture and live cpal output; Amagami
   (SLPS-25918) is playable through its title, menus, name entry and
   prologue. Not yet: IPU, VU1 timing, SPU2 sweep volumes.
6. **Front-end and speed** *(in progress)* — eframe/wgpu window with
   keyboard pad input (done); the interpreter runs at roughly 40% of
   real time after idle-loop skipping, with the EE interpreter and the
   rasterizer sharing the remaining cost. Next: cheaper per-pixel
   addressing, a cached decoder or JIT for the EE, and the wasm
   front-end.

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
