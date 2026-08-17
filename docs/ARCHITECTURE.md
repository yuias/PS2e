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
| CPU execution | Interpreter (`match`-based) as the reference; x86-64 recompiler for the EE behind the `jit` feature (`ee/jit/`, dynasm) | The interpreter stays authoritative and wasm-safe; the recompiler translates blocks that keep registers in the `Cpu` struct, calls back into the interpreter for anything it does not translate, and ends blocks at control flow so exceptions/branches keep the interpreter's semantics. The system retires a block, then advances the IOP/timers/vblank by the same cycles (IOP interleave is block-granular under the JIT). Both cores skip stepping while spinning in their kernel idle loops (only an interrupt can move them). `--no-jit` restores cycle-exact interleave; the debugger always steps the interpreter. |
| GS rendering | Software rasterizer over VRAM in the hardware page/block/column layout (`gs/layout.rs`), on a worker thread behind `GsFront` (`threads` feature) | Accuracy and debuggability first, same as PS1e. The real layout matters: games pack CLUTs and overlay formats in ways a linear model breaks. Everything the EE can observe (CSR/IMR, SIGNAL/FINISH, display registers) is decided on the EE side at enqueue time, so the worker only owns VRAM and stays deterministic. wgpu only uploads the final framebuffer. |
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
   keyboard pad input, TTY and register panels (done). Speed: EE x86-64
   recompiler with block linking, GS on a worker thread with the pixel
   pipeline split across a rayon pool for large primitives, idle-loop
   skipping on both cores; Amagami runs at ~1.9x real time headless (the
   OSD's blur-heavy boot screens at ~40%). Next: SPU2 audio quality (a
   constant noise under music, boot chime starvation), the IOP
   interpreter, and the wasm front-end.

## Component map (ps2-core)

```
ps2-core/src/
├── lib.rs        # Ps2System: EE + IOP interleave (8:1), vblank scheduling, idle skipping
├── bus.rs        # Both memory maps, MMIO dispatch, EE/IOP DMA, INTC/DMAC, CDVD, SIO2, TLB
├── sif.rs        # SIF mailboxes/flags/control + SIF0/SIF1 FIFOs
├── timers.rs     # EE timers (lazy counts, compare interrupts)
├── gif.rs        # GIF tag parser (packed/reglist/image) feeding the GS
├── vif.rs        # VIF1 command parser and UNPACK expansion into VU1 memory
├── vu1.rs        # VU micro interpreter (also serves VU0 macro mode)
├── prof.rs       # `profile` feature: TSC scopes, instruction/PC histograms
├── ee/
│   ├── mod.rs    # R5900 interpreter (128-bit GPRs, MMI, COP2 macro, branch delay slots)
│   ├── cop0.rs   # Status/Cause/EPC, exceptions, ERET, interrupt gating
│   ├── fpu.rs    # COP1 (non-IEEE single-precision; host f32 approximation for now)
│   └── jit/      # x86-64 recompiler: block cache (mod.rs), emitter (emit.rs), bus helpers, RWX arena
├── iop/
│   └── mod.rs    # R3000A interpreter (load delay slots, PS1-style COP0)
├── gs/
│   ├── front.rs  # EE-side GS: privileged registers, CSR/IMR, command stream to the worker
│   ├── mod.rs    # Registers, vertex kick, IMAGE/local transfers, VRAM accessors, scanout
│   ├── layout.rs # Hardware page/block/column addressing for every pixel format
│   └── raster.rs # Triangle/sprite rasterization, texture sampling, tests, blending
└── spu2/
    ├── mod.rs    # Registers, transfer engine, ADMA, mixer
    └── voice.rs  # ADPCM decode, ADSR envelopes
```

Not yet split out: `ipu/` (absent), a full 10-channel `dmac/` (channels live in
`bus.rs`), and an event scheduler (everything is catch-up ticks today).
