# PS2e

A PlayStation 2 emulator written in Rust. Software-rasterized GS with the
hardware VRAM layout, an x86-64 recompiler for the EE, SPU2 with reverb and
AutoDMA streaming, CDVD with drive timing and mechacon NVRAM, memory cards,
and an LLDB-compatible remote debugger on both cores.

A PlayStation 2 BIOS image (4 MiB, e.g. SCPH-50000) is required and not
included.

## Build

Requires a recent stable Rust toolchain.

```
cargo build --release
```

Produces `ps2e` in `target/release/`.

## Run

```
ps2e [--bios <path>] [--disc <image>]
```

A disc image is a raw ISO of 2048-byte sectors. 2352-byte CD images are not
supported — the PS2 titles this targets are DVDs.

Without `--disc` the BIOS browser runs. The machine starts running as soon
as the window opens.

"Insert disc..." in the Emulation menu swaps the disc the way the console
does: the drive opens, the file picker comes up, and the drive closes on
whatever was picked — cancelling puts the old disc back. Emulation never
stops, so a multi-disc title can change discs where it asks you to, and
from the browser the console picks the new disc up and boots it on its own.
It takes the drive a few seconds to spin a disc up and identify it, as it
does on hardware. "Boot disc..." is the same thing plus a power cycle, for
starting a disc over.

The window shows the display, with a menu bar for run control and a status
bar underneath. Emulation covers run/pause, step, reset (a power cycle; the
disc, memory card and mechacon NVRAM stay in), the two disc commands,
save/load state and screenshots; View toggles fullscreen, the display scaler, the deinterlacer,
internal 2x rendering and the debug panels — TTY console and CPU registers,
both hidden by default. Audio holds the master volume, and Help lists the
current key bindings.

The status bar names the image in the drive, and the window title carries the
disc's boot serial ("PS2e - SLPS-25418"): a PS2 disc has no printable game
title on it, so the serial stands in for one.

| Keys | |
|---|---|
| Arrows | D-pad |
| Z / X / S / D | Cross / Circle / Square / Triangle |
| W / E / R / U | L1 / L2 / R1 / R2 |
| 1 / 3 | L3 / R3 |
| V / C | Start / Select |
| F5 / F9 | Save / load state |
| F11 / F12 | Fullscreen (Esc leaves) / screenshot |

A gamepad drives the same pad in parallel with the keyboard, so either can
press any button. It is picked up automatically when one is plugged in.

The pad and the save/load shortcuts are rebindable; see the `[keys]`,
`[pad]` and `[hotkeys]` tables under [Configuration](#configuration).
Fullscreen and screenshot are fixed.

A screenshot writes the displayed frame as `screenshot_<epoch>.bmp` in the
working directory, the same encoding as headless `--screenshot`.

Save states snapshot the whole machine to `state0.sst` next to the memory
card image, zstd-compressed (about 13 MiB of a 40 MiB image, and roughly
0.4 s to write, which the machine pauses for). The BIOS, disc image and
memory card are not part of a state and carry over on load; a state saved
with a different BIOS loads with a warning.

## Configuration

`config.toml` is searched at `<exe_dir>/config/config.toml`, then
`~/.config/PS2e/config.toml`. A commented template is generated on first
run. CLI flags override the file.

```toml
bios = "path/to/bios.bin"   # falls back to assets/SCPH-50000.bin
volume = 0.5                # master volume, 0.0..1.0
memcard = "memcard0.ps2"    # created and formatted automatically
scaler = "sharp"            # nearest | linear | sharp | lanczos
deinterlace = "bwdif"       # weave | bob | blend | adaptive | bwdif
swap_fields = false         # flip which rows each field lands on
internal_2x = false         # true 2x edges on 3D geometry

[keys]                      # digital pad; egui key names
cross = "Z"
start = "V"

[pad]                       # same pad from a gamepad; gilrs button names
cross = "South"
start = "Start"

[hotkeys]                   # frontend shortcuts
save_state = "F5"
load_state = "F9"
```

Key names are the ones egui reports: letters and digits as themselves
(`"X"`, `"1"`), arrows as `"Up"`/`"Down"`/`"Left"`/`"Right"`, plus
`"Enter"`, `"Backspace"`, `"Space"`, `"F1"`..`"F35"`. Omitted buttons keep
their defaults, and an unrecognized name falls back to the default with a
warning in the log.

Gamepad names are the `gilrs` ones: `"South"`/`"East"`/`"North"`/`"West"`
for the action pad, `"DPadUp"`..`"DPadRight"`,
`"LeftTrigger"`/`"LeftTrigger2"` and the right-hand pair, `"Start"`,
`"Select"`, `"LeftThumb"`/`"RightThumb"`, `"Mode"`, `"C"`, `"Z"`.

Settings changed from the menus are written back when the window closes,
which replaces the template's comments with the plain values.

## Headless mode

```
ps2e --cycles <n> [--disc <image>] ...
```

Runs without a window for the given number of EE cycles and exits, printing
the kernel and game TTY output as it appears. This is the bring-up and
regression path: a screenshot from a fixed cycle count is bit-identical run
to run, which is what changes get checked against.

| Flag | |
|---|---|
| `--cycles N` | EE cycles to run, then exit |
| `--press` | Hold a pad button, `<button>@<cycle>[-<cycle>]`, repeatable |
| `--insert` | Open the drive and close it on a new image, `<path>@<cycle>` |
| `--save-state <p>@<n>` | Write a save state at that cycle |
| `--load-state <p>` | Start from a save state instead of the reset vector |
| `--screenshot <p>` | Write the final framebuffer as a BMP |
| `--screenshot-every N` | Also write `<p>_<n>.bmp` every N cycles |
| `--wav <p>` | Write the SPU2 output as a 48 kHz stereo WAV |
| `--dump <dir>` | EE and IOP RAM dumps after the run |
| `--memcard <p>` | Card image to load and persist |
| `--log <filter>` | Tracing filter, e.g. `info,ps2_core::tty=debug` |
| `--no-jit` | Interpret the EE instead of recompiling it |
| `--gs-inline` | Render on the emulation thread, no GS worker |
| `--internal-2x` | Render internally at 2x |

`--window` opens the window even when `--cycles` is given.

## Debugger (LLDB / GDB)

```
ps2e --cycles <n> --debug-ee 9000 --debug-iop 9001 --wait-debugger
```

Each core gets its own port and its own stub, speaking the gdb-remote serial
protocol with LLDB as the primary client; plain GDB works too.
`--wait-debugger` holds execution at the reset vector until a client
attaches. Halting either core halts the whole machine — the two run in
lockstep.

```
(lldb) gdb-remote localhost:9000
(lldb) breakpoint set --address 0x00082080
(lldb) continue
```

Registers, memory read/write, software breakpoints, single-stepping and
interrupt are supported. EE registers are 64-bit on the wire (mips64el);
the IOP is the PS1-style 32-bit layout. Disassembly requires an LLVM build
that includes the Mips target.

## Limitations

- The IPU (MPEG decoder) is not implemented, so full-motion video does not
  decode.
- The emulated controller is a digital pad: the keyboard and a gamepad both
  drive it, but the analog sticks read as centred.
- Memory cards respond in slot 1 only.
- A game asking for the next disc has not been tried against a real
  multi-disc title yet, though swapping mid-game works.
- VU1 has no timing model, and its microprogram runs to completion when
  kicked rather than in step with the EE.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Hardware behaviour worked
out during bring-up is written up in [docs/hw-notes.md](docs/hw-notes.md).

## License

MIT. See [LICENSE](LICENSE).
