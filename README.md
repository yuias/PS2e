# PS2e

A PlayStation 2 emulator written in Rust. Software-rasterized GS with the
hardware VRAM layout, x86-64 recompilers for the EE, IOP and VU1, an IPU
that decodes MPEG-2 video, SPU2 with reverb and AutoDMA streaming, CDVD with
drive timing and mechacon NVRAM, memory cards, and an LLDB-compatible remote
debugger on both cores.

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
disc, memory card and mechacon NVRAM stay in), the two disc commands, the
cheat toggle, save/load state and screenshots; View toggles fullscreen, the
side pane and the TTY console, and sizes the window to a 720p or 1080p
display; Help lists the current key bindings.

Settings live in the side pane on the right, which has three pages behind a
row of tabs: Settings (display scaler, aspect ratio, deinterlacer, field
order, internal 2x rendering, master volume, keyboard bindings), Memory (a
RAM viewer and the cheat-value scanner) and Registers (both CPUs'
register files). Only the
page on screen is fed by the emulator, so the two behind it cost nothing,
and the whole pane can be closed from the View menu. The pane's width and
whether it opens are remembered in `config.toml`.

View > Display size resizes the *window* until the display itself is 720 or
1080 pixels tall, at whatever aspect ratio is selected, so a window dragged
to some arbitrary size can be put back to a known one. The side pane and the
TTY panel keep their size across it; only the display grows or shrinks. A
window too big for the desktop is whatever the window manager makes of it.

The window's own size is remembered across runs, so a display size picked
here comes back on the next launch along with it. Fullscreen is not saved.

While a text field in the pane has focus it takes the keyboard, so typing an
address does not also press the pad buttons those keys are bound to. Esc, or
a click on empty pane, hands the keyboard back.

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

Settings > Input > Keyboard opens a controller diagram with a box against
every button: click one and press a key to bind it. A key bound to two
buttons is outlined in red, which is allowed but rarely meant. OK writes
the bindings back and they persist with the rest of the settings.

The pad and the save/load shortcuts are also rebindable by hand; see the
`[keys]`, `[pad]` and `[hotkeys]` tables under
[Configuration](#configuration). Fullscreen and screenshot are fixed.

A screenshot writes the displayed frame as `screenshot_<epoch>.bmp` in the
working directory. Headless `--screenshot` picks its encoding from the
extension instead, so it writes the same BMP for a `.bmp` path and a PNG
for `.png`.

Save states snapshot the whole machine to `state0.sst` next to
`config.toml`, or wherever `state` points, zstd-compressed (about 13 MiB of
a 40 MiB image, and roughly 0.4 s to write, which the machine pauses for).
The BIOS, disc image and memory card are not part of a state and carry over
on load; a state saved with a different BIOS loads with a warning in the
log.

## Configuration

`config.toml` is searched at `<exe_dir>/config/config.toml`, then
`~/.config/PS2e/config.toml`. A commented template is generated on first
run. CLI flags override the file.

```toml
bios = "path/to/bios.bin"   # falls back to assets/SCPH-50000.bin
region = "ntsc"             # video timing before SetGsCrt: ntsc | pal
volume = 0.5                # master volume, 0.0..1.0
memcard = "memcard0.ps2"    # blank until the BIOS browser formats it
state = "state0.sst"        # save state; next to config.toml by default
scaler = "sharp"            # nearest | linear | sharp | lanczos
aspect = "4:3"              # 4:3 | 16:9 | native
deinterlace = "bwdif"       # weave | bob | blend | adaptive | adaptivedebug
                            # | bwdif ("yadif" parses but is not in the menu)
swap_fields = false         # flip which rows each field lands on
internal_2x = false         # true 2x edges on 3D geometry
cheats = false              # apply <disc>.pnach next to the disc image
pane = true                 # open the side pane at startup
pane_width = 420.0          # and the width it opens at
window_width = 960.0        # window size at the last exit, in egui points
window_height = 640.0

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
| `--screenshot <p>` | Final framebuffer as `.bmp` or `.png`, by extension |
| `--screenshot-every N` | Also write `<stem>_<n>.<ext>` every N cycles |
| `--wav <p>` | Write the SPU2 output as a 48 kHz stereo WAV |
| `--dump <dir>` | EE/IOP RAM, GS VRAM, VU1 and SPU2 dumps after the run |
| `--memcard <p>` | Card image to load and persist |
| `--cheats` | Apply `<disc>.pnach` next to the image |
| `--watch` | Log writers of `<ee\|iop>:<addr>[,<len>]`, repeatable |
| `--log <filter>` | Tracing filter, e.g. `info,ps2_core::tty=debug` |
| `--no-jit` | Interpret EE, VU1 and IOP instead of recompiling them |
| `--gs-inline` | Render on the emulation thread, no GS worker |
| `--internal-2x` | Render internally at 2x |
| `--region <r>` | Video timing until `SetGsCrt`, `ntsc` (default) or `pal` |

`--window` opens the window even when `--cycles` is given. `--watch` steps
the interpreter so it can compare bytes between instructions, which is slow;
the frames it produces still match the recompilers'.

Every cycle count takes digits, `_` separators or e-notation, so `13e9`,
`13_000_000_000` and `13000000000` are the same number. `--cycles` is a
duration, but the cycles in `--press`, `--insert` and `--save-state` are
read off the machine's own counter — which a save state carries, so from
`--load-state` they are positions in that state's timeline, not offsets
from the start of the run.

`--region` only sets the refresh and horizontal-blank rates the machine
starts with. Software owns the CRTC from there: the kernel's `SetGsCrt`
programs SMODE1's `CMOD` field, and an NTSC BIOS does so during its own
boot, so this is a pre-boot default rather than a way to force 50 Hz --
running a PAL BIOS is. It is also not what software *detects* as the
console's region, which comes from the BIOS image's ROMVER.

## Debugger (LLDB / GDB)

```
ps2e --cycles <n> --debug-ee 9000 --debug-iop 9001 --wait-debugger
```

Each core gets its own port and its own stub, speaking the gdb-remote serial
protocol with LLDB as the primary client; plain GDB works too.
`--wait-debugger` holds execution at the reset vector until a client
attaches, and needs one of the two ports. Halting either core halts the
whole machine — the two run in lockstep. `--watch` is the cheaper way to
answer "who wrote this address" and is refused alongside either port.

```
(lldb) gdb-remote localhost:9000
(lldb) breakpoint set --address 0x00082080
(lldb) continue
```

Registers (read and write), memory read/write, software breakpoints,
single-stepping and interrupt are supported, along with write watchpoints,
which are polled per instruction and so catch DMA writes too; read and
access watchpoints are not. EE registers are 64-bit on the wire (mips64el);
the IOP is the PS1-style 32-bit layout. Disassembly requires an LLVM build
that includes the Mips target.

## Limitations

- The IPU decodes every command it has, and full-motion video plays, but it
  has been exercised against one title's movies only. `PACK` to 4-bit
  indexed output is not modelled and emits zeros.
- One controller, a DualShock 2 in port 1; port 2 reads as empty. A gamepad
  drives the buttons and both sticks; the keyboard drives the buttons only,
  so with no gamepad attached the sticks read as centred. Pressure-sensitive
  buttons report fully down or not at all, and vibration is acknowledged but
  goes nowhere.
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
