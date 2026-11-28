# AI-Agent & Automated Reverse-Engineering Workflow

`PS2e` includes a deterministic, lockstep control protocol (`ps2e --control-port <port>`) and a zero-dependency Rust CLI (`ps2ctl`) designed so both developers and AI agents can boot, inspect, debug, checkpoint, and visually verify **any** PlayStation 2 game without fragile GUI automation.

Run `ps2ctl --agent-guide` at any time to print a compact quick-reference playbook directly to stdout.

## Core Loop (Zero to 1:1 Verification)

### 1. Launch Headless Lockstep Session
```bash
ps2e --bios /path/to/bios.bin --disc /path/to/game.iso --control-port 9002 &
ps2ctl --wait 15 state json
```
- `--wait <secs>` retries until the control port is accepting connections (no shell `sleep` loops needed).
- `state json` returns a single-line JSON object (`ee_pc`, `iop_pc`, `cycles`, `frames`, `vblanks`, `region`, `field_hz`, `held`, `tray`).

### 2. Boot Once, Checkpoint Savestates, Iterate in Milliseconds
Use `until frame changed max <n>v` to advance past BIOS/logo/loading screens automatically without guessing frame counts, then save zstd-compressed checkpoints (or in-memory `@0`..`@15` slots):
```bash
ps2ctl --batch <<'EOF'
until frame changed max 900v
frame captures/title.png
seq start:10v none:60v cross:10v none:120v
savestate states/main_menu.ps2state
frame captures/main_menu.png
EOF
```
Every subsequent run reloads `states/main_menu.ps2state` in milliseconds and executes deterministically in lockstep.

### 3. Diagnose Hangs & Hardware State Without GDB
```bash
ps2ctl regs ee             # EE PC, COP0 (status/cause/epc), HI/LO, all 32 GPRs
ps2ctl regs iop            # IOP PC, COP0, all 32 GPRs
ps2ctl regs hw             # INTC, DMAC, VIF0/1 stall/CHCR, GIF, SIF flags/FIFOs
ps2ctl tty                 # EE/IOP kernel & game printf output
ps2ctl peek ee 00100000 64 # Side-effect-free memory dump
```

### 4. Discover & Track Game State in RAM
```bash
ps2ctl scan start ee 4 exact 100
ps2ctl run 60v
ps2ctl scan filter decreased
ps2ctl scan list 20
ps2ctl until ee <hexaddr> 4 changed max 300v
```

### 5. Pure-Rust 1:1 Visual Diff (No Python Needed)
Compare a PS2 ground-truth capture against a native port screenshot directly via `ps2ctl diff`:
```bash
ps2ctl diff captures/ps2_scene.png captures/native_scene.png captures/diff_heatmap.png
# Output:
# {"width":512,"height":448,"mae_rgb":4.210,"psnr_db":31.42,"diff_pct":1.85}
```
The optional 3rd argument writes a 4x amplified RGB error heatmap PNG so visual mismatches are immediately obvious.
