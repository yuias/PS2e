# Hardware notes from BIOS bring-up

Empirically derived facts, found by tracing SCPH-50000 execution and
disassembling the code at hang sites. Useful both for this emulator and for
the BIOS reimplementation project (`PS2BiosRebuild`).

## Reset / CPU discrimination

- The shared reset stub at 0xBFC00000 reads COP0 PRId: `PRId < 0x59` takes
  the IOP path (0xBFC02000), otherwise the EE path (0xBFC00800). EE PRId is
  0x2E20.
- The IOP path then splits again on PRId: `PRId < 0x10` selects the PS1
  compatibility register tables (POST stage 1) and ultimately boots the PS1
  kernel (TBIN). The PS2-revision IOP must report `0x10 <= PRId < 0x59`
  (we use 0x1F); the ROM then also consults 0x1F801450 bit 3.

## RDRAM init (InitRDRAM, ROM module `RDRAM` at offset 0x41000)

- DMAC ENABLER (0x1000F520) must read 0x1201 at reset. The BIOS uses the
  value as a board-revision key into a 12-byte-entry table (ROM 0x43C80) of
  RDRAM configurations; key miss returns -14 ("failed to initialize memory").
- 0x1000F410 takes commands with a busy bit (31) that must self-clear.
- MCH_RICM (0x1000F430) / MCH_DRD (0x1000F440) serial init handshake: with
  SOP=0, SA=0x21 (INIT) each of the 2 RDRAM devices answers 0x1F once;
  SA=0x23 (CNFGA) -> 0x0D0D; SA=0x24 (CNFGB) -> 0x0090; SA=0x40 (DEVID) ->
  RICM & 0x1F. RICM reads back 0 (busy clear).

## ROMGSCRT (GS/CRT controller at 0x1A00000x, via SBUS)

- Command interface: +0x00 execute (0x81/0x82), +0x02 subcommand
  (0x42 = write data, 0x43 = read data), +0x06 status (write 3 to clear,
  poll: bit 0 = busy, bit 1 = done, must become nonzero), +0x10 data.
- Status stub of 2 (done, not busy) satisfies every wait loop.
- If the data-read result equals 0x1D the ROM deliberately hangs (PS1-mode
  trap); any other value continues.

## SIF (EE<->IOP)

- 0x1D000060 read from the IOP must return the magic 0x1D000060 (its own
  address); SIFMAN treats anything with the top 20 bits clear as "no SBUS".
  The EE-side equivalent 0x1000F260 also reads 0x1D000060.
- Control register (0x1000F240 EE / 0x1D000040 IOP): EE reads OR in
  0xF0000102, IOP reads OR in 0xF0000002. The IOP sets bits 0x20 (SIF0
  path), 0x40 (SIF1 path), 0x80 (SIF2 path) by writing the bit value and
  polls that it sticks. The EE writes 0x100 (open) and later 0x40100.
- MSFLG/SMFLG are asymmetric: each side's write *sets* its own flag word
  and *clears* the peer's.
- IOP writes to SMFLG (0x1D000030) raise the EE INTC SBUS interrupt
  (bit 1); the EE kernel's handler folds the flags into its internal SREG
  array (EE RAM 0x937C0) and acknowledges by clearing them.
- The EE kernel's SBUS handler also implements a mailbox protocol over IOP
  RAM 0x3E0/0x3E4 keyed by MSFLG/SMFLG bits 30/31.

## SIF DMA packet framing

- SIF1 (EE->IOP): EE channel 6, source chain (CHCR 0x184: MOD=chain, TIE,
  STR; TTE clear). Each packet in the stream is: one quadword IOP tag
  {addr | flags (bit30 IRQ, bit31 end), word count, pad, pad} followed by
  the data, padded to a quadword boundary. IOP channel 10 consumes the tag
  and writes `count` words to `addr`.
- SIF0 (IOP->EE): the IOP send block at ch9 TADR is 16 bytes:
  {data addr | flags, word count, EE tag lo, EE tag hi}; TADR advances by
  16. The EE destination-chain tag {qwc | id | irq, dest addr} is pushed
  into the FIFO as its own quadword ahead of the data. EE channel 5
  (destination chain) pops the tag qword and stores the following qwords.
- After module init, the IOP's sifcmd answers the EE's two init packets
  (cmd 0x80000002, carrying the EE receive-buffer address) with
  cmd 0x80000001 SetSreg(0, 1); the EE OSD boot path polls
  SREG[0] & 0x10000 and later SMFLG bit 0x40000 (EESYNC boot-complete,
  raised from LOADCORE's boot callbacks).

## Interrupt expectations during boot

- EE INTC mask during the SIF wait: bit 1 (SBUS) and bit 12 (timer 3);
  EE DMAC mask: channels 5 and 7. The kernel tick runs off EE timer 3
  compare interrupts.
- IOP I_MASK: 0x1080D = VBLANK (0), CDVD (2), DMA (3), EVBLANK (11) and
  timer 5 (16). Thread delays rely on vblank and timer-5 target
  interrupts; without them every thread sleeps forever and boot stalls
  after module loading.
- IOP DMA completions report through DICR2 (enable bits 16+, flag bits
  24+ for channels 7-13) and I_STAT bit 3.

## EE timers (learned during OSDSYS bring-up)

- Tn_MODE: bits 0-1 CLKS, bit 7 CUE (count enable), bit 8 CMPE (compare
  interrupt enable), bit 10 EQUF (equal flag, W1C), bit 11 OVFF. Writing
  MODE clears COUNT.
- **CMPE gates the INTC line, not the flag**: a compare match always
  latches EQUF (no new event until a MODE write with bit 10 rearms it),
  but only raises INTC when bit 8 is set. The kernel *free-runs* T3 with
  MODE 0xC83 / COMP 0xFFFF at init — CUE on, CMPE off — and only enables
  the interrupt while its callback queue is loaded (0x583 + COMP =
  next-callback delay; back to 0x483 when the queue drains). Raising
  INTC on a CMPE-clear compare produces a spurious TIM3 exactly at the
  16-bit wrap, 65535 hblanks ≈ 4.2 s after kernel init, which crashes
  the callback dispatcher (below).
- The kernel schedules deferred callbacks (SIF handlers, alarms) through
  a byte queue drained by a T3-driven dispatcher (kernel 0x80002650,
  registered in the INTC vector table 0x800153C0 under bit 12). The
  dispatcher **unconditionally dispatches at least one callback per
  invocation** — it assumes T3 only interrupts while the queue is
  non-empty, so a spurious TIM3 jalr's through a NULL handler entry.
  Structures: pending count 0x80019CB0, byte queue 0x8001A1B8, pending
  bitmask (u64) 0x80019CA8, handler table 0x80019CB8 with 0x14-byte
  entries {fn, arg, gp, id:u16}. Handlers run on a dedicated stack via
  the trampoline at 0x81FE0 (lui sp,8; jalr; syscall -5 on return).

## EE kernel TLB usage

- After boot the kernel relies on real TLB mappings: MMIO mapped per-4KB
  at identity (0x10006000...), extended RAM mirrors, and high kernel
  pages. Direct address folding stops working once OSDSYS loads; the
  recorded tlbwi entries must actually be walked. Scratchpad is the
  entry with EntryLo0 bit 31 set.

## VIF1 / OSDSYS drawing path

- OSDSYS submits everything through VIF1 source chains. 3D packets kick
  with CHCR 0x145 (TTE set), but the 2D layer kicks with **CHCR 0x105 —
  TTE clear — while still carrying `[NOP, DIRECT n]` in the tag's upper
  64 bits**: the DMAC evidently delivers the tag upper half to VIF1 on
  chain transfers regardless of TTE, and the OSD depends on it. Chain
  tags and data live in the scratchpad (address bit 31 = SPR).
- The 2D layer is plain DIRECT -> GIF PACKED/A+D packets (no VU1
  needed). The 3D layer (backgrounds, towers) is UNPACK + MPG + MSCAL
  on VU1 — nothing draws from it until VU1 executes microprograms.
- **XGKICK kicks split packets**: an early kick sends a GIFtag with
  EOP=0 whose continuation (the vertex tag) is only written to VU1
  data memory by a later kick. An emulator that streams "until EOP"
  must abandon the packet when it runs off the written data, or the
  GIF stays mid-packet and every later transfer desyncs.
- OSD render structure: context 1 draws the 2D/text layer straight
  into the displayed buffer (FBP 0), context 0 draws into an offscreen
  buffer (FBP 0x46), PMODE=0x66 (read circuit 2 only, DISPFB2 -> FBP
  0). ~2000 textured prims/frame keep flowing even while the boot sits
  on its first interactive screen (no pad input, zeroed NVRAM).
- UNPACK input length depends on STCYCL (wl > cl row-fills whole
  writes) and, with the m flag, on STMASK (codes != 0 take no input) —
  getting either wrong desyncs the whole command stream.

## CDVD osdconfig (what makes the OSD skip first-boot setup)

Wire format decoded by PS2BiosRebuild (see
`N:\PS2BiosRebuild` analysis; SCPH-50000): a config block is 15 data
bytes + a one-byte sum mod 256; CDVDMAN verifies the sum and never
looks inside. The OSD opens area (1, 0) with count 2 — wire triple
`[0, 1, 2]` — reads two blocks, and retries open/read/close while the
returned status has bit 0x01 or 0x80 set.

- Block 0 is copied out uninterpreted. Block 1 carries the fields:
  byte +0 top three bits non-zero selects the "new generation" layout
  where byte +1 bits 0-4 are the language index; that index subscripts
  an 8-entry table with **no bounds check** (out of range = null string
  table = the OSD silently draws nothing — the old "fabricated config
  draws nothing" mystery).
- **Byte +2 bit 7 is the "configured" flag** (found by disassembling
  the expanded OSDSYS decoder at 0x203698; it returns the flag's
  inverse and never stores it in the config struct, which is why field
  sweeps missed it). Set: boot goes SCE splash -> PS logo -> browser
  menu. Clear: first-boot language setup.
- Byte +3 (low 8) + byte +2 bits 0-2 (high 3) = timezone offset in
  minutes; byte +2 bit 3 = +1h DST, bit 4 = 12h clock.

## GS lessons from the OSD boot screens

- HOST->LOCAL IMAGE transfers in PSMCT24 are a **packed 3-bytes-per-
  pixel stream with no 64-bit alignment**; a partial pixel carries
  across HWREG chunks. Treating it as 32-bit shifts every pixel's
  channels — the OSD's PS-logo frames (135x97, uploaded per frame)
  render as RGB-striped noise.
- CSM1 CLUTs live in VRAM as a 16x16 PSMCT32 image whose entry order
  has bits 3 and 4 swapped (8x2-entry tiles). The OSD uploads 16-entry
  font CLUTs as literal 8x2 IMAGE transfers, which lands entries 8-15
  one buffer row (64 words) down — a linear "row of 256" CLUT read
  returns garbage for them.
- The OSD runs the display interlaced with half-height field buffers
  (SMODE2 INT+FFMD, FRAME/DISPFB alternating FBP 0 and 0x46 per
  field, 640x224 each). Scanout must line-double one field to the
  448-line display; reading 448 consecutive lines walks into the
  other field's buffer and shows everything twice.
- OSD glyphs are PSMT4 atlases (font at one block-aligned-ish base,
  e.g. block 12037, TBW 4 = 256 px) with grayscale alpha-ramp CLUTs.

## SIO2 memory-card transfers (MCMAN's shapes)

- MCMAN drives SIO2 with a **block per sub-transfer**: DMA ch11/12 use
  BCR block size 0x24 words (144 bytes) x N blocks, each sub-command
  at its block's start and each reply padded to a full block. PADMAN
  instead packs its sub-transfers back to back in one block — the
  BCR block count tells them apart.
- DMA ch12 (SIO2out) is armed before CTRL starts the transfer; the
  hardware paces it with DRQ, an instant model must defer the copy.
- MCMAN checks one status word per transfer group: RECV1 must say
  "present" (0x1100) if ANY sub-transfer reached a device.
- Card reply conventions: 0x2B acknowledge; a settable terminator
  byte (0x27 sets it — reply carries the OLD one, so MCMAN issues it
  twice and checks the second); 0x66 = busy/absent marker everywhere
  (a multitap probe hitting the card must get 0x66 at reply byte 5 or
  XSIO2MAN invents a phantom tap with a nonsense slot). MagicGate
  auth (0x81 0xF0 sub): card-responds subs ack at reply[3] with 8
  data bytes and their XOR at [12]; console-sends subs (06/07/0B)
  just ack at the tail. SECRMAN delegates the crypto checks to the
  mechacon over S commands 0x80-0x8F — status-0 replies with zeroed
  16-byte challenges from 0x84/0x85 satisfy it.
- Page reads run SetReadSector (0x23, page number + XOR) then 0x43
  chunks of 128 + a 16-byte tail = 528 bytes per page (data + ECC),
  finished with ReadWriteEnd (0x81).

## CDVD disc reads and the disc key

- N 0x06/0x08 params: LSN and count as LE u32 at bytes 0-3/4-7.
  DvdRead (0x08) returns 2064-byte raw sectors: 12-byte header with
  the physical sector number (LBA + 0x30000), 2048 data, 4-byte EDC.
- IOP DMA ch3 (0x1F8010B0) drains the sector data and is armed before
  the N command — defer it or cdvdman reads zeros.
- **N 0x0C (sceCdReadKey)** must return a real key or the OSD rejects
  the disc with the red screen. The mechacon derives it from the boot
  serial (4 letters + 5 digits, e.g. SLPS-25918): with n = the digits
  as a number and l = the four letters' low 7 bits packed high-to-low,
  key[0..4] = LE32((n & 0x1FC00) >> 10 | (l & 0x1FFFFFF) << 7),
  key[4] = (n & 0x1F) << 3 | (l & 0xE000000) >> 25, and for command
  arg 75 key[14] = (n & 0x3E0) >> 2 | 0x04 (PCSX2's cdvdReadKey).
  cdvdman reads the key through the XOR-obfuscated register window:
  banks 0x2020-0x2024 / 0x2028-0x202C / 0x2030-0x2034 (5 bytes each,
  XORed with reg 0x2039), validity bits in 0x2038 (0x07 = all three),
  decrypt flag in 0x203A (0x05 for arg 75, else 0x01).
- With key and reads working, the boot chain runs: the OSD prints
  `ExecutePs2GameDisk`, walks the ISO (PVD sector 16, path table,
  directory, SYSTEM.CNF) and LoadExec restarts the kernel.
- **Drive status (reg 0x200A) must read 0x0A (PAUSE) when idle with a
  disc.** cdvdman's sceCdDiskReady polls that register and waits for
  exactly 0x0A (code at module offset ~0x1ef78 in this build: reads
  0xBF40200A, compares with 0x0A, blocking mode loops on an event flag
  until it matches, non-blocking returns 6 "not ready"). The OSD never
  tripped on this because ExecutePs2GameDisk reads without DiskReady.
- **DEC-SET (reg 0x203A, IOP-writable) arms drive-side decryption of
  the DMA'd sector data**: bit 0 = XOR each byte with disc key byte 4,
  bit 1 = rotate right by bits 4-6 (PCSX2's mechaDecryptBytes). The PS2
  logo area (lsn 0-11) is stored encrypted on disc (an all-0xF5 sector
  0 is XOR-key 0xF5 over zero padding); cdvdman writes 0x53 before
  PS2LOGO's 12-sector read and 0x00 after. Without this, PS2LOGO's
  logo check fails and it silently execs **rom0:OSDSYS instead of the
  game** — the boot "stall" was really the relaunched browser retrying
  LoadModule of the (absent, 0xFF) rom1 DVD-player modules forever.
  Debug hazards that mimicked real blockers: rmman2 polls S 0x1E every
  ~55 ms and MCMAN re-probes cards with mechacon MG groups (S 0x80-0x8F)
  on a seconds cadence — neither is boot-path traffic.
- EELOAD reads the boot ELF with N 0x06 for the first sectors, then
  switches to raw DVD N 0x08 for the bulk (2064-byte framing).

## EE kernel LoadExecPS2 path (game ELF boot)

- The LoadExecPS2 syscall handler (kernel 0x5744 area, this ROM) calls
  its Restart routine (prints `# Restart.` ... `# Restart Done.`),
  copies EELOAD from ROM to 0x00082000, sets EPC via `mtc0` and
  transfers control with **`eret` while neither Status.EXL nor ERL is
  set**. The R5900 eret still jumps to EPC in that state (ERL selects
  ErrorEPC, otherwise EPC — there is no "neither" fallthrough). An
  emulator that treats flagless eret as a no-op falls through into the
  handler's own epilogue, "returns" from the never-returning syscall
  into the just-cleared caller (OSDSYS at ~0x202aac), and the EE slides
  through zeroed RAM off the 32 MB end (the old pc 0x10000004 panic).
- Boot chain observed for a retail disc: OSDSYS `ExecutePs2GameDisk` →
  LoadExecPS2 → EELOAD (entry 0x82000, hit only via that eret) →
  chains rom0:PS2LOGO ("Restart Without Memory Clear", reads lsn 0-11)
  → LoadExecPS2(cdrom0:ELF) → EELOAD resets the IOP with
  "rom0:UDNL rom0:EELOADCNF" (twice, ~0.7s apart) → sceCdDiskReady →
  sceCdSearchFile/sceCdRead for the ELF.

## IOP silent reboot (sceSifIopReset)

- The EE sends SIFCMD cid 0x80000003 with the IOPRP argument string; the
  IOP reboots via UDNL without going through the ROM reset stub (no POST
  codes). In-flight SIF FIFO state must be discarded at that point or
  the new kernel's sifcmd handshake parses stale garbage and EELOAD
  retries forever.
- ROM1 (DVD player ROM, 0x1E000000 on both buses) should read like
  erased flash (0xFF) when absent.

## SIO2 (pads/memory cards)

- CTRL 0x1F808268: writing bit 0 starts a transfer; the bit must read
  back clear and I_STAT bit 17 must rise, or SIO2MAN spins forever.
  RECV1 (0x1F80826C) = 0x1D100 reports "no device" (0x1100 = present);
  RECV2 (0x1F808270) reads a constant 0xF.
- A transfer is described by SEND3 slots (0x1F808200+): port in bits
  0-1, byte length in bits 8-16; slot 0 terminates on the first zero.
  The first command byte selects the device class: 0x01 pad, 0x81
  memory card, 0x21 multitap. This BIOS's XSIO2MAN probes the multitap
  on ports 2/3 with `[21 12 ..]`/`[21 13 ..]` (len 6) and memory cards
  with `[81 11]`/`[81 52]`/`[81 F3]`; PADMAN polls `[01 42 00 00 00]`
  and settles for a digital pad after a short 0x43/0x45/0x4D config
  probe. A DualShock answering 0x42 with `FF 41 5A lo hi` (buttons
  active-low) is enough for continuous polling.
- **XSIO2MAN starts CTRL before the data arrives**: it kicks DMA ch11
  (SIO2in, 0x1F801540) *after* setting the start bit — the DMAC only
  moves data once DRQ asserts — and reads responses back with DMA ch12
  (SIO2out, 0x1F801550). It also writes the FIFO-reset bits (0x0C)
  *together with* the start bit in one CTRL write, so honoring the
  reset on that write throws away the DMA-delivered command bytes.
- With no device on a port, reads float high: respond 0xFF, not 0x00.

## VU0 macro mode (COP2)

- **The OSD transforms nearly all of its 3D geometry on the EE with
  VU0 macro instructions** and sends finished GIF packets via VIF1
  DIRECT (PATH2). Only ~7 UNPACKs and one MSCAL happen per boot —
  stubbing COP2 macro ops as NOPs makes every 3D vertex collapse to
  the matrix translation column (menu background invisible, browser
  black) while 2D text keeps working.
- COP2 macro instructions use the exact microcode field layout (dest
  21-24, ft 16-20, fs 11-15, fd 6-10, opcode 0-5), so a VU core can
  execute them directly. Dispatch: funct 0x30-0x35 are the lower
  integer ops (VIADD..VIOR); funct 0x3C-0x3F with op2 = (instr&3) |
  ((instr>>4)&0x7C) >= 0x30 are lower special2 (DIV, MOVE, MTIR,
  LQI, ...), everything else is the upper FMAC set. CFC2/CTC2 regs
  0-15 map to vi, 16 status, 17 mac, 18 clip, 20 R, 21 I, 22 Q;
  VPU-STAT (29) must read 0 ("never busy").

## OSD browser render pipeline

The browser scene is a feedback compositor: orbs and a full-res copy
of the previous screen accumulate in a feedback buffer (FBP 0xD2), a
full-screen sprite composites it back modulated by the background
tint, cloud strips (MODULATE of a 128x128 noise texture) draw over
it, and echo/zoom sprite passes through FBP 0x118 smear the result.
The clouds and the half-res downsample only run during the entry
transition; the steady state is the decayed feedback plus the orb
trail. With no disc and no memory cards the browser shows NO items,
so the near-black result is (close to) the authentic empty-browser
look — System Configuration renders its full 3D tower scene through
the same stack, which rules out a pipeline defect. Z-test was also
ruled out explicitly (forcing it off changes nothing).

## IOP load-delay pipeline (the "SCP" bug)

cdvdman copies S-command results with back-to-back `lwl`/`lwr` pairs.
The hardware forwards an in-flight (delay-slot) load to a following
lwl/lwr on the same register; an emulator that commits or hides the
pending load before executing the next instruction makes the second
half of the pair merge with the stale register value and silently
zeroes one byte per word. Symptom that found it: the OSD's Version
Information screen showed the console model as "SCP" — the model
string crossed the SIF as "SCP\0-50\0 00\0\0" (every 4th byte lost).
Any unaligned IOP memcpy hits the same path.

Related: this BIOS's Version screen reads the model via S command
0x17 (sceCdReadModelNumber): param = byte offset, result = [stat,
8 model chars], two calls (offsets 0 and 8). This cdvdman revision
also has an interrupt-driven S-command engine (a mailbox at ~0x3D81D
+ completion flag polled with DelayThread) and a register-window
result path (banks 0x2020-0x2034 XOR-obfuscated with 0x2039, valid
bits in 0x2038) that our FIFO-only model never triggers — worth
knowing if some path stops getting results.

## SPU2 transfer engine (what libspu2 / libsd wait on)

SLPS-25918's IOP sound module (`rspu2_driver`, a game-specific IRX
that embeds libspu2 + libsnd2 rather than LIBSD) polls the SPU2 like
this (disassembled from IOP RAM, `SPU:T/O [%s]` timeout strings mark
each wait):

- `SpuInit`: ATTR = 0, then 0x8000, then spin up to 0xF00 reads until
  `STATX & 0x7FF == 0` ("wait (reset)"). STATX must therefore echo the
  ATTR mode bits the PS1 way (bits 5:0 = ATTR 5:0, bit 7 = DMA request
  when a DMA mode is selected, 8/9 = read/write request, 10 = busy) —
  a constant "ready" value trips the timeout.
- Manual writes: up to 0x40 bytes are stored to STD (0x1AC) *before*
  ATTR mode is set to 1, then it spins until `STATX & 0x400` (busy)
  clears ("wait (SPU2_STATX_WRDY_M)"). So STD writes must land in RAM
  regardless of the current mode.
- Plain DMA (ch4 = core 0, ch7 = core 1; DMA regs at 0x1F8010C0 +
  core*0x440): BCR is written as two halfword stores (`sh 0x10` at
  +4, block count at +6) — sub-word DMA register access has to work.
  CHCR = 0x01000201. The completion handler for core 1 spins up to
  16M reads until `STATX & 0x80` is *set* ("wait (SPU2_STATX_DREQ)"),
  then clears ATTR bits 5:4 and waits for the readback to show 0.
- AutoDMA streaming (ADMAS = core bit, 1 KiB blocks: 512 bytes L then
  512 bytes R into the core's input area at halfword 0x2000 + core<<10,
  halves alternating): the DMA completion interrupt re-arms the next
  block from inside the handler. The block is consumed at 48 kHz (256
  stereo samples = 5.33 ms per KiB), and the completion IRQ must not
  fire before that — an instant completion turns the handler into an
  IOP interrupt storm that starves the RPC thread, which is what the
  "game hangs on its sound-init thread" symptom was.
  The ring must be written in stream order with a write cursor that
  alternates halves, starting at half 0 with the read position reset
  to 0 when ADMAS is switched on, and the IOP's completion must be
  tied to the moment the last block actually lands (each waiting block
  needs one more half boundary of playback). Filling "whichever half
  is free, the non-playing one first" swapped each 2 KiB kick's blocks
  and overwrote the playing half at a slowly sliding offset, and a
  byte-count completion time let the IOP's re-arm drift half a sample
  per kick: together a constant "trembling" smear under the music.
  Verified against the IOP's own decode: the game streams MUSIC.AFS
  ADX (CRI type-8 encrypted, vgmstream key "mituba" = 0x5a17/0x509f/
  0x5bfd, key stream advancing per frame across channels) through
  CRI_ADXI.IRX at volume 0x3D80/0x8000, and the SPU2 output now equals
  that stream to within BVOL rounding (`tools/adxcmp.py` decodes the
  track and reports gain/residual per window against a `--wav` capture).
- Voices interpolate with the 4-tap Gaussian table (phase = pitch
  counter bits 4..11, weights oldest-first, output centred two samples
  behind the newest); nearest sampling left strong aliasing above 8 kHz
  on the boot chime.
- Reverb: per-core ESA (0x2E0 H/L) .. EEA (0x33C, low halfword implied
  0xFFFF), the 22 address registers at 0x2E4.. in nocash's PS1 order as
  20-bit halfword offsets relative to ESA (the PS1 presets times four:
  the OSD's hall has dAPF1 = 0x38C = PS1 0xE3 * 4), coefficients vIIR..
  vRIN at 0x774 (+0x28 for core 1), EVOL at 0x764/0x78C, ATTR bit 7
  enables. MMIX routes per channel: bits 11/10 voice dry, 9/8 voice
  wet (VMIXEL/VMIXER select voices), 7/6 input dry, 5/4 input wet,
  3/2 external (core 0 -> core 1, AVOL) dry, 1/0 external wet; core 1's
  MVOL is the final one. Amagami uses MMIX 0x0FFC on core 1 — its ADX
  music (core 0 ADMA) stays dry, only core 1's voices get the hall
  (ESA 0xEDBE0, EVOL 0x3FFF); the OSD fades EVOL in under the chime.
- The OSD stops a voice by writing ADSR1/ADSR2 = 0 and then keying it
  off: release shift 0 drops the envelope to zero in two samples, so
  when a disc is present the SCE chime is cut hard at 5.0 s (right
  before the IOP reboot for the game), which is intended behaviour, not
  an underrun.
- IRQs: intrman 0x24/0x28 (DMA ch4/ch7), 9 (SPU IRQA). libspu2 also
  toggles ATTR bit 6 to arm/clear the IRQ ("wait (IRQ/ON)" / "IRQ/OFF"
  read the bit back).

## GS notes from Amagami (SLPS-25918)

- Its OP parks the 8-bit movie textures (PSMT8, bp 4480, dbw 10/8,
  re-uploaded every frame with a fresh 16x16 CSM1 CLUT) on top of the
  Z buffer (ZBUF bp 4480, PSMZ24, ZMSK=1) and still Z-tests full-screen
  copies at z = 0xFFFFFF with GEQUAL. Only a 24-bit compare passes:
  the upper byte of each word holds texture data.
- "NOW LOADING" text and its spinner are drawn from the display
  buffer's alpha plane: PSMT8H IMAGE uploads into bp 0 (dbw 10) at
  (0,0), CLUT re-uploaded to bp 10108 before each draw.
- 640x224 field buffers at bp 0 / 2240 (double buffered), extra
  buffers at 6720 / 8960 filled early by full-screen textured sprite
  copies of 2240 (a crossfade capture: at that moment 2240 still holds
  the PS2 logo). During the OP the particle triangles sample 6720/8960
  and 0/2240 (refraction look-ups with STQ) plus small PSMCT16/32
  textures at bp 11552/11488/11456/12384/12320.
- CLUT packing needs the real VRAM layout: 32-bit CSM1 CLUTs live at
  bp 10080/10092/10100/10103/10108/10120 — 3..12 blocks apart. On
  hardware a 16x16 PSMCT32 image is exactly four consecutive blocks
  (block table row 0: 0 1, row 1: 2 3), so these never overlap; a
  linear "one row per block" model made 10108 span 16 blocks and every
  palette upload clobbered its neighbours (glyphs came out on opaque
  boxes). Text glyphs: PSMT8H at bp 0 (tbw 10) with cbp 10108/10120,
  cld=1, index 0x40.. = coverage; the same CLUT slot is re-uploaded per
  use, so the CLUT cache is keyed by CBP/CPSM/CSM/CSA/TEXA and flushed
  on any transfer.
- The game idles in the EE kernel idle thread (`b .-32` over nops at
  0x81fc0) ~62% of the time and the IOP in `j .; nop` (0xae94) ~92%;
  both are skipped until an interrupt is pending without changing
  emulated timing (frames are bit-identical).

