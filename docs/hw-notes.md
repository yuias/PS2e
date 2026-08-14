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
  RECV1 (0x1F80826C) = 0x1D100 reports "no device"; RECV2 (0x1F808270)
  reads a constant 0xF; the out-FIFO (0x1F808264) reads 0xFF.
