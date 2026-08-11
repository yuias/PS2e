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
