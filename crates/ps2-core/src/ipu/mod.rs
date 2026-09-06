//! IPU: the image processing unit, an MPEG-2 macroblock decoder with its own
//! bitstream reader.
//!
//! The register file, the two FIFOs and the bit reader are here, along with
//! every command: the ones that only move bits (`BCLR`, `FDEC`, `SETIQ`,
//! `SETVQ`, `SETTH`), the variable-length decoder the software drives one
//! code at a time (`VDEC`), the block decoders (`BDEC` for one macroblock's
//! coefficients, `IDEC` for a whole intra slice through to pixels) and the
//! pixel converters (`CSC`, `PACK`). The code tables live in [`vlc`], the
//! transform and colour conversion in [`pixel`].
//!
//! The bitstream is read most-significant bit first out of quadwords in the
//! order they arrive from memory, so the FIFO stores plain byte arrays and the
//! reader indexes them as one flat bit string.
//!
//! A decode command can run out of bitstream anywhere inside a macroblock.
//! Rather than checkpoint the decoder's every local, each command is cut into
//! units (a macroblock, or the whole of a short command) that run inside a
//! transaction on the reader: a unit that starves is rolled back and retried
//! from its start when the FIFO is fed, and a unit that completes commits.

mod pixel;
mod vlc;

use pixel::{Convert, Samples};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, warn};

/// Quadwords the hardware's FIFOs hold. Ours are allowed to grow past this
/// so a DMA can complete inside the write that starts it; `IFC` and `OFC`
/// still report hardware-shaped counts.
const FIFO_DEPTH: usize = 8;

/// Bits in one quadword.
const QW_BITS: u32 = 128;

/// `IPU_CTRL` fields the register write leaves alone (the live status half)
/// and the ones it sets.
const CTRL_KEEP: u32 = 0x8000_FFFF;
const CTRL_WRITE: u32 = 0x47F3_0000;
const CTRL_RESET: u32 = 1 << 30;
const CTRL_BUSY: u32 = 1 << 31;
/// The decoder's status and picture parameters in `IPU_CTRL`: error code
/// and start code detected, intra DC precision, alternate scan, intra VLC
/// format, quantiser scale type, MPEG-1 and picture coding type.
const CTRL_CBP_SHIFT: u32 = 8;
const CTRL_ECD: u32 = 1 << 14;
const CTRL_SCD: u32 = 1 << 15;
const CTRL_IDP_SHIFT: u32 = 16;
const CTRL_AS: u32 = 1 << 20;
const CTRL_IVF: u32 = 1 << 21;
const CTRL_QST: u32 = 1 << 22;
const CTRL_MP1: u32 = 1 << 23;
const CTRL_PCT_SHIFT: u32 = 24;

/// The busy marker in the high word of `IPU_CMD` and `IPU_TOP`.
const BUSY32: u32 = 0x8000_0000;

/// Coefficient scan orders (ISO/IEC 13818-2 7.3): scan position to natural
/// `row * 8 + column` index.
const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
    41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23,
    30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
/// Natural index to zigzag position, for weights that arrive scanned.
const INV_ZIGZAG: [usize; 64] = {
    let mut inv = [0; 64];
    let mut i = 0;
    while i < 64 {
        inv[ZIGZAG[i]] = i;
        i += 1;
    }
    inv
};
const ALT_SCAN: [usize; 64] = [
    0, 8, 16, 24, 1, 9, 2, 10, 17, 25, 32, 40, 48, 56, 57, 49, 41, 33, 26, 18, 3, 11,
    4, 12, 19, 27, 34, 42, 50, 58, 35, 43, 51, 59, 20, 28, 5, 13, 6, 14, 21, 29, 36, 44,
    52, 60, 37, 45, 53, 61, 22, 30, 7, 15, 23, 31, 38, 46, 54, 62, 39, 47, 55, 63,
];

/// `quantiser_scale` for `q_scale_type = 1` (Table 7-6), by code.
const NON_LINEAR_SCALE: [i32; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48,
    52, 56, 64, 72, 80, 88, 96, 104, 112,
];

/// Why a decode unit stopped short.
#[derive(Debug, PartialEq)]
enum Halt {
    /// The FIFO ran dry; retry when it is fed.
    Starved,
    /// The bitstream held no valid code; `IPU_CTRL.ECD` reports it.
    Error,
}

/// What a [`Halt::Error`] was decoding when it gave up. A failed VLC decode
/// is nearly always a mistyped table row or a lost bit, and neither can be
/// told from the other without the table, the position and the bits, so
/// every error path records them for the warning [`Ipu::stepped`] prints.
#[derive(Clone, Copy, Default)]
struct Fault {
    /// The table whose lookup failed, or the rule that rejected the value.
    what: &'static str,
    /// Bit pointer into the window, and bits consumed since power-on.
    bp: u32,
    at: u64,
    /// The bits under the pointer, right-aligned, and how many are real
    /// (the FIFO may hold fewer than the 32 wanted).
    next: u32,
    have: u32,
}

type Step<T> = Result<T, Halt>;

/// Whether a command writes to the output FIFO: IDEC, BDEC, CSC, PACK.
/// The others (BCLR, VDEC, FDEC and the setup commands) only read.
fn emits(val: u32) -> bool {
    matches!(val >> 28, 0x1 | 0x2 | 0x7 | 0x8)
}

/// The reader state a rolled-back unit restores.
#[derive(Clone, Default)]
struct Checkpoint {
    window: [[u8; 16]; 2],
    fp: u32,
    bp: u32,
    advanced: u64,
    dc_pred: [i32; 3],
    qsc: u32,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Ipu {
    /// Input FIFO, quadwords in arrival order.
    fifo_in: VecDeque<[u8; 16]>,
    /// Output FIFO: decoded macroblocks, drained by DMA channel 3 or reads
    /// of `IPU_OUT_FIFO`.
    fifo_out: VecDeque<[u8; 16]>,
    /// The reader's two-quadword window, `fp` of them valid, `bp` bits into
    /// the first.
    window: [[u8; 16]; 2],
    fp: u32,
    bp: u32,
    /// Bits consumed since power-on; differences of it are code lengths.
    advanced: u64,
    /// `IPU_CTRL`, status half included.
    ctrl: u32,
    /// `IPU_CMD`: the last decode result and whether one is still running.
    cmd_data: u32,
    cmd_busy: bool,
    /// `IPU_TOP`: the 32 bits the bit pointer sits on, and its own busy flag.
    top: u32,
    top_busy: bool,
    /// Quantiser matrices (`SETIQ`), in the zigzag order they arrive, and
    /// the vector-quantiser CLUT (`SETVQ`).
    iq: Vec<u8>,
    niq: Vec<u8>,
    vqclut: Vec<u8>,
    /// `SETTH` thresholds.
    thresh: [u16; 2],
    /// DC predictors of the three components, carried between `BDEC`s.
    dc_pred: [i32; 3],
    /// `quantiser_scale_code` in force: from the command word, then from
    /// any macroblock `IDEC` meets that carries its own.
    qsc: u32,
    /// A command that ran out of bitstream, with how far it got. It resumes
    /// when the FIFO is fed.
    pending: Option<(u32, usize)>,
    /// The transaction a decode unit runs in: where to roll back to, and the
    /// quadwords it pulled from the FIFO that a rollback must put back.
    /// Never live across calls, so save states need not carry them.
    #[serde(skip)]
    ckpt: Checkpoint,
    #[serde(skip)]
    taken: Vec<[u8; 16]>,
    #[serde(skip)]
    tx: bool,
    /// Where the last [`Halt::Error`] came from. Diagnostic only.
    #[serde(skip)]
    fault: Option<Fault>,
    /// A command that finished inside `pop_out`, waiting to be reported.
    /// Never live across a bus access, so save states need not carry it.
    #[serde(skip)]
    drained_done: bool,
}

impl Ipu {
    pub fn new() -> Self {
        Self { iq: vec![0; 64], niq: vec![0; 64], vqclut: vec![0; 32], ..Self::default() }
    }

    /// Quadwords the software may see in each FIFO.
    fn ifc(&self) -> u32 {
        self.fifo_in.len().min(FIFO_DEPTH) as u32
    }

    fn ofc(&self) -> u32 {
        self.fifo_out.len().min(FIFO_DEPTH) as u32
    }

    // --- bit reader ------------------------------------------------------

    /// Take the next quadword out of the FIFO, remembering it if a unit may
    /// have to give it back.
    fn pull(&mut self) -> Option<[u8; 16]> {
        let q = self.fifo_in.pop_front()?;
        if self.tx {
            self.taken.push(q);
        }
        Some(q)
    }

    /// Pull quadwords in until `bits` more are readable. False means the
    /// FIFO ran dry and the caller must wait for more data.
    fn fill(&mut self, bits: u32) -> bool {
        while self.fp * QW_BITS < self.bp + bits {
            match self.pull() {
                Some(q) => {
                    self.window[self.fp as usize] = q;
                    self.fp += 1;
                }
                None => return false,
            }
        }
        true
    }

    /// Step the bit pointer, sliding the window when it leaves the first
    /// quadword. False means there was not enough data to step over.
    fn advance(&mut self, bits: u32) -> bool {
        if !self.fill(bits) {
            return false;
        }
        self.bp += bits;
        self.advanced += u64::from(bits);
        if self.bp >= QW_BITS {
            self.bp -= QW_BITS;
            if self.fp == 2 {
                self.window[0] = self.window[1];
                self.fp = 1;
            } else {
                match self.pull() {
                    Some(q) => {
                        self.window[0] = q;
                        self.fp = 1;
                    }
                    None => self.fp = 0,
                }
            }
        }
        true
    }

    /// One bit of the window, counted from the bit pointer.
    fn bit(&self, i: u32) -> u32 {
        let at = self.bp + i;
        let byte = self.window[(at / QW_BITS) as usize][((at % QW_BITS) / 8) as usize];
        u32::from(byte >> (7 - (at & 7)) & 1)
    }

    /// The next `n` bits without consuming them, most significant first.
    fn peek(&mut self, n: u32) -> Option<u32> {
        if !self.fill(n) {
            return None;
        }
        Some((0..n).fold(0u32, |acc, i| (acc << 1) | self.bit(i)))
    }

    /// Read one byte and consume it.
    fn take_byte(&mut self) -> Option<u8> {
        let v = self.peek(8)? as u8;
        self.advance(8);
        Some(v)
    }

    /// The decoder's reads: the same operations, starvation as an error so
    /// a unit can bail with `?`.
    fn look(&mut self, n: u32) -> Step<u32> {
        self.peek(n).ok_or(Halt::Starved)
    }

    fn bits(&mut self, n: u32) -> Step<u32> {
        let v = self.look(n)?;
        self.advance(n);
        Ok(v)
    }

    fn skip(&mut self, n: u32) -> Step<()> {
        if self.advance(n) { Ok(()) } else { Err(Halt::Starved) }
    }

    fn bytes(&mut self, buf: &mut [u8]) -> Step<()> {
        for b in buf {
            *b = self.bits(8)? as u8;
        }
        Ok(())
    }

    /// Decode one code with `t`. An invalid pattern is an error.
    fn vlc(&mut self, t: &vlc::Table) -> Step<u16> {
        let c = t.lookup(self.look(t.bits)?);
        if c.len == 0 {
            return Err(self.fail(t.name));
        }
        self.advance(u32::from(c.len));
        Ok(c.val)
    }

    /// Note where the bitstream stopped making sense and hand back the
    /// error to propagate. Peeking cannot move the bit pointer, so this is
    /// safe to call from anywhere inside a unit.
    fn fail(&mut self, what: &'static str) -> Halt {
        let mut have = 32;
        while have > 0 && self.peek(have).is_none() {
            have -= 8;
        }
        let next = self.peek(have).unwrap_or(0);
        self.fault = Some(Fault { what, bp: self.bp, at: self.advanced, next, have });
        Halt::Error
    }

    // --- transactions ----------------------------------------------------

    fn begin(&mut self) {
        self.ckpt = Checkpoint {
            window: self.window,
            fp: self.fp,
            bp: self.bp,
            advanced: self.advanced,
            dc_pred: self.dc_pred,
            qsc: self.qsc,
        };
        self.taken.clear();
        self.tx = true;
    }

    fn commit(&mut self) {
        self.taken.clear();
        self.tx = false;
    }

    fn rollback(&mut self) {
        for q in self.taken.drain(..).rev() {
            self.fifo_in.push_front(q);
        }
        let c = std::mem::take(&mut self.ckpt);
        self.window = c.window;
        self.fp = c.fp;
        self.bp = c.bp;
        self.advanced = c.advanced;
        self.dc_pred = c.dc_pred;
        self.qsc = c.qsc;
        self.tx = false;
    }

    /// Drive a command made of units: `unit(progress)` decodes one and says
    /// whether it was the last. A unit that starves is rolled back and the
    /// command parked at it; one that finds no valid code ends the command
    /// with `ECD` set. Returns whether the command completed.
    fn stepped(&mut self, val: u32, mut progress: usize, unit: fn(&mut Self, u32, usize) -> Step<bool>) -> bool {
        if progress == 0 {
            self.ctrl &= !(CTRL_ECD | CTRL_SCD);
        }
        loop {
            // The output FIFO is eight quadwords deep on hardware and a
            // full one holds the decoder until channel 3 (or a programmed
            // read) drains it. Without that back-pressure a CSC of a
            // thousand macroblocks finishes inside the write that starts
            // it, and a player that paces itself on the drain swallows the
            // whole stream in one burst. A unit is emitted whole, so the
            // check is between units, not inside one -- and only for the
            // commands that emit at all: VDEC produces no output, so
            // parking it on a full FIFO would deadlock a decode that
            // hardware lets straight through.
            if emits(val) && self.fifo_out.len() >= FIFO_DEPTH {
                self.stall(val, progress);
                return false;
            }
            self.begin();
            match unit(self, val, progress) {
                Ok(last) => {
                    self.commit();
                    if last {
                        return true;
                    }
                    progress += 1;
                }
                Err(Halt::Starved) => {
                    self.rollback();
                    self.stall(val, progress);
                    return false;
                }
                Err(Halt::Error) => {
                    self.commit();
                    self.ctrl |= CTRL_ECD;
                    let f = self.fault.take().unwrap_or_default();
                    warn!(target: "ps2_core::bus::ipu",
                        cmd = format_args!("{:#x}", val >> 28),
                        word = format_args!("{val:#010x}"), unit = progress,
                        table = f.what, bp = f.bp, at = f.at,
                        next = format_args!("{:0width$b}", f.next, width = f.have as usize),
                        ctrl = format_args!("{:#010x}", self.ctrl),
                        "no valid code in the bitstream");
                    return true;
                }
            }
        }
    }

    // --- registers -------------------------------------------------------

    pub fn read32(&mut self, addr: u32) -> u32 {
        match addr & 0x3C {
            0x00 => self.cmd_data,
            0x04 => u32::from(self.cmd_busy) * BUSY32,
            0x10 => self.ctrl(),
            0x20 => (self.bp & 0x7F) | (self.ifc() << 8) | (self.fp << 16),
            0x30 => self.top,
            0x34 => u32::from(self.top_busy) * BUSY32,
            _ => 0,
        }
    }

    pub fn read64(&mut self, addr: u32) -> u64 {
        u64::from(self.read32(addr)) | u64::from(self.read32(addr | 4)) << 32
    }

    /// `IPU_CTRL` with its live status fields folded in.
    fn ctrl(&self) -> u32 {
        let busy = u32::from(self.cmd_busy || self.pending.is_some()) * CTRL_BUSY;
        (self.ctrl & !(0xFF | CTRL_BUSY)) | self.ifc() | self.ofc() << 4 | busy
    }

    /// Returns true when a command finished, which raises the IPU interrupt.
    pub fn write32(&mut self, addr: u32, v: u32) -> bool {
        match addr & 0x3C {
            0x00 => return self.command(v),
            0x10 => {
                // RST is a strobe: it acts on the write and does not stay
                // set. Players read `IPU_CTRL` back and write it again to
                // change one field, so a stored RST would soft-reset the
                // IPU on every such write and throw the bitstream away in
                // the middle of a picture.
                self.ctrl = (v & CTRL_WRITE & !CTRL_RESET) | (self.ctrl & CTRL_KEEP);
                // The picture parameters the decoder runs on arrive here and
                // nowhere else, so a stream that decodes to garbage is
                // diagnosed from this line first.
                debug!(target: "ps2_core::bus::ipu",
                    write = format_args!("{v:#010x}"),
                    ctrl = format_args!("{:#010x}", self.ctrl), "IPU_CTRL");
                if v & CTRL_RESET != 0 {
                    self.reset();
                }
            }
            _ => {}
        }
        false
    }

    pub fn write64(&mut self, addr: u32, v: u64) -> bool {
        self.write32(addr, v as u32) | self.write32(addr | 4, (v >> 32) as u32)
    }

    /// `BCLR`: drop the bitstream and whatever was waiting on it.
    fn bclr(&mut self) {
        self.fifo_in.clear();
        self.window = [[0; 16]; 2];
        self.fp = 0;
        self.bp = 0;
        self.pending = None;
        self.cmd_busy = false;
        self.top_busy = false;
    }

    /// The soft reset `IPU_CTRL.RST` asks for: both FIFOs and the status.
    fn reset(&mut self) {
        self.bclr();
        self.fifo_out.clear();
        self.ctrl &= !(CTRL_ECD | CTRL_SCD | 0x3F << CTRL_CBP_SHIFT);
        self.top = 0;
    }

    // --- FIFOs -----------------------------------------------------------

    /// Whether the input FIFO holds its full hardware depth. Channel 4
    /// stalls here rather than running the whole source chain into it:
    /// MPEG players read `D4_MADR`/`D4_QWC` back as their position in the
    /// stream and rewind it by `IFC + FP` quadwords to re-feed what a
    /// `BCLR` threw away, so a channel that has already swallowed the
    /// file tells them the stream ended.
    pub fn in_full(&self) -> bool {
        self.fifo_in.len() >= FIFO_DEPTH
    }

    /// Whether a command is waiting for more bitstream. A unit runs in a
    /// transaction and a starved one rolls back, so a unit wider than the
    /// FIFO would deadlock against [`Ipu::in_full`]; channel 4 keeps
    /// feeding while this holds.
    pub fn starved(&self) -> bool {
        self.pending.is_some()
    }

    /// Feed one quadword to the input FIFO, from DMA channel 4 or a
    /// programmed write. Returns true when it let a stalled command finish.
    pub fn push_in(&mut self, q: [u8; 16]) -> bool {
        self.fifo_in.push_back(q);
        self.resume()
    }

    /// Take the oldest decoded quadword, for DMA channel 3 or a programmed
    /// read of the output FIFO.
    pub fn pop_out(&mut self) -> Option<[u8; 16]> {
        let was_full = self.fifo_out.len() >= FIFO_DEPTH;
        let q = self.fifo_out.pop_front();
        // Room again: a decoder parked on a full FIFO carries on, which is
        // what keeps a long command producing while channel 3 consumes.
        if was_full && q.is_some() && self.pending.is_some() {
            self.drained_done |= self.resume();
        }
        q
    }

    /// Whether a command completed inside [`Ipu::pop_out`] since this was
    /// last asked. The caller raises the interrupt; the FIFO cannot.
    pub fn took_done(&mut self) -> bool {
        std::mem::take(&mut self.drained_done)
    }

    /// Queue decoded data, a whole number of quadwords.
    fn emit(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len() % 16, 0);
        self.fifo_out.extend(bytes.as_chunks::<16>().0);
    }

    /// Retry the command that ran out of bitstream.
    fn resume(&mut self) -> bool {
        match self.pending {
            Some((val, progress)) => self.run(val, progress),
            None => false,
        }
    }

    // --- commands --------------------------------------------------------

    fn command(&mut self, val: u32) -> bool {
        self.cmd_busy = true;
        self.run(val, 0)
    }

    /// Execute `val`, picking up at `progress` (the meaning of which is the
    /// command's own). Returns true when it completed.
    fn run(&mut self, val: u32, progress: usize) -> bool {
        let op = val >> 28;
        let skip = val & 0x3F;
        let done = match op {
            // BCLR: drop the bitstream and restart at the given bit.
            0x0 => {
                self.bclr();
                self.bp = val & 0x7F;
                true
            }
            0x1 => self.stepped(val, progress, Self::idec),
            0x2 => self.stepped(val, progress, Self::bdec),
            0x3 => self.stepped(val, progress, Self::vdec),
            // FDEC: skip, then hand back the 32 bits under the pointer
            // without consuming them.
            0x4 => {
                if progress == 0 && !self.advance(skip) {
                    self.stall(val, 0);
                    return false;
                }
                match self.peek(32) {
                    Some(v) => {
                        self.cmd_data = v;
                        self.top = v;
                        true
                    }
                    None => {
                        self.stall(val, 1);
                        return false;
                    }
                }
            }
            // SETIQ / SETVQ: pull a table straight out of the bitstream.
            // Bit 27 of SETIQ picks the non-intra matrix.
            0x5 | 0x6 => {
                let len = if op == 0x5 { 64 } else { 32 };
                if progress == 0 && op == 0x5 && !self.advance(skip) {
                    self.stall(val, 0);
                    return false;
                }
                let mut i = progress.max(1) - 1;
                while i < len {
                    match self.take_byte() {
                        Some(b) => {
                            match op {
                                0x5 if val & (1 << 27) != 0 => self.niq[i] = b,
                                0x5 => self.iq[i] = b,
                                _ => self.vqclut[i] = b,
                            }
                            i += 1;
                        }
                        None => {
                            self.stall(val, i + 1);
                            return false;
                        }
                    }
                }
                true
            }
            0x7 => self.stepped(val, progress, Self::csc),
            0x8 => self.stepped(val, progress, Self::pack),
            // SETTH: two thresholds, straight from the command word.
            0x9 => {
                self.thresh = [(val & 0x1FF) as u16, ((val >> 16) & 0x1FF) as u16];
                true
            }
            _ => {
                warn!(target: "ps2_core::bus::ipu",
                    cmd = format_args!("{op:#x}"), "unknown command");
                true
            }
        };
        if done {
            debug!(target: "ps2_core::bus::ipu",
                cmd = format_args!("{op:#x}"), word = format_args!("{val:#010x}"),
                data = format_args!("{:#010x}", self.cmd_data),
                bp = self.bp, at = self.advanced,
                ifc = self.ifc(), ofc = self.fifo_out.len(), "command done");
            self.pending = None;
            self.cmd_busy = false;
            self.top_busy = false;
        }
        done
    }

    /// Park a command that ran out of bitstream; `push_in` resumes it.
    fn stall(&mut self, val: u32, progress: usize) {
        self.pending = Some((val, progress));
        self.cmd_busy = true;
        self.top_busy = matches!(val >> 28, 0x1 | 0x3 | 0x4);
    }

    // --- the decoder -----------------------------------------------------

    /// `VDEC`: one macroblock-layer code, chosen by bits 27..26 (address
    /// increment, macroblock type, motion code, dmvector). `IPU_CMD` gets
    /// the value in its low half and the code's length in its high half;
    /// `IPU_TOP` the 32 bits that follow.
    fn vdec(&mut self, val: u32, _: usize) -> Step<bool> {
        self.skip(val & 0x3F)?;
        let start = self.advanced;
        let v = match (val >> 26) & 3 {
            0 => {
                // A slice ends where the next `macroblock_address_increment`
                // would start. No code in any table is 23 zeros, so that
                // pattern is the zero padding and prefix of the next start
                // code (ISO/IEC 13818-2 6.2.4) rather than a bad code: the
                // pointer stays on it and `SCD` reports it, not `ECD`.
                // Players read this back to end the slice, and one told the
                // code was invalid instead rescans byte-wise from here and
                // walks straight past the start code, losing a whole slice.
                if self.look(23)? == 0 {
                    self.ctrl |= CTRL_SCD;
                    self.cmd_data = 0;
                    self.top = self.look(32)?;
                    return Ok(true);
                }
                u32::from(self.vlc(&vlc::MBA)?)
            }
            1 => {
                // PCT 0 is treated as an I-picture, as the hardware does
                // for software that never set it.
                let t = match (self.ctrl >> CTRL_PCT_SHIFT) & 7 {
                    0 | 1 => &*vlc::MBT_I,
                    2 => &*vlc::MBT_P,
                    3 => &*vlc::MBT_B,
                    4 => &*vlc::MBT_D,
                    _ => return Err(self.fail("picture coding type")),
                };
                u32::from(self.vlc(t)?)
            }
            2 => {
                // Table B.10 is B.1 with the sign folded into the last bit:
                // increment 2m+1 is +m, 2m is -m, and 1 is zero.
                let n = i32::from(self.vlc(&vlc::MBA)?);
                if n > 33 {
                    return Err(self.fail("B.10 escape"));
                }
                let m = if n % 2 == 1 { n / 2 } else { -(n / 2) };
                m as u16 as u32
            }
            _ => (i32::from(self.vlc(&vlc::DMV)?) - 1) as u16 as u32,
        };
        let len = (self.advanced - start) as u32;
        self.cmd_data = v & 0xFFFF | len << 16;
        self.top = self.look(32)?;
        Ok(true)
    }

    /// `BDEC`: the coefficient data of one macroblock, out as 16-bit samples
    /// (48 quadwords: Y 16x16, Cb 8x8, Cr 8x8). Bit 27 says it is intra,
    /// 26 resets the DC predictors first, 25 is a field DCT and 20..16 the
    /// quantiser scale code. A non-intra macroblock starts with its coded
    /// block pattern, which `IPU_CTRL.CBP` reports.
    fn bdec(&mut self, val: u32, _: usize) -> Step<bool> {
        self.skip(val & 0x3F)?;
        let intra = val & 1 << 27 != 0;
        let field = val & 1 << 25 != 0;
        self.qsc = (val >> 16) & 0x1F;
        if val & 1 << 26 != 0 {
            self.reset_dc_pred();
        }
        let cbp = if intra { 63 } else { u32::from(self.vlc(&vlc::CBP)?) };
        let mut blocks = [[0i32; 64]; 6];
        for (b, block) in blocks.iter_mut().enumerate() {
            if cbp & (32 >> b) != 0 {
                self.block(intra, b.saturating_sub(3), block)?;
                pixel::idct(block);
            }
        }
        if !intra {
            self.reset_dc_pred();
        }
        self.ctrl = (self.ctrl & !(0x3F << CTRL_CBP_SHIFT)) | cbp << CTRL_CBP_SHIFT;
        let lo = if intra { 0 } else { -256 };
        let mut out = [0u8; 768];
        for (i, s) in assemble_y(&blocks, field).into_iter().enumerate() {
            out[i * 2..i * 2 + 2].copy_from_slice(&(s.clamp(lo, 255) as i16).to_le_bytes());
        }
        for (i, s) in blocks[4].iter().chain(&blocks[5]).copied().enumerate() {
            let at = 512 + i * 2;
            out[at..at + 2].copy_from_slice(&(s.clamp(lo, 255) as i16).to_le_bytes());
        }
        self.emit(&out);
        self.cmd_data = 0;
        Ok(true)
    }

    /// `IDEC`: intra macroblocks straight to pixels, from the first
    /// macroblock's type through to the next start code. The first unit
    /// takes the skip and resets prediction; each later one begins with the
    /// address increment, whose absence (a start code's zero run has no
    /// valid code) ends the slice. Bits 20..16 seed the quantiser scale,
    /// 24 says each macroblock carries a `dct_type`, 25 flips the sign of
    /// the pixels, 26 dithers and 27 selects RGB16 over RGB32.
    fn idec(&mut self, val: u32, progress: usize) -> Step<bool> {
        if progress == 0 {
            self.skip(val & 0x3F)?;
            self.qsc = (val >> 16) & 0x1F;
            self.reset_dc_pred();
        } else {
            let mut gap = 0;
            loop {
                match self.vlc(&vlc::MBA) {
                    Ok(vlc::MBA_ESCAPE) => gap += 33,
                    Ok(vlc::MBA_STUFFING) => {}
                    Ok(n) => {
                        gap += n - 1;
                        break;
                    }
                    Err(Halt::Error) => {
                        self.end_of_slice()?;
                        return Ok(true);
                    }
                    Err(e) => return Err(e),
                }
            }
            if gap > 0 {
                self.reset_dc_pred();
            }
        }
        let modes = self.vlc(&vlc::MBT_I)?;
        if modes & vlc::MB_QUANT != 0 {
            self.qsc = self.bits(5)?;
        }
        let field = val & 1 << 24 != 0 && self.bits(1)? != 0;
        let mut blocks = [[0i32; 64]; 6];
        for (b, block) in blocks.iter_mut().enumerate() {
            self.block(true, b.saturating_sub(3), block)?;
            pixel::idct(block);
        }
        let mut s = Samples { y: [0; 256], cb: [0; 64], cr: [0; 64] };
        for (d, v) in s.y.iter_mut().zip(assemble_y(&blocks, field)) {
            *d = v.clamp(0, 255) as u8;
        }
        for (d, v) in s.cb.iter_mut().zip(blocks[4]) {
            *d = v.clamp(0, 255) as u8;
        }
        for (d, v) in s.cr.iter_mut().zip(blocks[5]) {
            *d = v.clamp(0, 255) as u8;
        }
        let rgb = pixel::to_rgb32(&s, Convert { th: self.thresh, sgn: val & 1 << 25 != 0 });
        self.emit_rgb(&rgb, val);
        self.cmd_data = 0;
        Ok(false)
    }

    /// Leave the bit pointer on the start code that ended an `IDEC` slice,
    /// with `IPU_TOP` showing it. Slices are byte-aligned, so the trailing
    /// bits of the last macroblock's byte are skipped when what follows is
    /// zero, then whole bytes until the `0x000001` prefix.
    fn end_of_slice(&mut self) -> Step<()> {
        if self.look(8)? == 0 {
            self.skip((8 - self.bp % 8) % 8)?;
            while self.look(24)? != 1 {
                self.skip(8)?;
            }
            self.ctrl |= CTRL_SCD;
        }
        self.top = self.look(32)?;
        Ok(())
    }

    /// `CSC`: macroblocks of 8-bit samples (24 quadwords each) from the
    /// input FIFO to pixels; bits 10..0 count them, 26 dithers, 27 picks
    /// RGB16. One unit per macroblock.
    fn csc(&mut self, val: u32, progress: usize) -> Step<bool> {
        if progress >= (val & 0x7FF) as usize {
            return Ok(true);
        }
        let mut raw = [0u8; 384];
        self.bytes(&mut raw)?;
        let rgb = pixel::to_rgb32(&Samples::from_bytes(&raw), Convert { th: self.thresh, sgn: false });
        self.emit_rgb(&rgb, val);
        Ok(false)
    }

    /// `PACK`: RGB32 macroblocks (64 quadwords each) from the input FIFO
    /// to RGB16 (bit 27 set) or, through the `SETVQ` palette, 4-bit
    /// indices. The latter is not modelled; it emits the right amount of
    /// zeros so the DMA the software armed still completes.
    fn pack(&mut self, val: u32, progress: usize) -> Step<bool> {
        if progress >= (val & 0x7FF) as usize {
            return Ok(true);
        }
        let mut raw = [0u8; 1024];
        self.bytes(&mut raw)?;
        if val & 1 << 27 != 0 {
            let rgb: [u32; 256] = std::array::from_fn(|i| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()));
            self.emit_rgb16(&rgb, val & 1 << 26 != 0);
        } else {
            warn!(target: "ps2_core::bus::ipu", "PACK to INDX4 not modelled; emitting zeros");
            self.emit(&[0u8; 128]);
        }
        Ok(false)
    }

    /// Push one macroblock of pixels in the format the command's OFM bit
    /// (27) selects, dithered if its DTE bit (26) says so.
    fn emit_rgb(&mut self, rgb: &[u32; 256], val: u32) {
        if val & 1 << 27 != 0 {
            self.emit_rgb16(rgb, val & 1 << 26 != 0);
        } else {
            let mut out = [0u8; 1024];
            for (i, p) in rgb.iter().enumerate() {
                out[i * 4..i * 4 + 4].copy_from_slice(&p.to_le_bytes());
            }
            self.emit(&out);
        }
    }

    fn emit_rgb16(&mut self, rgb: &[u32; 256], dte: bool) {
        let mut out = [0u8; 512];
        for (i, p) in pixel::to_rgb16(rgb, dte).iter().enumerate() {
            out[i * 2..i * 2 + 2].copy_from_slice(&p.to_le_bytes());
        }
        self.emit(&out);
    }

    fn reset_dc_pred(&mut self) {
        let idp = if self.ctrl & CTRL_MP1 != 0 { 0 } else { (self.ctrl >> CTRL_IDP_SHIFT) & 3 };
        self.dc_pred = [128 << idp; 3];
    }

    /// One block's coefficients (ISO/IEC 13818-2 7.2.2), dequantised (7.4)
    /// into `out` in natural order. `cc` is 0 for luma, 1 for Cb, 2 for Cr.
    fn block(&mut self, intra: bool, cc: usize, out: &mut [i32; 64]) -> Step<()> {
        let mp1 = self.ctrl & CTRL_MP1 != 0;
        let idp = if mp1 { 0 } else { (self.ctrl >> CTRL_IDP_SHIFT) & 3 };
        let scan = if self.ctrl & CTRL_AS != 0 { &ALT_SCAN } else { &ZIGZAG };
        let table = if intra && self.ctrl & CTRL_IVF != 0 { &*vlc::DCT_B15 } else { &*vlc::DCT_B14 };
        let qs = if self.ctrl & CTRL_QST != 0 { NON_LINEAR_SCALE[self.qsc as usize] } else { 2 * self.qsc as i32 };
        // The matrix arrived in zigzag order; weights are wanted by
        // natural position.
        let matrix: [u8; 64] = (if intra { &self.iq } else { &self.niq })[..].try_into().unwrap();
        let weight = |j: usize| i32::from(matrix[INV_ZIGZAG[j]]);

        let mut i = 0;
        if intra {
            let size = u32::from(self.vlc(if cc == 0 { &vlc::DC_LUMA } else { &vlc::DC_CHROMA })?);
            if size > 0 {
                let b = self.bits(size)? as i32;
                let diff = if b & 1 << (size - 1) != 0 { b } else { b - (1 << size) + 1 };
                self.dc_pred[cc] += diff;
            }
            out[0] = (self.dc_pred[cc] << (3 - idp)).clamp(-2048, 2047);
            i = 1;
        }
        // The first coefficient of a non-intra block cannot be End of
        // Block, so `1s` there is run 0, level +/-1.
        let mut first = !intra;
        loop {
            let (run, level) = if first && self.look(1)? == 1 {
                (0, if self.bits(2)? & 1 != 0 { -1 } else { 1 })
            } else {
                match self.vlc(table)? {
                    vlc::DCT_EOB => break,
                    vlc::DCT_ESCAPE => {
                        let run = self.bits(6)? as usize;
                        let level = if mp1 {
                            match self.bits(8)? as i32 {
                                0 => self.bits(8)? as i32,
                                128 => self.bits(8)? as i32 - 256,
                                l if l > 128 => l - 256,
                                l => l,
                            }
                        } else {
                            (self.bits(12)? as i32) << 20 >> 20
                        };
                        (run, level)
                    }
                    c => {
                        let level = i32::from(c & 0xFF);
                        (usize::from(c >> 8), if self.bits(1)? != 0 { -level } else { level })
                    }
                }
            };
            first = false;
            i += run;
            if i >= 64 {
                return Err(self.fail("coefficient run past the block"));
            }
            let j = scan[i];
            let (w, mag) = (weight(j), level.abs());
            let mut v = if intra { (mag * w * qs) >> 4 } else { ((2 * mag + 1) * w * qs) >> 5 };
            if level < 0 {
                v = -v;
            }
            // MPEG-1 makes every coefficient odd; MPEG-2 fixes the parity
            // of the block as a whole below.
            if mp1 && v != 0 && v & 1 == 0 {
                v -= v.signum();
            }
            out[j] = v.clamp(-2048, 2047);
            i += 1;
        }
        if !mp1 && out.iter().sum::<i32>() & 1 == 0 {
            out[63] ^= 1;
        }
        Ok(())
    }
}

/// Lay a macroblock's four luma blocks out as 16x16 samples. A frame DCT
/// tiles them; a field DCT interleaves the top pair into the even rows and
/// the bottom pair into the odd ones (6.1.3).
fn assemble_y(blocks: &[[i32; 64]; 6], field: bool) -> [i32; 256] {
    let mut y = [0i32; 256];
    for (b, block) in blocks[..4].iter().enumerate() {
        for r in 0..8 {
            let row = if field { (b >> 1) + 2 * r } else { (b >> 1) * 8 + r };
            let col = (b & 1) * 8;
            y[row * 16 + col..row * 16 + col + 8].copy_from_slice(&block[r * 8..r * 8 + 8]);
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A quadword whose bytes count up from `base`.
    fn qw(base: u8) -> [u8; 16] {
        std::array::from_fn(|i| base.wrapping_add(i as u8))
    }

    #[test]
    fn fdec_returns_the_bits_under_the_pointer() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        // FDEC with no skip: the first four bytes, most significant first.
        assert!(ipu.write32(0, 0x4000_0000));
        assert_eq!(ipu.read32(0), 0x0001_0203);
        // It does not consume them, so a second FDEC skipping 8 bits moves
        // the window on by exactly one byte.
        assert!(ipu.write32(0, 0x4000_0008));
        assert_eq!(ipu.read32(0), 0x0102_0304);
    }

    #[test]
    fn fdec_reads_across_a_quadword_boundary() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        ipu.push_in(qw(0x10));
        // Walk to the last byte of the first quadword, so the result
        // straddles the two.
        assert!(ipu.write32(0, 0x4000_0000));
        for _ in 0..15 {
            assert!(ipu.write32(0, 0x4000_0008));
        }
        assert_eq!(ipu.read32(0), 0x0F10_1112);
    }

    #[test]
    fn a_command_stalls_until_the_fifo_is_fed() {
        let mut ipu = Ipu::new();
        // Nothing in the FIFO: the command parks and IPU_CTRL stays busy.
        assert!(!ipu.write32(0, 0x4000_0000));
        assert_ne!(ipu.read32(0x10) & CTRL_BUSY, 0);
        assert_eq!(ipu.read32(4), BUSY32);
        assert!(ipu.push_in(qw(0)));
        assert_eq!(ipu.read32(0x10) & CTRL_BUSY, 0);
        assert_eq!(ipu.read32(0), 0x0001_0203);
    }

    #[test]
    fn setiq_takes_its_matrix_from_the_bitstream() {
        let mut ipu = Ipu::new();
        for i in 0..4 {
            ipu.push_in(qw(i * 16));
        }
        assert!(ipu.write32(0, 0x5000_0000));
        assert_eq!(ipu.iq[0], 0);
        assert_eq!(ipu.iq[63], 63);
        // The intra and non-intra matrices are separate.
        assert!(ipu.niq.iter().all(|&b| b == 0));
    }

    #[test]
    fn setiq_survives_being_fed_a_quadword_at_a_time() {
        // The movie player starves the FIFO constantly, so a table load has
        // to be resumable at any byte and still land byte-perfect.
        let mut ipu = Ipu::new();
        assert!(!ipu.write32(0, 0x5000_0000));
        let mut done = false;
        for i in 0..4 {
            assert!(!done, "finished before the last quadword arrived");
            done = ipu.push_in(qw(i * 16));
        }
        assert!(done);
        assert!(ipu.iq.iter().enumerate().all(|(i, &b)| b == i as u8));
    }

    #[test]
    fn a_read_that_outruns_the_fifo_resumes_without_losing_bits() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        // Two maximum skips leave the pointer near the end of the quadword,
        // so the 32 bits FDEC wants straddle a quadword that has not
        // arrived: the command parks mid-read and finishes on the feed.
        assert!(ipu.write32(0, 0x4000_0000 | 63));
        assert!(!ipu.write32(0, 0x4000_0000 | 63));
        assert!(ipu.push_in(qw(0x10)));
        assert_eq!(ipu.read32(0x20) & 0x7F, 126);
        assert_eq!(ipu.read32(0), 0xC404_4484);
    }

    #[test]
    fn bclr_drops_the_bitstream_and_sets_the_bit_pointer() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        assert!(ipu.write32(0, 0x0000_0009));
        assert_eq!(ipu.read32(0x20) & 0x7F, 9);
        assert_eq!(ipu.ifc(), 0);
    }

    #[test]
    fn ctrl_keeps_its_status_half_across_a_write() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        // Everything settable except the reset bit, which would empty the
        // FIFO out from under the check below.
        ipu.write32(0x10, !CTRL_RESET);
        // The written half is masked to the settable bits, and IFC still
        // reports the FIFO rather than what was written.
        assert_eq!(ipu.read32(0x10) & 0xF, 1);
        assert_eq!(ipu.read32(0x10) & !0xFFFF, CTRL_WRITE & !CTRL_RESET);
    }

    // --- decoder ---------------------------------------------------------

    /// Pack a string of bits (spaces ignored) into quadwords, zero-padded.
    fn stream(bits: &str) -> Vec<[u8; 16]> {
        let bits: Vec<u8> = bits.bytes().filter(|b| *b != b' ').map(|b| b - b'0').collect();
        let mut out = vec![[0u8; 16]; bits.len().div_ceil(128).max(1)];
        for (i, b) in bits.iter().enumerate() {
            out[i / 128][(i % 128) / 8] |= b << (7 - i % 8);
        }
        out
    }

    fn fed(bits: &str) -> Ipu {
        let mut ipu = Ipu::new();
        for q in stream(bits) {
            ipu.push_in(q);
        }
        ipu
    }

    /// Load flat quantiser matrices so a coefficient's value is easy to
    /// predict: every weight 16, both tables.
    fn flat_matrices(ipu: &mut Ipu) {
        ipu.iq = vec![16; 64];
        ipu.niq = vec![16; 64];
    }

    fn drain(ipu: &mut Ipu) -> Vec<u8> {
        let mut out = Vec::new();
        while let Some(q) = ipu.pop_out() {
            out.extend_from_slice(&q);
        }
        out
    }

    fn samples16(bytes: &[u8]) -> Vec<i16> {
        bytes.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect()
    }

    #[test]
    fn vdec_reports_the_value_the_length_and_what_follows() {
        // Address increments 1, 2, 3 and 7, then a marker word.
        let mut ipu = fed("1 011 010 00010 11111111000000001111111100000000");
        let vdec = 0x3000_0000;
        for (value, len) in [(1, 1), (2, 3), (3, 3), (7, 5)] {
            assert!(ipu.write32(0, vdec));
            assert_eq!(ipu.read32(0), len << 16 | value);
        }
        // The pointer has moved past every code, and IPU_TOP shows the
        // next 32 bits.
        assert_eq!(ipu.read32(0x20) & 0x7F, 12);
        assert_eq!(ipu.read32(0x30), 0xFF00_FF00);
        assert_eq!(ipu.read32(0), 0x0005_0007);
    }

    #[test]
    fn vdec_decodes_types_motion_and_dmvector() {
        // P-picture macroblock types: `1` is motion-compensated and coded,
        // `00001` quant and coded without motion.
        let mut ipu = fed("1 00001 011 0000 0011 000 11 0 00000000000000000000000000000000");
        ipu.write32(0x10, 2 << CTRL_PCT_SHIFT);
        let mbt = 0x3000_0000 | 1 << 26;
        assert!(ipu.write32(0, mbt));
        assert_eq!(ipu.read32(0) & 0xFFFF, u32::from(vlc::MB_FORWARD | vlc::MB_PATTERN));
        assert!(ipu.write32(0, mbt));
        assert_eq!(ipu.read32(0) & 0xFFFF, u32::from(vlc::MB_QUANT | vlc::MB_PATTERN));
        // Motion codes -1 and +16, as 16-bit two's complement.
        let mc = 0x3000_0000 | 2 << 26;
        assert!(ipu.write32(0, mc));
        assert_eq!(ipu.read32(0), 3 << 16 | 0xFFFF);
        assert!(ipu.write32(0, mc));
        assert_eq!(ipu.read32(0), 11 << 16 | 16);
        // dmvector -1 then 0.
        let dmv = 0x3000_0000 | 3 << 26;
        assert!(ipu.write32(0, dmv));
        assert_eq!(ipu.read32(0), 2 << 16 | 0xFFFF);
        assert!(ipu.write32(0, dmv));
        assert_eq!(ipu.read32(0), 1 << 16);
    }

    #[test]
    fn vdec_flags_a_pattern_no_code_claims() {
        // `0000 0010 000` is one of the gaps Table B.1 leaves, and it is
        // not a run of zeros, so it is a bad code rather than a slice end.
        let mut ipu = fed("00000010000 1111111111111111 0000000000000000");
        assert!(ipu.write32(0, 0x3000_0000));
        assert_ne!(ipu.read32(0x10) & CTRL_ECD, 0);
        assert_eq!(ipu.read32(0x10) & CTRL_SCD, 0);
    }

    #[test]
    fn vdec_ends_a_slice_on_the_start_code_rather_than_erroring() {
        // A slice ends with padding to the byte and then a start code. The
        // address increment that runs into it is a slice end, so `SCD` is
        // raised, `ECD` is not, and the pointer stays on the start code for
        // the player to read.
        let mut ipu = fed("1 0000000 00000000 00000000 00000001 00000001 11111111");
        // Consume the first increment so the pointer sits on the padding.
        assert!(ipu.write32(0, 0x3000_0000));
        assert_eq!(ipu.read32(0) & 0xFFFF, 1);
        assert!(ipu.write32(0, 0x3000_0000));
        assert_ne!(ipu.read32(0x10) & CTRL_SCD, 0);
        assert_eq!(ipu.read32(0x10) & CTRL_ECD, 0);
        assert_eq!(ipu.read32(0x20) & 0x7F, 1);
        assert!(ipu.write32(0, 0x4000_0007));
        assert_eq!(ipu.read32(0), 0x0000_0101);
    }

    /// An intra macroblock whose six blocks carry only DC: luma
    /// differentials in `y`, chroma zero. `dct_dc_size` 0 is `100` for luma
    /// and `00` for chroma; End of Block is `10`.
    fn dc_macroblock(y: [&str; 4]) -> String {
        let mut s = String::new();
        for code in y {
            s += code;
            s += " 10 ";
        }
        s += "00 10 00 10 ";
        s
    }

    #[test]
    fn bdec_reconstructs_an_intra_macroblock_from_its_dc_terms() {
        // Block 0: size 4, +8 (`110 1000`), so its samples are 136; the
        // prediction carries into block 1 (size 0), and block 2 takes -8
        // (`110 0111`) back to 128. Block 3 stays there.
        let mut ipu = fed(&dc_macroblock(["110 1000", "100", "110 0111", "100"]));
        flat_matrices(&mut ipu);
        let bdec = 0x2000_0000 | 1 << 27 | 1 << 26 | 1 << 16;
        assert!(ipu.write32(0, bdec));
        let out = drain(&mut ipu);
        assert_eq!(out.len(), 768);
        let y = samples16(&out[..512]);
        assert!(y[..8].iter().all(|&s| s == 136), "{:?}", &y[..16]);
        assert!(y[8..16].iter().all(|&s| s == 136));
        assert!(y[8 * 16..8 * 16 + 8].iter().all(|&s| s == 128));
        assert!(y[8 * 16 + 8..9 * 16].iter().all(|&s| s == 128));
        assert!(samples16(&out[512..]).iter().all(|&s| s == 128));
        assert_eq!(ipu.read32(0x10) >> CTRL_CBP_SHIFT & 0x3F, 63);
        assert_eq!(ipu.read32(0x10) & CTRL_ECD, 0);
    }

    #[test]
    fn bdec_reads_the_coded_block_pattern_of_a_non_intra_macroblock() {
        // Pattern `1010` codes only block 0; its one coefficient is the
        // first-position `1s` for run 0, level +1, then EOB. With weight
        // 16 and scale code 8 (scale 16) that is (2 + 1) * 16 * 16 / 32 =
        // 24 at DC, a flat 3 after the transform; mismatch control makes
        // the block sum odd through F[63], which rounds away.
        let mut ipu = fed("1010 10 10");
        flat_matrices(&mut ipu);
        assert!(ipu.write32(0, 0x2000_0000 | 8 << 16));
        assert_eq!(ipu.read32(0x10) >> CTRL_CBP_SHIFT & 0x3F, 32);
        let out = drain(&mut ipu);
        let y = samples16(&out[..512]);
        // Block 0 is flat; every other block is untouched.
        assert!(y[..8].iter().all(|&s| s == 3), "{:?}", &y[..8]);
        assert!(y[7 * 16..7 * 16 + 8].iter().all(|&s| s == 3));
        assert!(y[8..16].iter().all(|&s| s == 0));
        assert!(y[128..].iter().all(|&s| s == 0));
        assert!(samples16(&out[512..]).iter().all(|&s| s == 0));
    }

    #[test]
    fn bdec_dequantises_an_ac_coefficient_through_the_scan() {
        // Intra block 0: DC diff 0, then run 1 level 1 (`011` + sign 0):
        // scan position 2 is natural index 8, the first vertical
        // frequency, so the block darkens top to bottom. Scale code 31
        // (scale 62) makes the slope steep enough to be strict.
        let mut ipu = fed("100 011 0 10  100 10  100 10  100 10  00 10  00 10");
        flat_matrices(&mut ipu);
        assert!(ipu.write32(0, 0x2000_0000 | 1 << 27 | 1 << 26 | 31 << 16));
        let y = samples16(&drain(&mut ipu)[..512]);
        let column: Vec<i16> = (0..8).map(|r| y[r * 16]).collect();
        assert!(column.windows(2).all(|w| w[0] > w[1]), "{column:?}");
        assert!(column[0] > 128 && column[7] < 128);
        assert!(y[..8].iter().all(|&s| s == y[0]));
    }

    /// `intra_vlc_format = 1` swaps in Table B.15, where `0011 0` is run 1
    /// level 2 rather than B.14's run 4 level 1. Run 1 lands the
    /// coefficient at scan position 2, natural index 8: the first vertical
    /// frequency, so the block shades top to bottom and its rows stay flat.
    /// Run 4 would land it at natural index 2 and shade left to right.
    #[test]
    fn table_b15_puts_a_coefficient_where_b14_would_not() {
        let mb = "100 00110 0 0110  100 0110  100 0110  100 0110  00 0110  00 0110";
        let mut ipu = fed(mb);
        flat_matrices(&mut ipu);
        ipu.write32(0x10, CTRL_IVF);
        assert!(ipu.write32(0, 0x2000_0000 | 1 << 27 | 1 << 26 | 31 << 16));
        assert_eq!(ipu.read32(0x10) & CTRL_ECD, 0);
        let y = samples16(&drain(&mut ipu)[..512]);
        // Mismatch control moves F[63], which is worth a unit either way,
        // so a flat row is flat to within one.
        let row = &y[..8];
        assert!(row.iter().max().unwrap() - row.iter().min().unwrap() <= 1, "{row:?}");
        let column: Vec<i16> = (0..8).map(|r| y[r * 16]).collect();
        assert!(column.windows(2).all(|w| w[0] > w[1]), "{column:?}");
        assert!(column[0] > 128 && column[7] < 128);
    }

    /// A player changes one field of `IPU_CTRL` by reading it back and
    /// writing it again. `RST` is a strobe and reads as clear, so that
    /// round trip must leave the bitstream and the bit pointer alone.
    #[test]
    fn writing_ctrl_back_after_a_reset_does_not_reset_again() {
        let mut ipu = Ipu::new();
        ipu.push_in(qw(0));
        ipu.write32(0x10, CTRL_RESET | CTRL_IVF);
        assert_eq!(ipu.ifc(), 0, "the reset itself still empties the FIFO");
        ipu.push_in(qw(0));
        assert!(ipu.write32(0, 0x4000_0008));
        let ctrl = ipu.read32(0x10);
        assert_eq!(ctrl & CTRL_RESET, 0);
        ipu.write32(0x10, ctrl);
        // Bit pointer, window and picture parameters all survive.
        assert_eq!(ipu.read32(0x20) & 0x7F, 8);
        assert_ne!(ipu.read32(0x10) & CTRL_IVF, 0);
        assert!(ipu.write32(0, 0x4000_0000));
        assert_eq!(ipu.read32(0), 0x0102_0304);
    }

    #[test]
    fn a_field_dct_interleaves_the_luma_blocks() {
        let mut ipu = fed(&dc_macroblock(["110 1000", "100", "110 0111", "100"]));
        flat_matrices(&mut ipu);
        assert!(ipu.write32(0, 0x2000_0000 | 1 << 27 | 1 << 26 | 1 << 25 | 1 << 16));
        let y = samples16(&drain(&mut ipu)[..512]);
        for row in 0..16 {
            let want = if row % 2 == 0 { 136 } else { 128 };
            assert!(y[row * 16..row * 16 + 16].iter().all(|&s| s == want), "row {row}");
        }
    }

    #[test]
    fn a_starved_macroblock_is_retried_from_its_start() {
        // The same intra macroblock placed to straddle a quadword
        // boundary (BCLR parks the pointer at bit 120) and fed a quadword
        // at a time: the command parks on the first, whose eight bits are
        // not a whole DC code, and the output appears with the second.
        let bits = format!("{} {}", "0".repeat(120), dc_macroblock(["110 1000", "100", "110 0111", "100"]));
        let mut ipu = Ipu::new();
        flat_matrices(&mut ipu);
        assert!(ipu.write32(0, 120));
        assert!(!ipu.write32(0, 0x2000_0000 | 1 << 27 | 1 << 26 | 1 << 16));
        let mut done = false;
        for q in stream(&bits) {
            assert!(!done);
            assert!(ipu.fifo_out.is_empty());
            done = ipu.push_in(q);
        }
        assert!(done);
        let y = samples16(&drain(&mut ipu)[..512]);
        assert_eq!(y[0], 136);
        assert_eq!(y[255], 128);
    }

    #[test]
    fn idec_decodes_a_slice_and_stops_on_the_start_code() {
        // Two flat macroblocks (type `1`, no quant), the second reached by
        // increment `1`; then padding to a byte and a sequence end code.
        let mb = dc_macroblock(["100", "100", "100", "100"]);
        let bits = format!("1 {mb} 1 1 {mb}");
        let mut ipu = fed(&format!("{bits} {} 00000000 00000000 00000001 10110111 11111111",
            "0".repeat((8 - bits.replace(' ', "").len() % 8) % 8)));
        flat_matrices(&mut ipu);
        // Two macroblocks are 128 quadwords and the FIFO holds eight, so
        // the command parks and the drain is what carries it to the end.
        assert!(!ipu.write32(0, 0x1000_0000 | 1 << 16));
        let out = drain(&mut ipu);
        assert_eq!(out.len(), 2 * 1024);
        assert_eq!(ipu.read32(0x10) & CTRL_BUSY, 0);
        // Y 128, Cb/Cr 128 is (149 * 112 + 64) >> 7 = 130 on every channel.
        for p in out.chunks(4) {
            assert_eq!(p, [130, 130, 130, 0x80]);
        }
        assert_eq!(ipu.read32(0x30), 0x0000_01B7);
        assert_ne!(ipu.read32(0x10) & CTRL_SCD, 0);
        assert_eq!(ipu.read32(0x10) & CTRL_ECD, 0);
        // The next FDEC sees the start code too.
        assert!(ipu.write32(0, 0x4000_0000));
        assert_eq!(ipu.read32(0), 0x0000_01B7);
    }

    #[test]
    fn idec_can_write_rgb16_and_honours_a_quantiser_change() {
        // One macroblock with `01`: quant + intra, scale code 3 follows.
        let mb = dc_macroblock(["100", "100", "100", "100"]);
        let bits = format!("01 00011 {mb}");
        let mut ipu = fed(&format!("{bits} {} 00000000 00000000 00000001 00000000 00000000",
            "0".repeat((8 - bits.replace(' ', "").len() % 8) % 8)));
        flat_matrices(&mut ipu);
        assert!(!ipu.write32(0, 0x1000_0000 | 1 << 27 | 1 << 16));
        assert_eq!(ipu.qsc, 3);
        let out = drain(&mut ipu);
        assert_eq!(out.len(), 512);
        // 130 on every channel is 16 in five bits, alpha set.
        assert!(out.chunks(2).all(|p| u16::from_le_bytes([p[0], p[1]]) == 1 << 15 | 16 << 10 | 16 << 5 | 16));
    }

    #[test]
    fn csc_converts_raw_macroblocks_from_the_fifo() {
        let mut ipu = Ipu::new();
        // Two macroblocks: video black, then video white.
        for _ in 0..16 {
            ipu.push_in([16; 16]);
        }
        for _ in 0..8 {
            ipu.push_in([128; 16]);
        }
        for _ in 0..16 {
            ipu.push_in([235; 16]);
        }
        for _ in 0..8 {
            ipu.push_in([128; 16]);
        }
        assert!(!ipu.write32(0, 0x7000_0000 | 2));
        let out = drain(&mut ipu);
        assert_eq!(out.len(), 2048);
        assert!(out[..1024].chunks(4).all(|p| p == [0, 0, 0, 0x80]));
        assert!(out[1024..].chunks(4).all(|p| p == [255, 255, 255, 0x80]));
        assert_eq!(ipu.read32(0x10) & CTRL_BUSY, 0);
    }

    #[test]
    fn csc_waits_for_a_macroblock_that_has_not_all_arrived() {
        let mut ipu = Ipu::new();
        for _ in 0..23 {
            ipu.push_in([235; 16]);
        }
        assert!(!ipu.write32(0, 0x7000_0000 | 1));
        assert!(ipu.fifo_out.is_empty());
        // The macroblock converts as soon as its last quadword lands,
        // but the command only retires once its output has been taken.
        assert!(!ipu.push_in([128; 16]));
        assert_eq!(ipu.fifo_out.len(), 64);
        assert_eq!(drain(&mut ipu).len(), 1024);
        assert_eq!(ipu.read32(0x10) & CTRL_BUSY, 0);
    }

    #[test]
    fn pack_narrows_rgb32_to_rgb16() {
        let mut ipu = Ipu::new();
        let px = 0x80FF_8000u32.to_le_bytes();
        for _ in 0..64 {
            ipu.push_in(std::array::from_fn(|i| px[i % 4]));
        }
        assert!(!ipu.write32(0, 0x8000_0000 | 1 << 27 | 1));
        let out = drain(&mut ipu);
        assert_eq!(out.len(), 512);
        assert!(out.chunks(2).all(|p| u16::from_le_bytes([p[0], p[1]]) == 1 << 15 | 31 << 10 | 16 << 5));
    }

    /// The output FIFO is eight quadwords deep and holds the decoder when
    /// it is full. Without this a long CSC finishes inside the write that
    /// starts it and a player pacing itself on channel 3 eats its whole
    /// stream at once.
    #[test]
    fn a_full_output_fifo_holds_the_decoder_until_it_is_drained() {
        let mut ipu = Ipu::new();
        for _ in 0..48 {
            ipu.push_in([16; 16]);
        }
        // Two macroblocks; only the first can be converted before the FIFO
        // is full, and the second waits for room.
        assert!(!ipu.write32(0, 0x7000_0000 | 2));
        assert_eq!(ipu.fifo_out.len(), 64);
        assert_ne!(ipu.read32(0x10) & CTRL_BUSY, 0);
        for _ in 0..64 {
            ipu.pop_out().expect("the first macroblock is there");
        }
        // Taking it restarted the command, which converted the second.
        assert_eq!(ipu.fifo_out.len(), 64);
        assert_eq!(drain(&mut ipu).len(), 1024);
        assert_eq!(ipu.read32(0x10) & CTRL_BUSY, 0);
    }

    /// VDEC writes nothing, so a full output FIFO must not hold it: on
    /// hardware the decode goes straight through, and parking it here
    /// would deadlock a player that reads a header between macroblocks.
    #[test]
    fn a_full_output_fifo_does_not_hold_a_vdec() {
        let mut ipu = fed("00000000 00000000 00000001 00000011 1 0000000");
        ipu.fifo_out.extend(std::iter::repeat_n([0u8; 16], FIFO_DEPTH * 4));
        // Skip the start code, then read one macroblock_address_increment.
        assert!(ipu.write32(0, 0x3000_0000 | 32));
        assert_eq!(ipu.read32(0) & 0xFFFF, 1);
    }

    #[test]
    fn ofc_reports_the_output_fifo_and_reset_empties_it() {
        let mut ipu = fed(&dc_macroblock(["100", "100", "100", "100"]));
        flat_matrices(&mut ipu);
        assert!(ipu.write32(0, 0x2000_0000 | 1 << 27 | 1 << 26 | 1 << 16));
        assert_eq!(ipu.read32(0x10) >> 4 & 0xF, 8);
        assert_eq!(ipu.fifo_out.len(), 48);
        ipu.write32(0x10, CTRL_RESET);
        assert_eq!(ipu.read32(0x10) >> 4 & 0xF, 0);
        assert!(ipu.pop_out().is_none());
    }
}
