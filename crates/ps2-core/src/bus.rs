//! EE-side system bus: memory map, MMIO dispatch, TTY capture.
//!
//! Address translation is a direct segment fold for now: the kernel's TLB
//! mappings are essentially identity, so TLB instructions record entries
//! without remapping (see ARCHITECTURE.md).

use crate::Region;
use crate::gif::Gif;
use crate::gs::GsFront;
use crate::prof;
use crate::sif::Sif;
use crate::spu2::Spu2;
use crate::timers::Timers;
use crate::vif::Vif;
use crate::vu1::Vu1;
use std::collections::HashSet;
use tracing::{debug, trace, warn};
use serde::{Deserialize, Serialize};

pub const RAM_SIZE: usize = 32 * 1024 * 1024;
pub const BIOS_SIZE: usize = 4 * 1024 * 1024;
pub const SPAD_SIZE: usize = 16 * 1024;
/// Shadow register file for 0x1000_0000..0x1001_0000 MMIO.
const MMIO_SIZE: usize = 0x10000;

/// Number of RDRAM devices reported by the MCH init handshake.
const RDRAM_DEVICES: u32 = 2;

#[derive(Serialize, Deserialize)]
/// EE DMAC channel (only SIF0/SIF1 are modeled so far).
#[derive(Default)]
pub struct EeDmaChannel {
    pub chcr: u32,
    pub madr: u32,
    pub qwc: u32,
    pub tadr: u32,
    /// Current tag asked to stop the chain after its data.
    tag_end: bool,
    /// Return addresses pushed by `call` tags, popped by `ret`. Two deep,
    /// the same as the DMAC's ASR0/ASR1.
    asr: [u32; 2],
    asr_depth: u8,
    /// Scratchpad-side address, the SPR channels only. 14 bits.
    sadr: u32,
}

const EE_CHCR_STR: u32 = 1 << 8;
const EE_CHCR_TTE: u32 = 1 << 6;
const EE_CHCR_TIE: u32 = 1 << 7;

#[derive(Serialize, Deserialize)]
/// IOP DMA channel (SIF0 = ch9, SIF1 = ch10).
#[derive(Default)]
pub struct IopDmaChannel {
    pub madr: u32,
    pub bcr: u32,
    pub chcr: u32,
    pub tadr: u32,
    /// Words remaining in the current block (SIF0 send side).
    words_left: u32,
    /// Current tag asked to end the transfer after its block.
    tag_end: bool,
    /// SIF1 receive side: words remaining for the current IOP tag.
    recv_left: u32,
    recv_addr: u32,
    /// Destination address the current packet started at.
    recv_start: u32,
    recv_end: bool,
    /// Padding words to the packet's qword boundary, dropped after the data.
    recv_pad: u32,
}

const IOP_CHCR_BUSY: u32 = 1 << 24;

#[derive(Serialize, Deserialize)]
/// CDVD (MECHACON) model: S commands answer instantly; N commands read
/// sectors from an optional disc image, streamed from disk one request
/// at a time (PS2 images are far too large to hold in memory).
#[derive(Default)]
pub struct Cdvd {
    n_cmd: u8,
    n_params: Vec<u8>,
    /// CDVD-internal interrupt flags (reg 0x08, W1C).
    pub istat: u8,
    s_cmd: u8,
    s_params: Vec<u8>,
    s_results: Vec<u8>,
    s_result_pos: usize,
    /// Config session state (S 0x40 open .. 0x43 close), addressing one of
    /// the three NVRAM config areas: write mode, area, block budget, cursor.
    config_write: bool,
    config_area: u8,
    config_blocks: u8,
    config_index: u8,
    /// Mechacon NVRAM (1 KiB EEPROM). Holds the OSD's configuration —
    /// including the "initialized" flag that decides whether the boot runs
    /// the first-time setup (PS logo, PS2 logo, language wizard) — plus
    /// region parameters and the i.Link id. Persisted to `nvram_path`.
    nvram: Vec<u8>,
    /// Frontend-owned; re-attached after a state load, not part of one.
    #[serde(skip)]
    nvram_path: Option<std::path::PathBuf>,
    /// Disc image (2048-byte sectors), read on demand.
    #[serde(skip)]
    pub disc: Option<std::fs::File>,
    /// Sector data staged for DMA channel 3, and the drain cursor.
    read_buf: Vec<u8>,
    read_pos: usize,
    /// Disc key from N 0x0C, served through the XOR-obfuscated register
    /// window 0x2020-0x2034 (validity bits at 0x2038, XOR byte at 0x2039,
    /// decrypt flag at 0x203A).
    key: [u8; 15],
    key_flag: u8,
    key_valid: bool,
    /// DEC-SET (reg 0x3A, written by cdvdman): drive-side decryption of
    /// DMA'd sector data. Bit 0 = XOR with key[4], bit 1 = rotate right
    /// by bits 4-6. The PS2 logo area (lsn 0-11) is stored encrypted and
    /// PS2LOGO refuses to boot the game unless the read decrypts it.
    dec_set: u8,
    /// An N command is in flight: the completion interrupt is deferred by
    /// the drive-latency model (the bus delivers it via `finish_n`).
    n_busy: bool,
    /// Head position after the last read, for the seek-time model.
    last_lsn: u32,
    /// EE cycle at which the drive has spun up and identified the disc:
    /// until then the status reports SPIN and the disc type "detecting".
    /// Set at power-on and again on each IOP reboot, so cdvdman's re-init
    /// (sceCdDiskReady) holds the boot like the hardware drive check does
    /// — that hold is what lets the boot chime's reverb tail ring out
    /// before the fresh libsd zeroes the SPU.
    ready_at: u64,
    /// EE cycle at which the disc in the drive has been identified. Unlike
    /// [`Cdvd::ready_at`] an IOP reboot leaves it alone (the type stays
    /// known); only a tray close re-runs the identification.
    identified_at: u64,
    /// The drive is open: nothing is readable and the status reports OPEN
    /// until the tray closes again.
    tray_open: bool,
}

/// Initial spin-up/identification time after power-on, and again after an
/// IOP reboot ([`Cdvd::ready_at`]).
const CDVD_SPINUP: u64 = 6500 * CDVD_MS;
const CDVD_RESETTLE: u64 = 1200 * CDVD_MS;

/// ISO sector payload size; DVD reads wrap it in a 2064-byte raw sector.
const ISO_SECTOR: u64 = 2048;

/// How long the EE takes to recognise a raised INTC line, in EE cycles.
/// Long enough for a spin loop reading INTC_STAT to get a look in (the
/// OSD's vertical-blank wait reads it every five cycles), short beside
/// anything the interrupt timing itself feeds.
const INTC_LATENCY: u64 = 32;

/// EE cycles per millisecond, for the drive-latency model.
const CDVD_MS: u64 = 294_912;

impl Cdvd {
    /// Open the drive and hand back whatever was in it. Per-disc state
    /// (the key, the DEC-SET mode, the head position) goes with it.
    pub fn open_tray(&mut self) -> Option<std::fs::File> {
        self.tray_open = true;
        self.key = [0; 15];
        self.key_flag = 0;
        self.key_valid = false;
        self.dec_set = 0;
        self.last_lsn = 0;
        self.read_buf.clear();
        self.read_pos = 0;
        self.disc.take()
    }

    /// Close the drive on `disc` (`None` leaves it empty) and re-run the
    /// spin-up and identification the hardware does on a fresh disc.
    pub fn close_tray(&mut self, disc: Option<std::fs::File>, now: u64) {
        self.tray_open = false;
        self.disc = disc;
        self.ready_at = now + CDVD_SPINUP;
        self.identified_at = now + CDVD_SPINUP;
    }

    /// Whether the drive is open.
    pub fn tray_open(&self) -> bool {
        self.tray_open
    }

    /// The boot serial of the disc in the drive, formatted the way it is
    /// printed on the disc ("SLPS-25418"). `None` when the drive is empty
    /// or the image has no readable ISO9660 SYSTEM.CNF.
    pub fn boot_serial(&mut self) -> Option<String> {
        let s = self.disc_serial()?;
        let text = std::str::from_utf8(&s).ok()?;
        Some(format!("{}-{}", &text[..4], &text[4..]))
    }

    /// Take the disc and the NVRAM file location from `live`: a save state
    /// does not carry either, so a restored drive keeps what is physically
    /// in it.
    pub(crate) fn carry_over(&mut self, live: &mut Cdvd) {
        self.disc = live.disc.take();
        self.nvram_path = live.nvram_path.take();
    }

    /// Execute an N command. CdRead (0x06) and DvdRead (0x08) stage the
    /// requested sectors into `read_buf` for DMA channel 3. Returns the
    /// drive latency (EE cycles) until the completion interrupt: real
    /// drives seek and stream slowly, and boot-time audio/visuals (the
    /// chime riding out while the OSD loads the game, the PS2 logo dwell)
    /// depend on those delays.
    fn n_execute(&mut self) -> u64 {
        let cmd = self.n_cmd;
        let lsn = u32::from_le_bytes([
            self.n_params.first().copied().unwrap_or(0),
            self.n_params.get(1).copied().unwrap_or(0),
            self.n_params.get(2).copied().unwrap_or(0),
            self.n_params.get(3).copied().unwrap_or(0),
        ]);
        let count = u32::from_le_bytes([
            self.n_params.get(4).copied().unwrap_or(0),
            self.n_params.get(5).copied().unwrap_or(0),
            self.n_params.get(6).copied().unwrap_or(0),
            self.n_params.get(7).copied().unwrap_or(0),
        ]);
        match cmd {
            0x06 | 0x08 => {
                debug!(target: "ps2_core::iop::cdvd",
                    cmd = format_args!("{cmd:#04x}"), lsn, count, "disc read");
                self.read_buf.clear();
                self.read_pos = 0;
                // Latency: a distance-graded seek and ~4x-DVD streaming
                // (0.4 ms/sector); the spin-up is in `CDVD_READY_AT`.
                let dist = u64::from(lsn.abs_diff(self.last_lsn));
                let seek = match dist {
                    0 => CDVD_MS / 4,
                    1..16 => CDVD_MS,
                    16..4096 => 10 * CDVD_MS,
                    4096..65536 => 30 * CDVD_MS,
                    _ => 80 * CDVD_MS,
                };
                let xfer = u64::from(count.min(4096)) * (2 * CDVD_MS / 5);
                self.last_lsn = lsn.wrapping_add(count);
                let latency = seek + xfer;
                if self.tray_open {
                    return latency;
                }
                let Some(disc) = self.disc.as_mut() else {
                    return latency;
                };
                use std::io::{Read, Seek, SeekFrom};
                for i in 0..count.min(4096) {
                    let mut data = [0u8; ISO_SECTOR as usize];
                    let ok = disc
                        .seek(SeekFrom::Start((lsn as u64 + i as u64) * ISO_SECTOR))
                        .and_then(|_| disc.read_exact(&mut data))
                        .is_ok();
                    if !ok {
                        warn!(target: "ps2_core::iop::cdvd", lsn = lsn + i, "read past end of disc image");
                    }
                    if cmd == 0x08 {
                        // Raw DVD sector: 12-byte ID/IED/CPR header with the
                        // physical sector number (LBA + 0x30000), 2048 data,
                        // 4-byte EDC. cdvdman checks the number and strips
                        // the framing.
                        let phys = lsn + i + 0x30000;
                        let mut hdr = [0u8; 12];
                        hdr[0] = 0x20; // layer 0
                        hdr[1] = (phys >> 16) as u8;
                        hdr[2] = (phys >> 8) as u8;
                        hdr[3] = phys as u8;
                        self.read_buf.extend_from_slice(&hdr);
                        self.read_buf.extend_from_slice(&data);
                        self.read_buf.extend_from_slice(&[0; 4]);
                    } else {
                        self.read_buf.extend_from_slice(&data);
                    }
                }
                if self.dec_set != 0 {
                    let shift = (self.dec_set >> 4) & 7;
                    let key4 = self.key[4];
                    for b in &mut self.read_buf {
                        if self.dec_set & 1 != 0 {
                            *b ^= key4;
                        }
                        if self.dec_set & 2 != 0 {
                            *b = b.rotate_right(shift.into());
                        }
                    }
                }
                latency
            }
            // sceCdReadKey: derive the disc key from the boot executable's
            // serial the way the mechacon does; the OSD recomputes it and
            // compares before accepting the disc.
            0x0C => {
                let arg2 = u32::from(self.n_params.get(3).copied().unwrap_or(0))
                    | u32::from(self.n_params.get(4).copied().unwrap_or(0)) << 8;
                let serial = self.disc_serial();
                self.key = [0; 15];
                if let Some(s) = serial {
                    let numbers: u32 = std::str::from_utf8(&s[4..9])
                        .ok()
                        .and_then(|d| d.parse().ok())
                        .unwrap_or(0);
                    let letters = u32::from(s[3] & 0x7F)
                        | u32::from(s[2] & 0x7F) << 7
                        | u32::from(s[1] & 0x7F) << 14
                        | u32::from(s[0] & 0x7F) << 21;
                    let key_0_3 = ((numbers & 0x1FC00) >> 10) | ((letters & 0x01FF_FFFF) << 7);
                    self.key[..4].copy_from_slice(&key_0_3.to_le_bytes());
                    self.key[4] =
                        (((numbers & 0x1F) << 3) | ((letters & 0x0E00_0000) >> 25)) as u8;
                    if arg2 == 75 {
                        self.key[14] = (((numbers & 0x3E0) >> 2) | 0x04) as u8;
                    }
                }
                self.key_flag = match arg2 {
                    75 => 0x05,
                    4246 => {
                        self.key[..5].copy_from_slice(&[0x07, 0xF7, 0xF2, 0x01, 0x00]);
                        0x01
                    }
                    _ => 0x01,
                };
                self.key_valid = true;
                debug!(target: "ps2_core::iop::cdvd",
                    arg2,
                    serial = format_args!("{:?}", serial.map(|s| String::from_utf8_lossy(&s).into_owned())),
                    key = format_args!("{:02x?}", self.key),
                    "read disc key");
                // A quick mechacon exchange: PS2LOGO polls its completion
                // on a coarse ~0.8 s delay loop, so any latency beyond the
                // first check costs a whole extra round of black screen.
                CDVD_MS / 8
            }
            _ => {
                debug!(target: "ps2_core::iop::cdvd",
                    cmd = format_args!("{cmd:#04x}"),
                    params = format_args!("{:02x?}", self.n_params),
                    "N command (quick)");
                CDVD_MS / 8
            }
        }
    }

    fn read_sector_raw(&mut self, lsn: u64, buf: &mut [u8]) -> bool {
        use std::io::{Read, Seek, SeekFrom};
        let Some(disc) = self.disc.as_mut() else {
            return false;
        };
        disc.seek(SeekFrom::Start(lsn * ISO_SECTOR))
            .and_then(|_| disc.read_exact(buf))
            .is_ok()
    }

    /// Extract the boot serial ("SLPS25918"-style: 4 letters + 5 digits)
    /// from SYSTEM.CNF via a minimal ISO9660 walk.
    fn disc_serial(&mut self) -> Option<[u8; 9]> {
        let mut pvd = [0u8; 2048];
        if !self.read_sector_raw(16, &mut pvd) || &pvd[1..6] != b"CD001" {
            return None;
        }
        // Root directory record sits at PVD offset 156.
        let extent = u32::from_le_bytes(pvd[158..162].try_into().unwrap()) as u64;
        let mut dir = [0u8; 2048];
        if !self.read_sector_raw(extent, &mut dir) {
            return None;
        }
        let mut off = 0usize;
        let cnf_extent = loop {
            if off >= 2048 || dir[off] == 0 {
                return None;
            }
            let len = dir[off] as usize;
            let name_len = dir[off + 32] as usize;
            let name = &dir[off + 33..off + 33 + name_len];
            if name.starts_with(b"SYSTEM.CNF") {
                break u32::from_le_bytes(dir[off + 2..off + 6].try_into().unwrap()) as u64;
            }
            off += len;
        };
        let mut cnf = [0u8; 2048];
        if !self.read_sector_raw(cnf_extent, &mut cnf) {
            return None;
        }
        // BOOT2 = cdrom0:\SLPS_259.18;1 -> 4 letters, then every digit.
        let text = String::from_utf8_lossy(&cnf);
        let boot = text.lines().find(|l| l.contains("BOOT2"))?;
        let path = boot.split(['\\', ':', '/']).last()?;
        let mut out = [0u8; 9];
        let mut letters = path.bytes().filter(u8::is_ascii_alphabetic);
        let mut digits = path.split(';').next()?.bytes().filter(u8::is_ascii_digit);
        for slot in &mut out[..4] {
            *slot = letters.next()?;
        }
        for slot in &mut out[4..] {
            *slot = digits.next()?;
        }
        Some(out)
    }

    /// Staged sector bytes not yet drained by DMA channel 3.
    fn read_remaining(&self) -> usize {
        self.read_buf.len().saturating_sub(self.read_pos)
    }

    /// Drain staged sector data for DMA channel 3.
    fn dma_read(&mut self, out: &mut [u8]) {
        for b in out.iter_mut() {
            *b = self.read_buf.get(self.read_pos).copied().unwrap_or(0);
            self.read_pos += 1;
        }
    }
}

/// A config block on the wire is 15 data bytes plus their sum mod 256;
/// CDVDMAN verifies the sum and rejects the block before the OSD sees it.
/// NVRAM size (mechacon EEPROM, 8 Kbit).
const NVRAM_SIZE: usize = 1024;

/// NVRAM layout for v1.70+ BIOSes (both our reference images are v2.x);
/// offsets and per-area block caps as on the real EEPROM. Area 1 holds the
/// OSD's config: block 1 is the language/timezone block whose byte +2 bit 7
/// is the "initialized" flag — while it is clear the boot runs the
/// first-time setup (PS logo and PS2 logo screens, then the language
/// wizard) instead of going straight to the browser/disc.
const NVRAM_CONFIG_AREAS: [(usize, u8); 3] = [(0x270, 4), (0x2B0, 2), (0x200, 7)];
const NVRAM_REGPARAMS: usize = 0x180;
const NVRAM_ILINK_ID: usize = 0x1E0;
const NVRAM_LANGUAGE: usize = 0x2B0 + 0x10;

/// A factory-fresh Japanese NVRAM: region parameters, an i.Link id, and the
/// default language block WITHOUT the initialized flag, so the first boot
/// runs the OSD setup like a new console. Completing the wizard persists
/// the real configuration.
fn nvram_defaults() -> Vec<u8> {
    let mut nv = vec![0u8; NVRAM_SIZE];
    // "JJjpnJJ": PStwo region parameters for Japan.
    nv[NVRAM_REGPARAMS..NVRAM_REGPARAMS + 7].copy_from_slice(b"JJjpnJJ");
    // Dummy i.Link id + its checksum, as expected by libcdvd.
    nv[NVRAM_ILINK_ID..NVRAM_ILINK_ID + 8]
        .copy_from_slice(&[0x00, 0xAC, 0xFF, 0xFF, 0xFF, 0xFF, 0xB9, 0x86]);
    nv[NVRAM_ILINK_ID + 8..NVRAM_ILINK_ID + 10].copy_from_slice(&[0x00, 0x18]);
    // Default language block (Japanese, JST); byte +2 bit 7 stays clear.
    nv[NVRAM_LANGUAGE..NVRAM_LANGUAGE + 16].copy_from_slice(&[
        0x20, 0x20, 0, 0, 0, 0x70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x30,
    ]);
    nv
}

impl Cdvd {
    /// Load NVRAM from `path` (created with factory defaults when missing
    /// or invalid) and persist subsequent writes back to it.
    pub(crate) fn load_nvram(&mut self, path: std::path::PathBuf) {
        let loaded = std::fs::read(&path).ok().filter(|d| d.len() == NVRAM_SIZE);
        let fresh = loaded.is_none();
        self.nvram = loaded.unwrap_or_else(nvram_defaults);
        // Like the real libcdvd expects: region parameters and a language
        // block must exist; refill a wiped image with the defaults (the
        // initialized flag stays clear, so the first-boot setup still runs).
        if self.nvram[NVRAM_LANGUAGE..NVRAM_LANGUAGE + 16].iter().all(|&b| b == 0)
            || self.nvram[NVRAM_REGPARAMS..NVRAM_REGPARAMS + 12].iter().all(|&b| b == 0)
        {
            let d = nvram_defaults();
            self.nvram[NVRAM_REGPARAMS..NVRAM_REGPARAMS + 12]
                .copy_from_slice(&d[NVRAM_REGPARAMS..NVRAM_REGPARAMS + 12]);
            self.nvram[NVRAM_ILINK_ID..NVRAM_ILINK_ID + 10]
                .copy_from_slice(&d[NVRAM_ILINK_ID..NVRAM_ILINK_ID + 10]);
            self.nvram[NVRAM_LANGUAGE..NVRAM_LANGUAGE + 16]
                .copy_from_slice(&d[NVRAM_LANGUAGE..NVRAM_LANGUAGE + 16]);
        }
        self.nvram_path = Some(path);
        if fresh {
            self.save_nvram();
        }
    }

    fn save_nvram(&self) {
        if let Some(p) = &self.nvram_path
            && let Err(e) = std::fs::write(p, &self.nvram)
        {
            warn!(target: "ps2_core::iop::cdvd", path = %p.display(), error = %e, "cannot persist NVRAM");
        }
    }

    /// NVRAM range of the current config-session block, if the session's
    /// area, budget and per-area block cap all allow it.
    fn config_nvram_range(&self) -> Option<std::ops::Range<usize>> {
        if self.config_index >= self.config_blocks {
            return None;
        }
        let (base, cap) = *NVRAM_CONFIG_AREAS.get(self.config_area as usize)?;
        if self.config_index >= cap {
            return None;
        }
        let start = base + self.config_index as usize * 16;
        (start + 16 <= self.nvram.len()).then(|| start..start + 16)
    }

    /// Execute an S command; fills the result FIFO.
    fn s_execute(&mut self) {
        let cmd = self.s_cmd;
        debug!(target: "ps2_core::iop::cdvd",
            cmd = format_args!("{cmd:#04x}"),
            params = format_args!("{:02x?}", self.s_params),
            unread = self.s_results.len().saturating_sub(self.s_result_pos),
            "S command");
        self.s_results.clear();
        self.s_result_pos = 0;
        match cmd {
            // Mechacon version query (sub-command in param 0): status +
            // version bytes.
            0x03 => self.s_results.extend_from_slice(&[0, 3, 6, 2]),
            // sceCdReadClock: stat + BCD sec/min/hour/pad/day/month/year.
            0x08 => self
                .s_results
                .extend_from_slice(&[0, 0, 0, 0, 0, 1, 1, 0x25]),
            // Forbid/permit DVD player: canonical result is 5.
            0x15 | 0x16 => self.s_results.push(5),
            // sceCdReadModelNumber: param is a byte offset into the model
            // string; result is [stat, model bytes from that offset].
            0x17 => {
                const MODEL: &[u8; 16] = b"SCPH-50000\0\0\0\0\0\0";
                let off = self.s_params.first().copied().unwrap_or(0) as usize & 0xF;
                self.s_results.push(0);
                self.s_results
                    .extend_from_slice(&MODEL[off..(off + 8).min(16)]);
            }
            // OpenConfig: params are [read/write, area, block count]. The
            // OSD opens (0, 1, 2) to read its config from NVRAM area 1.
            0x40 => {
                self.config_write = self.s_params.first().copied().unwrap_or(0) == 1;
                self.config_area = self.s_params.get(1).copied().unwrap_or(0);
                self.config_blocks = self.s_params.get(2).copied().unwrap_or(0);
                self.config_index = 0;
                self.s_results.push(0);
            }
            // ReadConfig: one 16-byte block per call, straight from the
            // NVRAM area (the checksum byte lives in NVRAM as written).
            // Out-of-range areas or indices read as zeroes.
            0x41 => {
                let block = self
                    .config_nvram_range()
                    .map(|r| {
                        let mut b = [0u8; 16];
                        b.copy_from_slice(&self.nvram[r]);
                        b
                    })
                    .unwrap_or([0; 16]);
                self.config_index += 1;
                self.s_results.extend_from_slice(&block);
            }
            // WriteConfig: store the 16-byte block into NVRAM and persist
            // (this is how the first-boot wizard saves its settings).
            0x42 => {
                if self.config_write
                    && let Some(r) = self.config_nvram_range()
                    && self.s_params.len() >= 16
                {
                    self.nvram[r].copy_from_slice(&self.s_params[..16]);
                    self.save_nvram();
                }
                self.config_index += 1;
                self.s_results.push(0);
            }
            // CloseConfig: ends the session.
            0x43 => {
                self.config_write = false;
                self.config_area = 0;
                self.config_blocks = 0;
                self.config_index = 0;
                self.s_results.push(0);
            }
            // ReadNVM: params [addr_hi, addr_lo] in 16-bit words; result
            // [status, data_hi, data_lo].
            0x0A => {
                let addr = ((self.s_params.first().copied().unwrap_or(0) as usize) << 8
                    | self.s_params.get(1).copied().unwrap_or(0) as usize)
                    * 2;
                if addr + 1 < NVRAM_SIZE {
                    self.s_results
                        .extend_from_slice(&[0, self.nvram[addr + 1], self.nvram[addr]]);
                } else {
                    self.s_results.push(0xFF);
                }
            }
            // WriteNVM: params [addr_hi, addr_lo, data_hi, data_lo].
            0x0B => {
                let addr = ((self.s_params.first().copied().unwrap_or(0) as usize) << 8
                    | self.s_params.get(1).copied().unwrap_or(0) as usize)
                    * 2;
                if addr + 1 < NVRAM_SIZE && self.s_params.len() >= 4 {
                    self.nvram[addr + 1] = self.s_params[2];
                    self.nvram[addr] = self.s_params[3];
                    self.save_nvram();
                    self.s_results.push(0);
                } else {
                    self.s_results.push(0xFF);
                }
            }
            // BootCertify: accepted.
            0x1A => self.s_results.push(1),
            // MagicGate mechacon commands, used by SECRMAN for memory-card
            // authentication. No real crypto: status 0 with zeroed payloads
            // satisfies SECRMAN's shape checks. 0x84/0x85 hand out the
            // console-side challenges as [status, 16 bytes].
            0x84 | 0x85 => self.s_results.extend_from_slice(&[0; 17]),
            0x80..=0x8F => self.s_results.push(0),
            _ => {
                trace!(target: "ps2_core::iop::cdvd", cmd = format_args!("{cmd:#04x}"), "unhandled S command (returning 0)");
                self.s_results.push(0);
            }
        }
    }

    pub fn read(&mut self, addr: u32, now: u64) -> u32 {
        let loaded = self.disc.is_some() && !self.tray_open;
        let spinning_up = loaded && now < self.ready_at.max(CDVD_SPINUP);
        let v = match addr & 0x3F {
            0x04 => self.n_cmd as u32,
            // N status: busy while an N command's latency runs, else
            // ready with no data.
            0x05 => {
                if self.n_busy {
                    0x80
                } else {
                    0x40
                }
            }
            0x06 => 0, // error
            0x08 => self.istat as u32,
            // Drive status: spinning up after power-on, reading while an
            // N command is in flight, paused-on-disc when idle, stopped
            // without a disc. cdvdman's sceCdDiskReady waits for exactly
            // 0x0A (PAUSE); reporting SPIN forever stalls EELOAD's game
            // boot.
            0x0A => {
                if self.tray_open {
                    0x01
                } else if spinning_up {
                    0x02
                } else if self.n_busy && loaded {
                    0x06
                } else if loaded {
                    0x0A
                } else {
                    0
                }
            }
            0x0B => 0,
            // Disc type: detecting during the initial spin-up only (a
            // reboot resettle keeps the known type — EELOAD reads it to
            // pick its boot path and would fall back to the browser on
            // "detecting"), then PS2 DVD (0x14) when an image is loaded.
            0x0F => {
                if !loaded {
                    0
                } else if now < self.identified_at.max(CDVD_SPINUP) {
                    0x01
                } else {
                    0x14
                }
            }
            0x16 => self.s_cmd as u32,
            // S status: bit 6 set when the result FIFO is empty.
            0x17 => {
                if self.s_result_pos >= self.s_results.len() {
                    0x40
                } else {
                    0
                }
            }
            0x18 => {
                let v = self.s_results.get(self.s_result_pos).copied().unwrap_or(0);
                self.s_result_pos += 1;
                v as u32
            }
            // Disc key window: three 5-byte banks (0x20-0x24, 0x28-0x2C,
            // 0x30-0x34), validity bits at 0x38, XOR byte at 0x39.
            r @ (0x20..=0x24 | 0x28..=0x2C | 0x30..=0x34) => {
                let i = (r - 0x20 - (r - 0x20) / 8 * 3) as usize;
                self.key.get(i).copied().unwrap_or(0) as u32
            }
            0x38 => {
                if self.key_valid {
                    0x07
                } else {
                    0
                }
            }
            0x39 => 0,
            0x3A => self.key_flag as u32,
            _ => 0,
        };
        trace!(target: "ps2_core::iop::cdvd", addr = format_args!("{addr:#04x}"), value = format_args!("{v:#x}"), "read");
        v
    }

    /// Returns the drive latency when the write issued an N command; the
    /// bus delivers the completion ([`Cdvd::finish_n`]) after that many
    /// EE cycles.
    pub fn write(&mut self, addr: u32, v: u32) -> Option<u64> {
        trace!(target: "ps2_core::iop::cdvd", addr = format_args!("{addr:#04x}"), value = format_args!("{v:#x}"), "write");
        match addr & 0x3F {
            0x04 => {
                self.n_cmd = v as u8;
                let latency = self.n_execute();
                self.n_params.clear();
                self.n_busy = true;
                return Some(latency);
            }
            0x05 => self.n_params.push(v as u8),
            0x08 => self.istat &= !(v as u8),
            0x16 => {
                self.s_cmd = v as u8;
                self.s_execute();
                self.s_params.clear();
            }
            0x17 => self.s_params.push(v as u8),
            0x3A => {
                debug!(target: "ps2_core::iop::cdvd", value = format_args!("{v:#04x}"), "DEC-SET");
                self.dec_set = v as u8;
            }
            _ => {}
        }
        None
    }

    /// The in-flight N command's drive latency elapsed: raise the internal
    /// completion flags (the bus raises the IOP interrupt line).
    pub fn finish_n(&mut self) {
        self.n_busy = false;
        self.istat |= 3; // command complete + data ready
    }
}

#[derive(Serialize, Deserialize)]
/// SIO2 (pad/memory-card serial controller) with one digital-capable
/// DualShock in port 0. A transfer is described by the SEND3 slots
/// (port in bit 0, byte length in bits 8-17); command bytes arrive in
/// the in-FIFO (PIO or DMA ch11) and responses leave through the
/// out-FIFO (PIO or DMA ch12). The first command byte selects the
/// device class: 0x01 pad, 0x81 memory card, 0x21 multitap.
pub struct Sio2 {
    send3: [u32; 16],
    fifo_in: Vec<u8>,
    fifo_out: Vec<u8>,
    out_pos: usize,
    ctrl: u32,
    recv1: u32,
    /// RECV2/RECV3 and the two FIFO position registers: plain storage that
    /// reads back what was written, which is all the hardware does with them.
    recv2: u32,
    recv3: u32,
    fifo_pos: [u32; 2],
    /// Interrupt status. Bit 0 latches on transfer completion and the ISR
    /// clears it by writing the bit back.
    istat: u32,
    /// Buttons currently held, active-high in PS1 bit order
    /// (SELECT=0, L3, R3, START, UP, RIGHT, DOWN, LEFT,
    ///  L2, R2, L1, R1, TRIANGLE, CIRCLE, CROSS, SQUARE=15).
    pub buttons: u16,
    /// Transfer started while the in-FIFO was empty: waiting for DMA ch11
    /// to deliver the command bytes (the DMAC only moves data once the
    /// start bit asserts DRQ, so CTRL can legitimately come first).
    pending: bool,
    /// DMA ch11 block geometry (block bytes, block count). With more than
    /// one block, each sub-transfer's command sits at the start of its own
    /// block and its reply is padded to a full block (MCMAN's layout);
    /// a single block packs the sub-transfers back to back (PADMAN's).
    in_block: (usize, usize),
    /// DualShock command-0x43 config mode.
    pad_config: bool,
    /// The mode the pad reports and polls in: `PAD_DIGITAL`, `PAD_ANALOG`
    /// (adds four stick bytes) or `PAD_DS2` (also twelve pressure bytes).
    pad_mode: u8,
    /// The vibration slots as command 0x4D last left them.
    motor_map: [u8; 2],
    /// Memory card in slot 1 (SIO2 port 2).
    pub memcard: MemCard,
}

/// Pad modes, which are also the id byte a poll answers with.
const PAD_DIGITAL: u8 = 0x41;
const PAD_ANALOG: u8 = 0x73;
const PAD_DS2: u8 = 0x79;
/// Held-button bits in [`Sio2::buttons`], in the order a DualShock 2 reports
/// their pressure.
const PAD_PRESSURE_ORDER: [u16; 12] = [
    1 << 5,  // right
    1 << 7,  // left
    1 << 4,  // up
    1 << 6,  // down
    1 << 12, // triangle
    1 << 13, // circle
    1 << 14, // cross
    1 << 15, // square
    1 << 10, // L1
    1 << 11, // R1
    1 << 8,  // L2
    1 << 9,  // R2
];

/// RECV1 bits SIO2MAN checks after a transfer. The third nibble counts the
/// ports the packet addressed; the high bits tag a port whose device did not
/// answer. One open port with everything present is `0x1100`.
const SIO2_RECV1_ONE_PORT: u32 = 0x100;
const SIO2_RECV1_TWO_PORTS: u32 = 0x200;
const SIO2_RECV1_ALL_PRESENT: u32 = 0x1000;
const SIO2_RECV1_PORT1_MISSING: u32 = 0x1D000;
const SIO2_RECV1_PORT2_MISSING: u32 = 0x2D000;
const SIO2_RECV1_CONNECTED: u32 = SIO2_RECV1_ONE_PORT | SIO2_RECV1_ALL_PRESENT;

#[derive(Serialize, Deserialize)]
/// A PS2 memory card: 16384 pages of 512 data + 16 ECC bytes, stored as
/// contiguous 528-byte pages (the common .ps2 image layout). MCMAN talks
/// to it with `[0x81, cmd, ...]` sub-transfers; replies carry an 0x2B
/// acknowledge and end with a settable terminator byte. The MagicGate
/// authentication (cmd 0xF0) is answered formally, without real crypto,
/// mirroring what PCSX2 does — MCMAN only checks the message shapes and
/// the XOR of its own challenge bytes.
pub struct MemCard {
    pub data: Vec<u8>,
    /// Set whenever a write or erase lands, so the host can persist.
    pub dirty: bool,
    terminator: u8,
    /// Page selected by SetSector, and the byte cursor within it that
    /// ReadData/WriteData advance.
    sector: u32,
    progress: usize,
}

pub const MEMCARD_PAGE: usize = 528;
pub const MEMCARD_PAGES: usize = 16384;
/// Pages per erase block.
const MEMCARD_BLOCK: usize = 16;
/// VIF_STAT.FDR: the VIF FIFO's transfer direction.
const VIF_STAT_FDR: u64 = 1 << 23;
/// Sector size the PS1 read command works in, unrelated to the PS2 page size.
const PS1_SECTOR: usize = 128;

impl Default for MemCard {
    fn default() -> Self {
        Self {
            // Erased flash reads all-ones; the OSD offers to format it.
            data: vec![0xFF; MEMCARD_PAGE * MEMCARD_PAGES],
            dirty: false,
            terminator: 0x55,
            sector: 0,
            progress: 0,
        }
    }
}

impl MemCard {
    fn pos(&self) -> usize {
        (self.sector as usize * MEMCARD_PAGE + self.progress) % (MEMCARD_PAGE * MEMCARD_PAGES)
    }

    /// Build the `len`-byte reply for one `[0x81, op, ...]` sub-transfer.
    ///
    /// Every reply opens with two bytes clocked out while the host sends the
    /// 0x81 and the command byte: a present card drives them low, an empty
    /// slot leaves the line high. Most commands then pad with zeros and close
    /// with an 0x2B acknowledge plus the terminator.
    fn respond(&mut self, cmd: &[u8], len: usize) -> Vec<u8> {
        let op = cmd.get(1).copied().unwrap_or(0);
        let mut r: Vec<u8> = vec![0x00, 0x00];
        let term = self.terminator;
        // Pad to `n - 2` bytes, then acknowledge and terminate.
        let ack = |r: &mut Vec<u8>, n: usize| {
            r.resize(n.saturating_sub(2).max(r.len()), 0);
            r.push(0x2B);
            r.push(term);
        };
        match op {
            // Probe / write-delete-end / read-write-end / erase-block: bare
            // acknowledge. Erase clears the block holding the selected page.
            0x11 | 0x12 | 0x81 | 0x82 => {
                if op == 0x82 {
                    let block = (self.sector as usize / MEMCARD_BLOCK) * MEMCARD_BLOCK;
                    let start = (block * MEMCARD_PAGE) % self.data.len();
                    let end = (start + MEMCARD_BLOCK * MEMCARD_PAGE).min(self.data.len());
                    self.data[start..end].fill(0xFF);
                    self.dirty = true;
                }
                ack(&mut r, 4);
            }
            // Boot-time probe MCMAN issues before the auth handshake.
            0xBF | 0xF7 => ack(&mut r, 5),
            // MagicGate session reset; also forces the terminator back to 0x55.
            0xF3 => {
                self.terminator = 0x55;
                r.resize(3, 0);
                r.push(0x2B);
                r.push(0x55);
            }
            // SetSector for erase/write/read: 4-byte page + checksum.
            0x21 | 0x22 | 0x23 => {
                self.sector = u32::from(cmd.get(2).copied().unwrap_or(0))
                    | u32::from(cmd.get(3).copied().unwrap_or(0)) << 8
                    | u32::from(cmd.get(4).copied().unwrap_or(0)) << 16
                    | u32::from(cmd.get(5).copied().unwrap_or(0)) << 24;
                self.progress = 0;
                ack(&mut r, 9);
            }
            // GetSpecs: sector size, pages per erase block, page count.
            0x26 => {
                let specs = [0x00u8, 0x02, 0x10, 0x00, 0x00, 0x40, 0x00, 0x00];
                r.push(0x2B);
                r.extend(specs);
                r.push(specs.iter().fold(0u8, |a, &b| a ^ b));
                r.push(term);
            }
            // SetTerminator: the reply already carries the new byte.
            0x27 => {
                self.terminator = cmd.get(2).copied().unwrap_or(0x55);
                r.push(0x00);
                r.push(0x2B);
                r.push(self.terminator);
            }
            0x28 => {
                // GetTerminator.
                r.push(0x2B);
                r.push(term);
                r.push(term);
            }
            // WriteData: [0x81, 0x42, size, data.., xor]; the card echoes
            // zeros for the payload and answers with its own checksum.
            0x42 => {
                let size = cmd.get(2).copied().unwrap_or(0) as usize;
                r.push(0x00);
                r.push(0x2B);
                let mut xor = 0u8;
                for i in 0..size {
                    let b = cmd.get(3 + i).copied().unwrap_or(0);
                    xor ^= b;
                    let p = self.pos();
                    self.data[p] = b;
                    self.progress += 1;
                    r.push(0x00);
                }
                self.dirty = true;
                r.push(xor);
                r.push(term);
            }
            // ReadData: [0x81, 0x43, size]; reply carries the page bytes
            // plus their XOR before the terminator.
            0x43 => {
                let size = cmd.get(2).copied().unwrap_or(0) as usize;
                r.push(0x00);
                r.push(0x2B);
                let mut xor = 0u8;
                for _ in 0..size {
                    let b = self.data[self.pos()];
                    self.progress += 1;
                    xor ^= b;
                    r.push(b);
                }
                r.push(xor);
                r.push(term);
            }
            // PS1 card read, used to tell a PS1 card from a PS2 one. The
            // sector is a raw 128-byte offset, nothing like the PS2 layout.
            0x52 => {
                let hi = cmd.get(4).copied().unwrap_or(0);
                let lo = cmd.get(5).copied().unwrap_or(0);
                let start = (u32::from(hi) << 8 | u32::from(lo)) as usize * PS1_SECTOR;
                r.extend([0x5A, 0x5D, 0x00, 0x00, 0x5C, 0x5D, hi, lo]);
                let mut xor = hi ^ lo;
                for i in 0..PS1_SECTOR {
                    let b = self.data[(start + i) % self.data.len()];
                    xor ^= b;
                    r.push(b);
                }
                r.push(xor);
                r.push(0x47);
            }
            // MagicGate auth: formal replies only. Modes that carry eight
            // console bytes answer with their XOR; the rest just acknowledge.
            0xF0 => {
                let mode = cmd.get(2).copied().unwrap_or(0);
                match mode {
                    0x01 | 0x02 | 0x04 | 0x0F | 0x11 | 0x13 => {
                        r.push(0x00);
                        r.push(0x2B);
                        let mut xor = 0u8;
                        for i in 0..8 {
                            xor ^= cmd.get(3 + i).copied().unwrap_or(0);
                            r.push(0x00);
                        }
                        r.push(xor);
                        r.push(term);
                    }
                    // These carry eight bytes but want the plain acknowledge.
                    0x06 | 0x07 | 0x0B => ack(&mut r, 14),
                    _ => ack(&mut r, 5),
                }
            }
            _ => {
                debug!(target: "ps2_core::iop::sio2",
                    op = format_args!("{op:#04x}"), "unhandled memcard command");
                ack(&mut r, len);
            }
        }
        r.resize(len, 0);
        r
    }
}

impl Default for Sio2 {
    fn default() -> Self {
        Self {
            send3: [0; 16],
            fifo_in: Vec::new(),
            fifo_out: Vec::new(),
            out_pos: 0,
            ctrl: 0,
            recv1: 0,
            // "Command OK", the only value the hardware is ever seen with.
            recv2: 0xF,
            recv3: 0,
            fifo_pos: [0; 2],
            istat: 0,
            buttons: 0,
            pending: false,
            in_block: (0, 0),
            pad_config: false,
            pad_mode: PAD_DIGITAL,
            motor_map: [0xFF; 2],
            memcard: MemCard::default(),
        }
    }
}

impl Sio2 {
    /// Latch the completion interrupt. Returns true when the IOP's line
    /// should be pulsed, i.e. the previous one has been acknowledged.
    pub fn raise_irq(&mut self) -> bool {
        let first = self.istat == 0;
        self.istat |= 1;
        if !first {
            debug!(target: "ps2_core::iop::sio2", "transfer completed with the last interrupt unacknowledged");
        }
        first
    }

    /// Fold one addressed port into RECV1: the third nibble counts the ports
    /// the packet touched, and a device that did not answer tags its port.
    fn note_port(recv1: &mut u32, present: bool, port: u32) {
        if *recv1 & SIO2_RECV1_ONE_PORT != 0 {
            *recv1 &= !SIO2_RECV1_ONE_PORT;
            *recv1 |= SIO2_RECV1_TWO_PORTS;
        } else {
            *recv1 |= SIO2_RECV1_ONE_PORT;
        }
        *recv1 |= SIO2_RECV1_ALL_PRESENT;
        if !present {
            *recv1 |= if port == 0 {
                SIO2_RECV1_PORT1_MISSING
            } else {
                SIO2_RECV1_PORT2_MISSING
            };
        }
    }

    /// Execute the queued transfer: walk SEND3, consume the in-FIFO and
    /// synthesize each sub-transfer's response.
    fn run_transfer(&mut self) {
        self.fifo_out.clear();
        self.out_pos = 0;
        let mut pos = 0usize;
        let mut recv1 = 0u32;
        let (block_bytes, blocks) = self.in_block;
        let blocked = blocks > 1;
        for (i, slot) in self.send3.into_iter().enumerate() {
            if slot == 0 {
                break;
            }
            let port = slot & 1;
            let len = ((slot >> 8) & 0x3FF) as usize;
            if len == 0 {
                break;
            }
            if blocked {
                pos = i * block_bytes;
            }
            let end = (pos + len).min(self.fifo_in.len());
            let cmd: Vec<u8> = self.fifo_in[pos.min(end)..end].to_vec();
            pos += len;
            debug!(target: "ps2_core::iop::sio2",
                port, len, cmd = format_args!("{:02x?}", &cmd[..cmd.len().min(4)]),
                "sub-transfer");
            // The leading byte picks the device class; both live on port 0.
            match (port, cmd.first().copied()) {
                (0, Some(0x01)) => {
                    self.pad_respond(&cmd, len);
                    Self::note_port(&mut recv1, true, port);
                }
                (0, Some(0x81)) => {
                    let r = self.memcard.respond(&cmd, len);
                    self.fifo_out.extend(r);
                    // MCMAN checks one status word for the whole packet and
                    // the card overwrites it, the way the hardware does.
                    recv1 = SIO2_RECV1_CONNECTED;
                }
                (_, Some(0x21)) => {
                    // Multitap query. No tap is modelled, and the card behind
                    // the port answers 0x66 ("not a tap") at byte 5 — replying
                    // 0xFF there makes XSIO2MAN think a tap with a nonsense
                    // slot is present. MTAPMAN only looks at the port-1 bit.
                    let mut r = vec![0xFFu8; len];
                    if len > 5 {
                        r[5] = 0x66;
                    }
                    self.fifo_out.extend(r);
                    Self::note_port(&mut recv1, false, 0);
                }
                _ => {
                    // No device: the line floats high, so reads return 0xFF.
                    self.fifo_out.extend(std::iter::repeat(0xFFu8).take(len));
                    Self::note_port(&mut recv1, false, port);
                }
            }
            if blocked {
                // Each sub-transfer's reply fills its own block.
                self.fifo_out.resize((i + 1) * block_bytes, 0);
            }
        }
        self.in_block = (0, 0);
        self.recv1 = recv1;
        self.fifo_in.clear();
    }

    fn pad_respond(&mut self, cmd: &[u8], len: usize) {
        let op = cmd.get(1).copied().unwrap_or(0);
        let id: u8 = if self.pad_config { 0xF3 } else { self.pad_mode };
        let b = !self.buttons;
        let mut r = vec![0xFF, id, 0x5A];
        if self.pad_config {
            // Config-mode queries. A DualShock 2 answers these with fixed
            // constants; a driver that reads zeros back decides the pad is
            // not a DS2 and stays in digital mode -- and some (Ace Combat
            // 5's DS2U.IRX) then ignore its input entirely.
            let arg = cmd.get(3).copied().unwrap_or(0);
            r.extend(match op {
                // Set VREF param: a fixed acknowledgement.
                0x40 => [0x00, 0x00, 0x02, 0x00, 0x00, 0x5A],
                // Button query: only a pad that has been switched to analog
                // answers; a digital one returns zeros.
                0x41 if self.pad_mode != PAD_DIGITAL => [0xFF, 0xFF, 0x03, 0x00, 0x00, 0x5A],
                // Set response bytes: the mask picks the poll format, and
                // with it the id the pad answers polls with.
                0x4F => {
                    let mask = u32::from(cmd.get(3).copied().unwrap_or(0))
                        | u32::from(cmd.get(4).copied().unwrap_or(0)) << 8
                        | u32::from(cmd.get(5).copied().unwrap_or(0)) << 16;
                    self.pad_mode = match mask {
                        0x00_003F => PAD_ANALOG,
                        0x03_FFFF => PAD_DS2,
                        _ => PAD_DIGITAL,
                    };
                    [0x00, 0x00, 0x00, 0x00, 0x00, 0x5A]
                }
                // Query model: DS2, current mode, one mode entry.
                0x45 => [0x03, 0x02, u8::from(self.pad_mode != PAD_DIGITAL), 0x02, 0x01, 0x00],
                // Query act: the two actuators' descriptions.
                0x46 if arg == 0 => [0x00, 0x00, 0x01, 0x02, 0x00, 0x0A],
                0x46 => [0x00, 0x00, 0x01, 0x01, 0x01, 0x14],
                // Query comb: one combination driving two actuators.
                0x47 => [0x00, 0x00, 0x02, 0x00, 0x01, 0x00],
                // Query mode: the digital and analog mode ids.
                0x4C if arg == 0 => [0x00, 0x00, 0x00, 0x04, 0x00, 0x00],
                0x4C => [0x00, 0x00, 0x00, 0x07, 0x00, 0x00],
                // Vibration mapping: each slot answers with what it held
                // before this write, then takes the new value.
                0x4D => {
                    let old = self.motor_map;
                    self.motor_map = [
                        cmd.get(3).copied().unwrap_or(0xFF),
                        cmd.get(4).copied().unwrap_or(0xFF),
                    ];
                    [old[0], old[1], 0xFF, 0xFF, 0xFF, 0xFF]
                }
                _ => [0; 6],
            });
        } else if op == 0x42 || op == 0x43 {
            r.extend([b as u8, (b >> 8) as u8]);
            if self.pad_mode != PAD_DIGITAL {
                // Centered sticks: rx, ry, lx, ly.
                r.extend([0x7F; 4]);
            }
            if self.pad_mode == PAD_DS2 {
                // Pressure. A key is either fully down or not pressed at
                // all, and a driver that only reads these sees nothing if
                // they stay zero while the digital bits say otherwise.
                r.extend(
                    PAD_PRESSURE_ORDER
                        .map(|m| if self.buttons & m != 0 { 0xFF } else { 0x00 }),
                );
            }
        }
        // Mode changes take effect after the frame that carries them.
        if op == 0x43 {
            self.pad_config = cmd.get(3) == Some(&1);
        } else if self.pad_config && op == 0x44 {
            self.pad_mode = if cmd.get(3) == Some(&1) { PAD_ANALOG } else { PAD_DIGITAL };
        }
        r.resize(len, 0);
        self.fifo_out.extend(r);
    }

    /// Run a start that was waiting for its DMA-delivered command bytes.
    /// Returns true when the transfer executed (raises the interrupt).
    fn dma_in_done(&mut self) -> bool {
        if self.pending {
            self.pending = false;
            self.run_transfer();
            true
        } else {
            false
        }
    }

    pub fn read(&mut self, addr: u32) -> u32 {
        match addr & 0xFF {
            0x00..=0x3F => self.send3[((addr >> 2) & 0xF) as usize],
            0x64 => {
                let v = self.fifo_out.get(self.out_pos).copied().unwrap_or(0);
                self.out_pos += 1;
                v as u32
            }
            0x68 => self.ctrl,
            0x6C => self.recv1,
            0x70 => self.recv2,
            0x74 => self.recv3,
            0x78 | 0x7C => self.fifo_pos[usize::from(addr & 4 != 0)],
            0x80 => self.istat,
            _ => 0,
        }
    }

    /// Returns true when the write completed a transfer (raises the SIO2
    /// interrupt line).
    pub fn write(&mut self, addr: u32, v: u32) -> bool {
        match addr & 0xFF {
            0x00..=0x3F => self.send3[((addr >> 2) & 0xF) as usize] = v,
            0x60 => self.fifo_in.push(v as u8),
            0x68 => {
                // Bits 2/3 reset the FIFOs — but only honor them without
                // the start bit: SIO2MAN sets both in one write after DMA
                // has already loaded the in-FIFO.
                if v & 0xC != 0 && v & 1 == 0 {
                    self.fifo_in.clear();
                    self.fifo_out.clear();
                    self.out_pos = 0;
                    self.pending = false;
                }
                self.ctrl = v & !1;
                if v & 1 != 0 {
                    if self.fifo_in.is_empty() {
                        self.pending = true;
                    } else {
                        self.run_transfer();
                        return true;
                    }
                }
            }
            0x70 => self.recv2 = v,
            0x74 => self.recv3 = v,
            0x78 | 0x7C => self.fifo_pos[usize::from(addr & 4 != 0)] = v,
            // ISTAT acknowledges the bits written back as 1.
            0x80 => self.istat &= !v,
            _ => {}
        }
        false
    }
}

#[derive(Serialize, Deserialize)]
/// IOP root counter (0-2: 16-bit PS1-style, 3-5: 32-bit).
#[derive(Default, Clone, Copy)]
struct IopTimer {
    base: u32,
    base_cycle: u64,
    mode: u32,
    target: u32,
    last_check: u64,
}

/// Longest gap between periodic ticks, whatever the computed events say.
const MAX_TICK_GAP: u64 = 8192;

impl IopTimer {
    /// IOP sysclock ticks are EE cycles / 8; mode bit 8 selects the external
    /// clock (pixel for counter 0, hblank for counters 1/3), bit 9 is /8 on
    /// counter 2, and wide timers add a sysclock prescaler in bits 13-14.
    /// EE cycles per COUNT tick (mirrors [`IopTimer::count`]).
    fn cycles_per_tick(&self, idx: usize, region: Region) -> u64 {
        let external = self.mode & (1 << 8) != 0;
        8 * match idx {
            0 if external => 3,
            1 | 3 if external => region.iop_hblank_div(),
            2 if self.mode & (1 << 9) != 0 => 8,
            3.. => match (self.mode >> 13) & 3 {
                0 => 1,
                1 => 8,
                2 => 16,
                _ => 256,
            },
            _ => 1,
        }
    }

    /// Earliest cycle at which this timer reaches its target or wraps.
    fn next_event(&self, idx: usize, now: u64, region: Region) -> u64 {
        let size = if idx < 3 { 1u64 << 16 } else { 1u64 << 32 };
        let count = u64::from(self.count(idx, now, region));
        let to_target = (u64::from(self.target) + size - count) % size;
        let to_target = if to_target == 0 { size } else { to_target };
        let to_wrap = size - count;
        let p = self.cycles_per_tick(idx, region);
        let phase = now.saturating_sub(self.base_cycle) % p;
        now + to_target.min(to_wrap) * p - phase
    }

    /// Restart the count from where it stands, so a change of clock rate
    /// does not reinterpret the span already elapsed.
    fn rebase(&mut self, idx: usize, now: u64, region: Region) {
        self.base = self.count(idx, now, region);
        self.base_cycle = now;
    }

    fn count(&self, idx: usize, now: u64, region: Region) -> u32 {
        let sys = now.saturating_sub(self.base_cycle) / 8;
        let external = self.mode & (1 << 8) != 0;
        let ticks = match idx {
            // ~13.5 MHz dot clock, coarsely approximated.
            0 if external => sys / 3,
            1 | 3 if external => sys / region.iop_hblank_div(),
            2 if self.mode & (1 << 9) != 0 => sys / 8,
            3.. => match (self.mode >> 13) & 3 {
                0 => sys,
                1 => sys / 8,
                2 => sys / 16,
                _ => sys / 256,
            },
            _ => sys,
        };
        let count = self.base as u64 + ticks;
        if idx < 3 {
            count as u32 & 0xFFFF
        } else {
            count as u32
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Bus {
    #[serde(with = "serde_bytes")]
    pub ram: Box<[u8]>,
    pub bios: Box<[u8]>,
    #[serde(with = "serde_bytes")]
    pub spad: Box<[u8]>,
    /// IOP RAM as seen from the EE at 0x1C00_0000 (2 MiB).
    #[serde(with = "serde_bytes")]
    pub iop_ram: Box<[u8]>,
    /// Earliest cycle at which [`Bus::tick_timers`] has work (timer events,
    /// deferred DMA completions, SPU2 samples); any write that can move an
    /// event resets it to 0. See `Ps2System::machine_cycle`.
    pub timers_due: u64,
    /// Shadow storage for EE MMIO registers we don't model yet: reads return
    /// the last written value so BIOS read-modify-write sequences behave.
    #[serde(with = "serde_bytes")]
    mmio: Box<[u8]>,
    /// Snapshotted separately: the renderer may live on a worker thread.
    /// The placeholder a load builds is the inline one, so deserializing
    /// never starts a renderer thread just to throw it away.
    #[serde(skip, default = "GsFront::inline")]
    pub gs: GsFront,
    pub gif: Gif,
    pub ipu: crate::ipu::Ipu,
    pub vif0: Vif,
    pub vif1: Vif,
    pub vu1: Vu1,
    /// VU0 register state for COP2 macro mode (reuses the VU core; its
    /// micro/data memories stand in for VU0's 4 KiB ones).
    pub vu0: Vu1,
    pub timers: Timers,
    pub sif: Sif,
    /// IOP scratchpad (1 KiB at 0x1F800000).
    #[serde(with = "serde_bytes")]
    pub iop_spad: Box<[u8]>,
    pub spu2: Spu2,
    /// IOP DMA ch4 (SPU2 core 0) and ch7 (core 1).
    pub iop_dma_spu: [IopDmaChannel; 2],
    /// Shadow storage for IOP MMIO (0x1F801000..0x1F810000), same idea as
    /// the EE shadow.
    #[serde(with = "serde_bytes")]
    iop_mmio: Box<[u8]>,
    /// IOP interrupt controller: I_STAT / I_MASK / I_CTRL.
    pub iop_i_stat: u32,
    pub iop_i_mask: u32,
    pub iop_i_ctrl: u32,
    /// IOP root counters.
    iop_timers: [IopTimer; 6],
    /// EE INTC.
    pub intc_stat: u32,
    pub intc_mask: u32,
    /// Cycle from which the EE may recognise the INTC line, or `u64::MAX`
    /// while nothing is pending (see [`Bus::intc_changed`]).
    intc_ready_at: u64,
    /// EE DMAC: VIF1 (ch1), GIF (ch2), SIF0 (ch5) and SIF1 (ch6),
    /// interrupt status/mask.
    pub dma_vif0: EeDmaChannel,
    pub dma_vif1: EeDmaChannel,
    pub dma_ipu_to: EeDmaChannel,
    pub dma_gif: EeDmaChannel,
    /// Scratchpad channels: 8 copies out of the SPR, 9 into it.
    pub dma_spr_from: EeDmaChannel,
    pub dma_spr_to: EeDmaChannel,
    pub dma_sif0: EeDmaChannel,
    pub dma_sif1: EeDmaChannel,
    pub d_stat: u32,
    pub d_mask: u32,
    /// IOP DMA: SIF0 (ch9), SIF1 (ch10), interrupt control.
    pub iop_dma_sif0: IopDmaChannel,
    pub iop_dma_sif1: IopDmaChannel,
    pub iop_dicr: u32,
    pub iop_dicr2: u32,
    pub iop_dma_sio2in: IopDmaChannel,
    pub iop_dma_sio2out: IopDmaChannel,
    pub iop_dma_cdvd: IopDmaChannel,
    /// DMA ch12 armed before the SIO2 transfer produced its response; the
    /// copy runs when the transfer executes (hardware waits on DRQ).
    sio2out_deferred: bool,
    /// DMA ch3 armed before the CDVD read staged its sectors.
    cdvd_dma_deferred: bool,
    /// EE cycle when the in-flight CDVD N command completes.
    cdvd_done_at: Option<u64>,
    /// sceSifIopReset commands seen, for the drive re-settle policy.
    iop_resets: u32,
    pub cdvd: Cdvd,
    pub sio2: Sio2,
    /// Current EE cycle count, updated by the system before each step.
    pub now: u64,
    /// Video timing region; software moves it by programming SMODE1 (see
    /// [`Bus::set_region_from_smode1`]).
    pub region: Region,
    /// Kernel TTY output captured from the EE SIO TXFIFO (observation only).
    pub tty_buffer: String,
    /// Current TTY line, flushed to the log on '\n'.
    tty_line: String,
    /// RDRAM init handshake state (MCH_RICM/MCH_DRD).
    rdram_sdevid: u32,
    /// Unmapped addresses already reported, to keep the log readable.
    #[serde(skip)]
    warned_unmapped: HashSet<u32>,
    /// EE TLB entries (raw registers) and a 4 KiB-granular lookup cache.
    #[serde(with = "serde_big_array::BigArray")]
    ee_tlb: [(u32, u32, u32, u32); 48],
    /// (vaddr page | 1) -> phys page; 0 = invalid slot.
    tlb_cache: Box<[(u32, u32)]>,
    /// RAM pages (4 KiB) holding recompiled code; a write to one queues the
    /// address in `dirty_code_writes` for the recompiler to drop the blocks
    /// covering it (kernel data shares pages with kernel code, so whole
    /// pages would thrash).
    pub(crate) code_pages: Box<[bool]>,
    pub(crate) dirty_code_writes: Vec<u32>,
    /// A TLB rewrite invalidated every recompiled block.
    pub(crate) jit_flush_needed: bool,
    /// Addresses of `ram` and `code_pages` for the recompiler's inline RAM
    /// fast paths (the boxes never move; kept as integers so Bus stays Send).
    pub(crate) ram_ptr: usize,
    pub(crate) code_pages_ptr: usize,
    /// Instruction-fetch page cache: virtual page tag and its RAM offset
    /// (tag 1 never matches an aligned page).
    fetch_tag: u32,
    fetch_base: usize,
    /// Deferred EE DMAC completion interrupts: (D_STAT bit, due cycle).
    /// Data moves instantly but completion must not fire inside the very
    /// instruction that started the transfer.
    dma_irq_queue: Vec<(u32, u64)>,
}

impl Bus {
    pub fn new(bios: Vec<u8>, gs_threaded: bool, region: Region) -> Self {
        assert_eq!(bios.len(), BIOS_SIZE);
        let mut mmio = vec![0u8; MMIO_SIZE].into_boxed_slice();
        // DMAC ENABLER resets to 0x1201; the BIOS uses it as a board-revision
        // key into its RDRAM configuration table during InitRDRAM.
        write_le::<4>(&mut mmio, 0xF590, 0x1201);
        let mut bus = Self {
            ram: vec![0u8; RAM_SIZE].into_boxed_slice(),
            bios: bios.into_boxed_slice(),
            spad: vec![0u8; SPAD_SIZE].into_boxed_slice(),
            iop_ram: vec![0u8; 2 * 1024 * 1024].into_boxed_slice(),
            timers_due: 0,
            mmio,
            gs: if gs_threaded { GsFront::new() } else { GsFront::inline() },
            gif: Gif::new(),
            vif0: Vif::new(),
            vif1: Vif::new(),
            vu0: Vu1::new(),
            vu1: Vu1::new(),
            region,
            timers: Timers::new(),
            sif: Sif::new(),
            iop_spad: vec![0u8; 1024].into_boxed_slice(),
            spu2: Spu2::new(),
            iop_dma_spu: Default::default(),
            iop_mmio: vec![0u8; 0x10000].into_boxed_slice(),
            iop_i_stat: 0,
            iop_i_mask: 0,
            iop_i_ctrl: 0,
            iop_timers: [IopTimer::default(); 6],
            intc_stat: 0,
            intc_ready_at: u64::MAX,
            intc_mask: 0,
            dma_vif0: EeDmaChannel::default(),
            dma_vif1: EeDmaChannel::default(),
            dma_gif: EeDmaChannel::default(),
            dma_spr_from: EeDmaChannel::default(),
            dma_spr_to: EeDmaChannel::default(),
            ipu: crate::ipu::Ipu::new(),
            dma_ipu_to: EeDmaChannel::default(),
            dma_sif0: EeDmaChannel::default(),
            dma_sif1: EeDmaChannel::default(),
            d_stat: 0,
            d_mask: 0,
            iop_dma_sif0: IopDmaChannel::default(),
            iop_dma_sif1: IopDmaChannel::default(),
            iop_dicr: 0,
            iop_dicr2: 0,
            iop_dma_sio2in: IopDmaChannel::default(),
            iop_dma_sio2out: IopDmaChannel::default(),
            iop_dma_cdvd: IopDmaChannel::default(),
            sio2out_deferred: false,
            cdvd_dma_deferred: false,
            cdvd_done_at: None,
            iop_resets: 0,
            cdvd: Cdvd { nvram: nvram_defaults(), ..Cdvd::default() },
            sio2: Sio2::default(),
            now: 0,
            tty_buffer: String::new(),
            tty_line: String::new(),
            rdram_sdevid: 0,
            warned_unmapped: HashSet::new(),
            ee_tlb: [(0, 0, 0, 0); 48],
            tlb_cache: vec![(0u32, 0u32); 1024].into_boxed_slice(),
            code_pages: vec![false; RAM_SIZE >> 12].into_boxed_slice(),
            dirty_code_writes: Vec::new(),
            jit_flush_needed: false,
            ram_ptr: 0,
            code_pages_ptr: 0,
            fetch_tag: 1,
            fetch_base: 0,
            dma_irq_queue: Vec::new(),
        };
        bus.ram_ptr = bus.ram.as_mut_ptr() as usize;
        bus.code_pages_ptr = bus.code_pages.as_ptr() as usize;
        bus
    }

    /// Queue an EE DMAC completion interrupt a little into the future.
    /// Report an unmodelled corner of the machine once, so the log names the
    /// gap without drowning in it.
    fn warn_stub(&mut self, addr: u32, what: &str) {
        if self.warned_unmapped.insert(addr & !0xFFF) {
            warn!(target: "ps2_core::bus::stub", addr = format_args!("{addr:#010x}"), what,
                "not modelled (reported once per page)");
        }
    }

    /// A DMA channel with nothing behind it: complete the transfer so the
    /// software does not wait on it forever, and say so.
    fn ee_dma_stub<const N: usize>(&mut self, addr: u32, ch: u32, v: u64, what: &str) {
        let off = (addr & 0xFFFF) as usize;
        if v as u32 & EE_CHCR_STR == 0 {
            write_le::<N>(&mut self.mmio, off, v);
            return;
        }
        self.warn_stub(addr, what);
        write_le::<N>(&mut self.mmio, off, v & !(EE_CHCR_STR as u64));
        self.ee_dma_irq(ch);
    }

    fn ee_dma_irq(&mut self, ch: u32) {
        self.dma_irq_queue.push((1 << ch, self.now + 1024));
        self.timers_due = 0;
    }

    /// Record a TLB entry (from tlbwi) and flush the translation cache.
    pub fn ee_tlb_write(&mut self, idx: usize, mask: u32, hi: u32, lo0: u32, lo1: u32) {
        if idx < 48 {
            self.ee_tlb[idx] = (mask, hi, lo0, lo1);
            self.tlb_cache.fill((0, 0));
            self.fetch_tag = 1;
            // Blocks were keyed by virtual pc under the old mapping.
            self.jit_flush_needed = true;
        }
    }

    /// Physical RAM address a virtual EE address maps to, if RAM at all.
    pub fn ram_phys_of(&mut self, vaddr: u32) -> Option<u32> {
        let phys = self.translate(vaddr);
        ((phys as usize) < RAM_SIZE).then_some(phys)
    }

    /// A physical RAM address was written: if recompiled code lives on its
    /// page, queue the address so the recompiler can drop blocks covering
    /// it. The dispatcher drains the queue after every block, so it stays
    /// short even for a hot variable next to code.
    #[inline(always)]
    fn note_ram_write(&mut self, addr: usize) {
        if self.code_pages[addr >> 12] {
            let a = (addr & !7) as u32;
            if self.dirty_code_writes.last() != Some(&a) {
                self.dirty_code_writes.push(a);
            }
        }
    }

    /// Walk the TLB for a mapped-segment address. Returns a physical
    /// address; scratchpad hits map into a reserved range above VRAM-visible
    /// physical space (0x7000_0000 window preserved).
    fn tlb_lookup(&mut self, vaddr: u32) -> u32 {
        let slot = ((vaddr >> 12) & 1023) as usize;
        let (tag, base) = self.tlb_cache[slot];
        if tag == (vaddr >> 12) | 0x8000_0000 {
            return (base << 12) | (vaddr & 0xFFF);
        }
        for &(mask, hi, lo0, lo1) in &self.ee_tlb {
            if hi == 0 && lo0 == 0 && lo1 == 0 {
                continue;
            }
            let page_size = ((mask >> 13) + 1) << 12;
            let pair_mask = !(page_size * 2 - 1);
            if (vaddr & pair_mask) != (hi & pair_mask) {
                continue;
            }
            if lo0 & 0x8000_0000 != 0 {
                // Scratchpad entry: 16 KiB window.
                return 0x7000_0000 | (vaddr & 0x3FFF);
            }
            let odd = vaddr & page_size != 0;
            let lo = if odd { lo1 } else { lo0 };
            if lo & 2 == 0 {
                continue; // invalid half
            }
            let phys = ((lo >> 6) << 12) | (vaddr & (page_size - 1));
            self.tlb_cache[slot] = ((vaddr >> 12) | 0x8000_0000, phys >> 12);
            return phys;
        }
        // No mapping: fall back to a direct fold so early boot keeps working.
        if self.warned_unmapped.insert(vaddr & !0xFFF) {
            warn!(target: "ps2_core::bus", vaddr = format_args!("{vaddr:#010x}"), "EE access with no TLB mapping (direct fold)");
        }
        vaddr & 0x1FFF_FFFF
    }

    /// Translate an EE virtual address: KSEG0/1 fold directly, everything
    /// else goes through the TLB (with common fixed mappings fast-pathed).
    #[inline]
    fn translate(&mut self, vaddr: u32) -> u32 {
        match vaddr {
            // KSEG0 / KSEG1: unmapped segments.
            0x8000_0000..=0xBFFF_FFFF => vaddr & 0x1FFF_FFFF,
            // Scratchpad window (kernel TLB entry 0, effectively fixed).
            0x7000_0000..=0x7000_3FFF => vaddr,
            // Identity-mapped low RAM (kuseg): skip the walk.
            0x0000_0000..=0x01FF_FFFF => vaddr,
            _ => self.tlb_lookup(vaddr),
        }
    }

    #[inline]
    pub fn read8(&mut self, vaddr: u32) -> u8 {
        self.read::<1>(vaddr) as u8
    }
    #[inline]
    pub fn read16(&mut self, vaddr: u32) -> u16 {
        self.read::<2>(vaddr) as u16
    }
    #[inline]
    pub fn read32(&mut self, vaddr: u32) -> u32 {
        self.read::<4>(vaddr) as u32
    }
    #[inline]
    pub fn read64(&mut self, vaddr: u32) -> u64 {
        self.read::<8>(vaddr)
    }
    /// 128-bit read (lq); address is 16-byte aligned by the caller.
    pub fn read128(&mut self, vaddr: u32) -> [u64; 2] {
        [self.read::<8>(vaddr), self.read::<8>(vaddr + 8)]
    }

    #[inline]
    pub fn write8(&mut self, vaddr: u32, v: u8) {
        self.write::<1>(vaddr, v as u64)
    }
    #[inline]
    pub fn write16(&mut self, vaddr: u32, v: u16) {
        self.write::<2>(vaddr, v as u64)
    }
    #[inline]
    pub fn write32(&mut self, vaddr: u32, v: u32) {
        self.write::<4>(vaddr, v as u64)
    }
    #[inline]
    pub fn write64(&mut self, vaddr: u32, v: u64) {
        self.write::<8>(vaddr, v)
    }
    pub fn write128(&mut self, vaddr: u32, v: [u64; 2]) {
        self.write::<8>(vaddr, v[0]);
        self.write::<8>(vaddr + 8, v[1]);
    }

    /// Instruction fetch: RAM pages hit a one-entry page cache; anything
    /// else (BIOS, unmapped) takes the data-read path.
    #[inline]
    pub fn fetch32(&mut self, vaddr: u32) -> u32 {
        if vaddr & !0xFFF == self.fetch_tag {
            return read_le::<4>(&self.ram, self.fetch_base + (vaddr & 0xFFF) as usize) as u32;
        }
        let addr = self.translate(vaddr);
        if (addr as usize) < RAM_SIZE {
            self.fetch_tag = vaddr & !0xFFF;
            self.fetch_base = (addr & !0xFFF) as usize;
            return read_le::<4>(&self.ram, addr as usize) as u32;
        }
        self.read32(vaddr)
    }

    // --- debugger access (side-effect-free) ------------------------------

    /// Side-effect-free EE-side read for the debugger. Takes `&mut self`
    /// only because translation feeds the TLB cache (invisible to software).
    /// MMIO is served only where the read is a plain load of state (see
    /// [`Bus::peek_mmio`]); anything that would advance a device returns
    /// `None`.
    pub fn peek8(&mut self, vaddr: u32) -> Option<u8> {
        let addr = self.translate(vaddr);
        match addr {
            0x0000_0000..=0x01FF_FFFF => Some(self.ram[addr as usize]),
            0x7000_0000..=0x7000_3FFF => Some(self.spad[(addr & 0x3FFF) as usize]),
            0x1C00_0000..=0x1C1F_FFFF => Some(self.iop_ram[(addr & 0x1F_FFFF) as usize]),
            0x1FC0_0000..=0x1FFF_FFFF => Some(self.bios[(addr & 0x3F_FFFF) as usize]),
            _ => self.peek_mmio(addr),
        }
    }

    /// Side-effect-free MMIO byte read for the debugger.
    ///
    /// Only registers whose read path is a plain load of state are served,
    /// so that a peek can never advance the machine: FIFOs, the RDRAM
    /// handshake at MCH_DRD and everything not listed stay refused
    /// (`None`), rather than being answered with a value a real read would
    /// not have produced.
    fn peek_mmio(&mut self, addr: u32) -> Option<u8> {
        // GS privileged registers are 64 bits wide; `priv_read` is a shadow
        // load (CSR's write-1-to-clear lives on the write path).
        if let 0x1200_0000..=0x1200_1FFF = addr {
            let v = self.gs.priv_read(addr & !0x7);
            return Some((v >> ((addr & 7) * 8)) as u8);
        }
        let reg = addr & !0x3;
        let chan = |c: &EeDmaChannel, reg: u32| match reg & 0x30 {
            0x00 => Some(c.chcr),
            0x10 => Some(c.madr),
            0x20 => Some(c.qwc),
            0x30 => Some(c.tadr),
            _ => None,
        };
        let v = match reg {
            // EE timers: the count derives from `now`, nothing latches.
            0x1000_0000..=0x1000_1FFF => self.timers.read(reg, self.now, self.region),
            // EE DMAC channels and its interrupt status/mask.
            0x1000_8000..=0x1000_803F => chan(&self.dma_vif0, reg)?,
            0x1000_9000..=0x1000_903F => chan(&self.dma_vif1, reg)?,
            0x1000_A000..=0x1000_A03F => chan(&self.dma_gif, reg)?,
            0x1000_D000..=0x1000_D03F => chan(&self.dma_spr_from, reg)?,
            0x1000_D080 => self.dma_spr_from.sadr,
            0x1000_D400..=0x1000_D43F => chan(&self.dma_spr_to, reg)?,
            0x1000_D480 => self.dma_spr_to.sadr,
            0x1000_B400..=0x1000_B43F => chan(&self.dma_ipu_to, reg)?,
            0x1000_C000..=0x1000_C03F => chan(&self.dma_sif0, reg)?,
            0x1000_C400..=0x1000_C43F => chan(&self.dma_sif1, reg)?,
            0x1000_E010 => self.d_stat | (self.d_mask << 16),
            // EE INTC.
            0x1000_F000 => self.intc_stat,
            0x1000_F010 => self.intc_mask,
            // SIF mailboxes, flags and the control handshake.
            0x1000_F200..=0x1000_F26F => self.sif.ee_read(reg),
            _ => return None,
        };
        Some((v >> ((addr & 3) * 8)) as u8)
    }

    /// Side-effect-free EE-side write for the debugger. ROM and MMIO are
    /// refused (`false`).
    pub fn poke8(&mut self, vaddr: u32, v: u8) -> bool {
        let addr = self.translate(vaddr);
        match addr {
            0x0000_0000..=0x01FF_FFFF => {
                self.ram[addr as usize] = v;
                self.note_ram_write(addr as usize);
            }
            0x7000_0000..=0x7000_3FFF => self.spad[(addr & 0x3FFF) as usize] = v,
            0x1C00_0000..=0x1C1F_FFFF => self.iop_ram[(addr & 0x1F_FFFF) as usize] = v,
            _ => return false,
        }
        true
    }

    /// Side-effect-free IOP-side read for the debugger.
    pub fn iop_peek8(&self, vaddr: u32) -> Option<u8> {
        if vaddr >= 0xFFFE_0000 {
            return None; // KSEG2 cache control
        }
        let addr = vaddr & 0x1FFF_FFFF;
        match addr {
            0x0000_0000..=0x007F_FFFF => Some(self.iop_ram[(addr & 0x1F_FFFF) as usize]),
            0x1F80_0000..=0x1F80_03FF => Some(self.iop_spad[(addr & 0x3FF) as usize]),
            0x1FC0_0000..=0x1FFF_FFFF => Some(self.bios[(addr & 0x3F_FFFF) as usize]),
            _ => None,
        }
    }

    /// Side-effect-free IOP-side write for the debugger.
    pub fn iop_poke8(&mut self, vaddr: u32, v: u8) -> bool {
        if vaddr >= 0xFFFE_0000 {
            return false;
        }
        let addr = vaddr & 0x1FFF_FFFF;
        match addr {
            0x0000_0000..=0x007F_FFFF => self.iop_ram[(addr & 0x1F_FFFF) as usize] = v,
            0x1F80_0000..=0x1F80_03FF => self.iop_spad[(addr & 0x3FF) as usize] = v,
            _ => return false,
        }
        true
    }

    fn read<const N: usize>(&mut self, vaddr: u32) -> u64 {
        let addr = self.translate(vaddr);
        match addr {
            0x0000_0000..=0x01FF_FFFF => read_le::<N>(&self.ram, addr as usize),
            0x7000_0000..=0x7000_3FFF => read_le::<N>(&self.spad, (addr & 0x3FFF) as usize),
            0x1000_0000..=0x1000_FFFF => self.read_mmio::<N>(addr),
            // VU memory windows. Each VU's pair of 4 KB (VU0) or 16 KB
            // (VU1) memories mirrors within its own window.
            0x1100_8000..=0x1100_BFFF => {
                read_le::<N>(&self.vu1.micro, (addr & 0x3FFF) as usize)
            }
            0x1100_C000..=0x1100_FFFF => {
                read_le::<N>(&self.vu1.data, (addr & 0x3FFF) as usize)
            }
            0x1100_0000..=0x1100_3FFF => {
                read_le::<N>(&self.vu0.micro, addr as usize & self.vu0.micro_mask)
            }
            0x1100_4000..=0x1100_7FFF => {
                read_le::<N>(&self.vu0.data, addr as usize & (self.vu0.data.len() - 1))
            }
            0x1200_0000..=0x1200_1FFF => self.read_gs_priv::<N>(addr),
            0x1C00_0000..=0x1C1F_FFFF => read_le::<N>(&self.iop_ram, (addr & 0x1F_FFFF) as usize),
            0x1F80_0000..=0x1F80_FFFF => {
                // IOP MMIO window as seen from the EE.
                self.warn_stub(addr, "IOP MMIO window seen from the EE");
                0
            }
            0x1FC0_0000..=0x1FFF_FFFF => read_le::<N>(&self.bios, (addr & 0x3F_FFFF) as usize),
            // CDVD (MECHACON) registers, also visible from the EE: the OSD
            // peeks N-status directly.
            0x1F40_2000..=0x1F40_203F => self.cdvd.read(addr, self.now) as u64,
            // ROM1 (DVD player ROM): absent, reads like erased flash.
            0x1E00_0000..=0x1E3F_FFFF => u64::MAX >> (64 - 8 * N as u32),
            // SBUS CRT-controller command interface used by ROMGSCRT:
            // +0x06 status (bit1 = command done, bit0 = busy), +0x10 data.
            0x1A00_0000..=0x1A00_FFFF => {
                self.warn_stub(addr, "SBUS");
                match addr & 0xFF {
                    0x06 => 2,
                    _ => 0,
                }
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::bus", addr = format_args!("{addr:#010x}"), size = N, "read from unmapped address (reported once)");
                }
                0
            }
        }
    }

    fn write<const N: usize>(&mut self, vaddr: u32, v: u64) {
        let addr = self.translate(vaddr);
        match addr {
            0x0000_0000..=0x01FF_FFFF => {
                write_le::<N>(&mut self.ram, addr as usize, v);
                self.note_ram_write(addr as usize);
            }
            0x7000_0000..=0x7000_3FFF => write_le::<N>(&mut self.spad, (addr & 0x3FFF) as usize, v),
            0x1000_0000..=0x1000_FFFF => self.write_mmio::<N>(addr, v),
            0x1100_8000..=0x1100_BFFF => {
                self.vu1.write_micro((addr & 0x3FFF) as usize, &v.to_le_bytes()[..N])
            }
            0x1100_C000..=0x1100_FFFF => {
                write_le::<N>(&mut self.vu1.data, (addr & 0x3FFF) as usize, v)
            }
            0x1100_0000..=0x1100_3FFF => {
                let m = self.vu0.micro_mask;
                self.vu0.write_micro(addr as usize & m, &v.to_le_bytes()[..N])
            }
            0x1100_4000..=0x1100_7FFF => {
                let m = self.vu0.data.len() - 1;
                write_le::<N>(&mut self.vu0.data, addr as usize & m, v)
            }
            0x1200_0000..=0x1200_1FFF => self.write_gs_priv::<N>(addr, v),
            0x1C00_0000..=0x1C1F_FFFF => {
                write_le::<N>(&mut self.iop_ram, (addr & 0x1F_FFFF) as usize, v)
            }
            0x1F80_0000..=0x1F80_FFFF => {
                self.warn_stub(addr, "IOP MMIO window seen from the EE");
            }
            0x1FC0_0000..=0x1FFF_FFFF => {
                warn!(target: "ps2_core::bus", addr = format_args!("{addr:#010x}"), "write to BIOS ROM ignored");
            }
            0x1A00_0000..=0x1A00_FFFF => {
                self.warn_stub(addr, "SBUS");
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), size = N, "write to unmapped address (reported once)");
                }
            }
        }
    }

    fn read_mmio<const N: usize>(&mut self, addr: u32) -> u64 {
        let off = (addr & 0xFFFF) as usize;
        match addr & !0x3 {
            0x1000_0000..=0x1000_1FFF => self.timers.read(addr, self.now, self.region) as u64,
            // SIO_ISR: no pending serial interrupts.
            0x1000_F130 => 0,
            // MCH_RICM reads back as 0 (busy bit clear = operation done).
            0x1000_F430 => {
                trace!(target: "ps2_core::bus::mch", "RICM read -> 0");
                0
            }
            // MCH_DRD: RDRAM init handshake, mirrors the documented sequence.
            0x1000_F440 => {
                let ricm = read_le::<4>(&self.mmio, 0xF430) as u32;
                let sop = (ricm >> 6) & 0xF;
                let sa = (ricm >> 16) & 0xFFF;
                trace!(target: "ps2_core::bus::mch", ricm = format_args!("{ricm:#010x}"), sop, sa = format_args!("{sa:#x}"), "DRD read");
                if sop == 0 {
                    match sa {
                        0x21 => {
                            // INIT: each device answers once.
                            if self.rdram_sdevid < RDRAM_DEVICES {
                                self.rdram_sdevid += 1;
                                0x1F
                            } else {
                                0
                            }
                        }
                        0x23 => 0x0D0D,               // CNFGA
                        0x24 => 0x0090,               // CNFGB
                        0x40 => (ricm & 0x1F) as u64, // DEVID
                        _ => 0,
                    }
                } else {
                    0
                }
            }
            // IPU registers. IPU_CMD and IPU_TOP are 64-bit, with a busy
            // marker in the upper half, so route the width through.
            0x1000_2000..=0x1000_203F => {
                if N == 8 {
                    self.ipu.read64(addr)
                } else {
                    u64::from(self.ipu.read32(addr))
                }
            }
            // IPU output FIFO. Nothing decodes yet, so it stays empty.
            0x1000_7000..=0x1000_700F => 0,
            // IPU_TO (ch4): the input FIFO is fed by the channel, and the
            // channel is drained by whatever command is waiting on it.
            0x1000_B400 => self.dma_ipu_to.chcr as u64,
            0x1000_B410 => self.dma_ipu_to.madr as u64,
            0x1000_B420 => self.dma_ipu_to.qwc as u64,
            0x1000_B430 => self.dma_ipu_to.tadr as u64,
            // GIF_STAT. Channel 2 and the FIFO both run to completion inside
            // the write that starts them, so the GIF is idle whenever the EE
            // can look: no path queued or active, FIFO empty, EE-to-GS
            // direction. The mask bits would come from GIF_MODE, which
            // nothing has needed yet.
            0x1000_3020 => 0,
            // VIF0_STAT / VIF1_STAT. Same story — the parser is never caught
            // mid-code. FDR is kept as written; GS-to-EE readback through the
            // FIFO is not modelled, and a set FDR is warned about on write.
            0x1000_3800 | 0x1000_3C00 => read_le::<4>(&self.mmio, off & !3) & VIF_STAT_FDR,
            0x1000_8000 => self.dma_vif0.chcr as u64,
            0x1000_8010 => self.dma_vif0.madr as u64,
            0x1000_8020 => self.dma_vif0.qwc as u64,
            0x1000_8030 => self.dma_vif0.tadr as u64,
            // EE DMAC: VIF0 (ch0), VIF1 (ch1), GIF (ch2), SIF0 (ch5) / SIF1 (ch6),
            // interrupt status. Transfers run to completion inside the CHCR
            // write, so a poll of the start bit always sees it clear.
            0x1000_9000 => self.dma_vif1.chcr as u64,
            0x1000_9010 => self.dma_vif1.madr as u64,
            0x1000_9020 => self.dma_vif1.qwc as u64,
            0x1000_9030 => self.dma_vif1.tadr as u64,
            0x1000_A000 => self.dma_gif.chcr as u64,
            0x1000_A010 => self.dma_gif.madr as u64,
            0x1000_A020 => self.dma_gif.qwc as u64,
            0x1000_A030 => self.dma_gif.tadr as u64,
            0x1000_D000 => self.dma_spr_from.chcr as u64,
            0x1000_D010 => self.dma_spr_from.madr as u64,
            0x1000_D020 => self.dma_spr_from.qwc as u64,
            0x1000_D030 => self.dma_spr_from.tadr as u64,
            0x1000_D080 => self.dma_spr_from.sadr as u64,
            0x1000_D400 => self.dma_spr_to.chcr as u64,
            0x1000_D410 => self.dma_spr_to.madr as u64,
            0x1000_D420 => self.dma_spr_to.qwc as u64,
            0x1000_D430 => self.dma_spr_to.tadr as u64,
            0x1000_D480 => self.dma_spr_to.sadr as u64,
            0x1000_C000 => self.dma_sif0.chcr as u64,
            0x1000_C010 => self.dma_sif0.madr as u64,
            0x1000_C020 => self.dma_sif0.qwc as u64,
            0x1000_C400 => self.dma_sif1.chcr as u64,
            0x1000_C410 => self.dma_sif1.madr as u64,
            0x1000_C420 => self.dma_sif1.qwc as u64,
            0x1000_C430 => self.dma_sif1.tadr as u64,
            0x1000_E010 => (self.d_stat | (self.d_mask << 16)) as u64,
            // EE INTC.
            0x1000_F000 => self.intc_stat as u64,
            0x1000_F010 => self.intc_mask as u64,
            // SIF registers (mailboxes, flags, control handshake).
            0x1000_F200..=0x1000_F26F => self.sif.ee_read(addr) as u64,
            // DMAC ENABLER reads back the value written to ENABLEW.
            0x1000_F520 => read_le::<4>(&self.mmio, 0xF590),
            _ => {
                let v = read_le::<N>(&self.mmio, off);
                trace!(target: "ps2_core::bus::mmio", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "MMIO read (shadow)");
                v
            }
        }
    }

    fn write_mmio<const N: usize>(&mut self, addr: u32, v: u64) {
        match addr & !0x3 {
            0x1000_0000..=0x1000_1FFF => {
                self.timers.write(addr, v as u32, self.now);
                self.timers_due = 0;
                return;
            }
            // EE SIO TXFIFO: the kernel's debug output channel. Pure
            // observation — never affects execution.
            0x1000_F180 => {
                self.tty_push(v as u8);
                return;
            }
            // GIF channel: starting a transfer runs it to completion.
            0x1000_A000 => {
                self.dma_gif.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_gif();
                }
                return;
            }
            0x1000_A010 => {
                self.dma_gif.madr = v as u32;
                return;
            }
            0x1000_A020 => {
                self.dma_gif.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_A030 => {
                self.dma_gif.tadr = v as u32;
                return;
            }
            // Channels with nothing behind them. They complete immediately so
            // nothing waits forever, and each names itself once in the log.
            // VIF0 channel: uploads microcode and data into VU0 and can
            // start it, the same way channel 1 drives VU1.
            0x1000_8000 => {
                self.dma_vif0.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_vif0();
                }
                return;
            }
            0x1000_8010 => {
                self.dma_vif0.madr = v as u32;
                return;
            }
            0x1000_8020 => {
                self.dma_vif0.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_8030 => {
                self.dma_vif0.tadr = v as u32;
                return;
            }
            // VIF0 FIFO: programmed writes feed the same parser.
            0x1000_4000..=0x1000_4FF0 => {
                for i in 0..(N as u32 / 4) {
                    let w = (v >> (32 * i)) as u32;
                    let Bus { vif0, gs, gif, vu0, .. } = self;
                    vif0.push_word(gs, gif, vu0, w);
                }
                return;
            }
            0x1000_B000 => {
                self.ee_dma_stub::<N>(addr, 3, v, "IPU_FROM channel (no IPU)");
                return;
            }
            0x1000_C800 => {
                self.ee_dma_stub::<N>(addr, 7, v, "SIF2 channel");
                return;
            }
            // Scratchpad channels: like the others, a set start bit runs
            // the whole transfer inside this write.
            0x1000_D000 => {
                self.dma_spr_from.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_spr_from();
                }
                return;
            }
            0x1000_D010 => {
                self.dma_spr_from.madr = v as u32;
                return;
            }
            0x1000_D020 => {
                self.dma_spr_from.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_D030 => {
                self.dma_spr_from.tadr = v as u32;
                return;
            }
            0x1000_D080 => {
                self.dma_spr_from.sadr = v as u32 & 0x3FFF;
                return;
            }
            0x1000_D400 => {
                self.dma_spr_to.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_spr_to();
                }
                return;
            }
            0x1000_D410 => {
                self.dma_spr_to.madr = v as u32;
                return;
            }
            0x1000_D420 => {
                self.dma_spr_to.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_D430 => {
                self.dma_spr_to.tadr = v as u32;
                return;
            }
            0x1000_D480 => {
                self.dma_spr_to.sadr = v as u32 & 0x3FFF;
                return;
            }
            // VIF1 channel: starting a transfer runs it to completion.
            0x1000_9000 => {
                self.dma_vif1.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_vif1();
                }
                return;
            }
            0x1000_9010 => {
                self.dma_vif1.madr = v as u32;
                return;
            }
            0x1000_9020 => {
                self.dma_vif1.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_9030 => {
                self.dma_vif1.tadr = v as u32;
                return;
            }
            // IPU registers. A finished command raises the IPU interrupt.
            0x1000_2000..=0x1000_203F => {
                let done = if N == 8 {
                    self.ipu.write64(addr, v)
                } else {
                    self.ipu.write32(addr, v as u32)
                };
                if done {
                    self.intc_stat |= 1 << 8;
                    self.intc_changed();
                }
                return;
            }
            // IPU input FIFO written by programmed I/O: assembled in the
            // shadow and handed over a quadword at a time, like the GIF's.
            0x1000_7010..=0x1000_701F => {
                let off = (addr & 0xFFFF) as usize;
                write_le::<N>(&mut self.mmio, off, v);
                if (addr as usize & 0xF) + N >= 16 {
                    let base = off & !0xF;
                    let mut q = [0u8; 16];
                    q.copy_from_slice(&self.mmio[base..base + 16]);
                    if self.ipu.push_in(q) {
                        self.intc_stat |= 1 << 8;
                        self.intc_changed();
                    }
                }
                return;
            }
            // IPU_TO (ch4): feed the decoder's input FIFO.
            0x1000_B400 => {
                self.dma_ipu_to.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_ipu_to();
                }
                return;
            }
            0x1000_B410 => {
                self.dma_ipu_to.madr = v as u32;
                return;
            }
            0x1000_B420 => {
                self.dma_ipu_to.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_B430 => {
                self.dma_ipu_to.tadr = v as u32;
                return;
            }
            // GIF FIFO: PATH3 fed by programmed writes instead of channel 2
            // (the BIOS initialises the GS register file this way). The
            // quadword is assembled in place and handed to the same parser
            // once its last byte lands.
            0x1000_6000..=0x1000_6FF0 => {
                let off = (addr & 0xFFFF) as usize;
                write_le::<N>(&mut self.mmio, off, v);
                if (addr as usize & 0xF) + N >= 16 {
                    let base = off & !0xF;
                    let lo = read_le::<8>(&self.mmio, base);
                    let hi = read_le::<8>(&self.mmio, base + 8);
                    self.gif.process(&mut self.gs, lo, hi);
                }
                return;
            }
            // VIF1_STAT is read-only apart from FDR, which picks the FIFO's
            // direction. Nothing has needed the GS-to-EE side yet.
            0x1000_3C00 if v & u64::from(VIF_STAT_FDR) != 0 => {
                warn!(target: "ps2_core::bus::vif", "VIF1 GS-to-EE readback requested (not modelled)");
            }
            // VIF1 FIFO: direct programmed writes feed the same parser.
            0x1000_5000..=0x1000_5FF0 => {
                for i in 0..(N as u32 / 4) {
                    let w = (v >> (32 * i)) as u32;
                    self.vif1
                        .push_word(&mut self.gs, &mut self.gif, &mut self.vu1, w);
                }
                return;
            }
            // EE DMAC SIF channels: starting a transfer pumps it to completion.
            0x1000_C000 => {
                self.dma_sif0.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_sif();
                }
                return;
            }
            0x1000_C010 => {
                self.dma_sif0.madr = v as u32;
                return;
            }
            0x1000_C020 => {
                self.dma_sif0.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_C400 => {
                self.dma_sif1.chcr = v as u32;
                if v as u32 & EE_CHCR_STR != 0 {
                    self.pump_sif();
                }
                return;
            }
            0x1000_C410 => {
                self.dma_sif1.madr = v as u32;
                return;
            }
            0x1000_C420 => {
                self.dma_sif1.qwc = v as u32 & 0xFFFF;
                return;
            }
            0x1000_C430 => {
                self.dma_sif1.tadr = v as u32;
                return;
            }
            // D_STAT: low half write-1-to-clear, high half toggles the mask.
            0x1000_E010 => {
                self.d_stat &= !(v as u32 & 0xFFFF);
                self.d_mask ^= (v as u32 >> 16) & 0xFFFF;
                return;
            }
            // INTC: STAT is write-1-to-clear, MASK is write-1-to-toggle.
            0x1000_F000 => {
                self.intc_stat &= !(v as u32);
                self.intc_changed();
                return;
            }
            0x1000_F010 => {
                self.intc_mask ^= v as u32 & 0xFFFF;
                self.intc_changed();
                return;
            }
            0x1000_F200..=0x1000_F26F => {
                self.sif.ee_write(addr, v as u32);
                return;
            }
            // RDRAM controller command register: busy bit (31) self-clears.
            0x1000_F410 => {
                trace!(target: "ps2_core::bus::mch", value = format_args!("{v:#010x}"), "F410 write");
                write_le::<4>(&mut self.mmio, 0xF410, v & !0x8000_0000);
                return;
            }
            // MCH_RICM: busy bit self-clears; INIT restarts device counting.
            0x1000_F430 => {
                let sa = ((v >> 16) & 0xFFF) as u32;
                let sbc = ((v >> 6) & 0xF) as u32;
                trace!(target: "ps2_core::bus::mch", value = format_args!("{v:#010x}"), sop = sbc, sa = format_args!("{sa:#x}"), "RICM write");
                if sa == 0x21 && sbc == 1 && (read_le::<4>(&self.mmio, 0xF440) >> 7) & 1 == 0 {
                    self.rdram_sdevid = 0;
                }
                write_le::<4>(&mut self.mmio, 0xF430, v & !0x8000_0000);
                return;
            }
            _ => {}
        }
        trace!(target: "ps2_core::bus::mmio", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "MMIO write (shadow)");
        write_le::<N>(&mut self.mmio, (addr & 0xFFFF) as usize, v);
    }

    fn read_gs_priv<const N: usize>(&mut self, addr: u32) -> u64 {
        let v = self.gs.priv_read(addr & !0x7);
        if N == 8 {
            v
        } else if addr & 4 != 0 {
            v >> 32
        } else {
            v & 0xFFFF_FFFF
        }
    }

    fn write_gs_priv<const N: usize>(&mut self, addr: u32, v: u64) {
        let v = if N == 8 {
            v
        } else {
            // 32-bit access: merge into the 64-bit register.
            let cur = self.gs.priv_read(addr & !0x7);
            if addr & 4 != 0 {
                (cur & 0xFFFF_FFFF) | (v << 32)
            } else {
                (cur & !0xFFFF_FFFF) | (v & 0xFFFF_FFFF)
            }
        };
        self.gs.priv_write(addr & !0x7, v);
        if addr & 0x1FF0 == 0x0010 {
            self.set_region_from_smode1(v);
        }
        self.gs_sync_int();
    }

    /// Follow the CRTC mode software programmed: SMODE1's CMOD field selects
    /// the composite encoder, and the kernel's `SetGsCrt` writes it from the
    /// caller's `pal_ntsc` argument. Other values are the progressive and
    /// DTV modes, whose refresh is not encoded here, so they leave the rate
    /// as it is.
    fn set_region_from_smode1(&mut self, v: u64) {
        let region = match (v >> 13) & 3 {
            2 => Region::Ntsc,
            3 => Region::Pal,
            _ => return,
        };
        if self.region == region {
            return;
        }
        debug!(target: "ps2_core::gs::crtc", ?region, "video timing changed");
        // Counts are reconstructed from (now - base_cycle), so the hblank
        // divisor changing would reinterpret the whole elapsed span. Freeze
        // each timer at the count it has under the old region first.
        let (now, old) = (self.now, self.region);
        self.timers.rebase(now, old);
        for (t, timer) in self.iop_timers.iter_mut().enumerate() {
            timer.rebase(t, now, old);
        }
        self.region = region;
    }

    /// Fold a pending GS interrupt edge into EE INTC bit 0.
    fn gs_sync_int(&mut self) {
        if self.gs.intc_pending {
            self.gs.intc_pending = false;
            self.intc_stat |= 1;
            self.intc_changed();
        }
    }

    // --- periodic events -------------------------------------------------

    /// Edge-detect timer interrupts on both sides. Called periodically.
    pub fn tick_timers(&mut self) {
        // Deliver a due CDVD N-command completion: internal flags, the IOP
        // interrupt line, and any DMA that was armed while the drive worked.
        if let Some(at) = self.cdvd_done_at
            && at <= self.now
        {
            self.cdvd_done_at = None;
            self.cdvd.finish_n();
            self.iop_i_stat |= 1 << 2;
            if self.cdvd_dma_deferred && self.cdvd.read_remaining() > 0 {
                self.cdvd_dma_deferred = false;
                self.do_cdvd_dma();
            }
        }
        // Deliver due deferred DMA completion interrupts.
        let mut i = 0;
        while i < self.dma_irq_queue.len() {
            if self.dma_irq_queue[i].1 <= self.now {
                self.d_stat |= self.dma_irq_queue[i].0;
                self.dma_irq_queue.swap_remove(i);
            } else {
                i += 1;
            }
        }
        let spu_done = {
            let _p = prof::scope(prof::Slot::Spu2);
            self.spu2.tick(self.now)
        };
        for (core, done) in spu_done.into_iter().enumerate() {
            if done {
                self.iop_dma_spu[core].chcr &= !IOP_CHCR_BUSY;
                self.iop_dma_irq(if core == 0 { 4 } else { 7 });
            }
        }
        // A voice crossing IRQA during the mix raises the line here, not
        // only at the next register access.
        if self.spu2.take_irq() {
            self.iop_i_stat |= 1 << 9;
        }
        self.intc_stat |= self.timers.check_irqs(self.now, self.region);
        self.intc_changed();
        const IRQ_BITS: [u32; 6] = [4, 5, 6, 14, 15, 16];
        let mut fired = 0u32;
        let now = self.now;
        let region = self.region;
        // Next time anything here can happen (bounded, as a safety net).
        let mut due = (now + MAX_TICK_GAP)
            .min(self.timers.next_event(now, region))
            .min(self.spu2.next_due());
        if let Some(at) = self.cdvd_done_at {
            due = due.min(at);
        }
        for &(_, at) in &self.dma_irq_queue {
            due = due.min(at);
        }
        for (t, timer) in self.iop_timers.iter_mut().enumerate() {
            if timer.mode == 0 {
                continue; // never configured
            }
            due = due.min(timer.next_event(t, now, region));
            let before = timer.count(t, timer.last_check, region);
            let after = timer.count(t, now, region);
            timer.last_check = now;
            if before == after {
                continue;
            }
            let target = timer.target;
            let crossed = if before <= after {
                before < target && target <= after
            } else {
                target > before || target <= after
            };
            // The reached-target/reached-max flags latch even with the IRQ
            // disabled — timrman's GetTimerStatus pollers depend on them.
            // Bits 4/5 gate the interrupt for target/overflow respectively.
            if crossed {
                timer.mode |= 1 << 11; // reached target
                if timer.mode & (1 << 3) != 0 {
                    timer.base = 0;
                    timer.base_cycle = now;
                }
                if timer.mode & (1 << 4) != 0 {
                    fired |= 1 << IRQ_BITS[t];
                }
            }
            if after < before {
                timer.mode |= 1 << 12; // reached max (wrap)
                if timer.mode & (1 << 5) != 0 {
                    fired |= 1 << IRQ_BITS[t];
                }
            }
        }
        self.iop_i_stat |= fired;
        self.timers_due = due.max(now + 1);
    }

    /// Vertical blank begin/end: EE INTC bits 2/3, IOP I_STAT bits 0/11,
    /// GS CSR VSINT.
    pub fn vblank(&mut self, begin: bool) {
        if begin {
            self.intc_stat |= 1 << 2;
            self.intc_changed();
            self.iop_i_stat |= 1 << 0;
            self.gs.vblank();
            self.gs_sync_int();
        } else {
            self.intc_stat |= 1 << 3;
            self.intc_changed();
            self.iop_i_stat |= 1 << 11;
        }
    }

    /// Run the VIF1 channel (ch1) to completion: normal or source chain.
    /// With TTE set, each chain tag's upper 64 bits carry two vifcodes.
    /// Channel 0: the same parser as channel 1, pointed at VU0. There is no
    /// PATH2 behind VIF0, so a DIRECT in this stream would be a mis-parse.
    fn pump_vif0(&mut self) {
        let _p = prof::scope(prof::Slot::Vif1);
        let mut guard = 0u32;
        while self.dma_vif0.chcr & EE_CHCR_STR != 0 {
            guard += 1;
            if guard > 1_000_000 {
                warn!(target: "ps2_core::bus::dma", "VIF0 DMA hit its iteration limit");
                break;
            }
            if self.dma_vif0.qwc > 0 {
                let q = self.ee_dma_read128(self.dma_vif0.madr);
                for w in q {
                    self.vif0
                        .push_word(&mut self.gs, &mut self.gif, &mut self.vu0, w);
                }
                self.dma_vif0.madr = self.dma_vif0.madr.wrapping_add(16);
                self.dma_vif0.qwc -= 1;
                continue;
            }
            // Block finished.
            let chain = (self.dma_vif0.chcr >> 2) & 3 == 1;
            if !chain || self.dma_vif0.tag_end {
                self.dma_vif0.chcr &= !EE_CHCR_STR;
                self.dma_vif0.tag_end = false;
                self.ee_dma_irq(0);
                debug!(target: "ps2_core::bus::dma", "VIF0 DMA done");
                break;
            }
            // Source-chain tag.
            let tag = self.ee_dma_read128(self.dma_vif0.tadr);
            let qwc = tag[0] & 0xFFFF;
            let id = (tag[0] >> 28) & 7;
            let irq = tag[0] & 0x8000_0000 != 0;
            // Keep bit 31: it selects the scratchpad (SPR) as the source.
            let addr = tag[1] & 0xFFFF_FFF0;
            trace!(
                target: "ps2_core::bus::dma",
                tadr = format_args!("{:#010x}", self.dma_vif0.tadr),
                id, qwc,
                addr = format_args!("{addr:#010x}"),
                chcr = format_args!("{:#x}", self.dma_vif0.chcr),
                tag_hi = format_args!("{:08x} {:08x}", tag[2], tag[3]),
                "VIF1 tag"
            );
            match id {
                0 => {
                    self.dma_vif0.madr = addr;
                    self.dma_vif0.tadr = self.dma_vif0.tadr.wrapping_add(16);
                    self.dma_vif0.tag_end = true;
                }
                1 => {
                    self.dma_vif0.madr = self.dma_vif0.tadr.wrapping_add(16);
                    self.dma_vif0.tadr = self.dma_vif0.madr.wrapping_add(qwc * 16);
                }
                2 => {
                    self.dma_vif0.madr = self.dma_vif0.tadr.wrapping_add(16);
                    self.dma_vif0.tadr = addr;
                }
                3 | 4 => {
                    self.dma_vif0.madr = addr;
                    self.dma_vif0.tadr = self.dma_vif0.tadr.wrapping_add(16);
                }
                // call: run the block at `addr`, remembering where to
                // come back to. ret pops that back off.
                5 => {
                    self.dma_vif0.madr = self.dma_vif0.tadr.wrapping_add(16);
                    let back = self.dma_vif0.madr.wrapping_add(qwc * 16);
                    let d = self.dma_vif0.asr_depth as usize;
                    if d < 2 {
                        self.dma_vif0.asr[d] = back;
                        self.dma_vif0.asr_depth += 1;
                    } else {
                        warn!(target: "ps2_core::bus::dma", "chain call stack overflow");
                    }
                    self.dma_vif0.tadr = addr;
                }
                6 => {
                    self.dma_vif0.madr = self.dma_vif0.tadr.wrapping_add(16);
                    if self.dma_vif0.asr_depth > 0 {
                        self.dma_vif0.asr_depth -= 1;
                        self.dma_vif0.tadr = self.dma_vif0.asr[self.dma_vif0.asr_depth as usize];
                    } else {
                        self.dma_vif0.tag_end = true;
                    }
                }
                7 => {
                    self.dma_vif0.madr = self.dma_vif0.tadr.wrapping_add(16);
                    self.dma_vif0.tag_end = true;
                }
                _ => {
                    warn!(target: "ps2_core::bus::dma", id, "unhandled VIF0 chain tag id");
                    self.dma_vif0.chcr &= !EE_CHCR_STR;
                    break;
                }
            }
            if irq && self.dma_vif0.chcr & EE_CHCR_TIE != 0 {
                self.dma_vif0.tag_end = true;
            }
            // TTE decides whether the tag's upper 64 bits enter the VIF
            // stream. Feeding them unconditionally would turn them into
            // data mid-UNPACK; no chain seen so far clears TTE anyway.
            if self.dma_vif0.chcr & EE_CHCR_TTE != 0 {
                self.vif0
                    .push_word(&mut self.gs, &mut self.gif, &mut self.vu0, tag[2]);
                self.vif0
                    .push_word(&mut self.gs, &mut self.gif, &mut self.vu0, tag[3]);
            }
            self.dma_vif0.qwc = qwc;
        }
    }

    /// Run the GIF channel (ch2) to completion: normal or source chain.
    fn pump_vif1(&mut self) {
        let _p = prof::scope(prof::Slot::Vif1);
        let mut guard = 0u32;
        while self.dma_vif1.chcr & EE_CHCR_STR != 0 {
            guard += 1;
            if guard > 1_000_000 {
                warn!(target: "ps2_core::bus::dma", "VIF1 DMA hit its iteration limit");
                break;
            }
            if self.dma_vif1.qwc > 0 {
                let q = self.ee_dma_read128(self.dma_vif1.madr);
                for w in q {
                    self.vif1
                        .push_word(&mut self.gs, &mut self.gif, &mut self.vu1, w);
                }
                self.dma_vif1.madr = self.dma_vif1.madr.wrapping_add(16);
                self.dma_vif1.qwc -= 1;
                continue;
            }
            // Block finished.
            let chain = (self.dma_vif1.chcr >> 2) & 3 == 1;
            if !chain || self.dma_vif1.tag_end {
                self.dma_vif1.chcr &= !EE_CHCR_STR;
                self.dma_vif1.tag_end = false;
                self.ee_dma_irq(1);
                debug!(target: "ps2_core::bus::dma", "VIF1 DMA done");
                break;
            }
            // Source-chain tag.
            let tag = self.ee_dma_read128(self.dma_vif1.tadr);
            let qwc = tag[0] & 0xFFFF;
            let id = (tag[0] >> 28) & 7;
            let irq = tag[0] & 0x8000_0000 != 0;
            // Keep bit 31: it selects the scratchpad (SPR) as the source.
            let addr = tag[1] & 0xFFFF_FFF0;
            trace!(
                target: "ps2_core::bus::dma",
                tadr = format_args!("{:#010x}", self.dma_vif1.tadr),
                id, qwc,
                addr = format_args!("{addr:#010x}"),
                chcr = format_args!("{:#x}", self.dma_vif1.chcr),
                tag_hi = format_args!("{:08x} {:08x}", tag[2], tag[3]),
                "VIF1 tag"
            );
            match id {
                0 => {
                    self.dma_vif1.madr = addr;
                    self.dma_vif1.tadr = self.dma_vif1.tadr.wrapping_add(16);
                    self.dma_vif1.tag_end = true;
                }
                1 => {
                    self.dma_vif1.madr = self.dma_vif1.tadr.wrapping_add(16);
                    self.dma_vif1.tadr = self.dma_vif1.madr.wrapping_add(qwc * 16);
                }
                2 => {
                    self.dma_vif1.madr = self.dma_vif1.tadr.wrapping_add(16);
                    self.dma_vif1.tadr = addr;
                }
                3 | 4 => {
                    self.dma_vif1.madr = addr;
                    self.dma_vif1.tadr = self.dma_vif1.tadr.wrapping_add(16);
                }
                // call: run the block at `addr`, remembering where to
                // come back to. ret pops that back off.
                5 => {
                    self.dma_vif1.madr = self.dma_vif1.tadr.wrapping_add(16);
                    let back = self.dma_vif1.madr.wrapping_add(qwc * 16);
                    let d = self.dma_vif1.asr_depth as usize;
                    if d < 2 {
                        self.dma_vif1.asr[d] = back;
                        self.dma_vif1.asr_depth += 1;
                    } else {
                        warn!(target: "ps2_core::bus::dma", "chain call stack overflow");
                    }
                    self.dma_vif1.tadr = addr;
                }
                6 => {
                    self.dma_vif1.madr = self.dma_vif1.tadr.wrapping_add(16);
                    if self.dma_vif1.asr_depth > 0 {
                        self.dma_vif1.asr_depth -= 1;
                        self.dma_vif1.tadr = self.dma_vif1.asr[self.dma_vif1.asr_depth as usize];
                    } else {
                        self.dma_vif1.tag_end = true;
                    }
                }
                7 => {
                    self.dma_vif1.madr = self.dma_vif1.tadr.wrapping_add(16);
                    self.dma_vif1.tag_end = true;
                }
                _ => {
                    warn!(target: "ps2_core::bus::dma", id, "unhandled VIF1 chain tag id");
                    self.dma_vif1.chcr &= !EE_CHCR_STR;
                    break;
                }
            }
            if irq && self.dma_vif1.chcr & EE_CHCR_TIE != 0 {
                self.dma_vif1.tag_end = true;
            }
            // TTE decides whether the tag's upper 64 bits enter the VIF
            // stream. Feeding them unconditionally would turn them into
            // data mid-UNPACK; no chain seen so far clears TTE anyway.
            if self.dma_vif1.chcr & EE_CHCR_TTE != 0 {
                self.vif1
                    .push_word(&mut self.gs, &mut self.gif, &mut self.vu1, tag[2]);
                self.vif1
                    .push_word(&mut self.gs, &mut self.gif, &mut self.vu1, tag[3]);
            }
            self.dma_vif1.qwc = qwc;
        }
    }

    /// Run the GIF channel (ch2) to completion: normal or source chain.
    /// Channel 4: move quadwords from memory into the IPU's input FIFO,
    /// letting each one wake a command that ran out of bitstream. Supports
    /// the source-chain tags the MPEG players build their streams from.
    fn pump_ipu_to(&mut self) {
        let mut irq = false;
        let mut guard = 0u32;
        while self.dma_ipu_to.chcr & EE_CHCR_STR != 0 {
            guard += 1;
            if guard > 1_000_000 {
                warn!(target: "ps2_core::bus::dma", "IPU_TO DMA hit its iteration limit");
                break;
            }
            if self.dma_ipu_to.qwc > 0 {
                let w = self.ee_dma_read128(self.dma_ipu_to.madr);
                let mut q = [0u8; 16];
                for (i, word) in w.iter().enumerate() {
                    q[i * 4..i * 4 + 4].copy_from_slice(&word.to_le_bytes());
                }
                irq |= self.ipu.push_in(q);
                self.dma_ipu_to.madr = self.dma_ipu_to.madr.wrapping_add(16);
                self.dma_ipu_to.qwc -= 1;
                continue;
            }
            let chain = (self.dma_ipu_to.chcr >> 2) & 3 == 1;
            if !chain || self.dma_ipu_to.tag_end {
                self.dma_ipu_to.chcr &= !EE_CHCR_STR;
                self.dma_ipu_to.tag_end = false;
                self.ee_dma_irq(4);
                debug!(target: "ps2_core::bus::dma", "IPU_TO DMA done");
                break;
            }
            let tag = self.ee_dma_read128(self.dma_ipu_to.tadr);
            let qwc = tag[0] & 0xFFFF;
            let id = (tag[0] >> 28) & 7;
            let addr = tag[1] & 0xFFFF_FFF0;
            match id {
                0 => {
                    self.dma_ipu_to.madr = addr;
                    self.dma_ipu_to.tadr = self.dma_ipu_to.tadr.wrapping_add(16);
                    self.dma_ipu_to.tag_end = true;
                }
                1 => {
                    self.dma_ipu_to.madr = self.dma_ipu_to.tadr.wrapping_add(16);
                    self.dma_ipu_to.tadr = self.dma_ipu_to.madr.wrapping_add(qwc * 16);
                }
                2 => {
                    self.dma_ipu_to.madr = self.dma_ipu_to.tadr.wrapping_add(16);
                    self.dma_ipu_to.tadr = addr;
                }
                3 | 4 => {
                    self.dma_ipu_to.madr = addr;
                    self.dma_ipu_to.tadr = self.dma_ipu_to.tadr.wrapping_add(16);
                }
                // call: run the block at `addr`, remembering where to
                // come back to. ret pops that back off.
                5 => {
                    self.dma_ipu_to.madr = self.dma_ipu_to.tadr.wrapping_add(16);
                    let back = self.dma_ipu_to.madr.wrapping_add(qwc * 16);
                    let d = self.dma_ipu_to.asr_depth as usize;
                    if d < 2 {
                        self.dma_ipu_to.asr[d] = back;
                        self.dma_ipu_to.asr_depth += 1;
                    } else {
                        warn!(target: "ps2_core::bus::dma", "chain call stack overflow");
                    }
                    self.dma_ipu_to.tadr = addr;
                }
                6 => {
                    self.dma_ipu_to.madr = self.dma_ipu_to.tadr.wrapping_add(16);
                    if self.dma_ipu_to.asr_depth > 0 {
                        self.dma_ipu_to.asr_depth -= 1;
                        self.dma_ipu_to.tadr = self.dma_ipu_to.asr[self.dma_ipu_to.asr_depth as usize];
                    } else {
                        self.dma_ipu_to.tag_end = true;
                    }
                }
                7 => {
                    self.dma_ipu_to.madr = self.dma_ipu_to.tadr.wrapping_add(16);
                    self.dma_ipu_to.tag_end = true;
                }
                _ => {
                    warn!(target: "ps2_core::bus::dma", id, "unhandled IPU_TO tag");
                    self.dma_ipu_to.chcr &= !EE_CHCR_STR;
                    break;
                }
            }
            self.dma_ipu_to.qwc = qwc;
        }
        if irq {
            self.intc_stat |= 1 << 8;
            self.intc_changed();
        }
    }

    /// Channel 9 (toSPR): quadwords from main memory into the scratchpad.
    /// MADR walks memory, SADR the 16 KB scratchpad, wrapping inside it.
    /// The DMAC reads memory here, so chain mode is an ordinary source
    /// chain with its tags at TADR.
    fn pump_spr_to(&mut self) {
        let mut guard = 0u32;
        while self.dma_spr_to.chcr & EE_CHCR_STR != 0 {
            guard += 1;
            if guard > 1_000_000 {
                warn!(target: "ps2_core::bus::dma", "toSPR DMA hit its iteration limit");
                break;
            }
            if self.dma_spr_to.qwc > 0 {
                let q = self.ee_dma_read128(self.dma_spr_to.madr);
                self.spad_write128(self.dma_spr_to.sadr, q);
                self.dma_spr_to.sadr = self.dma_spr_to.sadr.wrapping_add(16) & 0x3FFF;
                self.dma_spr_to.madr = self.dma_spr_to.madr.wrapping_add(16);
                self.dma_spr_to.qwc -= 1;
                continue;
            }
            if !self.spr_chain(true) {
                break;
            }
        }
    }

    /// Channel 8 (fromSPR): the other direction, scratchpad to memory.
    /// Its chain tags sit in the scratchpad at SADR, ahead of the data
    /// they describe, since that is the side the DMAC reads.
    fn pump_spr_from(&mut self) {
        let mut guard = 0u32;
        while self.dma_spr_from.chcr & EE_CHCR_STR != 0 {
            guard += 1;
            if guard > 1_000_000 {
                warn!(target: "ps2_core::bus::dma", "fromSPR DMA hit its iteration limit");
                break;
            }
            if self.dma_spr_from.qwc > 0 {
                let q = self.spad_read128(self.dma_spr_from.sadr);
                self.ee_dma_write128(self.dma_spr_from.madr, q);
                self.dma_spr_from.sadr = self.dma_spr_from.sadr.wrapping_add(16) & 0x3FFF;
                self.dma_spr_from.madr = self.dma_spr_from.madr.wrapping_add(16);
                self.dma_spr_from.qwc -= 1;
                continue;
            }
            if !self.spr_chain(false) {
                break;
            }
        }
    }

    /// One chain step for an SPR channel with its data run exhausted:
    /// finish the transfer, or read the next tag and set up the run it
    /// describes. Returns whether the channel is still running.
    ///
    /// Both are simple two-word chains — a count and an address, no
    /// call/ret — but they read their tags from opposite sides: `to`
    /// (channel 9) from memory at TADR, channel 8 from the scratchpad.
    fn spr_chain(&mut self, to: bool) -> bool {
        let ch = if to { &self.dma_spr_to } else { &self.dma_spr_from };
        let (mode, tag_end, tte) = ((ch.chcr >> 2) & 3, ch.tag_end, ch.chcr & EE_CHCR_TTE != 0);
        let reg = if to { 0x1000_D400 } else { 0x1000_D000 };
        if tte {
            // Transferring the tag as well shifts every run by a
            // quadword, so this is a wrong result, not a missing one.
            self.warn_stub(reg, "SPR chain with TTE set");
        }
        if mode != 1 || tag_end {
            if mode == 2 {
                self.warn_stub(reg, "interleaved SPR transfer");
            }
            let ch = if to { &mut self.dma_spr_to } else { &mut self.dma_spr_from };
            ch.chcr &= !EE_CHCR_STR;
            ch.tag_end = false;
            self.ee_dma_irq(if to { 9 } else { 8 });
            return false;
        }
        let tag = if to {
            self.ee_dma_read128(self.dma_spr_to.tadr)
        } else {
            self.spad_read128(self.dma_spr_from.sadr)
        };
        let qwc = tag[0] & 0xFFFF;
        let id = (tag[0] >> 28) & 7;
        let irq = tag[0] & 0x8000_0000 != 0;
        let addr = tag[1] & 0xFFFF_FFF0;
        let ch = if to { &mut self.dma_spr_to } else { &mut self.dma_spr_from };
        if !to {
            // Destination chain: the tag names where in memory its run
            // lands, and the data follows it in the scratchpad. Only
            // cnts, cnt and end are defined on this side.
            if !matches!(id, 0 | 1 | 7) {
                warn!(target: "ps2_core::bus::dma", id, "unhandled fromSPR tag");
                ch.chcr &= !EE_CHCR_STR;
                return false;
            }
            ch.madr = addr;
            ch.sadr = ch.sadr.wrapping_add(16) & 0x3FFF;
            ch.tag_end = id == 7;
        } else {
            // Source chain, the same ids every memory-reading channel
            // follows.
            match id {
                0 => {
                    ch.madr = addr;
                    ch.tadr = ch.tadr.wrapping_add(16);
                    ch.tag_end = true;
                }
                1 => {
                    ch.madr = ch.tadr.wrapping_add(16);
                    ch.tadr = ch.madr.wrapping_add(qwc * 16);
                }
                2 => {
                    ch.madr = ch.tadr.wrapping_add(16);
                    ch.tadr = addr;
                }
                3 | 4 => {
                    ch.madr = addr;
                    ch.tadr = ch.tadr.wrapping_add(16);
                }
                5 => {
                    ch.madr = ch.tadr.wrapping_add(16);
                    let back = ch.madr.wrapping_add(qwc * 16);
                    let d = ch.asr_depth as usize;
                    if d < 2 {
                        ch.asr[d] = back;
                        ch.asr_depth += 1;
                    } else {
                        warn!(target: "ps2_core::bus::dma", "chain call stack overflow");
                    }
                    ch.tadr = addr;
                }
                6 => {
                    ch.madr = ch.tadr.wrapping_add(16);
                    if ch.asr_depth > 0 {
                        ch.asr_depth -= 1;
                        ch.tadr = ch.asr[ch.asr_depth as usize];
                    } else {
                        ch.tag_end = true;
                    }
                }
                _ => {
                    ch.madr = ch.tadr.wrapping_add(16);
                    ch.tag_end = true;
                }
            }
        }
        ch.tag_end |= irq && ch.chcr & EE_CHCR_TIE != 0;
        ch.qwc = qwc;
        true
    }

    /// One quadword of the scratchpad, by byte address; SADR is 14 bits,
    /// so the address is already inside it.
    fn spad_read128(&self, sadr: u32) -> [u32; 4] {
        let a = (sadr & 0x3FF0) as usize;
        [
            read_le::<4>(&self.spad, a) as u32,
            read_le::<4>(&self.spad, a + 4) as u32,
            read_le::<4>(&self.spad, a + 8) as u32,
            read_le::<4>(&self.spad, a + 12) as u32,
        ]
    }

    fn spad_write128(&mut self, sadr: u32, q: [u32; 4]) {
        let a = (sadr & 0x3FF0) as usize;
        for (i, w) in q.iter().enumerate() {
            write_le::<4>(&mut self.spad, a + i * 4, *w as u64);
        }
    }

    /// Write one quadword for EE-side DMA, taking bit 31 of the address
    /// as the scratchpad the way [`Bus::ee_dma_read128`] does.
    fn ee_dma_write128(&mut self, addr: u32, q: [u32; 4]) {
        if addr & 0x8000_0000 != 0 {
            self.spad_write128(addr, q);
            return;
        }
        let a = (addr & 0x1FFF_FFF0) as usize;
        if a + 16 > RAM_SIZE {
            if self.warned_unmapped.insert(addr & !0xFFF) {
                warn!(target: "ps2_core::bus::dma", addr = format_args!("{addr:#010x}"), "EE DMA write outside RAM (reported once per page)");
            }
            return;
        }
        for (i, w) in q.iter().enumerate() {
            write_le::<4>(&mut self.ram, a + i * 4, *w as u64);
        }
        self.note_ram_write(a);
    }

    fn pump_gif(&mut self) {
        let _p = prof::scope(prof::Slot::Gif);
        let mut guard = 0u32;
        while self.dma_gif.chcr & EE_CHCR_STR != 0 {
            guard += 1;
            if guard > 1_000_000 {
                warn!(target: "ps2_core::bus::dma", "GIF DMA hit its iteration limit");
                break;
            }
            if self.dma_gif.qwc > 0 {
                let q = self.ee_dma_read128(self.dma_gif.madr);
                let lo = q[0] as u64 | ((q[1] as u64) << 32);
                let hi = q[2] as u64 | ((q[3] as u64) << 32);
                self.gif.process(&mut self.gs, lo, hi);
                self.dma_gif.madr = self.dma_gif.madr.wrapping_add(16);
                self.dma_gif.qwc -= 1;
                continue;
            }
            // Block finished.
            let chain = (self.dma_gif.chcr >> 2) & 3 == 1;
            if !chain || self.dma_gif.tag_end {
                self.dma_gif.chcr &= !EE_CHCR_STR;
                self.dma_gif.tag_end = false;
                self.ee_dma_irq(2);
                debug!(target: "ps2_core::bus::dma", "GIF DMA done");
                break;
            }
            // Source-chain tag.
            let tag = self.ee_dma_read128(self.dma_gif.tadr);
            let qwc = tag[0] & 0xFFFF;
            let id = (tag[0] >> 28) & 7;
            let irq = tag[0] & 0x8000_0000 != 0;
            // Keep bit 31: it selects the scratchpad (SPR) as the source.
            let addr = tag[1] & 0xFFFF_FFF0;
            match id {
                0 => {
                    self.dma_gif.madr = addr;
                    self.dma_gif.tadr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tag_end = true;
                }
                1 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tadr = self.dma_gif.madr.wrapping_add(qwc * 16);
                }
                2 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tadr = addr;
                }
                3 | 4 => {
                    self.dma_gif.madr = addr;
                    self.dma_gif.tadr = self.dma_gif.tadr.wrapping_add(16);
                }
                // call: run the block at `addr`, remembering where to
                // come back to. ret pops that back off.
                5 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    let back = self.dma_gif.madr.wrapping_add(qwc * 16);
                    let d = self.dma_gif.asr_depth as usize;
                    if d < 2 {
                        self.dma_gif.asr[d] = back;
                        self.dma_gif.asr_depth += 1;
                    } else {
                        warn!(target: "ps2_core::bus::dma", "chain call stack overflow");
                    }
                    self.dma_gif.tadr = addr;
                }
                6 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    if self.dma_gif.asr_depth > 0 {
                        self.dma_gif.asr_depth -= 1;
                        self.dma_gif.tadr = self.dma_gif.asr[self.dma_gif.asr_depth as usize];
                    } else {
                        self.dma_gif.tag_end = true;
                    }
                }
                7 => {
                    self.dma_gif.madr = self.dma_gif.tadr.wrapping_add(16);
                    self.dma_gif.tag_end = true;
                }
                _ => {
                    warn!(target: "ps2_core::bus::dma", id, "unhandled GIF chain tag id");
                    self.dma_gif.tag_end = true;
                }
            }
            if irq && self.dma_gif.chcr & EE_CHCR_TIE != 0 {
                self.dma_gif.tag_end = true;
            }
            if self.dma_gif.chcr & EE_CHCR_TTE != 0 {
                // TTE on the GIF channel sends the tag's upper 64 bits.
                let lo = tag[2] as u64 | ((tag[3] as u64) << 32);
                self.gif.process(&mut self.gs, lo, 0);
            }
            self.dma_gif.qwc = qwc;
        }
        self.gs_sync_int();
    }

    // --- SIF DMA ---------------------------------------------------------

    /// EE interrupt lines: INT0 = INTC, INT1 = DMAC.
    pub fn ee_int0_pending(&self) -> bool {
        self.now >= self.intc_ready_at
    }
    /// Re-arm the INTC line after any change to its status or mask.
    ///
    /// The EE does not see an interrupt in the cycle its source raises it:
    /// the line has to reach the core and be recognised at an instruction
    /// boundary. Modelling that window matters because software polls
    /// INTC_STAT directly — the OSD waits for vertical blank by clearing
    /// bit 2 and spinning on it, and with no window at all the kernel's
    /// own handler acknowledges the bit before the spin can ever read it.
    fn intc_changed(&mut self) {
        if self.intc_stat & self.intc_mask == 0 {
            self.intc_ready_at = u64::MAX;
        } else if self.intc_ready_at == u64::MAX {
            self.intc_ready_at = self.now + INTC_LATENCY;
        }
    }

    pub fn ee_int1_pending(&self) -> bool {
        self.d_stat & self.d_mask & 0x3FF != 0
    }

    /// Read one quadword for EE-side DMA. Bit 31 of MADR/TADR/tag
    /// addresses selects the scratchpad (SPR) instead of main RAM — the
    /// OSD builds its display lists there.
    fn ee_dma_read128(&mut self, addr: u32) -> [u32; 4] {
        if addr & 0x8000_0000 != 0 {
            let a = (addr & 0x3FF0) as usize; // 16 KiB scratchpad, wraps
            return [
                read_le::<4>(&self.spad, a) as u32,
                read_le::<4>(&self.spad, a + 4) as u32,
                read_le::<4>(&self.spad, a + 8) as u32,
                read_le::<4>(&self.spad, a + 12) as u32,
            ];
        }
        let a = (addr & 0x1FFF_FFF0) as usize;
        if a + 16 <= RAM_SIZE {
            [
                read_le::<4>(&self.ram, a) as u32,
                read_le::<4>(&self.ram, a + 4) as u32,
                read_le::<4>(&self.ram, a + 8) as u32,
                read_le::<4>(&self.ram, a + 12) as u32,
            ]
        } else {
            if self.warned_unmapped.insert(addr & !0xFFF) {
                warn!(target: "ps2_core::bus::sifdma", addr = format_args!("{addr:#010x}"), "EE DMA read outside RAM (reported once per page)");
            }
            [0; 4]
        }
    }

    /// Raise an IOP DMA completion interrupt: channels 0-6 report through
    /// DICR, 7-13 through DICR2 (enable bits 16+, flag bits 24+).
    /// IOP BCR: block size in words (low 16) x block count (high 16).
    fn iop_bcr_bytes(bcr: u32) -> usize {
        ((bcr & 0xFFFF).max(1) as usize) * ((bcr >> 16).max(1) as usize) * 4
    }

    /// Run a kicked SPU2 DMA (ch4 = core 0, ch7 = core 1) against the SPU2
    /// model. CHCR bit 0 set = IOP RAM to SPU. The channel stays busy until
    /// the model's completion time; `tick_timers` retires it.
    fn do_spu2_dma(&mut self, core: usize) {
        let ch = &self.iop_dma_spu[core];
        let bytes = Self::iop_bcr_bytes(ch.bcr);
        let to_spu = ch.chcr & 1 != 0;
        let start = (ch.madr & 0x1F_FFFF) as usize;
        let len = self.iop_ram.len();
        let mut buf = vec![0u8; bytes];
        if to_spu {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = self.iop_ram[(start + i) % len];
            }
        }
        self.spu2.dma(core, to_spu, &mut buf, self.now);
        self.timers_due = 0;
        if !to_spu {
            for (i, b) in buf.into_iter().enumerate() {
                self.iop_ram[(start + i) % len] = b;
            }
        }
        if self.spu2.take_irq() {
            self.iop_i_stat |= 1 << 9;
        }
    }

    /// Drain staged CDVD sector data into RAM for DMA ch3.
    fn do_cdvd_dma(&mut self) {
        let bytes = Self::iop_bcr_bytes(self.iop_dma_cdvd.bcr);
        let start = (self.iop_dma_cdvd.madr & 0x1F_FFFF) as usize;
        let len = self.iop_ram.len();
        let mut chunk = vec![0u8; bytes];
        self.cdvd.dma_read(&mut chunk);
        for (i, b) in chunk.into_iter().enumerate() {
            self.iop_ram[(start + i) % len] = b;
        }
        debug!(target: "ps2_core::iop::cdvd", bytes,
            madr = format_args!("{:#x}", self.iop_dma_cdvd.madr),
            "DMA ch3");
        self.iop_dma_irq(3);
    }

    /// Drain the SIO2 out-FIFO into RAM for DMA ch12.
    fn do_sio2_out(&mut self) {
        let bytes = Self::iop_bcr_bytes(self.iop_dma_sio2out.bcr);
        let start = (self.iop_dma_sio2out.madr & 0x1F_FFFF) as usize;
        for i in 0..bytes {
            let b = self.sio2.read(0x1F80_8264) as u8;
            let len = self.iop_ram.len();
            self.iop_ram[(start + i) % len] = b;
        }
        debug!(target: "ps2_core::iop::sio2",
            bytes,
            madr = format_args!("{:#x}", self.iop_dma_sio2out.madr),
            head = format_args!("{:02x?}", &self.iop_ram[start..(start + 8).min(start + bytes)]),
            "DMA out");
        self.iop_dma_irq(12);
    }

    fn iop_dma_irq(&mut self, ch: u32) {
        let (reg, bit) = if ch < 7 {
            (&mut self.iop_dicr, ch)
        } else {
            (&mut self.iop_dicr2, ch - 7)
        };
        let enabled = *reg & (1 << (16 + bit)) != 0;
        *reg |= 1 << (24 + bit);
        if enabled {
            // IOP DMA interrupt line.
            self.iop_i_stat |= 1 << 3;
        }
        debug!(target: "ps2_core::bus::sifdma", ch, enabled, "IOP DMA complete");
    }

    /// Move as much SIF traffic as the armed channels allow. Runs transfers
    /// to completion synchronously; timing comes later if software needs it.
    pub fn pump_sif(&mut self) {
        let _p = prof::scope(prof::Slot::Sif);
        // Safety valve against malformed chains.
        for _ in 0..4096 {
            let mut progressed = false;
            progressed |= self.pump_sif1_ee();
            progressed |= self.pump_sif1_iop();
            progressed |= self.pump_sif0_iop();
            progressed |= self.pump_sif0_ee();
            if !progressed {
                return;
            }
        }
        warn!(target: "ps2_core::bus::sifdma", "SIF pump hit its iteration limit");
    }

    /// EE SIF1 (ch6): source chain from EE RAM into fifo1.
    fn pump_sif1_ee(&mut self) -> bool {
        let mut progressed = false;
        while self.dma_sif1.chcr & EE_CHCR_STR != 0 {
            if self.dma_sif1.qwc > 0 {
                for _ in 0..self.dma_sif1.qwc {
                    let q = self.ee_dma_read128(self.dma_sif1.madr);
                    self.sif.fifo1.extend(q);
                    self.dma_sif1.madr = self.dma_sif1.madr.wrapping_add(16);
                }
                self.dma_sif1.qwc = 0;
                progressed = true;
                if self.dma_sif1.tag_end {
                    self.dma_sif1.chcr &= !EE_CHCR_STR;
                    self.dma_sif1.tag_end = false;
                    self.ee_dma_irq(6);
                    debug!(target: "ps2_core::bus::sifdma", "EE SIF1 chain done");
                }
            } else {
                // Fetch the next source-chain tag.
                let tag = self.ee_dma_read128(self.dma_sif1.tadr);
                let qwc = tag[0] & 0xFFFF;
                let id = (tag[0] >> 28) & 7;
                let irq = tag[0] & 0x8000_0000 != 0;
                // Keep bit 31: it selects the scratchpad (SPR) as the source.
            let addr = tag[1] & 0xFFFF_FFF0;
                trace!(
                    target: "ps2_core::bus::sifdma",
                    tadr = format_args!("{:#010x}", self.dma_sif1.tadr),
                    qwc, id, irq,
                    "EE SIF1 tag"
                );
                if self.dma_sif1.chcr & EE_CHCR_TTE != 0 {
                    // Transfer the tag's upper 64 bits (the IOP-side tag).
                    self.sif.fifo1.push_back(tag[2]);
                    self.sif.fifo1.push_back(tag[3]);
                }
                match id {
                    0 => {
                        // refe: data at ADDR, end after this block.
                        self.dma_sif1.madr = addr;
                        self.dma_sif1.tadr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tag_end = true;
                    }
                    1 => {
                        // cnt: data follows the tag.
                        self.dma_sif1.madr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tadr = self.dma_sif1.madr.wrapping_add(qwc * 16);
                    }
                    2 => {
                        // next: data follows, next tag at ADDR.
                        self.dma_sif1.madr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tadr = addr;
                    }
                    3 | 4 => {
                        // ref/refs: data at ADDR, tags are sequential.
                        self.dma_sif1.madr = addr;
                        self.dma_sif1.tadr = self.dma_sif1.tadr.wrapping_add(16);
                    }
                    7 => {
                        // end: data follows, then stop.
                        self.dma_sif1.madr = self.dma_sif1.tadr.wrapping_add(16);
                        self.dma_sif1.tag_end = true;
                    }
                    _ => {
                        warn!(target: "ps2_core::bus::sifdma", id, "unhandled EE source-chain tag id");
                        self.dma_sif1.tag_end = true;
                    }
                }
                if irq && self.dma_sif1.chcr & EE_CHCR_TIE != 0 {
                    self.dma_sif1.tag_end = true;
                }
                self.dma_sif1.qwc = qwc;
                progressed = true;
                if qwc == 0 && self.dma_sif1.tag_end {
                    self.dma_sif1.chcr &= !EE_CHCR_STR;
                    self.dma_sif1.tag_end = false;
                    self.ee_dma_irq(6);
                }
            }
        }
        progressed
    }

    /// IOP SIF1 (ch10): fifo1 into IOP RAM, guided by embedded 2-word tags.
    fn pump_sif1_iop(&mut self) -> bool {
        let mut progressed = false;
        while self.iop_dma_sif1.chcr & IOP_CHCR_BUSY != 0 {
            let ch = &mut self.iop_dma_sif1;
            if ch.recv_left == 0 && ch.recv_pad == 0 {
                if self.sif.fifo1.len() < 4 {
                    break;
                }
                // The IOP-side tag occupies a full quadword in the stream:
                // {addr|flags, word count, pad, pad}, then the data follows,
                // itself padded up to a qword boundary.
                let w0 = self.sif.fifo1.pop_front().unwrap();
                let w1 = self.sif.fifo1.pop_front().unwrap();
                self.sif.fifo1.pop_front();
                self.sif.fifo1.pop_front();
                ch.recv_addr = w0 & 0xFF_FFFF;
                ch.recv_start = ch.recv_addr;
                ch.recv_left = w1;
                ch.recv_pad = (4 - (w1 & 3)) & 3;
                ch.recv_end = w0 & 0xC000_0000 != 0;
                trace!(
                    target: "ps2_core::bus::sifdma",
                    addr = format_args!("{:#010x}", ch.recv_addr),
                    words = w1,
                    pad = ch.recv_pad,
                    end = ch.recv_end,
                    "IOP SIF1 tag"
                );
                progressed = true;
            }
            while ch.recv_left > 0 && !self.sif.fifo1.is_empty() {
                let w = self.sif.fifo1.pop_front().unwrap();
                let a = (ch.recv_addr & 0x1F_FFFC) as usize;
                write_le::<4>(&mut self.iop_ram, a, w as u64);
                ch.recv_addr = ch.recv_addr.wrapping_add(4);
                ch.recv_left -= 1;
                progressed = true;
            }
            if ch.recv_left > 0 {
                break; // wait for more data
            }
            while ch.recv_pad > 0 && !self.sif.fifo1.is_empty() {
                self.sif.fifo1.pop_front();
                ch.recv_pad -= 1;
                progressed = true;
            }
            if ch.recv_pad > 0 {
                break;
            }
            if ch.recv_end {
                ch.recv_end = false;
                ch.chcr &= !IOP_CHCR_BUSY;
                let start = ch.recv_start;
                self.iop_dma_irq(10);
                // An sceSifIopReset command (cid 0x80000003) means the IOP
                // is about to reboot silently via UDNL: drop in-flight SIF
                // state so the new kernel starts with clean FIFOs.
                let cid = read_le::<4>(&self.iop_ram, ((start + 8) & 0x1F_FFFC) as usize) as u32;
                if cid & 0x8000_0000 != 0 {
                    let payload: Vec<u32> = (0..6)
                        .map(|i| {
                            read_le::<4>(&self.iop_ram, ((start + 16 + i * 4) & 0x1F_FFFC) as usize)
                                as u32
                        })
                        .collect();
                    debug!(
                        target: "ps2_core::bus::sifcmd",
                        cid = format_args!("{cid:#010x}"),
                        payload = format_args!("{payload:08x?}"),
                        "EE->IOP command"
                    );
                }
                if cid == 0x8000_0003 {
                    debug!(target: "ps2_core::bus::sifdma", "IOP reset command: flushing SIF state");
                    // From the third reboot on (PS2LOGO -> game), the
                    // rebooted cdvdman re-checks the drive and it settles a
                    // while later; that hold is what lets the logo sound's
                    // reverb tail ring before the game's libsd clears the
                    // SPU. The two earlier reboots (OSD boot, and PS2LOGO's
                    // own right before its mechacon sequence) must stay
                    // fast: a drive hold after PS2LOGO's reboot blocks its
                    // ReadKey and logo read, and its coarse ~0.46 s polling
                    // rounds turn any such wait into whole extra rounds of
                    // black screen in the vsync-scheduled intro.
                    self.iop_resets += 1;
                    if self.iop_resets >= 3 {
                        self.cdvd.ready_at = self.now + CDVD_RESETTLE;
                    }
                    self.sif.fifo0.clear();
                    self.sif.fifo1.clear();
                    self.iop_dma_sif0 = IopDmaChannel::default();
                    let chcr = self.iop_dma_sif1.chcr;
                    self.iop_dma_sif1 = IopDmaChannel::default();
                    self.iop_dma_sif1.chcr = chcr;
                }
            }
        }
        progressed
    }

    /// IOP SIF0 (ch9): IOP RAM into fifo0, guided by 2-word tags at TADR.
    fn pump_sif0_iop(&mut self) -> bool {
        let mut progressed = false;
        while self.iop_dma_sif0.chcr & IOP_CHCR_BUSY != 0 {
            let ch = &mut self.iop_dma_sif0;
            if ch.words_left == 0 {
                // 16-byte send block: {data addr | flags, word count,
                // EE tag lo, EE tag hi}. The EE-side destination tag rides
                // ahead of the data as its own quadword.
                let t = (ch.tadr & 0x1F_FFFC) as usize;
                let w0 = read_le::<4>(&self.iop_ram, t) as u32;
                let w1 = read_le::<4>(&self.iop_ram, t + 4) as u32;
                let ee_tag_lo = read_le::<4>(&self.iop_ram, t + 8) as u32;
                let ee_tag_hi = read_le::<4>(&self.iop_ram, t + 12) as u32;
                ch.madr = w0 & 0xFF_FFFF;
                ch.words_left = ((w1 & 0xFF_FFFF) + 3) & !3;
                ch.tag_end = w0 & 0xC000_0000 != 0;
                ch.tadr = ch.tadr.wrapping_add(16);
                self.sif.fifo0.extend([ee_tag_lo, ee_tag_hi, 0, 0]);
                trace!(
                    target: "ps2_core::bus::sifdma",
                    madr = format_args!("{:#010x}", ch.madr),
                    words = ch.words_left,
                    ee_tag = format_args!("{ee_tag_lo:08x} {ee_tag_hi:08x}"),
                    end = ch.tag_end,
                    "IOP SIF0 tag"
                );
                if ch.words_left == 0 && ch.tag_end {
                    ch.chcr &= !IOP_CHCR_BUSY;
                    ch.tag_end = false;
                    self.iop_dma_irq(9);
                    progressed = true;
                    continue;
                }
            }
            while ch.words_left > 0 {
                let a = (ch.madr & 0x1F_FFFC) as usize;
                self.sif
                    .fifo0
                    .push_back(read_le::<4>(&self.iop_ram, a) as u32);
                ch.madr = ch.madr.wrapping_add(4);
                ch.words_left -= 1;
                progressed = true;
            }
            if ch.tag_end {
                ch.tag_end = false;
                ch.chcr &= !IOP_CHCR_BUSY;
                self.iop_dma_irq(9);
            }
        }
        progressed
    }

    /// EE SIF0 (ch5): fifo0 into EE RAM as a destination chain.
    fn pump_sif0_ee(&mut self) -> bool {
        let mut progressed = false;
        while self.dma_sif0.chcr & EE_CHCR_STR != 0 {
            if self.dma_sif0.qwc == 0 {
                if self.dma_sif0.tag_end {
                    self.dma_sif0.tag_end = false;
                    self.dma_sif0.chcr &= !EE_CHCR_STR;
                    self.ee_dma_irq(5);
                    debug!(target: "ps2_core::bus::sifdma", "EE SIF0 chain done");
                    progressed = true;
                    continue;
                }
                if self.sif.fifo0.len() < 4 {
                    break;
                }
                let w0 = self.sif.fifo0.pop_front().unwrap();
                let w1 = self.sif.fifo0.pop_front().unwrap();
                self.sif.fifo0.pop_front();
                self.sif.fifo0.pop_front();
                self.dma_sif0.qwc = w0 & 0xFFFF;
                self.dma_sif0.madr = w1 & 0x1FFF_FFF0;
                let id = (w0 >> 28) & 7;
                let irq = w0 & 0x8000_0000 != 0;
                self.dma_sif0.tag_end = id == 7 || (irq && self.dma_sif0.chcr & EE_CHCR_TIE != 0);
                trace!(
                    target: "ps2_core::bus::sifdma",
                    madr = format_args!("{:#010x}", self.dma_sif0.madr),
                    qwc = self.dma_sif0.qwc,
                    end = self.dma_sif0.tag_end,
                    "EE SIF0 tag"
                );
                progressed = true;
            }
            while self.dma_sif0.qwc > 0 && self.sif.fifo0.len() >= 4 {
                let a = (self.dma_sif0.madr & 0x1FFF_FFF0) as usize;
                for i in 0..4 {
                    let w = self.sif.fifo0.pop_front().unwrap();
                    if a + 16 <= RAM_SIZE {
                        write_le::<4>(&mut self.ram, a + i * 4, w as u64);
                        self.note_ram_write(a);
                    }
                }
                self.dma_sif0.madr = self.dma_sif0.madr.wrapping_add(16);
                self.dma_sif0.qwc -= 1;
                progressed = true;
            }
            if self.dma_sif0.qwc > 0 {
                break; // wait for more data
            }
        }
        progressed
    }

    // --- IOP side --------------------------------------------------------

    #[inline]
    pub fn iop_read8(&mut self, vaddr: u32) -> u8 {
        self.iop_read::<1>(vaddr) as u8
    }
    #[inline]
    pub fn iop_read16(&mut self, vaddr: u32) -> u16 {
        self.iop_read::<2>(vaddr) as u16
    }
    #[inline]
    /// Re-establish everything a save state deliberately leaves out: the
    /// raw pointers the recompiler's inline RAM paths use (the boxes moved
    /// with the load), the instruction-fetch cache, and the recompiler's
    /// view of which pages hold code.
    pub(crate) fn after_load(&mut self) {
        self.ram_ptr = self.ram.as_mut_ptr() as usize;
        self.code_pages_ptr = self.code_pages.as_ptr() as usize;
        self.fetch_tag = 1;
        self.code_pages.fill(false);
        self.dirty_code_writes.clear();
        self.jit_flush_needed = true;
    }

    pub fn iop_read32(&mut self, vaddr: u32) -> u32 {
        self.iop_read::<4>(vaddr)
    }
    #[inline]
    pub fn iop_write8(&mut self, vaddr: u32, v: u8) {
        self.iop_write::<1>(vaddr, v as u32)
    }
    #[inline]
    pub fn iop_write16(&mut self, vaddr: u32, v: u16) {
        self.iop_write::<2>(vaddr, v as u32)
    }
    #[inline]
    pub fn iop_write32(&mut self, vaddr: u32, v: u32) {
        self.iop_write::<4>(vaddr, v)
    }

    /// Instruction fetch. Code only ever runs from IOP RAM or the BIOS, so
    /// those two are read directly and everything else defers to the full
    /// [`Bus::iop_read32`] dispatch.
    #[inline]
    pub fn iop_fetch32(&mut self, vaddr: u32) -> u32 {
        let addr = vaddr & 0x1FFF_FFFF;
        let (mem, off) = if addr < 0x0080_0000 {
            (&self.iop_ram, (addr & 0x1F_FFFF) as usize)
        } else if addr >= 0x1FC0_0000 {
            (&self.bios, (addr & 0x3F_FFFF) as usize)
        } else {
            return self.iop_read32(vaddr);
        };
        u32::from_le_bytes(mem[off..off + 4].try_into().unwrap())
    }

    pub fn iop_irq_pending(&self) -> bool {
        self.iop_i_ctrl & 1 != 0 && (self.iop_i_stat & self.iop_i_mask) != 0
    }

    fn iop_read<const N: usize>(&mut self, vaddr: u32) -> u32 {
        // KSEG2 (cache control etc.) is not mapped to physical memory.
        if vaddr >= 0xFFFE_0000 {
            trace!(target: "ps2_core::iop::bus", vaddr = format_args!("{vaddr:#010x}"), "KSEG2 read (stub)");
            return 0;
        }
        let addr = vaddr & 0x1FFF_FFFF;
        match addr {
            // 2 MiB RAM, mirrored through the first 8 MiB.
            0x0000_0000..=0x007F_FFFF => {
                read_le::<N>(&self.iop_ram, (addr & 0x1F_FFFF) as usize) as u32
            }
            0x1F80_0000..=0x1F80_03FF => {
                read_le::<N>(&self.iop_spad, (addr & 0x3FF) as usize) as u32
            }
            0x1F80_1000..=0x1F80_FFFF => self.iop_read_mmio::<N>(addr),
            0x1D00_0000..=0x1D00_00FF => self.sif.iop_read(addr),
            0x1F40_2000..=0x1F40_203F => self.cdvd.read(addr, self.now),
            // ROM1 (DVD player ROM): not present; reads like erased flash so
            // presence/version checks fail instead of "succeeding" with zeros.
            0x1E00_0000..=0x1E3F_FFFF => u32::MAX,
            0x1F90_0000..=0x1F90_0FFF => self.spu2.read::<N>((addr & 0xFFF) as usize),
            0x1FC0_0000..=0x1FFF_FFFF => {
                read_le::<N>(&self.bios, (addr & 0x3F_FFFF) as usize) as u32
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), size = N, "IOP read from unmapped address (reported once)");
                }
                0
            }
        }
    }

    fn iop_write<const N: usize>(&mut self, vaddr: u32, v: u32) {
        if vaddr >= 0xFFFE_0000 {
            trace!(target: "ps2_core::iop::bus", vaddr = format_args!("{vaddr:#010x}"), value = format_args!("{v:#x}"), "KSEG2 write (stub)");
            return;
        }
        let addr = vaddr & 0x1FFF_FFFF;
        match addr {
            0x0000_0000..=0x007F_FFFF => {
                write_le::<N>(&mut self.iop_ram, (addr & 0x1F_FFFF) as usize, v as u64)
            }
            0x1F80_0000..=0x1F80_03FF => {
                write_le::<N>(&mut self.iop_spad, (addr & 0x3FF) as usize, v as u64)
            }
            0x1F80_1000..=0x1F80_FFFF => self.iop_write_mmio::<N>(addr, v),
            0x1D00_0000..=0x1D00_00FF => {
                self.sif.iop_write(addr, v);
                // The IOP raising SMFLG interrupts the EE (INTC SBUS); the
                // EE handler folds the flags into its SREG array and acks.
                if addr & 0xF0 == 0x30 {
                    self.intc_stat |= 1 << 1;
                    self.intc_changed();
                }
            }
            0x1F40_2000..=0x1F40_203F => {
                if let Some(latency) = self.cdvd.write(addr, v) {
                    // The N command completes (and interrupts the IOP)
                    // after the drive latency, from tick_timers.
                    self.cdvd_done_at = Some(self.now + latency);
                    self.timers_due = 0;
                }
            }
            0x1F90_0000..=0x1F90_0FFF => {
                self.spu2.write::<N>((addr & 0xFFF) as usize, v);
                self.timers_due = 0;
                if self.spu2.take_irq() {
                    self.iop_i_stat |= 1 << 9;
                }
            }
            0x1FC0_0000..=0x1FFF_FFFF => {
                warn!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), "IOP write to BIOS ROM ignored");
            }
            _ => {
                if self.warned_unmapped.insert(addr) {
                    warn!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), size = N, "IOP write to unmapped address (reported once)");
                }
            }
        }
    }

    /// IOP timers: 0-2 at 0x1F8011x0 (16-bit), 3-5 at 0x1F8014{8,9,A}0
    /// (32-bit). Lazy counts off the EE cycle counter at the IOP clock (/8).
    fn iop_timer_index(addr: u32) -> Option<usize> {
        match addr & 0xFFF0 {
            0x1100 => Some(0),
            0x1110 => Some(1),
            0x1120 => Some(2),
            0x1480 => Some(3),
            0x1490 => Some(4),
            0x14A0 => Some(5),
            _ => None,
        }
    }

    fn iop_read_mmio<const N: usize>(&mut self, addr: u32) -> u32 {
        let off = (addr & 0xFFFF) as usize;
        if let Some(t) = Self::iop_timer_index(addr) {
            let (now, region) = (self.now, self.region);
            let timer = &mut self.iop_timers[t];
            return match addr & 0xF {
                0x0 => timer.count(t, now, region),
                0x4 => {
                    // Reading MODE clears the reached-target flags.
                    let v = timer.mode;
                    timer.mode &= !(0x1800);
                    v
                }
                0x8 => timer.target,
                _ => 0,
            };
        }
        match addr {
            0x1F80_8200..=0x1F80_82FF => self.sio2.read(addr),
            0x1F80_1070 => self.iop_i_stat,
            0x1F80_1074 => self.iop_i_mask,
            0x1F80_1078 => {
                // I_CTRL reads clear the master enable bit, as on PS1.
                let v = self.iop_i_ctrl;
                self.iop_i_ctrl &= !1;
                v
            }
            0x1F80_10B0 => self.iop_dma_cdvd.madr,
            0x1F80_10B4 => self.iop_dma_cdvd.bcr,
            0x1F80_10B8 => self.iop_dma_cdvd.chcr,
            // IOP DMA: SIF0 (ch9) and SIF1 (ch10), interrupt control.
            0x1F80_1520 => self.iop_dma_sif0.madr,
            0x1F80_1524 => self.iop_dma_sif0.bcr,
            0x1F80_1528 => self.iop_dma_sif0.chcr,
            0x1F80_152C => self.iop_dma_sif0.tadr,
            0x1F80_1530 => self.iop_dma_sif1.madr,
            0x1F80_1534 => self.iop_dma_sif1.bcr,
            0x1F80_1538 => self.iop_dma_sif1.chcr,
            0x1F80_153C => self.iop_dma_sif1.tadr,
            0x1F80_1540 => self.iop_dma_sio2in.madr,
            0x1F80_1544 => self.iop_dma_sio2in.bcr,
            0x1F80_1548 => self.iop_dma_sio2in.chcr,
            0x1F80_1550 => self.iop_dma_sio2out.madr,
            0x1F80_1554 => self.iop_dma_sio2out.bcr,
            0x1F80_1558 => self.iop_dma_sio2out.chcr,
            // SPU2 channels get sub-word accesses (libspu2 writes BCR's
            // block count with a halfword store at +2).
            0x1F80_10C0..=0x1F80_10CB | 0x1F80_1500..=0x1F80_150B => {
                let ch = &self.iop_dma_spu[usize::from(addr >= 0x1F80_1500)];
                let reg = match addr & 0xC {
                    0x0 => ch.madr,
                    0x4 => ch.bcr,
                    _ => ch.chcr,
                };
                extract_sub_word::<N>(reg, addr)
            }
            0x1F80_10F4 => self.iop_dicr,
            0x1F80_1574 => self.iop_dicr2,
            // USB OHCI. No host controller is modelled and nothing is
            // plugged in, so the register block is the plain shadow below
            // apart from these two reads.
            //
            // HcRevision identifies the controller; zero would say there is
            // none. HcCommandStatus's HCR bit is the one a driver blocks on:
            // USBD.IRX sets it and polls until the reset finishes, so it has
            // to read back clear or the module never finishes loading and
            // every RPC waiting on it (the EE's boot, for titles that load
            // USBD) waits forever.
            0x1F80_1600 => 0x10,
            0x1F80_1608 => read_le::<N>(&self.iop_mmio, off) as u32 & !1,
            0x1F80_160C..=0x1F80_16FF => {
                self.warn_stub(addr, "USB host controller");
                read_le::<N>(&self.iop_mmio, off) as u32
            }
            _ => {
                let v = read_le::<N>(&self.iop_mmio, off) as u32;
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "IOP MMIO read (shadow)");
                v
            }
        }
    }

    fn iop_write_mmio<const N: usize>(&mut self, addr: u32, v: u32) {
        if let Some(t) = Self::iop_timer_index(addr) {
            self.timers_due = 0;
            let timer = &mut self.iop_timers[t];
            match addr & 0xF {
                0x0 => {
                    timer.base = v;
                    timer.base_cycle = self.now;
                }
                0x4 => {
                    timer.mode = v;
                    // Writing MODE restarts the counter, as on hardware.
                    timer.base = 0;
                    timer.base_cycle = self.now;
                    timer.last_check = self.now;
                }
                0x8 => timer.target = v,
                _ => {}
            }
            return;
        }
        match addr {
            0x1F80_8200..=0x1F80_82FF => {
                if self.sio2.write(addr, v) {
                    // Transfer completion raises the SIO2 interrupt line.
                    if self.sio2.raise_irq() {
                        self.iop_i_stat |= 1 << 17;
                    }
                    if self.sio2out_deferred {
                        self.sio2out_deferred = false;
                        self.do_sio2_out();
                    }
                }
            }
            // I_STAT write acknowledges: keeps only bits written as 1.
            0x1F80_1070 => self.iop_i_stat &= v,
            0x1F80_1074 => self.iop_i_mask = v,
            0x1F80_1078 => self.iop_i_ctrl = v,
            // CDVD DMA (ch3): drain staged sector data into IOP RAM. The
            // channel is usually armed before the N read command (hardware
            // paces it with DRQ), so hold the copy until data is staged.
            0x1F80_10B0 => self.iop_dma_cdvd.madr = v & 0xFF_FFFF,
            0x1F80_10B4 => self.iop_dma_cdvd.bcr = v,
            0x1F80_10B8 => {
                self.iop_dma_cdvd.chcr = v & !IOP_CHCR_BUSY;
                if v & IOP_CHCR_BUSY != 0 {
                    // Sectors staged by a still-busy drive are not visible
                    // yet: the copy waits for the completion like hardware
                    // waits on DRQ.
                    if self.cdvd.n_busy || self.cdvd.read_remaining() == 0 {
                        self.cdvd_dma_deferred = true;
                    } else {
                        self.do_cdvd_dma();
                    }
                }
            }
            // SPU2 DMA (ch4 = core 0, ch7 = core 1): data moves now, the
            // completion interrupt fires when the SPU2 model says so.
            0x1F80_10C0..=0x1F80_10CB | 0x1F80_1500..=0x1F80_150B => {
                let core = usize::from(addr >= 0x1F80_1500);
                let ch = &mut self.iop_dma_spu[core];
                match addr & 0xC {
                    0x0 => ch.madr = merge_sub_word::<N>(ch.madr, addr, v) & 0xFF_FFFF,
                    0x4 => ch.bcr = merge_sub_word::<N>(ch.bcr, addr, v),
                    _ => {
                        ch.chcr = merge_sub_word::<N>(ch.chcr, addr, v);
                        if ch.chcr & IOP_CHCR_BUSY != 0 {
                            self.do_spu2_dma(core);
                        }
                    }
                }
            }
            0x1F80_1520 => self.iop_dma_sif0.madr = v & 0xFF_FFFF,
            0x1F80_1524 => self.iop_dma_sif0.bcr = v,
            0x1F80_1528 => {
                self.iop_dma_sif0.chcr = v;
                if v & IOP_CHCR_BUSY != 0 {
                    self.pump_sif();
                }
            }
            0x1F80_152C => self.iop_dma_sif0.tadr = v & 0xFF_FFFF,
            0x1F80_1530 => self.iop_dma_sif1.madr = v & 0xFF_FFFF,
            0x1F80_1534 => self.iop_dma_sif1.bcr = v,
            0x1F80_1538 => {
                self.iop_dma_sif1.chcr = v;
                if v & IOP_CHCR_BUSY != 0 {
                    self.pump_sif();
                }
            }
            0x1F80_153C => self.iop_dma_sif1.tadr = v & 0xFF_FFFF,
            // SIO2 DMA: ch11 feeds the in-FIFO, ch12 drains the out-FIFO.
            // Transfers complete instantly (the FIFO model has no timing).
            0x1F80_1540 => self.iop_dma_sio2in.madr = v & 0xFF_FFFF,
            0x1F80_1544 => self.iop_dma_sio2in.bcr = v,
            0x1F80_1548 => {
                self.iop_dma_sio2in.chcr = v & !IOP_CHCR_BUSY;
                if v & IOP_CHCR_BUSY != 0 {
                    let bcr = self.iop_dma_sio2in.bcr;
                    let bytes = Self::iop_bcr_bytes(bcr);
                    self.sio2.in_block = (((bcr & 0xFFFF) * 4) as usize, (bcr >> 16) as usize);
                    let start = (self.iop_dma_sio2in.madr & 0x1F_FFFF) as usize;
                    for i in 0..bytes {
                        let b = self.iop_ram[(start + i) % self.iop_ram.len()];
                        self.sio2.fifo_in.push(b);
                    }
                    debug!(target: "ps2_core::iop::sio2", bytes, "DMA in");
                    if self.sio2.dma_in_done() {
                        if self.sio2.raise_irq() {
                            self.iop_i_stat |= 1 << 17;
                        }
                        if self.sio2out_deferred {
                            self.sio2out_deferred = false;
                            self.do_sio2_out();
                        }
                    }
                    self.iop_dma_irq(11);
                }
            }
            0x1F80_1550 => self.iop_dma_sio2out.madr = v & 0xFF_FFFF,
            0x1F80_1554 => self.iop_dma_sio2out.bcr = v,
            0x1F80_1558 => {
                self.iop_dma_sio2out.chcr = v & !IOP_CHCR_BUSY;
                if v & IOP_CHCR_BUSY != 0 {
                    // The hardware channel waits on the SIO2's DRQ: if the
                    // transfer hasn't produced its response yet (the driver
                    // may arm this DMA before CTRL), hold the copy until it
                    // runs.
                    if self.sio2.out_pos >= self.sio2.fifo_out.len()
                        && (self.sio2.pending || !self.sio2.fifo_in.is_empty())
                    {
                        self.sio2out_deferred = true;
                    } else {
                        self.do_sio2_out();
                    }
                }
            }
            // DICR/DICR2: enables in bits 16-23, flags (W1C) in bits 24-30.
            0x1F80_10F4 => {
                self.iop_dicr =
                    (v & 0x00FF_FFFF) | (self.iop_dicr & !(v & 0x7F00_0000) & 0x7F00_0000);
            }
            0x1F80_1574 => {
                self.iop_dicr2 =
                    (v & 0x00FF_FFFF) | (self.iop_dicr2 & !(v & 0x7F00_0000) & 0x7F00_0000);
            }
            // POST: boot progress byte from the IOP BIOS.
            0x1F80_2070 => {
                debug!(target: "ps2_core::iop::bus", stage = format_args!("{:#04x}", v as u8), "POST");
            }
            _ => {
                trace!(target: "ps2_core::iop::bus", addr = format_args!("{addr:#010x}"), value = format_args!("{v:#x}"), "IOP MMIO write (shadow)");
                write_le::<N>(&mut self.iop_mmio, (addr & 0xFFFF) as usize, v as u64);
            }
        }
    }

    fn tty_push(&mut self, byte: u8) {
        let c = byte as char;
        if c == '\n' {
            debug!(target: "ps2_core::tty", "{}", self.tty_line);
            self.tty_line.clear();
        } else if byte.is_ascii() && !c.is_control() {
            self.tty_line.push(c);
        }
        self.tty_buffer.push(c);
    }
}

#[inline]
fn read_le<const N: usize>(mem: &[u8], offset: usize) -> u64 {
    // One bounds check and one load, instead of a byte loop.
    let mut b = [0u8; 8];
    b[..N].copy_from_slice(&mem[offset..offset + N]);
    u64::from_le_bytes(b)
}

/// Read `N` bytes of a 32-bit register at the byte lane selected by `addr`.
#[inline]
fn extract_sub_word<const N: usize>(reg: u32, addr: u32) -> u32 {
    let shift = (addr & 3) * 8;
    let mask = if N >= 4 { u32::MAX } else { (1u32 << (8 * N as u32)) - 1 };
    (reg >> shift) & mask
}

/// Merge an `N`-byte write into a 32-bit register at the byte lane selected
/// by `addr`.
#[inline]
fn merge_sub_word<const N: usize>(reg: u32, addr: u32, v: u32) -> u32 {
    let shift = (addr & 3) * 8;
    let mask = if N >= 4 { u32::MAX } else { ((1u32 << (8 * N as u32)) - 1) << shift };
    (reg & !mask) | ((v << shift) & mask)
}

#[inline]
fn write_le<const N: usize>(mem: &mut [u8], offset: usize, v: u64) {
    mem[offset..offset + N].copy_from_slice(&v.to_le_bytes()[..N]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bus() -> Bus {
        Bus::new(vec![0u8; BIOS_SIZE], false, Region::Ntsc)
    }

    #[test]
    fn ram_read_write_roundtrip() {
        let mut b = bus();
        b.write32(0x0010_0000, 0xDEAD_BEEF);
        assert_eq!(b.read32(0x0010_0000), 0xDEAD_BEEF);
        // KSEG0/KSEG1 mirrors reach the same storage.
        assert_eq!(b.read32(0x8010_0000), 0xDEAD_BEEF);
        assert_eq!(b.read32(0xA010_0000), 0xDEAD_BEEF);
    }

    #[test]
    fn scratchpad_is_isolated_from_ram() {
        let mut b = bus();
        b.write32(0x7000_0000, 0x1234_5678);
        assert_eq!(b.read32(0x7000_0000), 0x1234_5678);
        assert_ne!(b.read32(0x0000_0000), 0x1234_5678);
    }

    /// Channel 9 moves quadwords from memory into the scratchpad and
    /// leaves both addresses past the run.
    #[test]
    fn to_spr_copies_memory_into_the_scratchpad() {
        let mut b = bus();
        for i in 0..8u32 {
            b.write32(0x0010_0000 + i * 4, 0x1000 + i);
        }
        b.write32(0x1000_D410, 0x0010_0000); // MADR
        b.write32(0x1000_D480, 0x0100); // SADR
        b.write32(0x1000_D420, 2); // QWC
        b.write32(0x1000_D400, EE_CHCR_STR);
        for i in 0..8u32 {
            assert_eq!(b.read32(0x7000_0100 + i * 4), 0x1000 + i);
        }
        assert_eq!(b.read32(0x1000_D400) & EE_CHCR_STR, 0);
        assert_eq!(b.read32(0x1000_D420), 0);
        assert_eq!(b.read32(0x1000_D410), 0x0010_0020);
        assert_eq!(b.read32(0x1000_D480), 0x0120);
    }

    /// Channel 8 the other way, and its start bit clears the same way.
    #[test]
    fn from_spr_copies_the_scratchpad_into_memory() {
        let mut b = bus();
        for i in 0..4u32 {
            b.write32(0x7000_0200 + i * 4, 0x2000 + i);
        }
        b.write32(0x1000_D010, 0x0020_0000); // MADR
        b.write32(0x1000_D080, 0x0200); // SADR
        b.write32(0x1000_D020, 1); // QWC
        b.write32(0x1000_D000, EE_CHCR_STR);
        for i in 0..4u32 {
            assert_eq!(b.read32(0x0020_0000 + i * 4), 0x2000 + i);
        }
        assert_eq!(b.read32(0x1000_D000) & EE_CHCR_STR, 0);
        assert_eq!(b.read32(0x1000_D080), 0x0210);
    }

    /// Channel 9's chain reads its tags from memory at TADR: one cnt of
    /// a quadword, then end.
    #[test]
    fn to_spr_follows_a_source_chain() {
        let mut b = bus();
        // cnt, qwc 1, then the quadword it covers.
        b.write32(0x0030_0000, 1 | (1 << 28));
        for i in 0..4u32 {
            b.write32(0x0030_0010 + i * 4, 0x3000 + i);
        }
        b.write32(0x0030_0020, 7 << 28); // end, qwc 0
        b.write32(0x1000_D430, 0x0030_0000); // TADR
        b.write32(0x1000_D480, 0); // SADR
        b.write32(0x1000_D400, EE_CHCR_STR | (1 << 2)); // chain mode
        for i in 0..4u32 {
            assert_eq!(b.read32(0x7000_0000 + i * 4), 0x3000 + i);
        }
        assert_eq!(b.read32(0x1000_D400) & EE_CHCR_STR, 0);
    }

    /// Channel 8's chain reads its tags from the scratchpad instead, each
    /// naming where in memory the quadwords behind it land.
    #[test]
    fn from_spr_follows_a_chain_out_of_the_scratchpad() {
        let mut b = bus();
        // end tag: one quadword, destination 0x0040_0000.
        b.write32(0x7000_0000, 1 | (7 << 28));
        b.write32(0x7000_0004, 0x0040_0000);
        for i in 0..4u32 {
            b.write32(0x7000_0010 + i * 4, 0x4000 + i);
        }
        b.write32(0x1000_D080, 0); // SADR
        b.write32(0x1000_D000, EE_CHCR_STR | (1 << 2)); // chain mode
        for i in 0..4u32 {
            assert_eq!(b.read32(0x0040_0000 + i * 4), 0x4000 + i);
        }
        assert_eq!(b.read32(0x1000_D000) & EE_CHCR_STR, 0);
        assert_eq!(b.read32(0x1000_D080), 0x0020);
    }

    #[test]
    fn tty_capture() {
        let mut b = bus();
        for c in b"hi\n" {
            b.write8(0x1000_F180, *c);
        }
        assert_eq!(b.tty_buffer, "hi\n");
    }

    #[test]
    fn rdram_init_handshake() {
        let mut b = bus();
        // SOP=0, SA=0x21 (INIT): first two reads answer 0x1F, then 0.
        b.write32(0x1000_F430, 0x21 << 16);
        assert_eq!(b.read32(0x1000_F440), 0x1F);
        assert_eq!(b.read32(0x1000_F440), 0x1F);
        assert_eq!(b.read32(0x1000_F440), 0);
    }

    #[test]
    fn debugger_peeks_gs_priv_and_refuses_the_rdram_handshake() {
        let mut b = bus();
        // SMODE1 (write-only to software) still reads back for a debugger.
        b.write64(0xB200_0010, 0x0000_0000_C74F_0F1E);
        assert_eq!(b.peek8(0xB200_0010), Some(0x1E));
        assert_eq!(b.peek8(0xB200_0013), Some(0xC7));
        // SYNCHV, which is otherwise only visible in the shadow array.
        b.write64(0xB200_0060, 0x00C7_800A_1500_0801);
        assert_eq!(b.peek8(0xB200_0066), Some(0xC7));

        b.write32(0x1000_F010, 0x0000_0400);
        assert_eq!(b.peek8(0xB000_F011), Some(0x04));

        // MCH_DRD advances the RDRAM device count on read, so it stays
        // refused rather than being answered from a peek.
        assert_eq!(b.peek8(0xB000_F440), None);
    }
}
