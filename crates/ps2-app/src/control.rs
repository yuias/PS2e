//! Interactive control port for headless automation.
//!
//! Designed for script and LLM operators: a line-based text protocol over
//! TCP where the emulator runs in *lockstep* — it advances only when a
//! `run` or `press` command says so, so every observation is deterministic
//! and repeatable. One command per line; the reply is `ok`/`err` followed
//! by payload lines, terminated by a single `.` line (payload lines
//! starting with `.` are dot-stuffed, SMTP-style).
//!
//! The bundled `ps2ctl` binary wraps one command per invocation, so a shell
//! (or a tool-using agent) can drive a session statelessly:
//!
//! ```text
//! ps2ctl press circle 30    # hold CIRCLE for 30 frames
//! ps2ctl frame shot.png     # dump what the TV shows
//! ps2ctl peek 00100000 64   # inspect EE memory, side-effect-free
//! ```
//!
//! This is the interactive counterpart to the `--cycles` batch mode: the
//! two are separate modes, because a scripted `--press circle@150e9` and a
//! client deciding when to press cannot both own the run. Cycle counts
//! reported here are the same absolute EE cycles the batch flags take, so a
//! number read off `state` can be pasted into a `--press` or `--save-state`
//! recipe verbatim.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use ps2_core::cheats::Group;
use ps2_core::{EE_CLOCK_HZ, Ps2System};

use crate::cheatfile;
use crate::pad;

/// Emulation granularity, matching the headless loop's slice. Held buttons
/// are re-applied and TTY drained on these boundaries, so a `run` behaves
/// the way the same span of `--press` scripting does.
const SLICE: u64 = 1_000_000;

/// TTY text kept between `tty` commands. A session that prints for hours
/// must not grow the buffer without bound; the oldest text goes first.
const TTY_CAP: usize = 1 << 20;

/// Cap on a single `peek`, in bytes. The reply is hex, so this is also
/// what keeps one command from returning a megabyte of text.
const PEEK_MAX: u32 = 4096;

/// Largest `peekb` reply / `peekm` total, in bytes: the whole EE RAM.
const PEEK_BYTES_MAX: u32 = 32 * 1024 * 1024;

const BUTTON_NAMES: [(&str, u16); 16] = [
    ("select", pad::SELECT),
    ("l3", pad::L3),
    ("r3", pad::R3),
    ("start", pad::START),
    ("up", pad::UP),
    ("right", pad::RIGHT),
    ("down", pad::DOWN),
    ("left", pad::LEFT),
    ("l2", pad::L2),
    ("r2", pad::R2),
    ("l1", pad::L1),
    ("r1", pad::R1),
    ("triangle", pad::TRIANGLE),
    ("circle", pad::CIRCLE),
    ("cross", pad::CROSS),
    ("square", pad::SQUARE),
];

fn buttons_to_names(mask: u16) -> String {
    let names: Vec<&str> =
        BUTTON_NAMES.iter().filter(|(_, b)| mask & b != 0).map(|(n, _)| *n).collect();
    if names.is_empty() { "none".into() } else { names.join("+") }
}

/// Parse `circle+cross` into a button mask, case-insensitively.
fn parse_buttons(s: &str) -> Result<u16, String> {
    s.split('+').try_fold(0u16, |acc, name| {
        pad::mask_by_name(&name.to_ascii_lowercase()).map(|b| acc | b)
    })
}

/// Parse `10` (frames), `2s` (seconds) or `50000c` (EE cycles) into cycles.
fn parse_duration(s: &str, cycles_per_frame: u64) -> Result<u64, String> {
    let (num, unit) = match s.chars().last() {
        Some('s') => (&s[..s.len() - 1], EE_CLOCK_HZ),
        Some('c') => (&s[..s.len() - 1], 1),
        _ => (s, cycles_per_frame),
    };
    let n: f64 = num.parse().map_err(|_| format!("bad duration '{s}'"))?;
    if !n.is_finite() || n <= 0.0 {
        return Err(format!("bad duration '{s}'"));
    }
    Ok((n * unit as f64) as u64)
}

fn parse_addr(s: &str) -> Result<u32, String> {
    u32::from_str_radix(s.trim_start_matches("0x"), 16).map_err(|_| format!("bad address '{s}'"))
}

/// Which core's bus view a `peek`/`poke` uses. The EE is the default
/// because it is what the bring-up workflow looks at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Core {
    Ee,
    Iop,
}

/// Split an optional leading `ee`/`iop` off a peek/poke argument list.
fn split_core<'a>(args: &'a [&'a str]) -> (Core, &'a [&'a str]) {
    match args.first() {
        Some(&"ee") => (Core::Ee, &args[1..]),
        Some(&"iop") => (Core::Iop, &args[1..]),
        _ => (Core::Ee, args),
    }
}

fn peek8(sys: &mut Ps2System, core: Core, addr: u32) -> Option<u8> {
    match core {
        Core::Ee => sys.bus.peek8(addr),
        Core::Iop => sys.bus.iop_peek8(addr),
    }
}

fn poke8(sys: &mut Ps2System, core: Core, addr: u32, v: u8) -> bool {
    match core {
        Core::Ee => sys.bus.poke8(addr, v),
        Core::Iop => sys.bus.iop_poke8(addr, v),
    }
}

/// Standard base64 with `=` padding (no dependency for one function).
fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 { TABLE[(b2 & 0x3f) as usize] as char } else { '=' });
    }
    out
}

/// Decode a `base64_encode` payload, for the tests only. A 256-entry lookup
/// built once: a per-character `position()` scan is too slow for a 43 MiB
/// test payload in debug.
#[cfg(test)]
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    static LOOKUP: std::sync::OnceLock<[i8; 256]> = std::sync::OnceLock::new();
    let lookup = LOOKUP.get_or_init(|| {
        let mut l = [-1i8; 256];
        for (i, &b) in TABLE.iter().enumerate() {
            l[b as usize] = i as i8;
        }
        l
    });
    let mut out = Vec::with_capacity(s.len() / 4 * 3 + 3);
    let mut bits = 0u32;
    let mut nbits = 0u32;
    for c in s.trim_end_matches('=').bytes() {
        let v = lookup[c as usize];
        if v < 0 {
            return None;
        }
        bits = (bits << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
        }
    }
    Some(out)
}

/// Backing slice and offset for the directly addressable region containing
/// `addr`, already truncated to the end of the run reachable from there (a
/// TLB page for EE RAM, a mirror period for IOP RAM): `slice.len() - offset`
/// is therefore always a safe chunk length. `None` where there is none
/// (`read_range` falls back to `peek8`).
fn ram_window(sys: &mut Ps2System, core: Core, addr: u32) -> Option<(&[u8], usize)> {
    match core {
        Core::Ee => match addr {
            0x7000_0000..=0x7000_3FFF => Some((&sys.bus.spad, (addr & 0x3FFF) as usize)),
            _ => {
                let phys = sys.bus.ram_phys_of(addr)? as usize;
                // TLB pages need not be physically contiguous, so the chunk
                // stops at the page end even though the RAM array continues.
                let page_end = (phys | 0xFFF) + 1;
                Some((&sys.bus.ram[..page_end], phys))
            }
        },
        Core::Iop => {
            if addr >= 0xFFFE_0000 {
                return None; // KSEG2 cache control
            }
            match addr & 0x1FFF_FFFF {
                a @ 0x0000_0000..=0x007F_FFFF => Some((&sys.bus.iop_ram, (a & 0x1F_FFFF) as usize)),
                a @ 0x1F80_0000..=0x1F80_03FF => Some((&sys.bus.iop_spad, (a & 0x3FF) as usize)),
                _ => None,
            }
        }
    }
}

/// `len` bytes from `addr` on `core`. RAM, scratchpad and IOP RAM are
/// copied as slices (a per-byte loop over 32 MiB would dominate the read);
/// anything else goes through `peek8` a byte at a time, and the first
/// unreadable byte fails the whole read (base64 has no room for a per-byte
/// marker). The readability rule is therefore exactly `peek`'s: whatever
/// `peek` shows as `--` makes this fail too.
fn read_range(sys: &mut Ps2System, core: Core, addr: u32, len: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(len as usize);
    let mut a = addr;
    let mut remaining = len;
    while remaining > 0 {
        if let Some((slice, offset)) = ram_window(sys, core, a) {
            let n = remaining.min((slice.len() - offset) as u32);
            out.extend_from_slice(&slice[offset..offset + n as usize]);
            a = a.wrapping_add(n);
            remaining -= n;
        } else {
            let b = peek8(sys, core, a).ok_or_else(|| format!("address {a:#010x} not readable"))?;
            out.push(b);
            a = a.wrapping_add(1);
            remaining -= 1;
        }
    }
    Ok(out)
}

pub struct Reply {
    pub ok: bool,
    /// Payload lines, without the status line or the terminator.
    pub payload: String,
    pub quit: bool,
}

impl Reply {
    fn ok(payload: impl Into<String>) -> Self {
        Reply { ok: true, payload: payload.into(), quit: false }
    }
    fn err(msg: impl Into<String>) -> Self {
        Reply { ok: false, payload: msg.into(), quit: false }
    }
}

/// Command executor: protocol state independent of the transport, so the
/// whole command surface is unit-testable without sockets.
#[derive(Default)]
pub struct Controller {
    /// Buttons held across `run` commands (`input set`).
    held: u16,
    /// Kernel and IOP TTY text accumulated since the last `tty`.
    tty: String,
    /// The pnach model behind `cheat list`. Toggles live here and in the
    /// machine's table only: headless runs never write the user's
    /// `cheats.toml`, so a scripted session cannot change what the window
    /// shows next time.
    cheats: Vec<Group>,
    /// The pnach the current list came from, for `cheat reload`.
    cheat_file: Option<PathBuf>,
    /// The disc lifted out by `disc open`, so a bare `disc close` can put
    /// the same one back the way a real tray does.
    tray: Option<std::fs::File>,
}

impl Controller {
    /// Adopt the pnach loaded for a disc named on the command line.
    pub fn set_cheats(&mut self, path: PathBuf, groups: Vec<Group>) {
        self.cheat_file = Some(path);
        self.cheats = groups;
    }

    /// Install the adopted list, with the master switch in its starting
    /// position. The list goes in either way, so `cheat list` shows what
    /// the disc has before anything is switched on.
    pub fn install_cheats(&mut self, sys: &mut Ps2System, enabled: bool) {
        self.push_cheats(sys);
        sys.set_cheats_enabled(enabled);
    }

    /// Advance emulation, keeping the held buttons applied and draining the
    /// per-slice outputs the batch loop also drains.
    fn advance(&mut self, sys: &mut Ps2System, cycles: u64) {
        let target = sys.cycles.saturating_add(cycles);
        while sys.cycles < target {
            sys.bus.sio2.buttons = self.held;
            sys.run((target - sys.cycles).min(SLICE));
            self.collect(sys);
        }
    }

    /// Take what the machine produced during a slice. SPU2 output is
    /// discarded rather than accumulated: nothing here writes a WAV, and an
    /// undrained queue grows for the length of the session.
    fn collect(&mut self, sys: &mut Ps2System) {
        let mut text = sys.take_tty();
        text.push_str(&sys.take_iop_tty());
        if !text.is_empty() {
            self.tty.push_str(&text);
            if self.tty.len() > TTY_CAP {
                let mut cut = self.tty.len() - TTY_CAP;
                while cut < self.tty.len() && !self.tty.is_char_boundary(cut) {
                    cut += 1;
                }
                self.tty.drain(..cut);
            }
        }
        sys.bus.spu2.take_output();
    }

    /// Rebuild the machine's cheat table from the local model.
    fn push_cheats(&self, sys: &mut Ps2System) {
        sys.set_cheats(self.cheats.clone());
    }

    pub fn execute(&mut self, sys: &mut Ps2System, line: &str, debugger_owns: bool) -> Reply {
        let mut words = line.split_whitespace();
        let cmd = words.next().unwrap_or("");
        let args: Vec<&str> = words.collect();
        // The debugger and the control port must not both drive execution
        // (loadstate mutates it just as much as running does).
        if debugger_owns && matches!(cmd, "run" | "press" | "loadstate") {
            return Reply::err("debugger attached; execution is owned by the debugger");
        }
        let cpf = sys.region().cycles_per_frame();
        match (cmd, args.as_slice()) {
            ("help", _) => Reply::ok(HELP.trim_end()),
            ("state", _) => Reply::ok(format!(
                "ee_pc={:#010x} iop_pc={:#010x} cycles={} frames={} held={} tray={}",
                sys.ee.pc,
                sys.iop.pc,
                sys.cycles,
                sys.cycles / cpf,
                buttons_to_names(self.held),
                if sys.bus.cdvd.tray_open() { "open" } else { "closed" },
            )),
            // Absolute form, so a cycle read off `state` (or out of a gate
            // recipe) can be arrived at exactly rather than by arithmetic.
            ("run", ["to", at]) => match crate::parse_cycle(at) {
                Ok(at) if at <= sys.cycles => {
                    Reply::err(format!("already at cycle {} (past {at})", sys.cycles))
                }
                Ok(at) => {
                    self.advance(sys, at - sys.cycles);
                    Reply::ok(format!("at cycle {}, ee_pc={:#010x}", sys.cycles, sys.ee.pc))
                }
                Err(e) => Reply::err(e),
            },
            ("run", [dur]) => match parse_duration(dur, cpf) {
                Ok(cycles) => {
                    self.advance(sys, cycles);
                    Reply::ok(format!(
                        "ran {cycles} cycles to {}, ee_pc={:#010x}",
                        sys.cycles, sys.ee.pc
                    ))
                }
                Err(e) => Reply::err(e),
            },
            ("press", [buttons, dur]) => match (parse_buttons(buttons), parse_duration(dur, cpf)) {
                (Ok(mask), Ok(cycles)) => {
                    let from = sys.cycles;
                    let prev = self.held;
                    self.held |= mask;
                    self.advance(sys, cycles);
                    self.held = prev;
                    sys.bus.sio2.buttons = self.held;
                    Reply::ok(format!(
                        "pressed {} over cycles {from}-{}",
                        buttons_to_names(mask),
                        sys.cycles
                    ))
                }
                (Err(e), _) | (_, Err(e)) => Reply::err(e),
            },
            ("input", ["set", buttons]) => match parse_buttons(buttons) {
                Ok(mask) => {
                    self.held = mask;
                    sys.bus.sio2.buttons = mask;
                    Reply::ok(format!("holding {}", buttons_to_names(mask)))
                }
                Err(e) => Reply::err(e),
            },
            ("input", ["clear"]) => {
                self.held = 0;
                sys.bus.sio2.buttons = 0;
                Reply::ok("holding none")
            }
            ("peek", rest) => {
                let (core, rest) = split_core(rest);
                let [addr, len] = rest else {
                    return Reply::err("usage: peek [ee|iop] <hexaddr> <len>");
                };
                let (addr, len) = match (parse_addr(addr), len.parse::<u32>()) {
                    (Ok(a), Ok(l)) if l > 0 && l <= PEEK_MAX => (a, l),
                    (Err(e), _) => return Reply::err(e),
                    _ => return Reply::err(format!("bad length (1..{PEEK_MAX})")),
                };
                let mut out = String::new();
                for base in (0..len).step_by(16) {
                    let row: Vec<String> = (base..(base + 16).min(len))
                        .map(|i| match peek8(sys, core, addr.wrapping_add(i)) {
                            Some(b) => format!("{b:02x}"),
                            None => "--".into(),
                        })
                        .collect();
                    out.push_str(&format!("{:#010x}: {}\n", addr.wrapping_add(base), row.join(" ")));
                }
                Reply::ok(out.trim_end().to_string())
            }
            ("peekb", rest) => {
                let (core, rest) = split_core(rest);
                let [addr, len] = rest else {
                    return Reply::err("usage: peekb [ee|iop] <hexaddr> <len>");
                };
                let (addr, len) = match (parse_addr(addr), len.parse::<u32>()) {
                    (Ok(a), Ok(l)) if l >= 1 && l <= PEEK_BYTES_MAX => (a, l),
                    (Err(e), _) => return Reply::err(e),
                    _ => return Reply::err(format!("bad length (1-{PEEK_BYTES_MAX})")),
                };
                match read_range(sys, core, addr, len) {
                    Ok(bytes) => Reply::ok(base64_encode(&bytes)),
                    Err(e) => Reply::err(e),
                }
            }
            ("peekm", rest) => {
                let (core, rest) = split_core(rest);
                if rest.is_empty() {
                    return Reply::err("peekm needs at least one <hexaddr>:<len>");
                }
                // Every range is parsed and sized before any is read, so a
                // bad range later in the list does not read the earlier ones.
                let mut ranges = Vec::with_capacity(rest.len());
                let mut total: u64 = 0;
                for r in rest {
                    let bad = || format!("bad range '{r}' (want <hexaddr>:<len>)");
                    let Some((a, l)) = r.split_once(':') else {
                        return Reply::err(bad());
                    };
                    let (addr, len) = match (parse_addr(a), l.parse::<u32>()) {
                        (Ok(addr), Ok(len)) if len >= 1 => (addr, len),
                        _ => return Reply::err(bad()),
                    };
                    total += u64::from(len);
                    if total > u64::from(PEEK_BYTES_MAX) {
                        return Reply::err(format!("bad length (1-{PEEK_BYTES_MAX})"));
                    }
                    ranges.push((addr, len));
                }
                let mut out = String::new();
                for (addr, len) in ranges {
                    match read_range(sys, core, addr, len) {
                        Ok(bytes) => {
                            out.push_str(&base64_encode(&bytes));
                            out.push('\n');
                        }
                        Err(e) => return Reply::err(e),
                    }
                }
                Reply::ok(out.trim_end().to_string())
            }
            ("poke", rest) => {
                let (core, rest) = split_core(rest);
                let [addr, hex] = rest else {
                    return Reply::err("usage: poke [ee|iop] <hexaddr> <hexbytes>");
                };
                let addr = match parse_addr(addr) {
                    Ok(a) => a,
                    Err(e) => return Reply::err(e),
                };
                if hex.is_empty() || hex.len() % 2 != 0 {
                    return Reply::err("bad hex (whole bytes, big-endian byte order)");
                }
                let bytes: Option<Vec<u8>> = (0..hex.len())
                    .step_by(2)
                    .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
                    .collect();
                let Some(bytes) = bytes else {
                    return Reply::err("bad hex");
                };
                for (i, b) in bytes.iter().enumerate() {
                    if !poke8(sys, core, addr.wrapping_add(i as u32), *b) {
                        return Reply::err(format!(
                            "address {:#010x} not writable",
                            addr.wrapping_add(i as u32)
                        ));
                    }
                }
                Reply::ok(format!("wrote {} bytes", bytes.len()))
            }
            // The tray is split into its two halves, so a script can leave
            // the drive open across several `run`s and watch how the game
            // reacts before the next disc goes in.
            ("disc", ["open"]) => {
                self.tray = sys.bus.cdvd.open_tray();
                Reply::ok("drive open")
            }
            ("disc", ["close"]) => {
                let had = self.tray.is_some();
                sys.bus.cdvd.close_tray(self.tray.take(), sys.cycles);
                Reply::ok(if had { "drive closed" } else { "drive closed, empty" })
            }
            ("disc", ["close", path]) => match std::fs::File::open(path) {
                Ok(f) => {
                    self.tray = None;
                    sys.bus.cdvd.close_tray(Some(f), sys.cycles);
                    let serial = sys.bus.cdvd.boot_serial().unwrap_or_else(|| "unknown".into());
                    let pnach = cheatfile::path_for(Path::new(path));
                    self.cheats = cheatfile::load(&pnach);
                    let n = self.cheats.len();
                    self.cheat_file = Some(pnach);
                    self.push_cheats(sys);
                    Reply::ok(format!("drive closed on {path} ({serial}); {n} cheats"))
                }
                Err(e) => Reply::err(format!("open {path}: {e}")),
            },
            ("cheat", ["apply", state @ ("on" | "off")]) => {
                sys.set_cheats_enabled(*state == "on");
                Reply::ok(format!("cheats {state}"))
            }
            ("cheat", ["list"]) => {
                let mut out = format!(
                    "apply: {}\n",
                    if sys.cheats().enabled { "on" } else { "off" }
                );
                for (i, g) in self.cheats.iter().enumerate() {
                    let name = if g.name.is_empty() { "(unnamed)" } else { g.name.as_str() };
                    let warn =
                        if g.warnings.is_empty() { String::new() } else { format!(", {} rejected", g.warnings.len()) };
                    out.push_str(&format!(
                        "{i:>3}  {}  {name} [{} codes{warn}]\n",
                        if g.enabled { "on " } else { "off" },
                        g.cheats.len(),
                    ));
                }
                Reply::ok(out.trim_end())
            }
            ("cheat", [state @ ("on" | "off"), index]) => {
                let Ok(i) = index.parse::<usize>() else {
                    return Reply::err(format!("bad cheat index '{index}'"));
                };
                let Some(group) = self.cheats.get_mut(i) else {
                    return Reply::err(format!("no cheat {i} (see 'cheat list')"));
                };
                group.enabled = *state == "on";
                let name = group.name.clone();
                // Flip the live table in place rather than rebuilding it:
                // rebuilding re-arms every one-shot, so toggling one cheat
                // would re-fire the start-up writes of all the others.
                sys.set_group_enabled(&name, *state == "on");
                let shown = if name.is_empty() { "(unnamed)".into() } else { name };
                Reply::ok(format!("cheat {i} '{shown}' {state} (in memory only)"))
            }
            ("cheat", ["reload"]) => {
                let Some(path) = self.cheat_file.clone() else {
                    return Reply::err("no pnach (start with --disc, or 'disc close <path>')");
                };
                self.cheats = cheatfile::load(&path);
                let n = self.cheats.len();
                self.push_cheats(sys);
                Reply::ok(format!("{n} cheats from {}", path.display()))
            }
            ("tty", _) => Reply::ok(std::mem::take(&mut self.tty)),
            ("frame", [path]) => {
                let (w, h, rgba) = sys.framebuffer();
                if w == 0 || h == 0 {
                    return Reply::err("no display yet (run at least one frame)");
                }
                match crate::write_image(path, w, h, &rgba) {
                    Ok(()) => Reply::ok(format!("{w}x{h} -> {path}")),
                    Err(e) => Reply::err(format!("write {path}: {e}")),
                }
            }
            ("frameb", rest) => {
                if rest.len() > 1 {
                    return Reply::err("usage: frameb [rgb24|png]");
                }
                let fmt = rest.first().copied().unwrap_or("rgb24");
                if fmt != "rgb24" && fmt != "png" {
                    return Reply::err(format!("bad format '{fmt}' (rgb24 or png)"));
                }
                let (w, h, rgba) = sys.framebuffer();
                if w == 0 || h == 0 {
                    return Reply::err("no display yet (run at least one frame)");
                }
                let base64 = if fmt == "rgb24" {
                    base64_encode(&crate::rgb24(&rgba))
                } else {
                    match crate::png_bytes(w, h, &rgba) {
                        Ok(bytes) => base64_encode(&bytes),
                        Err(e) => return Reply::err(format!("encode: {e}")),
                    }
                };
                Reply::ok(format!("{w} {h} {fmt}\n{base64}"))
            }
            // Raw rather than an image: PS2 VRAM is one 4 MiB pool holding
            // buffers of several pixel formats at once, so there is no one
            // picture of it. This is the same blob `--dump` writes.
            ("vram", [path]) => {
                let vram = sys.bus.gs.vram();
                match std::fs::write(path, &vram) {
                    Ok(()) => Reply::ok(format!("{} bytes -> {path}", vram.len())),
                    Err(e) => Reply::err(format!("write {path}: {e}")),
                }
            }
            ("savestate", [path]) => match sys.save_state() {
                Ok(data) => match crate::state::write(Path::new(path), &data) {
                    Ok(len) => {
                        Reply::ok(format!("cycle {}, {len} bytes -> {path}", sys.cycles))
                    }
                    Err(e) => Reply::err(format!("write {path}: {e}")),
                },
                Err(e) => Reply::err(format!("save state failed: {e}")),
            },
            ("loadstate", [path]) => match crate::state::read(Path::new(path)) {
                Ok(data) => match sys.load_state(&data) {
                    Ok(()) => {
                        // The state carries its own pad state; the held set
                        // is the operator's and outlives it.
                        sys.bus.sio2.buttons = self.held;
                        Reply::ok(format!("cycle {}, ee_pc={:#010x}", sys.cycles, sys.ee.pc))
                    }
                    Err(e) => Reply::err(e),
                },
                Err(e) => Reply::err(format!("read {path}: {e}")),
            },
            ("quit", _) => Reply { ok: true, payload: "bye".into(), quit: true },
            _ => Reply::err(format!("unknown command '{line}' (try 'help')")),
        }
    }
}

const HELP: &str = "\
state                  ee/iop pc, EE cycle, frame, held buttons, tray
run <n>[s|c]           advance n frames (s=seconds, c=EE cycles), inputs held
run to <cycle>         advance to an absolute EE cycle (5e9 shorthand ok)
press <btn[+btn]> <n>  hold buttons for n frames on top of the held set
input set <btn[+btn]>  hold buttons until changed (applied during run)
input clear            release all held buttons
peek [ee|iop] <hexaddr> <len>    hex dump memory (side-effect-free, MMIO --)
peekb [ee|iop] <hexaddr> <len>   memory as one base64 line (up to 32 MiB;
                       err if any byte is one peek would show as --)
peekm [ee|iop] <hexaddr>:<len> ...
                       one base64 line per range, 32 MiB in total
poke [ee|iop] <hexaddr> <hex>    write bytes to RAM/scratchpad
frame <path>           write the display as .png or .bmp
frameb [rgb24|png]     display over the socket: <w> <h> <fmt>, then one
                       base64 line (top-down RGB24, or a PNG file)
vram <path>            write the raw 4 MiB GS VRAM (no image: mixed formats)
disc open              open the tray, keeping the disc that was in it
disc close [path]      close it on a new image, else on the one lifted out
cheat list             pnach sections for the disc, and their enable state
cheat apply on|off     master switch
cheat on|off <n>       toggle section n (in memory; cheats.toml untouched)
cheat reload           re-read the pnach
tty                    kernel/IOP TTY accumulated since the last 'tty'
savestate <path>       snapshot the machine (zstd, as --save-state writes)
loadstate <path>       restore a snapshot
quit                   shut the emulator down
";

/// Ceiling on the blocking reply write in [`ControlServer::pump`], so a
/// client that never drains its socket gets dropped instead of wedging the
/// emulator forever.
const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// TCP transport: accepts one client at a time, reads newline-terminated
/// commands, writes dot-terminated replies.
pub struct ControlServer {
    listener: TcpListener,
    client: Option<TcpStream>,
    buf: Vec<u8>,
    pub controller: Controller,
}

impl ControlServer {
    pub fn bind(port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        listener.set_nonblocking(true)?;
        tracing::info!("control port listening on {}", listener.local_addr()?);
        Ok(Self { listener, client: None, buf: Vec::new(), controller: Controller::default() })
    }

    pub fn port(&self) -> u16 {
        self.listener.local_addr().map(|a| a.port()).unwrap_or(0)
    }

    /// Service the connection; executes at most one command per call.
    /// Returns false once a `quit` command has been executed.
    pub fn pump(&mut self, sys: &mut Ps2System, debugger_owns: bool) -> bool {
        if self.client.is_none() {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    stream.set_nonblocking(true).ok();
                    stream.set_nodelay(true).ok();
                    self.client = Some(stream);
                    self.buf.clear();
                }
                Err(_) => return true, // includes WouldBlock: nothing to do
            }
        }
        let Some(stream) = &mut self.client else {
            return true;
        };
        let mut chunk = [0u8; 1024];
        loop {
            match stream.read(&mut chunk) {
                Ok(0) => {
                    self.client = None;
                    return true;
                }
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(_) => {
                    self.client = None;
                    return true;
                }
            }
        }
        let Some(nl) = self.buf.iter().position(|&b| b == b'\n') else {
            return true;
        };
        let line: Vec<u8> = self.buf.drain(..nl + 1).collect();
        let line = String::from_utf8_lossy(&line).trim().to_string();
        if line.is_empty() {
            return true;
        }
        let reply = self.controller.execute(sys, &line, debugger_owns);
        let mut out = String::new();
        out.push_str(if reply.ok { "ok\n" } else { "err\n" });
        for l in reply.payload.lines() {
            // Dot-stuff payload lines so `.` can never terminate early.
            if l.starts_with('.') {
                out.push('.');
            }
            out.push_str(l);
            out.push('\n');
        }
        out.push_str(".\n");
        // Blocking write: under lockstep the client is always reading, so
        // this costs nothing, and it is the only way a multi-megabyte reply
        // survives instead of hitting WouldBlock mid-write and dropping the
        // client. The write timeout bounds the remaining risk: a client
        // that connects and never reads gets dropped instead of wedging
        // the emulator.
        if let Some(stream) = &mut self.client {
            stream.set_nonblocking(false).ok();
            stream.set_write_timeout(Some(WRITE_TIMEOUT)).ok();
            let sent = stream.write_all(out.as_bytes());
            stream.set_write_timeout(None).ok();
            stream.set_nonblocking(true).ok();
            if sent.is_err() {
                self.client = None;
            }
        }
        !reply.quit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zeroed 4 MiB "BIOS": both cores execute nops from the reset
    /// vector, which is all these tests need out of the machine.
    fn sys() -> Ps2System {
        Ps2System::new(vec![0u8; 4 * 1024 * 1024]).unwrap()
    }

    fn temp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("ps2e-ctl-{}-{tag}", std::process::id()))
    }

    /// A machine with the inline GS renderer (`frameb`'s tests need
    /// synchronous privileged writes; the threaded front only publishes at
    /// vblank).
    fn inline_sys() -> Ps2System {
        Ps2System::new_with(vec![0u8; 4 << 20], false).unwrap()
    }

    /// Program a 64x8 progressive PSMCT32 display and paint one red pixel at
    /// (5, 2). Inline `push` applies privileged writes immediately, so no
    /// `run` is needed before reading the frame back.
    fn display_64x8(sys: &mut Ps2System) {
        sys.bus.gs.priv_write(0x1200_0000, 1); // PMODE EN1
        sys.bus.gs.priv_write(0x1200_0020, 0); // SMODE2 progressive
        sys.bus.gs.priv_write(0x1200_0070, 1 << 9); // DISPFB1: FBP 0, FBW 1 = 64px, PSMCT32
        sys.bus.gs.priv_write(0x1200_0080, (63u64 << 32) | (7u64 << 44)); // DISPLAY1 64x8
        sys.bus.gs.inline_gs().unwrap().write_psmct32(0, 1, 5, 2, 0x0000_00FF);
    }

    /// Decode a PNG (file or in-memory) to its raw RGB8 pixel bytes.
    fn decode_png(r: impl std::io::Read + std::io::Seek) -> Vec<u8> {
        let mut reader = png::Decoder::new(std::io::BufReader::new(r)).read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
        let info = reader.next_frame(&mut buf).unwrap();
        buf[..info.buffer_size()].to_vec()
    }

    #[test]
    fn run_advances_by_frames() {
        let (mut sys, mut c) = (sys(), Controller::default());
        let cpf = sys.region().cycles_per_frame();
        let r = c.execute(&mut sys, "run 2", false);
        assert!(r.ok, "{}", r.payload);
        assert!(sys.cycles >= 2 * cpf);
        assert!(c.execute(&mut sys, "state", false).payload.contains("frames=2"));
    }

    #[test]
    fn run_units_and_absolute_target() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(c.execute(&mut sys, "run 5000000c", false).ok);
        assert_eq!(sys.cycles, 5_000_000);
        // `run to` lands on the cycle exactly, which is what makes a
        // number read off `state` usable in a batch recipe.
        assert!(c.execute(&mut sys, "run to 8e6", false).ok);
        assert_eq!(sys.cycles, 8_000_000);
        // Going backwards is a mistake, not a no-op.
        assert!(!c.execute(&mut sys, "run to 1e6", false).ok);
    }

    #[test]
    fn press_restores_the_held_set() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(c.execute(&mut sys, "input set up", false).ok);
        let r = c.execute(&mut sys, "press CROSS+start 1", false);
        assert!(r.ok, "{}", r.payload);
        assert_eq!(sys.bus.sio2.buttons, pad::UP);
        assert!(c.execute(&mut sys, "input clear", false).ok);
        assert_eq!(sys.bus.sio2.buttons, 0);
    }

    #[test]
    fn peek_poke_round_trip_on_both_cores() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(c.execute(&mut sys, "poke 00100000 deadbeef", false).ok);
        let r = c.execute(&mut sys, "peek 00100000 4", false);
        assert!(r.payload.contains("de ad be ef"), "{}", r.payload);
        // The kseg mirror is the same memory.
        let r = c.execute(&mut sys, "peek ee 80100000 4", false);
        assert!(r.payload.contains("de ad be ef"), "{}", r.payload);
        // The IOP has its own RAM, so the EE's write is not there.
        assert!(c.execute(&mut sys, "poke iop 00010000 55", false).ok);
        let r = c.execute(&mut sys, "peek iop 00010000 1", false);
        assert!(r.payload.contains("55"), "{}", r.payload);
        // A register with no side-effect-free read (the VIF0 FIFO) renders
        // as -- rather than being dispatched.
        let r = c.execute(&mut sys, "peek 10004000 4", false);
        assert!(r.payload.contains("--"), "{}", r.payload);
    }

    #[test]
    fn base64_round_trips() {
        let cases: [(&[u8], &str); 7] = [
            (b"", ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
            (b"fooba", "Zm9vYmE="),
            (b"foobar", "Zm9vYmFy"),
        ];
        for (data, want) in cases {
            assert_eq!(base64_encode(data), want, "encoding {data:?}");
            assert_eq!(base64_decode(want).unwrap(), data, "decoding {want:?}");
        }
    }

    #[test]
    fn peekb_matches_peek_and_covers_all_of_ram() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(c.execute(&mut sys, "poke 00100000 deadbeef", false).ok);
        let want = vec![0xde, 0xad, 0xbe, 0xef];

        let r = c.execute(&mut sys, "peekb 00100000 4", false);
        assert!(r.ok, "{}", r.payload);
        assert_eq!(base64_decode(&r.payload).unwrap(), want);

        // The kseg0 mirror is the same memory, on either core spelling.
        let r = c.execute(&mut sys, "peekb 80100000 4", false);
        assert_eq!(base64_decode(&r.payload).unwrap(), want);
        let r = c.execute(&mut sys, "peekb ee 00100000 4", false);
        assert_eq!(base64_decode(&r.payload).unwrap(), want);

        let r = c.execute(&mut sys, "peekb 00000000 33554432", false);
        assert!(r.ok, "{}", r.payload);
        assert_eq!(base64_decode(&r.payload).unwrap(), sys.bus.ram.to_vec());

        assert!(!c.execute(&mut sys, "peekb 00000000 33554433", false).ok);
        // The VIF0 FIFO has no side-effect-free read, same as `peek`'s `--`.
        assert!(!c.execute(&mut sys, "peekb 10004000 4", false).ok);
        assert!(!c.execute(&mut sys, "peekb 00100000 0", false).ok);
    }

    #[test]
    fn peekb_reads_scratchpad_and_the_iop() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(c.execute(&mut sys, "poke ee 70000000 aa", false).ok);
        let r = c.execute(&mut sys, "peekb 70000000 1", false);
        assert_eq!(base64_decode(&r.payload).unwrap(), vec![0xaa]);

        assert!(c.execute(&mut sys, "poke iop 00010000 55", false).ok);
        let r = c.execute(&mut sys, "peekb iop 00010000 1", false);
        assert_eq!(base64_decode(&r.payload).unwrap(), vec![0x55]);
        // The 2 MiB mirror is the same underlying IOP RAM.
        let r = c.execute(&mut sys, "peekb iop 00210000 1", false);
        assert_eq!(base64_decode(&r.payload).unwrap(), vec![0x55]);

        assert!(c.execute(&mut sys, "poke iop 001ffffe 1122", false).ok);
        assert!(c.execute(&mut sys, "poke iop 00000000 3344", false).ok);
        // Crosses the mirror end, so the read spans two window calls.
        let r = c.execute(&mut sys, "peekb iop 001ffffe 4", false);
        assert_eq!(base64_decode(&r.payload).unwrap(), vec![0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn peekb_stops_a_chunk_at_the_tlb_page_boundary() {
        let (mut sys, mut c) = (sys(), Controller::default());
        // Two adjacent virtual 4 KiB pages mapped to non-adjacent physical
        // pages: a chunk that ran past the page end instead of
        // re-translating would read the wrong bytes here.
        let phys0 = 0x0010_0000u32;
        let phys1 = 0x0030_0000u32;
        let lo0 = ((phys0 >> 12) << 6) | 2; // valid
        let lo1 = ((phys1 >> 12) << 6) | 2;
        sys.bus.ee_tlb_write(0, 0, 0x2000_0000, lo0, lo1);

        sys.bus.ram[(phys0 + 0xFFE) as usize] = 0x11;
        sys.bus.ram[(phys0 + 0xFFF) as usize] = 0x22;
        sys.bus.ram[phys1 as usize] = 0x33;
        sys.bus.ram[(phys1 + 1) as usize] = 0x44;

        let want: Vec<u8> =
            (0..4).map(|i| peek8(&mut sys, Core::Ee, 0x2000_0FFE + i).unwrap()).collect();
        assert_eq!(want, vec![0x11, 0x22, 0x33, 0x44]);

        let r = c.execute(&mut sys, "peekb 20000ffe 4", false);
        assert!(r.ok, "{}", r.payload);
        assert_eq!(base64_decode(&r.payload).unwrap(), want);
    }

    #[test]
    fn peekm_returns_one_line_per_range() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(c.execute(&mut sys, "poke 00100000 deadbeef", false).ok);
        assert!(c.execute(&mut sys, "poke ee 70000000 cc", false).ok);
        // Distinct known bytes at each address, so a wrong order or a bug
        // that reads every range from the first address cannot pass by
        // accident (lengths alone would not catch either).
        let r = c.execute(&mut sys, "peekm 00100000:4 bfc00000:2 70000000:1", false);
        assert!(r.ok, "{}", r.payload);
        let lines: Vec<&str> = r.payload.lines().collect();
        assert_eq!(lines.len(), 3, "{:?}", lines);
        assert_eq!(base64_decode(lines[0]).unwrap(), vec![0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(base64_decode(lines[1]).unwrap(), sys.bus.bios[..2].to_vec());
        assert_eq!(base64_decode(lines[2]).unwrap(), vec![0xcc]);

        assert!(c.execute(&mut sys, "poke iop 00010000 11223344", false).ok);
        let r = c.execute(&mut sys, "peekm iop 00010000:2 00010002:2", false);
        assert!(r.ok, "{}", r.payload);
        let lines: Vec<&str> = r.payload.lines().collect();
        assert_eq!(lines.len(), 2, "{:?}", lines);
        assert_eq!(base64_decode(lines[0]).unwrap(), vec![0x11, 0x22]);
        assert_eq!(base64_decode(lines[1]).unwrap(), vec![0x33, 0x44]);

        assert!(!c.execute(&mut sys, "peekm", false).ok);
        assert!(!c.execute(&mut sys, "peekm 10004000:4", false).ok);
        assert!(!c.execute(&mut sys, "peekm 00100000:0", false).ok);
        assert!(!c.execute(&mut sys, "peekm 00100000:4 bfc00000:0", false).ok);
    }

    #[test]
    fn frameb_matches_the_png_written_by_frame() {
        let mut sys = inline_sys();
        let mut c = Controller::default();
        display_64x8(&mut sys);

        let r = c.execute(&mut sys, "frameb", false);
        assert!(r.ok, "{}", r.payload);
        let mut lines = r.payload.lines();
        assert_eq!(lines.next(), Some("64 8 rgb24"));
        let rgb = base64_decode(lines.next().unwrap()).unwrap();
        assert_eq!(rgb.len(), 64 * 8 * 3);
        let px = |x: usize, y: usize| rgb[(y * 64 + x) * 3..(y * 64 + x) * 3 + 3].to_vec();
        assert_eq!(px(5, 2), vec![0xff, 0, 0]);
        assert_eq!(px(0, 0), vec![0, 0, 0]);

        let path = temp_path("frameb.png");
        let p = path.to_str().unwrap();
        assert!(c.execute(&mut sys, &format!("frame {p}"), false).ok);
        let file = std::fs::File::open(&path).unwrap();
        assert_eq!(decode_png(file), rgb);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn frameb_png_decodes_to_the_same_pixels() {
        let mut sys = inline_sys();
        let mut c = Controller::default();
        display_64x8(&mut sys);

        let rgb = base64_decode(c.execute(&mut sys, "frameb", false).payload.lines().nth(1).unwrap())
            .unwrap();

        let r = c.execute(&mut sys, "frameb png", false);
        assert!(r.ok, "{}", r.payload);
        let mut lines = r.payload.lines();
        assert_eq!(lines.next(), Some("64 8 png"));
        let png_bytes = base64_decode(lines.next().unwrap()).unwrap();
        assert_eq!(decode_png(std::io::Cursor::new(png_bytes)), rgb);
    }

    #[test]
    fn frameb_rejects_unknown_formats_and_a_blank_display() {
        let mut sys = inline_sys();
        let mut c = Controller::default();
        // DISPLAY unprogrammed gives height 0, same failure `frame` reports.
        assert!(!c.execute(&mut sys, "frameb", false).ok);

        display_64x8(&mut sys);
        assert!(!c.execute(&mut sys, "frameb bmp", false).ok);
        assert!(c.execute(&mut sys, "frameb", false).ok);
    }

    #[test]
    fn savestate_loadstate_round_trip() {
        let (mut sys, mut c) = (sys(), Controller::default());
        let path = temp_path("state.sst");
        let p = path.to_str().unwrap();

        assert!(c.execute(&mut sys, "run 1", false).ok);
        let at = sys.cycles;
        let r = c.execute(&mut sys, &format!("savestate {p}"), false);
        assert!(r.ok, "{}", r.payload);

        assert!(c.execute(&mut sys, "run 1", false).ok);
        assert_ne!(sys.cycles, at);

        let r = c.execute(&mut sys, &format!("loadstate {p}"), false);
        assert!(r.ok, "{}", r.payload);
        assert_eq!(sys.cycles, at);

        // Saving is observation; loading rewinds execution, so only the
        // second is refused while a debugger owns the machine.
        assert!(c.execute(&mut sys, &format!("savestate {p}"), true).ok);
        assert!(!c.execute(&mut sys, &format!("loadstate {p}"), true).ok);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn cheat_toggles_reach_the_machine() {
        let (mut sys, mut c) = (sys(), Controller::default());
        let groups = ps2_core::cheats::parse(
            "[Health]\npatch=1,EE,00100000,word,00000063\n\
             [Lives]\npatch=1,EE,00100004,word,00000009\n",
        );
        assert_eq!(groups.len(), 2, "pnach did not parse into two sections");
        c.set_cheats(temp_path("none.pnach"), groups);
        c.push_cheats(&mut sys);
        assert!(c.execute(&mut sys, "cheat apply on", false).ok);

        let r = c.execute(&mut sys, "cheat list", false);
        assert!(r.payload.contains("apply: on"), "{}", r.payload);
        assert!(r.payload.contains("Health"), "{}", r.payload);

        // A cheat lands at the vblank inside the first frame.
        assert!(c.execute(&mut sys, "run 1", false).ok);
        let r = c.execute(&mut sys, "peek 00100000 4", false);
        assert!(r.payload.contains("63 00 00 00"), "{}", r.payload);

        // Off means the value the machine holds is left alone.
        assert!(c.execute(&mut sys, "cheat off 0", false).ok);
        assert!(c.execute(&mut sys, "poke 00100000 00", false).ok);
        assert!(c.execute(&mut sys, "run 1", false).ok);
        let r = c.execute(&mut sys, "peek 00100000 1", false);
        assert!(r.payload.contains("00"), "{}", r.payload);

        // The master switch stops everything without touching the list.
        assert!(c.execute(&mut sys, "cheat apply off", false).ok);
        let r = c.execute(&mut sys, "cheat list", false);
        assert!(r.payload.contains("apply: off"), "{}", r.payload);
    }

    #[test]
    fn debugger_owns_execution() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(!c.execute(&mut sys, "run 1", true).ok);
        assert!(!c.execute(&mut sys, "press start 1", true).ok);
        // Observation stays available while it is attached.
        assert!(c.execute(&mut sys, "peek 00100000 4", true).ok);
        assert!(c.execute(&mut sys, "state", true).ok);
    }

    #[test]
    fn tty_returns_only_new_output() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert_eq!(c.execute(&mut sys, "tty", false).payload, "");
        c.tty.push_str("hello\n");
        assert_eq!(c.execute(&mut sys, "tty", false).payload, "hello\n");
        assert_eq!(c.execute(&mut sys, "tty", false).payload, "");
    }

    #[test]
    fn unknown_and_malformed_commands_error() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(!c.execute(&mut sys, "dance", false).ok);
        assert!(!c.execute(&mut sys, "run zero", false).ok);
        assert!(!c.execute(&mut sys, "run -5", false).ok);
        assert!(!c.execute(&mut sys, "press nope 1", false).ok);
        assert!(!c.execute(&mut sys, "peek xyz 4", false).ok);
        assert!(!c.execute(&mut sys, "peek 00100000 999999", false).ok);
        assert!(!c.execute(&mut sys, "poke 00100000 abc", false).ok);
        assert!(!c.execute(&mut sys, "cheat on 0", false).ok);
        assert!(!c.execute(&mut sys, "cheat reload", false).ok);
    }

    #[test]
    fn quit_flag_propagates() {
        let (mut sys, mut c) = (sys(), Controller::default());
        assert!(c.execute(&mut sys, "quit", false).quit);
    }

    /// Regression for the nonblocking `write_all` this replaced: it dropped
    /// the client mid-reply once a large payload filled the socket send
    /// buffer, because the client sees the connection close before the
    /// reply completes.
    #[test]
    fn a_large_reply_reaches_the_client() {
        use std::io::{BufRead, BufReader};
        use std::thread;
        use std::time::Instant;

        let mut server = ControlServer::bind(0).unwrap();
        let port = server.port();
        let text = "x".repeat(1 << 20) + "\n";
        server.controller.tty = text.clone();
        let mut sys = sys();

        let handle = thread::spawn(move || {
            let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
            stream.write_all(b"tty\n").unwrap();
            // Give the server a chance to start writing before anyone
            // drains the socket, so the send buffer fills mid-reply.
            thread::sleep(Duration::from_millis(200));
            stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let mut reader = BufReader::new(stream);
            let mut payload_len = 0usize;
            let mut line = String::new();
            loop {
                line.clear();
                let n = reader.read_line(&mut line).unwrap();
                if n == 0 || line.trim_end() == "." {
                    break;
                }
                if line.trim_end() != "ok" {
                    payload_len += line.len();
                }
            }
            payload_len
        });

        let deadline = Instant::now() + Duration::from_secs(10);
        while !handle.is_finished() {
            assert!(Instant::now() < deadline, "client did not finish reading in time");
            server.pump(&mut sys, false);
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(handle.join().unwrap(), text.len());
    }
}
