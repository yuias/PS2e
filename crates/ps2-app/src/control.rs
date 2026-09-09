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
poke [ee|iop] <hexaddr> <hex>    write bytes to RAM/scratchpad
frame <path>           write the display as .png or .bmp
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
        if let Some(stream) = &mut self.client
            && stream.write_all(out.as_bytes()).is_err()
        {
            self.client = None;
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
}
