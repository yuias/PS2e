//! LLDB-first gdb-remote debug stub for the PS2e core.
//!
//! [`DebugServer`] exposes two independent gdb-remote targets — the EE and
//! the IOP — each on its own TCP port, both driving the one [`Ps2System`]
//! owned by the frontend. LLDB is the primary client (`qHostInfo`,
//! `qProcessInfo`, `qRegisterInfo` and `target.xml` are all implemented);
//! plain GDB works via the same `target.xml`.
//!
//! Integration model: the frontend keeps calling [`DebugServer::pump`] with a
//! cycle budget. With no client attached, `pump` returns immediately and the
//! frontend runs the emulator itself. Once a client attaches, execution is
//! *owned by the debugger*: the frontend must stop stepping the system and
//! `pump` executes instructions only while every attached client says
//! "continue".
//!
//! The cores are lock-stepped at the 8:1 clock ratio, so halting either
//! target halts the whole machine. Single-stepping the IOP steps the EE
//! through the interleave (up to 8 EE instructions), like stepping one
//! clock of the real machine. Write watchpoints (`Z2`) are polled after
//! every instruction against a byte snapshot — slow but exact, and they
//! catch DMA writes too.

mod packet;
mod registers;

use packet::{Item, Receiver};
use ps2_core::{EE_PER_IOP, Ps2System};
use registers::Target;
use std::collections::HashSet;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use tracing::{debug, info, warn};

/// Canonical breakpoint key for an EE virtual address: KSEG0/KSEG1 fold onto
/// the physical address, which the identity-mapped kuseg RAM window equals,
/// so a breakpoint set through one alias hits the others. TLB-mapped ranges
/// (scratchpad window, kseg2) stay exact-match.
fn ee_canonical(addr: u32) -> u32 {
    match addr {
        0x8000_0000..=0xBFFF_FFFF => addr & 0x1FFF_FFFF,
        _ => addr,
    }
}

/// Canonical key for an IOP address: classic MIPS segment fold.
fn iop_canonical(addr: u32) -> u32 {
    match addr >> 29 {
        4 => addr & 0x7FFF_FFFF, // kseg0
        5 => addr & 0x1FFF_FFFF, // kseg1
        _ => addr,
    }
}

fn canonical(t: Target, addr: u32) -> u32 {
    match t {
        Target::Ee => ee_canonical(addr),
        Target::Iop => iop_canonical(addr),
    }
}

/// Side-effect-free byte read on the target's own bus view.
fn peek(sys: &mut Ps2System, t: Target, addr: u32) -> Option<u8> {
    match t {
        Target::Ee => sys.bus.peek8(addr),
        Target::Iop => sys.bus.iop_peek8(addr),
    }
}

/// Side-effect-free byte write on the target's own bus view.
fn poke(sys: &mut Ps2System, t: Target, addr: u32, v: u8) -> bool {
    match t {
        Target::Ee => sys.bus.poke8(addr, v),
        Target::Iop => sys.bus.iop_poke8(addr, v),
    }
}

/// Debug stub owning up to two per-core gdb-remote servers.
pub struct DebugServer {
    ee: Option<Stub>,
    iop: Option<Stub>,
}

impl DebugServer {
    /// Bind the requested targets on `127.0.0.1`. Port 0 picks a free port
    /// (see [`DebugServer::ee_port`] / [`DebugServer::iop_port`]).
    pub fn bind(ee_port: Option<u16>, iop_port: Option<u16>) -> std::io::Result<Self> {
        Ok(Self {
            ee: ee_port.map(|p| Stub::bind(Target::Ee, p)).transpose()?,
            iop: iop_port.map(|p| Stub::bind(Target::Iop, p)).transpose()?,
        })
    }

    pub fn ee_port(&self) -> Option<u16> {
        self.ee.as_ref().map(Stub::port)
    }

    pub fn iop_port(&self) -> Option<u16> {
        self.iop.as_ref().map(Stub::port)
    }

    /// A client is attached to either target (the debugger owns execution).
    pub fn attached(&self) -> bool {
        self.stubs().any(|s| s.client.is_some())
    }

    /// Execution is suspended at a debug stop on either target.
    pub fn halted(&self) -> bool {
        self.stubs().any(|s| s.client.is_some() && s.halted)
    }

    fn stubs(&self) -> impl Iterator<Item = &Stub> {
        self.ee.iter().chain(self.iop.iter())
    }

    /// Service both connections and, while every attached client has us
    /// running, execute up to `budget_cycles` of emulation with breakpoint
    /// and watchpoint checks.
    pub fn pump(&mut self, sys: &mut Ps2System, budget_cycles: u64) {
        if let Some(stub) = &mut self.ee {
            stub.accept_new_client();
            stub.service_client(sys);
        }
        if let Some(stub) = &mut self.iop {
            stub.accept_new_client();
            stub.service_client(sys);
        }
        if self.attached() && !self.halted() {
            self.run_slice(sys, budget_cycles);
        }
    }

    /// Execute instructions until the budget runs out or a stop fires.
    fn run_slice(&mut self, sys: &mut Ps2System, budget_cycles: u64) {
        let end = sys.cycles + budget_cycles;
        while sys.cycles < end {
            if let Some(stub) = &mut self.ee
                && stub.client.is_some()
                && !std::mem::take(&mut stub.resume_skip)
                && stub.breakpoints.contains(&ee_canonical(sys.ee.pc))
            {
                stub.stop(b"T05thread:01;");
                return;
            }
            // The IOP only executes on every 8th EE cycle; check its
            // breakpoints (and consume its resume_skip) just before those.
            if sys.cycles.is_multiple_of(EE_PER_IOP)
                && let Some(stub) = &mut self.iop
                && stub.client.is_some()
                && !std::mem::take(&mut stub.resume_skip)
                && stub.breakpoints.contains(&iop_canonical(sys.iop.pc))
            {
                stub.stop(b"T05thread:01;");
                return;
            }
            sys.step();
            if let Some(stub) = &mut self.ee
                && stub.check_watchpoints(sys)
            {
                return;
            }
            if let Some(stub) = &mut self.iop
                && stub.check_watchpoints(sys)
            {
                return;
            }
        }
    }
}

/// One gdb-remote server bound to one core.
struct Stub {
    target: Target,
    listener: TcpListener,
    client: Option<Client>,
    /// Breakpoint addresses, stored segment-folded so a breakpoint set on
    /// a KSEG0 address also hits its KSEG1/KUSEG aliases.
    breakpoints: HashSet<u32>,
    /// Write watchpoints, polled against a snapshot after every step.
    watchpoints: Vec<Watchpoint>,
    /// Execution is suspended, waiting for debugger commands.
    halted: bool,
    /// Skip the breakpoint check for the first instruction after a resume so
    /// continuing from a breakpointed pc makes progress.
    resume_skip: bool,
    no_ack: bool,
}

struct Watchpoint {
    addr: u32,
    len: u32,
    old: Vec<u8>,
}

struct Client {
    stream: TcpStream,
    rx: Receiver,
}

impl Stub {
    fn bind(target: Target, port: u16) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        listener.set_nonblocking(true)?;
        info!(
            target: "ps2_debug",
            "{} gdb-remote stub listening on {}",
            target.hostname(),
            listener.local_addr()?
        );
        Ok(Self {
            target,
            listener,
            client: None,
            breakpoints: HashSet::new(),
            watchpoints: Vec::new(),
            halted: false,
            resume_skip: false,
            no_ack: false,
        })
    }

    fn port(&self) -> u16 {
        self.listener.local_addr().map(|a| a.port()).unwrap_or(0)
    }

    fn accept_new_client(&mut self) {
        if self.client.is_some() {
            return;
        }
        match self.listener.accept() {
            Ok((stream, addr)) => {
                info!(target: "ps2_debug", "{} debugger attached from {addr}", self.target.hostname());
                stream.set_nonblocking(true).ok();
                stream.set_nodelay(true).ok();
                self.client = Some(Client {
                    stream,
                    rx: Receiver::default(),
                });
                // Attaching halts the target, like gdbserver attaching to a
                // live process. The client's `?` query finds us stopped.
                self.halted = true;
                self.no_ack = false;
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {}
            Err(e) => warn!(target: "ps2_debug", "accept failed: {e}"),
        }
    }

    fn detach(&mut self, reason: &str) {
        info!(target: "ps2_debug", "{} debugger detached ({reason})", self.target.hostname());
        self.client = None;
        self.breakpoints.clear();
        self.watchpoints.clear();
        self.halted = false;
        self.no_ack = false;
    }

    /// Read pending bytes and handle every complete protocol item.
    fn service_client(&mut self, sys: &mut Ps2System) {
        let Some(client) = &mut self.client else {
            return;
        };
        let mut buf = [0u8; 4096];
        loop {
            match client.stream.read(&mut buf) {
                Ok(0) => {
                    self.detach("connection closed");
                    return;
                }
                Ok(n) => client.rx.push_bytes(&buf[..n]),
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) => {
                    let msg = format!("read failed: {e}");
                    self.detach(&msg);
                    return;
                }
            }
        }
        while let Some(item) = self.client.as_mut().and_then(|c| c.rx.next_item()) {
            match item {
                Item::Packet(payload) => {
                    self.send_raw(b"+");
                    self.handle_packet(sys, &payload);
                }
                Item::Corrupt => self.send_raw(b"-"),
                Item::Interrupt => {
                    if !self.halted {
                        self.halted = true;
                        // SIGINT
                        self.send_reply(b"T02thread:01;");
                    }
                }
                Item::Ack | Item::Nak => {}
            }
            if self.client.is_none() {
                break; // detached while handling (D / k)
            }
        }
    }

    /// Execute exactly one instruction of this stub's core. For the IOP that
    /// means running the interleave until its next 1-of-8 slot.
    fn step_target(&mut self, sys: &mut Ps2System) {
        match self.target {
            Target::Ee => sys.step(),
            Target::Iop => loop {
                let iop_slot = sys.cycles.is_multiple_of(EE_PER_IOP);
                sys.step();
                if iop_slot {
                    break;
                }
            },
        }
    }

    /// Enter the halted state and send the (deferred) stop reply.
    fn stop(&mut self, reply: &[u8]) {
        self.halted = true;
        self.send_reply(reply);
    }

    /// Re-snapshot all watchpoints, e.g. when resuming: memory edited while
    /// halted must not read as a program write.
    fn refresh_watchpoints(&mut self, sys: &mut Ps2System) {
        let t = self.target;
        for wp in &mut self.watchpoints {
            for i in 0..wp.len {
                wp.old[i as usize] = peek(sys, t, wp.addr.wrapping_add(i)).unwrap_or(0);
            }
        }
    }

    /// Compare watched bytes against their snapshot; on a change, halt and
    /// send the watch stop reply. Returns true when execution must stop.
    fn check_watchpoints(&mut self, sys: &mut Ps2System) -> bool {
        if self.client.is_none() || self.watchpoints.is_empty() {
            return false;
        }
        let t = self.target;
        let hit = self.watchpoints.iter().position(|wp| {
            (0..wp.len)
                .any(|i| peek(sys, t, wp.addr.wrapping_add(i)).unwrap_or(0) != wp.old[i as usize])
        });
        let Some(idx) = hit else {
            return false;
        };
        let addr = self.watchpoints[idx].addr;
        self.refresh_watchpoints(sys);
        self.stop(format!("T05watch:{addr:x};thread:01;").as_bytes());
        true
    }

    fn send_raw(&mut self, data: &[u8]) {
        if let Some(client) = &mut self.client
            && let Err(e) = client.stream.write_all(data)
        {
            let msg = format!("write failed: {e}");
            self.detach(&msg);
        }
    }

    fn send_reply(&mut self, payload: &[u8]) {
        debug!(target: "ps2_debug", "reply: {}", String::from_utf8_lossy(payload));
        let frame = packet::frame(payload);
        self.send_raw(&frame);
    }

    fn handle_packet(&mut self, sys: &mut Ps2System, payload: &[u8]) {
        // `X` carries escaped binary data; handle it before any utf8 view.
        if payload.first() == Some(&b'X') {
            let reply = handle_binary_write(sys, self.target, &payload[1..]);
            self.send_reply(&reply);
            return;
        }
        let text = String::from_utf8_lossy(payload).into_owned();
        debug!(target: "ps2_debug", "packet: {text}");
        // Commands that reply asynchronously (on the next stop) or change
        // connection state are handled here; everything else returns a reply.
        match text.as_bytes() {
            [b'c', ..] | [b'C', ..] => {
                self.refresh_watchpoints(sys);
                self.halted = false;
                self.resume_skip = true;
                return; // reply comes when we stop
            }
            [b's', ..] | [b'S', ..] => {
                self.refresh_watchpoints(sys);
                self.step_target(sys);
                self.send_reply(b"T05thread:01;");
                return;
            }
            b"D" => {
                self.send_reply(b"OK");
                self.detach("D packet");
                return;
            }
            b"k" => {
                // No process to kill: drop the connection, emulation resumes.
                self.detach("k packet");
                return;
            }
            _ => {}
        }
        if let Some(rest) = text.strip_prefix("vCont;") {
            match rest.as_bytes().first() {
                Some(b'c') | Some(b'C') => {
                    self.refresh_watchpoints(sys);
                    self.halted = false;
                    self.resume_skip = true;
                }
                Some(b's') | Some(b'S') => {
                    self.refresh_watchpoints(sys);
                    self.step_target(sys);
                    self.send_reply(b"T05thread:01;");
                }
                _ => self.send_reply(b""),
            }
            return;
        }
        let reply = self.reply_for(sys, &text);
        self.send_reply(&reply);
        if text == "QStartNoAckMode" {
            self.no_ack = true;
        }
    }

    /// Synchronous request/reply commands.
    fn reply_for(&mut self, sys: &mut Ps2System, text: &str) -> Vec<u8> {
        if text == "?" {
            return b"T05thread:01;".to_vec();
        }
        if text.starts_with("qSupported") {
            return b"PacketSize=4096;qXfer:features:read+;QStartNoAckMode+;\
                     swbreak+;vContSupported+"
                .to_vec();
        }
        match text {
            "QStartNoAckMode" => b"OK".to_vec(),
            "qHostInfo" => format!(
                "triple:{};ptrsize:4;endian:little;hostname:{};",
                packet::to_hex(self.target.triple().as_bytes()),
                packet::to_hex(self.target.hostname().as_bytes())
            )
            .into_bytes(),
            "qProcessInfo" => format!(
                "pid:1;parent-pid:1;real-uid:0;real-gid:0;effective-uid:0;\
                 effective-gid:0;triple:{};ostype:unknown;endian:little;ptrsize:4;",
                packet::to_hex(self.target.triple().as_bytes())
            )
            .into_bytes(),
            "qC" => b"QC1".to_vec(),
            "qAttached" => b"1".to_vec(),
            "qfThreadInfo" => b"m1".to_vec(),
            "qsThreadInfo" => b"l".to_vec(),
            "vCont?" => b"vCont;c;C;s;S".to_vec(),
            "g" => {
                let t = self.target;
                let mut hex = String::with_capacity(registers::g_len(t) * 2);
                for i in 0..registers::NUM_REGS {
                    let v = registers::read(sys, t, i);
                    hex.push_str(&packet::to_hex(&v.to_le_bytes()[..registers::size(t, i)]));
                }
                hex.into_bytes()
            }
            _ => self.reply_for_prefixed(sys, text),
        }
    }

    fn reply_for_prefixed(&mut self, sys: &mut Ps2System, text: &str) -> Vec<u8> {
        let t = self.target;
        if let Some(rest) = text.strip_prefix("qRegisterInfo") {
            return match usize::from_str_radix(rest, 16)
                .ok()
                .and_then(|i| registers::register_info(t, i))
            {
                Some(info) => info.into_bytes(),
                None => b"E45".to_vec(),
            };
        }
        if let Some(rest) = text.strip_prefix("qXfer:features:read:target.xml:") {
            return xfer_chunk(&registers::target_xml(t), rest);
        }
        if let Some(rest) = text.strip_prefix("G") {
            return match packet::from_hex(rest) {
                Some(bytes) if bytes.len() == registers::g_len(t) => {
                    let mut off = 0;
                    for i in 0..registers::NUM_REGS {
                        let sz = registers::size(t, i);
                        let mut le = [0u8; 8];
                        le[..sz].copy_from_slice(&bytes[off..off + sz]);
                        registers::write(sys, t, i, u64::from_le_bytes(le));
                        off += sz;
                    }
                    b"OK".to_vec()
                }
                _ => b"E01".to_vec(),
            };
        }
        if let Some(rest) = text.strip_prefix("p") {
            return match usize::from_str_radix(rest, 16) {
                Ok(i) if i < registers::NUM_REGS => {
                    let v = registers::read(sys, t, i);
                    packet::to_hex(&v.to_le_bytes()[..registers::size(t, i)]).into_bytes()
                }
                _ => b"E45".to_vec(),
            };
        }
        if let Some(rest) = text.strip_prefix("P") {
            let parsed = rest.split_once('=').and_then(|(idx, val)| {
                let i = usize::from_str_radix(idx, 16).ok()?;
                let bytes = packet::from_hex(val)?;
                Some((i, bytes))
            });
            return match parsed {
                Some((i, bytes))
                    if i < registers::NUM_REGS && bytes.len() == registers::size(t, i) =>
                {
                    let mut le = [0u8; 8];
                    le[..bytes.len()].copy_from_slice(&bytes);
                    registers::write(sys, t, i, u64::from_le_bytes(le));
                    b"OK".to_vec()
                }
                _ => b"E01".to_vec(),
            };
        }
        if let Some(rest) = text.strip_prefix("m") {
            return match parse_addr_len(rest) {
                Some((addr, len)) => read_memory(sys, t, addr, len),
                None => b"E01".to_vec(),
            };
        }
        if let Some(rest) = text.strip_prefix("M") {
            let parsed = rest
                .split_once(':')
                .and_then(|(range, hex)| Some((parse_addr_len(range)?, packet::from_hex(hex)?)));
            return match parsed {
                Some(((addr, len), bytes)) if bytes.len() as u32 == len => {
                    write_memory(sys, t, addr, &bytes)
                }
                _ => b"E01".to_vec(),
            };
        }
        if let Some(rest) = text.strip_prefix("Z") {
            return self.handle_breakpoint(sys, rest, true);
        }
        if let Some(rest) = text.strip_prefix("z") {
            return self.handle_breakpoint(sys, rest, false);
        }
        if text.starts_with("H") || text == "T1" {
            return b"OK".to_vec();
        }
        // Unknown packet: empty reply means "unsupported".
        b"".to_vec()
    }

    /// `Z<type>,<addr>,<kind>` / `z<type>,<addr>,<kind>`. Software and
    /// hardware breakpoints share one implementation (the interpreter checks
    /// pc every instruction). Type 2 (write watchpoint, `<kind>` = length)
    /// is polled; read/access watchpoints are unsupported.
    fn handle_breakpoint(&mut self, sys: &mut Ps2System, rest: &str, insert: bool) -> Vec<u8> {
        let mut parts = rest.split(',');
        let (ty, addr) = match (
            parts.next(),
            parts.next().and_then(|a| u32::from_str_radix(a, 16).ok()),
        ) {
            (Some(ty), Some(addr)) => (ty, addr),
            _ => return b"E01".to_vec(),
        };
        let kind = parts
            .next()
            .and_then(|k| u32::from_str_radix(k, 16).ok())
            .unwrap_or(4);
        match ty {
            "0" | "1" => {
                let key = canonical(self.target, addr);
                if insert {
                    self.breakpoints.insert(key);
                } else {
                    self.breakpoints.remove(&key);
                }
                b"OK".to_vec()
            }
            "2" => {
                let len = kind.clamp(1, 4096);
                if insert {
                    let t = self.target;
                    let old = (0..len)
                        .map(|i| peek(sys, t, addr.wrapping_add(i)).unwrap_or(0))
                        .collect();
                    self.watchpoints.push(Watchpoint { addr, len, old });
                } else {
                    self.watchpoints
                        .retain(|w| !(w.addr == addr && w.len == len));
                }
                b"OK".to_vec()
            }
            _ => b"".to_vec(), // read/access watchpoints unsupported
        }
    }
}

/// Serve one `qXfer` window (`<offset>,<length>` in hex) of a document:
/// `l` prefixes the final chunk, `m` a chunk with more data following.
fn xfer_chunk(doc: &str, range: &str) -> Vec<u8> {
    let Some((off, len)) = parse_addr_len(range) else {
        return b"E01".to_vec();
    };
    let bytes = doc.as_bytes();
    let start = (off as usize).min(bytes.len());
    let end = (start + len as usize).min(bytes.len());
    let mut out = Vec::with_capacity(end - start + 1);
    out.push(if end == bytes.len() { b'l' } else { b'm' });
    out.extend_from_slice(&bytes[start..end]);
    out
}

/// `X<addr>,<len>:<escaped binary>` — the write path LLDB prefers.
fn handle_binary_write(sys: &mut Ps2System, t: Target, rest: &[u8]) -> Vec<u8> {
    let Some(colon) = rest.iter().position(|&b| b == b':') else {
        return b"E01".to_vec();
    };
    let header = String::from_utf8_lossy(&rest[..colon]);
    let Some((addr, len)) = parse_addr_len(&header) else {
        return b"E01".to_vec();
    };
    let bytes = packet::unescape_binary(&rest[colon + 1..]);
    if bytes.len() as u32 != len {
        return b"E01".to_vec();
    }
    write_memory(sys, t, addr, &bytes)
}

/// Parse `<addr>,<len>` (both hex).
fn parse_addr_len(s: &str) -> Option<(u32, u32)> {
    let (a, l) = s.split_once(',')?;
    Some((
        u32::from_str_radix(a, 16).ok()?,
        u32::from_str_radix(l, 16).ok()?,
    ))
}

fn read_memory(sys: &mut Ps2System, t: Target, addr: u32, len: u32) -> Vec<u8> {
    let mut hex = String::with_capacity(len as usize * 2);
    for i in 0..len {
        match peek(sys, t, addr.wrapping_add(i)) {
            Some(b) => hex.push_str(&format!("{b:02x}")),
            // Partial reads are legal; an unmapped first byte is an error.
            None if i == 0 => return b"E01".to_vec(),
            None => break,
        }
    }
    hex.into_bytes()
}

fn write_memory(sys: &mut Ps2System, t: Target, addr: u32, bytes: &[u8]) -> Vec<u8> {
    for (i, b) in bytes.iter().enumerate() {
        if !poke(sys, t, addr.wrapping_add(i as u32), *b) {
            return b"E01".to_vec();
        }
    }
    b"OK".to_vec()
}
