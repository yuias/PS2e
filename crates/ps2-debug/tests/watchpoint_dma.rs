//! Does an IOP write watchpoint see a word that a DMA engine wrote?
//!
//! Watchpoints are polled against a byte snapshot after every
//! `Ps2System::step`, so a write by anything other than the core — a DMA
//! engine draining into RAM inside the same step — must trip them too, and
//! the stop must say which engine did it. The program below arms the SIO2
//! output channel (ch12) over a pre-filled word and expects the stop.

use ps2_core::Ps2System;
use ps2_debug::DebugServer;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;

const BIOS_SIZE: usize = 4 * 1024 * 1024;
const RESET: u32 = 0xBFC0_0000;
const BUDGET: u64 = 10_000;
const PUMP_LIMIT: usize = 200;

/// The watched word, inside the 32 bytes the DMA below writes.
const WATCHED: u32 = 0x0006_6010;

/// IOP program at the reset vector: `madr = 0x66000`, `bcr` = one block of
/// eight words, then kick ch12 and spin.
const PROGRAM: &[(usize, u32)] = &[
    (0x00, 0x3C08_1F80), // lui $t0, 0x1f80
    (0x04, 0x3C09_0006), // lui $t1, 0x0006
    (0x08, 0x3529_6000), // ori $t1, $t1, 0x6000
    (0x0c, 0xAD09_1550), // sw $t1, 0x1550($t0)   -- MADR
    (0x10, 0x3C0A_0001), // lui $t2, 0x0001
    (0x14, 0x354A_0008), // ori $t2, $t2, 8
    (0x18, 0xAD0A_1554), // sw $t2, 0x1554($t0)   -- BCR: 1 x 8 words
    (0x1c, 0x3C0B_0100), // lui $t3, 0x0100       -- CHCR busy
    (0x20, 0xAD0B_1558), // sw $t3, 0x1558($t0)   -- the DMA runs here
    (0x24, 0x1000_FFFF), // beq $0, $0, -1
    (0x28, 0x3412_000B), // ori $s2, $0, 11       -- non-nop delay slot
];

struct Session {
    sys: Ps2System,
    dbg: DebugServer,
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Session {
    /// Plant [`PROGRAM`] and attach a client to the IOP stub.
    fn new() -> Self {
        let mut bios = vec![0u8; BIOS_SIZE];
        for &(off, w) in PROGRAM {
            bios[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        let sys = Ps2System::new(bios).unwrap();
        let dbg = DebugServer::bind(None, Some(0)).unwrap();
        let port = dbg.iop_port().unwrap();
        let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.set_nonblocking(true).unwrap();
        stream.set_nodelay(true).unwrap();
        let mut s = Session {
            sys,
            dbg,
            stream,
            buf: Vec::new(),
        };
        for _ in 0..PUMP_LIMIT {
            s.dbg.pump(&mut s.sys, 0);
            if s.dbg.attached() {
                break;
            }
        }
        assert!(s.dbg.attached(), "stub never accepted the connection");
        assert_eq!(s.sys.iop.pc, RESET);
        s
    }

    fn send(&mut self, payload: &str) {
        let sum = payload.bytes().fold(0u8, |a, b| a.wrapping_add(b));
        let pkt = format!("${payload}#{sum:02x}");
        self.stream.write_all(pkt.as_bytes()).unwrap();
    }

    fn fill(&mut self) {
        let mut tmp = [0u8; 4096];
        loop {
            match self.stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => self.buf.extend_from_slice(&tmp[..n]),
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) => panic!("client read failed: {e}"),
            }
        }
    }

    fn take_packet(&mut self) -> Option<String> {
        let start = self.buf.iter().position(|&b| b == b'$')?;
        let hash = self.buf[start..].iter().position(|&b| b == b'#')? + start;
        if self.buf.len() < hash + 3 {
            return None;
        }
        let payload = String::from_utf8_lossy(&self.buf[start + 1..hash]).into_owned();
        self.buf.drain(..hash + 3);
        Some(payload)
    }

    fn recv(&mut self) -> String {
        for _ in 0..PUMP_LIMIT {
            self.fill();
            if let Some(p) = self.take_packet() {
                return p;
            }
            self.dbg.pump(&mut self.sys, BUDGET);
        }
        panic!("no reply from the stub within the pump limit");
    }

    fn roundtrip(&mut self, payload: &str) -> String {
        self.send(payload);
        self.recv()
    }
}

/// Decode a gdb `O` (console output) packet.
fn console_text(packet: &str) -> String {
    let hex = packet.strip_prefix('O').expect("not an O packet");
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap() as char)
        .collect()
}

#[test]
fn a_dma_write_trips_a_watchpoint_and_names_the_channel() {
    let mut s = Session::new();
    assert_eq!(s.roundtrip(&format!("M{WATCHED:x},4:ffffffff")), "OK");
    assert_eq!(s.roundtrip(&format!("Z2,{WATCHED:x},4")), "OK");
    s.send("c");
    // The console line naming the writer precedes the stop reply.
    let first = s.recv();
    assert!(first.starts_with('O'), "expected console output first, got {first}");
    let text = console_text(&first);
    assert!(text.contains("SIO2 out"), "writer not named: {text}");
    assert!(text.contains("0x66000"), "descriptor start missing: {text}");
    assert!(text.contains("0x20"), "descriptor size missing: {text}");
    let stop = s.recv();
    assert_eq!(stop, format!("T05watch:{WATCHED:x};thread:01;"));
    // The DMA ran inside the kick's step, so the stop lands just after it.
    assert_eq!(s.sys.iop.pc, RESET + 0x24);
}

#[test]
fn removing_the_watchpoint_forgets_its_dma_hook() {
    let mut s = Session::new();
    assert_eq!(s.roundtrip(&format!("Z2,{WATCHED:x},4")), "OK");
    assert_eq!(s.roundtrip(&format!("z2,{WATCHED:x},4")), "OK");
    assert!(s.sys.bus.dma_watch.is_empty());
}
