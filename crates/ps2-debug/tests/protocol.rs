//! End-to-end wire-protocol tests: a real `DebugServer` pumped on a
//! background thread, driven through real TCP clients like LLDB would.
//!
//! The system boots a zeroed 4 MiB "BIOS", so both cores execute nops from
//! the reset vector — enough to exercise stepping, breakpoints and memory.

use ps2_core::Ps2System;
use ps2_debug::DebugServer;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

struct Harness {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<Ps2System>>,
    ee_port: u16,
    iop_port: u16,
}

impl Harness {
    fn start() -> Self {
        let mut sys = Ps2System::new(vec![0u8; 4 * 1024 * 1024]).unwrap();
        let mut dbg = DebugServer::bind(Some(0), Some(0)).unwrap();
        let (ee_port, iop_port) = (dbg.ee_port().unwrap(), dbg.iop_port().unwrap());
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let thread = std::thread::spawn(move || {
            while !stop2.load(Ordering::Relaxed) {
                dbg.pump(&mut sys, 100_000);
                std::thread::sleep(Duration::from_millis(1));
            }
            sys
        });
        Self {
            stop,
            thread: Some(thread),
            ee_port,
            iop_port,
        }
    }

    fn finish(mut self) -> Ps2System {
        self.stop.store(true, Ordering::Relaxed);
        self.thread.take().unwrap().join().unwrap()
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

struct Client {
    stream: TcpStream,
}

impl Client {
    fn connect(port: u16) -> Self {
        let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream.set_nodelay(true).unwrap();
        Self { stream }
    }

    fn send(&mut self, payload: &str) {
        let sum: u8 = payload.bytes().fold(0u8, |a, b| a.wrapping_add(b));
        let pkt = format!("${payload}#{sum:02x}");
        self.stream.write_all(pkt.as_bytes()).unwrap();
    }

    /// Read the next framed packet payload, skipping acks and noise.
    fn recv(&mut self) -> String {
        let mut byte = [0u8; 1];
        loop {
            self.stream.read_exact(&mut byte).unwrap();
            if byte[0] == b'$' {
                break;
            }
        }
        let mut buf = Vec::new();
        loop {
            self.stream.read_exact(&mut byte).unwrap();
            if byte[0] == b'#' {
                break;
            }
            buf.push(byte[0]);
        }
        let mut sum = [0u8; 2];
        self.stream.read_exact(&mut sum).unwrap();
        String::from_utf8(buf).unwrap()
    }

    fn roundtrip(&mut self, payload: &str) -> String {
        self.send(payload);
        self.recv()
    }
}

#[test]
fn handshake_and_lldb_queries() {
    let h = Harness::start();
    let mut c = Client::connect(h.ee_port);
    assert!(c.roundtrip("qSupported").contains("qXfer:features:read+"));
    assert_eq!(c.roundtrip("?"), "T05thread:01;");
    assert_eq!(c.roundtrip("QStartNoAckMode"), "OK");
    assert!(c.roundtrip("qHostInfo").contains("endian:little"));
    assert_eq!(c.roundtrip("vMustReplyEmpty"), "");
    assert_eq!(c.roundtrip("qC"), "QC1");
}

#[test]
fn register_info_and_target_xml() {
    let h = Harness::start();
    let mut ee = Client::connect(h.ee_port);
    // EE GPRs are 64-bit, pc is 32-bit; index 38 is past the end.
    assert!(ee.roundtrip("qRegisterInfo0").contains("bitsize:64"));
    assert!(ee.roundtrip("qRegisterInfo25").contains("generic:pc"));
    assert!(ee.roundtrip("qRegisterInfo25").contains("bitsize:32"));
    assert_eq!(ee.roundtrip("qRegisterInfo26"), "E45");
    let xml = ee.roundtrip("qXfer:features:read:target.xml:0,4000");
    assert!(xml.starts_with('l'));
    assert!(xml.contains("mips64"));
    assert!(xml.contains("bitsize=\"64\""));

    let mut iop = Client::connect(h.iop_port);
    assert!(iop.roundtrip("qRegisterInfo0").contains("bitsize:32"));
    let xml = iop.roundtrip("qXfer:features:read:target.xml:0,4000");
    assert!(!xml.contains("bitsize=\"64\""));
}

#[test]
fn ee_register_read_write() {
    let h = Harness::start();
    let mut c = Client::connect(h.ee_port);
    // r1 is 64-bit wide on the wire.
    assert_eq!(c.roundtrip("P1=efbeadde01000000"), "OK");
    assert_eq!(c.roundtrip("p1"), "efbeadde01000000");
    // pc is 32-bit; writing it also resets next_pc.
    assert_eq!(c.roundtrip("P25=00001080"), "OK");
    assert_eq!(c.roundtrip("p25"), "00001080");
    // g packet: 32x8 + 4 + 8 + 8 + 4 + 4 + 4 = 288 bytes.
    assert_eq!(c.roundtrip("g").len(), 288 * 2);
    let sys = h.finish();
    assert_eq!(sys.ee.gpr[1][0], 0x1_dead_beef);
    assert_eq!(sys.ee.pc, 0x8010_0000);
    assert_eq!(sys.ee.next_pc, 0x8010_0004);
}

#[test]
fn memory_access_and_segment_aliasing() {
    let h = Harness::start();
    let mut c = Client::connect(h.ee_port);
    assert_eq!(c.roundtrip("M100000,4:11223344"), "OK");
    assert_eq!(c.roundtrip("m100000,4"), "11223344");
    // KSEG0 alias of the same RAM.
    assert_eq!(c.roundtrip("m80100000,4"), "11223344");
    // MMIO refuses debugger access.
    assert_eq!(c.roundtrip("m10000000,4"), "E01");

    let mut iop = Client::connect(h.iop_port);
    assert_eq!(iop.roundtrip("M9000,4:aabbccdd"), "OK");
    assert_eq!(iop.roundtrip("m9000,4"), "aabbccdd");
}

#[test]
fn ee_breakpoint_and_step() {
    let h = Harness::start();
    let mut c = Client::connect(h.ee_port);
    assert_eq!(c.roundtrip("?"), "T05thread:01;");
    // Set via the KSEG1 reset-vector address; stored segment-folded.
    assert_eq!(c.roundtrip("Z0,bfc00008,4"), "OK");
    c.send("c");
    assert_eq!(c.recv(), "T05thread:01;");
    assert_eq!(c.roundtrip("p25"), "0800c0bf");
    // Single step past the breakpoint.
    assert_eq!(c.roundtrip("s"), "T05thread:01;");
    assert_eq!(c.roundtrip("p25"), "0c00c0bf");
    assert_eq!(c.roundtrip("z0,bfc00008,4"), "OK");
}

#[test]
fn iop_breakpoint_hits_at_interleaved_rate() {
    let h = Harness::start();
    let mut c = Client::connect(h.iop_port);
    assert_eq!(c.roundtrip("Z0,bfc00010,4"), "OK");
    c.send("c");
    assert_eq!(c.recv(), "T05thread:01;");
    assert_eq!(c.roundtrip("p25"), "1000c0bf");
    let sys = h.finish();
    assert_eq!(sys.iop.pc, 0xBFC0_0010);
    // The EE kept running through the interleave (8 EE cycles per IOP step).
    assert!(sys.cycles >= 4 * 8);
}

#[test]
fn ee_write_watchpoint_catches_store() {
    let h = Harness::start();
    let mut c = Client::connect(h.ee_port);
    // Program at 0x100000: lui $1, 0x20; sw $1, 0($1)  -> writes 0x00200000.
    assert_eq!(c.roundtrip("M100000,8:2000013c000021ac"), "OK");
    assert_eq!(c.roundtrip("P25=00001000"), "OK");
    assert_eq!(c.roundtrip("Z2,200000,4"), "OK");
    c.send("c");
    assert_eq!(c.recv(), "T05watch:200000;thread:01;");
    assert_eq!(c.roundtrip("m200000,4"), "00002000");
    // Resuming after the refresh must not re-trigger on the same value:
    // interrupt instead of running forever.
    c.send("c");
    std::thread::sleep(Duration::from_millis(50));
    c.stream.write_all(&[0x03]).unwrap();
    assert_eq!(c.recv(), "T02thread:01;");
}

#[test]
fn detach_resumes_emulation() {
    let h = Harness::start();
    let mut c = Client::connect(h.ee_port);
    assert_eq!(c.roundtrip("Z0,bfc00100,4"), "OK");
    assert_eq!(c.roundtrip("D"), "OK");
    drop(c);
    // A fresh client attaches cleanly and the old breakpoint is gone.
    std::thread::sleep(Duration::from_millis(50));
    let mut c2 = Client::connect(h.ee_port);
    assert_eq!(c2.roundtrip("?"), "T05thread:01;");
}
