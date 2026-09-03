//! Does an EE breakpoint fire at *every* pc the core passes through?
//!
//! [`DebugServer::run_slice`] compares `sys.ee.pc` against the breakpoint set
//! *before* each `Ps2System::step`, so any address the core reaches and
//! leaves inside one `step` would be invisible to it — an accepted `Z0` that
//! never fires, which reads as "this code never runs". These tests drive the
//! real gdb-remote wire protocol over a loopback socket against a synthetic
//! EE program planted at the reset vector, and compare the stop sequence with
//! the program's known execution order.
//!
//! The pump runs on the test thread between socket operations, so the
//! sequence is deterministic — no timing, no worker thread.

use ps2_core::Ps2System;
use ps2_debug::DebugServer;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpStream;

const BIOS_SIZE: usize = 4 * 1024 * 1024;
const RESET: u32 = 0xBFC0_0000;

/// EE cycles per `pump` while waiting for a stop reply. Small enough that a
/// program which never hits a breakpoint fails fast instead of hanging.
const BUDGET: u64 = 10_000;
/// `pump` calls allowed per expected reply.
const PUMP_LIMIT: usize = 200;

/// A `Ps2System` and its `DebugServer` driven from one thread, with a real
/// TCP client on the loopback interface.
struct Session {
    sys: Ps2System,
    dbg: DebugServer,
    stream: TcpStream,
    buf: Vec<u8>,
}

impl Session {
    /// Plant `code` at the reset vector and attach a client to the EE stub.
    /// `code` is `(byte offset from 0xBFC00000, instruction)` pairs.
    fn new(code: &[(usize, u32)]) -> Self {
        Session::with_ram(code, &[])
    }

    /// As [`Session::new`], plus `ram` planted at absolute EE addresses
    /// before the client attaches.
    fn with_ram(code: &[(usize, u32)], ram: &[(u32, u32)]) -> Self {
        let mut bios = vec![0u8; BIOS_SIZE];
        for &(off, w) in code {
            bios[off..off + 4].copy_from_slice(&w.to_le_bytes());
        }
        let mut sys = Ps2System::new(bios).unwrap();
        for &(addr, w) in ram {
            sys.bus.write32(addr, w);
        }
        let dbg = DebugServer::bind(Some(0), None).unwrap();
        let port = dbg.ee_port().unwrap();
        let stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
        stream.set_nonblocking(true).unwrap();
        stream.set_nodelay(true).unwrap();
        let mut s = Session {
            sys,
            dbg,
            stream,
            buf: Vec::new(),
        };
        // Attaching halts the target; nothing executes until we say `c`.
        for _ in 0..PUMP_LIMIT {
            s.dbg.pump(&mut s.sys, 0);
            if s.dbg.attached() {
                break;
            }
        }
        assert!(s.dbg.attached(), "stub never accepted the connection");
        assert_eq!(s.sys.ee.pc, RESET, "EE must be halted at the reset vector");
        s
    }

    fn send(&mut self, payload: &str) {
        let sum = payload.bytes().fold(0u8, |a, b| a.wrapping_add(b));
        let pkt = format!("${payload}#{sum:02x}");
        self.stream.write_all(pkt.as_bytes()).unwrap();
    }

    /// Drain the socket into `buf`.
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

    /// Pop one framed packet payload, discarding acks and other noise.
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

    /// Pump until the stub answers. While halted the budget is inert, so the
    /// same call serves both synchronous queries and `c`.
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

    fn set_breakpoint(&mut self, addr: u32) {
        assert_eq!(self.roundtrip(&format!("Z0,{addr:x},4")), "OK");
    }

    /// Read one register through `p`; the reply is little-endian hex.
    fn reg(&mut self, index: usize) -> u64 {
        let hex = self.roundtrip(&format!("p{index:x}"));
        assert!(!hex.starts_with('E'), "register read failed: {hex}");
        let bytes: Vec<u8> = (0..hex.len() / 2)
            .map(|i| u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap())
            .collect();
        let mut le = [0u8; 8];
        le[..bytes.len()].copy_from_slice(&bytes);
        u64::from_le_bytes(le)
    }

    fn pc(&mut self) -> u32 {
        self.reg(37) as u32
    }

    /// Continue and return the pc of the next stop.
    fn continue_to_stop(&mut self) -> u32 {
        let reply = self.roundtrip("c");
        assert!(reply.starts_with("T05"), "unexpected stop reply: {reply}");
        self.pc()
    }

    /// Single-step and return the pc it lands on.
    fn step_to_stop(&mut self) -> u32 {
        let reply = self.roundtrip("s");
        assert!(reply.starts_with("T05"), "unexpected stop reply: {reply}");
        self.pc()
    }
}

/// Synthetic EE program at the reset vector, in `(offset, instruction)`
/// form. It covers the shapes that could plausibly hide a pc from a
/// pre-step check: 8-byte-aligned pairs the EE would dual-issue, a
/// not-taken and a taken branch with their delay slots, `jal` + delay slot,
/// a callee entry, `jr $ra` + delay slot, the return landing, and a
/// terminal self-branch.
///
/// Execution order (offsets from `RESET`):
/// `00 04 08 0c 10 14 18 1c 24 28 40 44 48 4c 2c 30 34 38 34 38 ...`
/// `20` (skipped by the taken branch) and `3c` (padding) are never reached.
const PROGRAM: &[(usize, u32)] = &[
    // 8-aligned couple: dual-issue candidates, no hazard.
    (0x00, 0x3408_0001), // ori $t0, $0, 1
    (0x04, 0x3409_0002), // ori $t1, $0, 2
    (0x08, 0x340A_0003), // ori $t2, $0, 3
    (0x0c, 0x340B_0004), // ori $t3, $0, 4
    (0x10, 0x1109_0002), // beq $t0, $t1, +2 -- not taken (1 != 2)
    (0x14, 0x340C_0005), // ori $t4, $0, 5   -- delay slot
    (0x18, 0x1108_0002), // beq $t0, $t0, +2 -- taken, to 0x24
    (0x1c, 0x340D_0006), // ori $t5, $0, 6   -- delay slot
    (0x20, 0x340E_0007), // ori $t6, $0, 7   -- branched over, never runs
    (0x24, 0x0FF0_0010), // jal 0xBFC00040
    (0x28, 0x340F_0008), // ori $t7, $0, 8   -- delay slot
    (0x2c, 0x3410_0009), // ori $s0, $0, 9   -- return landing
    (0x30, 0x3411_000A), // ori $s1, $0, 10
    // Terminal loop. The delay slot is deliberately not a nop: an all-nop
    // body would trip `check_idle_loop` and park the EE.
    (0x34, 0x1000_FFFF), // beq $0, $0, -1
    (0x38, 0x3412_000B), // ori $s2, $0, 11  -- delay slot
    // 0x3c: padding, never reached.
    (0x40, 0x3413_000C), // ori $s3, $0, 12  -- callee entry (8-aligned)
    (0x44, 0x3414_000D), // ori $s4, $0, 13
    (0x48, 0x03E0_0008), // jr $ra
    (0x4c, 0x3415_000E), // ori $s5, $0, 14  -- delay slot
];

/// Every word of [`PROGRAM`]'s address range, breakpointed or not.
fn program_words() -> Vec<u32> {
    (0..=0x4c).step_by(4).map(|o| RESET + o as u32).collect()
}

#[test]
fn every_executed_pc_stops_at_its_breakpoint() {
    let mut s = Session::new(PROGRAM);
    assert_eq!(s.roundtrip("?"), "T05thread:01;");

    for addr in program_words() {
        s.set_breakpoint(addr);
    }

    // A client is attached, so the stub must have forced single-issue: a
    // dual-issued couple would retire 0x04 inside the step taken for 0x00.
    s.dbg.pump(&mut s.sys, 0);
    assert!(
        s.sys.ee.single_issue,
        "the stub must disable dual-issue while a client is attached"
    );

    // The first `c` consumes `resume_skip`, which suppresses the check at
    // the attach pc (0xBFC00000) -- ordinary continue-from-here semantics.
    let expected: Vec<u32> = [
        0x04, 0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c, 0x24, 0x28, 0x40, 0x44, 0x48, 0x4c, 0x2c, 0x30,
        0x34, 0x38, 0x34,
    ]
    .iter()
    .map(|o| RESET + o)
    .collect();

    let mut seen = Vec::new();
    let mut loop_head_hits = 0;
    for _ in 0..64 {
        let pc = s.continue_to_stop();
        seen.push(pc);
        if pc == RESET + 0x34 {
            loop_head_hits += 1;
            if loop_head_hits == 2 {
                break;
            }
        }
    }

    // Report the holes before the ordering, so a failure names the addresses.
    let missing: Vec<String> = program_words()
        .into_iter()
        .filter(|a| !seen.contains(a))
        .map(|a| format!("{a:#010x}"))
        .collect();
    assert_eq!(
        missing,
        // 0x00 is the attach pc; 0x20 is branched over; 0x3c is padding.
        ["0xbfc00000", "0xbfc00020", "0xbfc0003c"],
        "unexpected breakpoint holes; observed stops: {seen:#010x?}"
    );
    assert_eq!(seen, expected, "stop order does not match execution order");
}

/// A software interrupt raised through Cause.IP0. `ee::Cpu::step` takes the
/// exception and runs the handler's first word in that one call, so the stub
/// has to enter the handler itself before it looks at the pc — otherwise the
/// vector entry is retired by a step whose check saw the interrupted pc.
const INTERRUPT_PROGRAM: &[(usize, u32)] = &[
    (0x00, 0x3408_0100), // ori  $t0, $0, 0x0100   -- Cause.IP0
    (0x04, 0x4088_6800), // mtc0 $t0, $13 (Cause)
    (0x08, 0x3C09_0041), // lui  $t1, 0x0041
    (0x0c, 0x3529_0101), // ori  $t1, $t1, 0x0101  -- IE|IM0|EIE|BEV
    (0x10, 0x4089_6000), // mtc0 $t1, $12 (Status)
    (0x14, 0x340A_0001), // ori  $t2, $0, 1        -- interrupted here
    (0x18, 0x1000_FFFF), // beq  $0, $0, -1        -- reached only on failure
    (0x1c, 0x340B_0002), // ori  $t3, $0, 2        -- delay slot
    // BEV is left set, so the interrupt vector is 0xBFC00200 + 0x200.
    (0x400, 0x3410_0001), // ori $s0, $0, 1
    (0x404, 0x3411_0002), // ori $s1, $0, 2
    (0x408, 0x1000_FFFF), // beq $0, $0, -1
    (0x40c, 0x3412_0003), // ori $s2, $0, 3        -- delay slot
];

#[test]
fn a_breakpoint_on_the_interrupt_vector_fires_before_it_runs() {
    let mut s = Session::new(INTERRUPT_PROGRAM);
    // 0x14 is deliberately left un-breakpointed so the single `resume_skip`
    // of the first `c` is spent on the attach pc and cannot be credited for
    // the vector entry being checked.
    for off in [0x18u32, 0x400, 0x404] {
        s.set_breakpoint(RESET + off);
    }

    let stop = s.continue_to_stop();
    assert_ne!(
        stop,
        RESET + 0x18,
        "no interrupt was taken; the vector test proves nothing"
    );
    assert_eq!(stop, RESET + 0x400, "the vector entry was stepped over");
    // Stopped *before* the handler's first word rather than after it: $s0 is
    // still 0, as is $t2 from the instruction the interrupt displaced.
    assert_eq!(s.reg(16), 0, "$s0 shows 0xBFC00400 already executed");
    assert_eq!(s.reg(10), 0, "$t2 shows 0xBFC00014 executed after all");
    // And the handler then runs normally from there.
    assert_eq!(s.continue_to_stop(), RESET + 0x404);
    assert_eq!(s.reg(16), 1, "$s0 shows 0xBFC00400 was skipped");
}

/// A resume consumes one pc check so that continuing from a breakpoint makes
/// progress. Entering a handler on the way is a different pc, and must not be
/// swallowed by that allowance — the stop before this one was a single step,
/// so the vector is the first address the resume reaches.
#[test]
fn a_resume_does_not_spend_its_skip_on_the_vector() {
    let mut s = Session::new(INTERRUPT_PROGRAM);
    for off in [0x10u32, 0x18, 0x400, 0x404] {
        s.set_breakpoint(RESET + off);
    }
    assert_eq!(s.continue_to_stop(), RESET + 0x10);
    // Status is written at 0x10, so the interrupt is pending from 0x14 on.
    assert_eq!(s.step_to_stop(), RESET + 0x14);
    assert_eq!(
        s.continue_to_stop(),
        RESET + 0x400,
        "the resume skipped the vector"
    );
}

/// Single-stepping has the same hazard: entering the handler must be a step
/// of its own, or one `s` from the interrupted instruction lands past the
/// vector's first word with that word already executed.
#[test]
fn a_single_step_lands_on_the_vector() {
    let mut s = Session::new(INTERRUPT_PROGRAM);
    s.set_breakpoint(RESET + 0x10);
    assert_eq!(s.continue_to_stop(), RESET + 0x10);
    // Stepping 0x10 writes Status, so the interrupt is pending at 0x14.
    assert_eq!(s.step_to_stop(), RESET + 0x14);
    assert_eq!(s.step_to_stop(), RESET + 0x400, "s stepped over the vector");
    assert_eq!(s.reg(16), 0, "$s0 shows 0xBFC00400 already executed");
    assert_eq!(s.step_to_stop(), RESET + 0x404);
    assert_eq!(s.reg(16), 1, "$s0 shows 0xBFC00400 was skipped");
}

// --- a caller's report: an entry in RAM that never stopped -----------------

/// Entry stub in ROM: set a stack pointer and call into RAM.
const CALL_INTO_RAM: &[(usize, u32)] = &[
    (0x00, 0x3C1D_0100), // lui   $sp, 0x0100      -- 16 MiB into RAM
    (0x04, 0x3C08_0018), // lui   $t0, 0x0018
    (0x08, 0x3508_1078), // ori   $t0, $t0, 0x1078
    (0x0c, 0x0100_F809), // jalr  $ra, $t0         -- into LOAD_MODULE
    (0x10, 0x3409_0001), // ori   $t1, $0, 1       -- delay slot
];

/// The words a caller reported as breakpointed-but-never-stopped, at the
/// address they reported them at. `0x00181078..0x00181094` are verbatim from
/// their RAM dump -- a prologue of seven back-to-back 64-bit stores, the
/// shape [`PROGRAM`] does not contain -- and the rest follows their
/// disassembly to the second call. `$2` is left non-negative so the `bltz`
/// falls through, as it did on their run.
const LOAD_MODULE: &[(u32, u32)] = &[
    (0x0018_1078, 0x27BD_FF70), // addiu $sp, $sp, -0x90   -- the entry that never stopped
    (0x0018_107c, 0xFFB6_0070), // sd    $22, 0x70($sp)
    (0x0018_1080, 0xFFB3_0040), // sd    $19, 0x40($sp)
    (0x0018_1084, 0x00E0_B02D), // move  $22, $7
    (0x0018_1088, 0xFFB1_0020), // sd    $17, 0x20($sp)
    (0x0018_108c, 0x0080_982D), // move  $19, $4
    (0x0018_1090, 0xFFB0_0010), // sd    $16, 0x10($sp)
    (0x0018_1094, 0x00A0_882D), // move  $17, $5
    (0x0018_1098, 0xFFBF_0080), // sd    $ra, 0x80($sp)
    (0x0018_109c, 0x00C0_802D), // move  $16, $6
    (0x0018_10a0, 0xFFB5_0060), // sd    $21, 0x60($sp)
    (0x0018_10a4, 0xFFB4_0050), // sd    $20, 0x50($sp)
    (0x0018_10a8, 0x0C06_03AC), // jal   0x00180eb0
    (0x0018_10ac, 0xFFB2_0030), // sd    $18, 0x30($sp)   -- delay slot
    (0x0018_10b0, 0x0440_0069), // bltz  $2, 0x00181258   -- not taken
    (0x0018_10b4, 0x3C02_FFFF), // lui   $2, 0xffff       -- delay slot
    (0x0018_10b8, 0x0C06_03EC), // jal   0x00180fb0       -- the call that stopped 12x
    (0x0018_10bc, 0x0000_0000), // nop                    -- delay slot
    // Terminal loop, with a non-nop delay slot so the idle-loop check does
    // not park the EE.
    (0x0018_10c0, 0x1000_FFFF), // beq   $0, $0, -1
    (0x0018_10c4, 0x3413_000C), // ori   $s3, $0, 12      -- delay slot
    // Callee at 0x00180eb0: returns a non-negative $2.
    (0x0018_0eb0, 0x03E0_0008), // jr    $ra
    (0x0018_0eb4, 0x3402_0001), // ori   $2, $0, 1        -- delay slot
    // Callee at 0x00180fb0, the one their breakpoint did fire on.
    (0x0018_0fb0, 0x27BD_FFB0), // addiu $sp, $sp, -0x50
    (0x0018_0fb4, 0x03E0_0008), // jr    $ra
    (0x0018_0fb8, 0x27BD_0050), // addiu $sp, $sp, 0x50   -- delay slot
];

/// Their table shows the entry at `0x00181078` and the three words after it
/// never stopping while the callee at `0x00180fb0` stopped every time. Run
/// their code at their addresses and breakpoint every word of it.
#[test]
fn a_reported_ram_prologue_stops_at_every_word() {
    let mut s = Session::with_ram(CALL_INTO_RAM, LOAD_MODULE);
    for &(addr, _) in LOAD_MODULE {
        s.set_breakpoint(addr);
    }

    let expected: Vec<u32> = vec![
        0x0018_1078,
        0x0018_107c,
        0x0018_1080,
        0x0018_1084,
        0x0018_1088,
        0x0018_108c,
        0x0018_1090,
        0x0018_1094,
        0x0018_1098,
        0x0018_109c,
        0x0018_10a0,
        0x0018_10a4,
        0x0018_10a8,
        0x0018_10ac,
        0x0018_0eb0,
        0x0018_0eb4,
        0x0018_10b0,
        0x0018_10b4,
        0x0018_10b8,
        0x0018_10bc,
        0x0018_0fb0,
        0x0018_0fb4,
        0x0018_0fb8,
        0x0018_10c0,
        0x0018_10c4,
    ];

    let mut seen = Vec::new();
    for _ in 0..expected.len() {
        seen.push(s.continue_to_stop());
    }

    let missing: Vec<String> = LOAD_MODULE
        .iter()
        .map(|&(a, _)| a)
        .filter(|a| !seen.contains(a))
        .map(|a| format!("{a:#010x}"))
        .collect();
    assert!(
        missing.is_empty(),
        "breakpoints never fired at {missing:?}; observed stops: {seen:#010x?}"
    );
    assert_eq!(seen, expected, "stop order does not match execution order");
}
