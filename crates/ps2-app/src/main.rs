//! Headless front-end: boot a BIOS, run for N cycles, stream kernel TTY
//! output to stdout. The egui + wgpu UI arrives with the GS milestone.

use std::io::Write;
use std::process::ExitCode;

use ps2_core::Ps2System;
use tracing_subscriber::EnvFilter;

struct Args {
    bios: String,
    cycles: u64,
    log: Option<String>,
    /// Directory to dump EE/IOP RAM into after the run (bring-up aid).
    dump: Option<String>,
    /// Write the final framebuffer as a BMP.
    screenshot: Option<String>,
    /// gdb-remote stub ports for the EE and IOP targets.
    debug_ee: Option<u16>,
    debug_iop: Option<u16>,
    /// Hold execution at the reset vector until a debugger attaches.
    wait_debugger: bool,
    /// Scripted pad input: (button mask, first cycle, last cycle).
    presses: Vec<(u16, u64, u64)>,
    /// Memory card image to load and persist (16384 x 528-byte pages).
    memcard: Option<String>,
    /// Disc image (2048-byte-sector ISO), streamed on demand.
    disc: Option<String>,
}

/// Default hold length for a scripted press, in EE cycles (~0.5 s).
const PRESS_HOLD: u64 = 150_000_000;

fn pad_button_bit(name: &str) -> Result<u16, String> {
    Ok(match name {
        "select" => 0,
        "l3" => 1,
        "r3" => 2,
        "start" => 3,
        "up" => 4,
        "right" => 5,
        "down" => 6,
        "left" => 7,
        "l2" => 8,
        "r2" => 9,
        "l1" => 10,
        "r1" => 11,
        "triangle" => 12,
        "circle" => 13,
        "cross" => 14,
        "square" => 15,
        _ => return Err(format!("unknown button '{name}'")),
    })
}

/// Parse "circle@6000000000" or "down@5e9-5.2e9"-style "<button>@<from>[-<to>]".
fn parse_press(spec: &str) -> Result<(u16, u64, u64), String> {
    let (name, range) = spec
        .split_once('@')
        .ok_or_else(|| format!("--press needs <button>@<cycle>, got '{spec}'"))?;
    let bit = pad_button_bit(name)?;
    let parse_n = |s: &str| -> Result<u64, String> {
        s.replace('_', "")
            .parse()
            .map_err(|e| format!("bad cycle '{s}': {e}"))
    };
    let (from, to) = match range.split_once('-') {
        Some((a, b)) => (parse_n(a)?, parse_n(b)?),
        None => {
            let a = parse_n(range)?;
            (a, a + PRESS_HOLD)
        }
    };
    Ok((1 << bit, from, to))
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        bios: "assets/SCPH-50000.bin".to_string(),
        cycles: 500_000_000,
        log: None,
        dump: None,
        screenshot: None,
        debug_ee: None,
        debug_iop: None,
        wait_debugger: false,
        presses: Vec::new(),
        memcard: None,
        disc: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bios" => args.bios = it.next().ok_or("--bios needs a path")?,
            "--cycles" => {
                args.cycles = it
                    .next()
                    .ok_or("--cycles needs a number")?
                    .replace('_', "")
                    .parse()
                    .map_err(|e| format!("bad --cycles: {e}"))?;
            }
            "--log" => args.log = Some(it.next().ok_or("--log needs a filter")?),
            "--dump" => args.dump = Some(it.next().ok_or("--dump needs a directory")?),
            "--screenshot" => args.screenshot = Some(it.next().ok_or("--screenshot needs a path")?),
            "--debug-ee" => {
                args.debug_ee = Some(parse_port(it.next().ok_or("--debug-ee needs a port")?)?)
            }
            "--debug-iop" => {
                args.debug_iop = Some(parse_port(it.next().ok_or("--debug-iop needs a port")?)?)
            }
            "--wait-debugger" => args.wait_debugger = true,
            "--press" => args
                .presses
                .push(parse_press(&it.next().ok_or("--press needs <button>@<cycle>")?)?),
            "--memcard" => args.memcard = Some(it.next().ok_or("--memcard needs a path")?),
            "--disc" => args.disc = Some(it.next().ok_or("--disc needs a path")?),
            "--help" | "-h" => {
                println!(
                    "usage: ps2-app [--bios <path>] [--cycles <n>] [--log <filter>]\n\
                     \n\
                     --bios           BIOS image (default assets/SCPH-50000.bin)\n\
                     --cycles         EE cycles to run (default 500_000_000)\n\
                     --log            tracing filter, e.g. 'info,ps2_core::tty=debug'\n\
                     --dump           directory for EE/IOP RAM dumps after the run\n\
                     --screenshot     write the final framebuffer as a BMP\n\
                     --debug-ee       gdb-remote stub port for the EE (LLDB-first)\n\
                     --debug-iop      gdb-remote stub port for the IOP\n\
                     --wait-debugger  hold at the reset vector until a debugger attaches\n\
                     --press          hold a pad button, <button>@<cycle>[-<cycle>]\n\
                     \x20                (circle, cross, up, down, start, ...; repeatable)\n\
                     --memcard        card image to load/persist (created if missing)\n\
                     --disc           disc image (2048-byte-sector ISO), streamed"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    if args.wait_debugger && args.debug_ee.is_none() && args.debug_iop.is_none() {
        return Err("--wait-debugger needs --debug-ee or --debug-iop".to_string());
    }
    Ok(args)
}

fn parse_port(s: String) -> Result<u16, String> {
    s.parse().map_err(|e| format!("bad port '{s}': {e}"))
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let filter = match &args.log {
        Some(f) => EnvFilter::new(f),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    let bios = match std::fs::read(&args.bios) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read BIOS '{}': {e}", args.bios);
            return ExitCode::FAILURE;
        }
    };
    let mut sys = match Ps2System::new(bios) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut debugger = match (args.debug_ee, args.debug_iop) {
        (None, None) => None,
        (ee, iop) => match ps2_debug::DebugServer::bind(ee, iop) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("error: cannot bind debug port: {e}");
                return ExitCode::FAILURE;
            }
        },
    };

    if let Some(path) = &args.disc {
        match std::fs::File::open(path) {
            Ok(f) => {
                sys.bus.cdvd.disc = Some(f);
                tracing::info!(path = %path, "disc image attached");
            }
            Err(e) => {
                eprintln!("error: cannot open disc '{path}': {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    if let Some(path) = &args.memcard {
        match std::fs::read(path) {
            Ok(img) if img.len() == sys.bus.sio2.memcard.data.len() => {
                sys.bus.sio2.memcard.data.copy_from_slice(&img);
                tracing::info!(path = %path, "memory card image loaded");
            }
            Ok(img) => {
                eprintln!(
                    "error: memcard '{path}' has {} bytes, expected {}",
                    img.len(),
                    sys.bus.sio2.memcard.data.len()
                );
                return ExitCode::FAILURE;
            }
            Err(_) => {
                tracing::info!(path = %path, "memcard image missing, starting blank");
            }
        }
    }

    tracing::info!(bios = %args.bios, cycles = args.cycles, "booting");

    // Run in slices so TTY output streams out as it appears.
    const SLICE: u64 = 1_000_000;
    let stdout = std::io::stdout();
    let mut remaining = args.cycles;
    let mut debugger_seen = false;
    while remaining > 0 {
        // While a debugger is attached (or awaited), it owns execution: the
        // stub runs the system from inside pump() and we only track cycles.
        if let Some(dbg) = &mut debugger {
            let before = sys.cycles;
            dbg.pump(&mut sys, remaining.min(SLICE));
            remaining -= (sys.cycles - before).min(remaining);
            debugger_seen |= dbg.attached();
            if dbg.attached() || (args.wait_debugger && !debugger_seen) {
                if !dbg.attached() || dbg.halted() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                flush_tty(&stdout, &mut sys);
                continue;
            }
        }
        let n = remaining.min(SLICE);
        sys.bus.sio2.buttons = args
            .presses
            .iter()
            .filter(|&&(_, from, to)| sys.cycles >= from && sys.cycles < to)
            .fold(0, |acc, &(mask, _, _)| acc | mask);
        sys.run(n);
        remaining -= n;
        flush_tty(&stdout, &mut sys);
    }

    tracing::info!(
        cycles = sys.cycles,
        prims = sys.bus.gs.prims_drawn,
        prims_tex = sys.bus.gs.prims_textured,
        pmode = format_args!("{:#x}", sys.bus.gs.pmode),
        dispfb1 = format_args!("{:#x}", sys.bus.gs.dispfb1),
        dispfb2 = format_args!("{:#x}", sys.bus.gs.dispfb2),
        frame0 = format_args!("{:#x}", sys.bus.gs.ctx[0].frame),
        frame1 = format_args!("{:#x}", sys.bus.gs.ctx[1].frame),
        zbuf0 = format_args!("{:#x}", sys.bus.gs.ctx[0].zbuf),
        test0 = format_args!("{:#x}", sys.bus.gs.ctx[0].test),
        test1 = format_args!("{:#x}", sys.bus.gs.ctx[1].test),
        ee_pc = format_args!("{:#010x}", sys.ee.pc),
        iop_pc = format_args!("{:#010x}", sys.iop.pc),
        iop_i_mask = format_args!("{:#x}", sys.bus.iop_i_mask),
        iop_i_ctrl = sys.bus.iop_i_ctrl,
        intc_mask = format_args!("{:#x}", sys.bus.intc_mask),
        d_mask = format_args!("{:#x}", sys.bus.d_mask),
        "run finished"
    );
    let psms: Vec<String> = sys
        .bus
        .gs
        .tex_psm_hist
        .iter()
        .enumerate()
        .filter(|&(_, &n)| n > 0)
        .map(|(psm, n)| format!("{psm:#04x}:{n}"))
        .collect();
    tracing::info!(psms = psms.join(" "), "texture samples per PSM");

    if let Some(path) = &args.memcard
        && sys.bus.sio2.memcard.dirty
    {
        if let Err(e) = std::fs::write(path, &sys.bus.sio2.memcard.data) {
            eprintln!("error: memcard save failed: {e}");
            return ExitCode::FAILURE;
        }
        tracing::info!(path = %path, "memory card image saved");
    }

    if let Some(dir) = &args.dump {
        let ee = format!("{dir}/ee_ram.bin");
        let iop = format!("{dir}/iop_ram.bin");
        let vram = format!("{dir}/gs_vram.bin");
        if let Err(e) = std::fs::write(&ee, &sys.bus.ram)
            .and_then(|_| std::fs::write(&iop, &sys.bus.iop_ram))
            .and_then(|_| std::fs::write(&vram, &sys.bus.gs.vram))
            .and_then(|_| std::fs::write(format!("{dir}/vu1_micro.bin"), &sys.bus.vu1.micro))
            .and_then(|_| std::fs::write(format!("{dir}/vu1_data.bin"), &sys.bus.vu1.data))
        {
            eprintln!("error: RAM dump failed: {e}");
            return ExitCode::FAILURE;
        }
        tracing::info!(dir = %dir, "dumped EE/IOP RAM and GS VRAM");
    }
    if let Some(path) = &args.screenshot {
        let (w, h, rgba) = sys.framebuffer();
        if let Err(e) = write_bmp(path, w, h, &rgba) {
            eprintln!("error: screenshot failed: {e}");
            return ExitCode::FAILURE;
        }
        tracing::info!(path = %path, w, h, "screenshot written");
    }
    ExitCode::SUCCESS
}

fn flush_tty(stdout: &std::io::Stdout, sys: &mut Ps2System) {
    let tty = sys.take_tty();
    if !tty.is_empty() {
        let mut out = stdout.lock();
        let _ = out.write_all(tty.as_bytes());
        let _ = out.flush();
    }
}

/// Minimal 24-bit bottom-up BMP writer.
fn write_bmp(path: &str, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
    let row = ((w * 3 + 3) & !3) as usize;
    let data_size = row * h as usize;
    let mut out = Vec::with_capacity(54 + data_size);
    out.extend_from_slice(b"BM");
    out.extend_from_slice(&(54 + data_size as u32).to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&54u32.to_le_bytes());
    out.extend_from_slice(&40u32.to_le_bytes());
    out.extend_from_slice(&w.to_le_bytes());
    out.extend_from_slice(&h.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24u16.to_le_bytes());
    out.extend_from_slice(&[0; 24]);
    for y in (0..h).rev() {
        let start = out.len();
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            out.push(rgba[o + 2]);
            out.push(rgba[o + 1]);
            out.push(rgba[o]);
        }
        out.resize(start + row, 0);
    }
    std::fs::write(path, out)
}
