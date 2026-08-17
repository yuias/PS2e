//! Native front-end for ps2-core.
//!
//! `--cycles N` runs headless and exits: this is the bring-up workflow used
//! for BIOS/game analysis (screenshots, `--press` scripting, `--wav`,
//! `--dump`, `--debug-ee`/`--debug-iop` gdb-remote stubs). Without
//! `--cycles` (or with an explicit `--window`) it opens an eframe/wgpu
//! window instead: the emulator runs on a worker thread paced against the
//! audio buffer, and the UI is a thin client over published snapshots (see
//! [`emu`]).
//!
//! `--debug-ee`/`--debug-iop` open LLDB/GDB gdb-remote stubs in both modes;
//! `--wait-debugger` additionally holds execution at reset until a debugger
//! attaches.

mod audio;
mod config;
mod emu;
mod pad;
mod ui;

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use ps2_core::Ps2System;
use tracing_subscriber::EnvFilter;

struct Args {
    bios: Option<String>,
    /// Headless when set; windowed run control ignores it.
    cycles: Option<u64>,
    /// Force windowed mode even when `--cycles` is also given.
    window: bool,
    log: Option<String>,
    /// Directory to dump EE/IOP RAM into after a headless run.
    dump: Option<String>,
    /// Write the final framebuffer as a BMP (headless only).
    screenshot: Option<String>,
    /// Also write a numbered BMP next to `screenshot` every N cycles.
    screenshot_every: Option<u64>,
    /// gdb-remote stub ports for the EE and IOP targets.
    debug_ee: Option<u16>,
    debug_iop: Option<u16>,
    /// Hold execution at the reset vector until a debugger attaches.
    wait_debugger: bool,
    /// Scripted pad input: (button mask, first cycle, last cycle). Headless.
    presses: Vec<(u16, u64, u64)>,
    /// Memory card image to load and persist (16384 x 528-byte pages).
    memcard: Option<String>,
    /// Disc image (2048-byte-sector ISO), streamed on demand.
    disc: Option<String>,
    /// Write the SPU2 output (48 kHz stereo) as a WAV file (headless only).
    wav: Option<String>,
}

/// Default hold length for a scripted press, in EE cycles (~0.5 s).
const PRESS_HOLD: u64 = 150_000_000;

/// Parse "circle@6000000000" or "down@5e9-5.2e9"-style "<button>@<from>[-<to>]".
fn parse_press(spec: &str) -> Result<(u16, u64, u64), String> {
    let (name, range) = spec
        .split_once('@')
        .ok_or_else(|| format!("--press needs <button>@<cycle>, got '{spec}'"))?;
    let mask = pad::mask_by_name(name)?;
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
    Ok((mask, from, to))
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        bios: None,
        cycles: None,
        window: false,
        log: None,
        dump: None,
        screenshot: None,
        screenshot_every: None,
        debug_ee: None,
        debug_iop: None,
        wait_debugger: false,
        presses: Vec::new(),
        memcard: None,
        disc: None,
        wav: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--bios" => args.bios = Some(it.next().ok_or("--bios needs a path")?),
            "--cycles" => {
                args.cycles = Some(
                    it.next()
                        .ok_or("--cycles needs a number")?
                        .replace('_', "")
                        .parse()
                        .map_err(|e| format!("bad --cycles: {e}"))?,
                );
            }
            "--window" => args.window = true,
            "--log" => args.log = Some(it.next().ok_or("--log needs a filter")?),
            "--dump" => args.dump = Some(it.next().ok_or("--dump needs a directory")?),
            "--screenshot" => args.screenshot = Some(it.next().ok_or("--screenshot needs a path")?),
            "--screenshot-every" => {
                args.screenshot_every = Some(
                    it.next()
                        .ok_or("--screenshot-every needs a cycle count")?
                        .replace('_', "")
                        .parse()
                        .map_err(|e| format!("bad --screenshot-every: {e}"))?,
                )
            }
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
            "--wav" => args.wav = Some(it.next().ok_or("--wav needs a path")?),
            "--help" | "-h" => {
                println!(
                    "usage: ps2-app [--bios <path>] [--cycles <n>] [--window] [--log <filter>]\n\
                     \n\
                     With no --cycles (or with --window), opens a window; otherwise runs\n\
                     headless for the given number of EE cycles and exits.\n\
                     \n\
                     --bios           BIOS image (default assets/SCPH-50000.bin)\n\
                     --cycles         EE cycles to run headlessly, then exit\n\
                     --window         open a window even when --cycles is given\n\
                     --log            tracing filter, e.g. 'info,ps2_core::tty=debug'\n\
                     --dump           directory for EE/IOP RAM dumps after a headless run\n\
                     --screenshot     write the final framebuffer as a BMP (headless)\n\
                     --screenshot-every  also write <screenshot>_<n>.bmp every N cycles\n\
                     --debug-ee       gdb-remote stub port for the EE (LLDB-first)\n\
                     --debug-iop      gdb-remote stub port for the IOP\n\
                     --wait-debugger  hold at the reset vector until a debugger attaches\n\
                     --press          hold a pad button, <button>@<cycle>[-<cycle>] (headless)\n\
                     \x20                (circle, cross, up, down, start, ...; repeatable)\n\
                     --memcard        card image to load/persist (created if missing)\n\
                     --disc           disc image (2048-byte-sector ISO), streamed\n\
                     --wav            write the SPU2 output as a 48 kHz stereo WAV (headless)"
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

    let (cfg, cfg_path) = config::Config::load();
    let windowed = args.window || args.cycles.is_none();

    let bios_path = args
        .bios
        .clone()
        .or_else(|| cfg.bios.as_ref().map(|p| p.display().to_string()))
        .unwrap_or_else(|| "assets/SCPH-50000.bin".to_string());
    let bios = match std::fs::read(&bios_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: cannot read BIOS '{bios_path}': {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut sys = match Ps2System::new(bios.clone()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let debugger = match (args.debug_ee, args.debug_iop) {
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

    // Headless keeps its original semantics: a card is only loaded/persisted
    // when `--memcard` is given. Windowed mode always mounts one (falling
    // back to the config's default location) so play sessions save by
    // default.
    let memcard_path: Option<PathBuf> = if windowed {
        Some(
            args.memcard
                .clone()
                .map(PathBuf::from)
                .unwrap_or_else(|| cfg.memcard_path(cfg_path.as_ref())),
        )
    } else {
        args.memcard.clone().map(PathBuf::from)
    };
    if let Some(path) = &memcard_path {
        match std::fs::read(path) {
            Ok(img) if img.len() == sys.bus.sio2.memcard.data.len() => {
                sys.bus.sio2.memcard.data.copy_from_slice(&img);
                tracing::info!(path = %path.display(), "memory card image loaded");
            }
            Ok(img) => {
                eprintln!(
                    "error: memcard '{}' has {} bytes, expected {}",
                    path.display(),
                    img.len(),
                    sys.bus.sio2.memcard.data.len()
                );
                return ExitCode::FAILURE;
            }
            Err(_) => {
                tracing::info!(path = %path.display(), "memcard image missing, starting blank");
            }
        }
    }

    if windowed {
        run_windowed(sys, bios, args, cfg, cfg_path, debugger, memcard_path)
    } else {
        run_headless(sys, &args, debugger, memcard_path)
    }
}

fn run_windowed(
    sys: Ps2System,
    bios: Vec<u8>,
    args: Args,
    cfg: config::Config,
    cfg_path: Option<PathBuf>,
    debugger: Option<ps2_debug::DebugServer>,
    memcard_path: Option<PathBuf>,
) -> ExitCode {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: eframe::egui::ViewportBuilder::default()
            .with_inner_size([960.0, 640.0])
            .with_title("PS2e"),
        ..Default::default()
    };
    let wait_debugger = args.wait_debugger;
    let result = eframe::run_native(
        "PS2e",
        options,
        Box::new(move |cc| {
            let worker_cfg = emu::WorkerConfig {
                bios,
                memcard_path,
                debugger,
                wait_debugger,
                volume: cfg.volume,
            };
            let emu = emu::spawn(sys, worker_cfg, cc.egui_ctx.clone());
            Ok(Box::new(ui::App::new(emu, cfg, cfg_path)))
        }),
    );
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run_headless(
    mut sys: Ps2System,
    args: &Args,
    mut debugger: Option<ps2_debug::DebugServer>,
    memcard_path: Option<PathBuf>,
) -> ExitCode {
    let cycles = args.cycles.expect("headless mode requires --cycles");
    tracing::info!(bios = ?args.bios, cycles, "booting");

    // Run in slices so TTY output streams out as it appears.
    const SLICE: u64 = 1_000_000;
    let stdout = std::io::stdout();
    let mut remaining = cycles;
    let mut debugger_seen = false;
    let mut audio: Vec<i16> = Vec::new();
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
        if args.wav.is_some() {
            audio.extend(sys.bus.spu2.take_output());
        } else {
            sys.bus.spu2.take_output();
        }
        if let (Some(every), Some(path)) = (args.screenshot_every, &args.screenshot)
            && sys.cycles / every != (sys.cycles - n) / every
        {
            let numbered = numbered_path(path, sys.cycles / every);
            let (w, h, rgba) = sys.framebuffer();
            if let Err(e) = write_bmp(&numbered, w, h, &rgba) {
                eprintln!("error: screenshot failed: {e}");
                return ExitCode::FAILURE;
            }
            tracing::info!(path = %numbered, cycles = sys.cycles, "screenshot written");
        }
    }

    tracing::info!(
        cycles = sys.cycles,
        prims = sys.bus.gs.prims_drawn,
        prims_tex = sys.bus.gs.prims_textured,
        pixels = sys.bus.gs.pixels_shaded,
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
    if let Some(report) = ps2_core::prof::report() {
        eprintln!("{report}");
    }

    if let Some(path) = &memcard_path
        && sys.bus.sio2.memcard.dirty
    {
        if let Err(e) = std::fs::write(path, &sys.bus.sio2.memcard.data) {
            eprintln!("error: memcard save failed: {e}");
            return ExitCode::FAILURE;
        }
        tracing::info!(path = %path.display(), "memory card image saved");
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
            .and_then(|_| std::fs::write(format!("{dir}/spu2_ram.bin"), &sys.bus.spu2.ram))
            .and_then(|_| std::fs::write(format!("{dir}/spu2_regs.bin"), sys.bus.spu2.regs_bytes()))
        {
            eprintln!("error: RAM dump failed: {e}");
            return ExitCode::FAILURE;
        }
        tracing::info!(dir = %dir, "dumped EE/IOP RAM and GS VRAM");
    }
    if let Some(path) = &args.wav {
        audio.extend(sys.bus.spu2.take_output());
        if let Err(e) = write_wav(path, &audio) {
            eprintln!("error: wav write failed: {e}");
            return ExitCode::FAILURE;
        }
        tracing::info!(path = %path, samples = audio.len() / 2, "wav written");
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

/// Write already-drained TTY text to stdout. Used by the windowed worker
/// thread, which reacquires the stdout handle each call (cheap: it is a
/// thin wrapper over the process-wide handle, not a fresh open).
pub(crate) fn print_tty(text: &str) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = out.write_all(text.as_bytes());
    let _ = out.flush();
}

/// `foo.bmp` -> `foo_<n>.bmp` (extension-less paths just get the suffix).
fn numbered_path(path: &str, n: u64) -> String {
    match path.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => format!("{stem}_{n}.{ext}"),
        _ => format!("{path}_{n}"),
    }
}

/// 16-bit stereo 48 kHz PCM WAV.
fn write_wav(path: &str, samples: &[i16]) -> std::io::Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&2u16.to_le_bytes()); // stereo
    out.extend_from_slice(&48_000u32.to_le_bytes());
    out.extend_from_slice(&(48_000u32 * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    std::fs::write(path, out)
}

/// Minimal 24-bit bottom-up BMP writer.
pub(crate) fn write_bmp(path: &str, w: u32, h: u32, rgba: &[u8]) -> std::io::Result<()> {
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
