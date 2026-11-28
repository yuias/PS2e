//! ps2ctl: one-shot or batch client for the PS2e control port.
//!
//! Sends commands to a `ps2e --control-port <port>` session and prints
//! the reply, so shells and tool-using agents can drive the emulator
//! statelessly or in a single pipelined connection:
//!
//! ```text
//! ps2ctl --wait 10 state json
//! ps2ctl press circle 30
//! ps2ctl frame shot.png
//! ps2ctl --port 9005 --batch < script.txt
//! ```
//!
//! Exit code 0 on `ok`, 1 on `err`, 2 on usage or connection problems.

use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

const DEFAULT_PORT: u16 = 9002;

const AGENT_GUIDE: &str = r#"PS2e AI-Agent Zero-to-1:1 Reverse-Engineering & Verification Playbook
=====================================================================

1. START LOCKSTEP HEADLESS SESSION (deterministic, zero-flake)
   ps2e --bios <bios.bin> --disc <game.iso> --control-port 9002 &
   ps2ctl --wait 15 state json

2. BOOT ONCE & CHECKPOINT SAVESTATES (50ms instant reload forever)
   # Advance until the screen changes instead of guessing frame counts:
   ps2ctl until frame changed max 900v
   ps2ctl frame boot_01.png
   # Chain inputs + waits in a single persistent connection:
   ps2ctl --batch <<'EOF'
   seq start:10v none:60v cross:10v none:120v
   savestate states/main_menu.ps2state
   frame captures/main_menu.png
   EOF

   # Every future experiment starts at the exact scene in milliseconds:
   ps2ctl loadstate states/main_menu.ps2state

3. DIAGNOSE HANGS OR UNKNOWN BEHAVIOR IN ONE COMMAND (no GDB required)
   ps2ctl state json          # ee_pc, iop_pc, cycles, vblanks, region, held
   ps2ctl regs ee             # EE PC, COP0 status/cause/epc, HI/LO, 32 GPRs
   ps2ctl regs iop            # IOP PC, COP0, 32 GPRs
   ps2ctl regs hw             # INTC, DMAC, VIF0/1 stall/CHCR, SIF flags/FIFOs
   ps2ctl tty                 # EE/IOP kernel & game printf log
   ps2ctl peek ee <addr> 64   # Side-effect-free hex dump of EE RAM

4. FIND GAME STATE IN RAM (health, coords, camera, timers, level IDs)
   ps2ctl scan start ee 4 exact 100    # Initial value scan
   ps2ctl run 60v                      # Advance game
   ps2ctl scan filter decreased        # Narrow candidates
   ps2ctl scan list 20                 # Print matching addresses
   ps2ctl until ee <addr> 4 changed max 300v

5. 1:1 VISUAL PARITY VERIFICATION (PS2 ground truth vs native port)
   ps2ctl frame captures/ps2_scene.png
   ps2ctl diff captures/ps2_scene.png captures/native_scene.png captures/diff.png
   # Prints JSON: {"width":...,"height":...,"mae_rgb":...,"psnr_db":...,"diff_pct":...}
   # Targets: mae_rgb < 12.0, psnr_db > 24.0 (tune per scene)

6. QUICK MEMORY MAP REFERENCE (PS2)
   EE RAM:        0x00000000 - 0x01FFFFFF (32 MiB; game ELF usually @ 0x00100000)
   EE Scratchpad: 0x70000000 - 0x70003FFF (16 KiB)
   EE MMIO:       0x10000000 - 0x1000FFFF (Timers, GIF, VIF, DMAC, INTC, SIF)
   IOP RAM:       0x00000000 - 0x001FFFFF (2 MiB on IOP bus; 0x1C000000 on EE)
"#;

fn decode_png_rgb(path: &str) -> Result<(u32, u32, Vec<u8>), String> {
    let file = File::open(path).map_err(|e| format!("open {path}: {e}"))?;
    let mut decoder = png::Decoder::new(BufReader::new(file));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| format!("decode {path}: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().ok_or("png buffer size overflow")?];
    let info = reader.next_frame(&mut buf).map_err(|e| format!("frame {path}: {e}"))?;
    let data = &buf[..info.buffer_size()];
    let w = info.width;
    let h = info.height;
    let mut rgb = Vec::with_capacity((w as usize) * (h as usize) * 3);
    match info.color_type {
        png::ColorType::Rgb => rgb.extend_from_slice(data),
        png::ColorType::Rgba => {
            for px in data.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        }
        png::ColorType::Grayscale => {
            for &g in data {
                rgb.extend_from_slice(&[g, g, g]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for px in data.chunks_exact(2) {
                rgb.extend_from_slice(&[px[0], px[0], px[0]]);
            }
        }
        other => return Err(format!("{path}: unsupported color type {other:?}")),
    }
    Ok((w, h, rgb))
}

fn run_diff(a_path: &str, b_path: &str, heatmap_path: Option<&str>) -> Result<(), String> {
    let (wa, ha, ra) = decode_png_rgb(a_path)?;
    let (wb, hb, rb) = decode_png_rgb(b_path)?;
    if wa == 0 || ha == 0 || wb == 0 || hb == 0 {
        return Err("empty image".into());
    }
    let n = (wa as usize) * (ha as usize);
    let mut sum_abs: u64 = 0;
    let mut sum_sq: u64 = 0;
    let mut diff_px: u64 = 0;
    let mut heat = if heatmap_path.is_some() { vec![0u8; n * 3] } else { Vec::new() };

    for y in 0..ha {
        let yb = ((u64::from(y) * u64::from(hb)) / u64::from(ha)) as usize;
        for x in 0..wa {
            let xb = ((u64::from(x) * u64::from(wb)) / u64::from(wa)) as usize;
            let ia = ((y as usize) * (wa as usize) + (x as usize)) * 3;
            let ib = (yb * (wb as usize) + xb) * 3;
            let dr = (ra[ia] as i32 - rb[ib] as i32).unsigned_abs();
            let dg = (ra[ia + 1] as i32 - rb[ib + 1] as i32).unsigned_abs();
            let db = (ra[ia + 2] as i32 - rb[ib + 2] as i32).unsigned_abs();
            sum_abs += u64::from(dr + dg + db);
            sum_sq += u64::from(dr * dr + dg * dg + db * db);
            if dr.max(dg).max(db) > 8 {
                diff_px += 1;
            }
            if !heat.is_empty() {
                heat[ia] = (dr * 4).min(255) as u8;
                heat[ia + 1] = (dg * 4).min(255) as u8;
                heat[ia + 2] = (db * 4).min(255) as u8;
            }
        }
    }

    let channels = (n as f64) * 3.0;
    let mae = sum_abs as f64 / channels;
    let mse = sum_sq as f64 / channels;
    let psnr = if mse <= 1e-12 { 99.99 } else { 10.0 * (255.0_f64 * 255.0 / mse).log10() };
    let diff_pct = (diff_px as f64 * 100.0) / (n as f64);

    if let Some(out) = heatmap_path {
        let f = File::create(out).map_err(|e| format!("create {out}: {e}"))?;
        let mut enc = png::Encoder::new(BufWriter::new(f), wa, ha);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().map_err(|e| format!("png header {out}: {e}"))?;
        writer.write_image_data(&heat).map_err(|e| format!("png write {out}: {e}"))?;
    }

    println!(
        "{{\"width\":{wa},\"height\":{ha},\"mae_rgb\":{mae:.3},\"psnr_db\":{psnr:.2},\"diff_pct\":{diff_pct:.2}}}"
    );
    Ok(())
}

fn connect(port: u16, wait_secs: f64) -> TcpStream {
    let deadline = Instant::now() + Duration::from_secs_f64(wait_secs.max(0.0));
    loop {
        match TcpStream::connect(("127.0.0.1", port)) {
            Ok(s) => return s,
            Err(e) if Instant::now() < deadline => {
                let _ = e;
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                eprintln!("ps2ctl: cannot connect to 127.0.0.1:{port}: {e}");
                eprintln!("is `ps2e --control-port {port}` running?");
                std::process::exit(2);
            }
        }
    }
}

fn send_one(writer: &mut TcpStream, reader: &mut BufReader<TcpStream>, cmd: &str) -> bool {
    writer.write_all(format!("{cmd}\n").as_bytes()).expect("send command");
    let mut status = String::new();
    if reader.read_line(&mut status).expect("read status") == 0 {
        eprintln!("ps2ctl: connection closed before status");
        std::process::exit(2);
    }
    let ok = status.trim() == "ok";
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line).expect("read reply") == 0 {
            eprintln!("ps2ctl: connection closed mid-reply");
            std::process::exit(2);
        }
        let l = line.trim_end_matches(['\r', '\n']);
        if l == "." {
            break;
        }
        // Reverse the server's dot-stuffing.
        println!("{}", l.strip_prefix('.').unwrap_or(l));
    }
    ok
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut port = DEFAULT_PORT;
    let mut wait_secs = 0.0f64;
    let mut batch = false;

    while let Some(flag) = args.first().map(String::as_str) {
        match flag {
            "--port" => {
                args.remove(0);
                port = args.first().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage());
                args.remove(0);
            }
            "--wait" => {
                args.remove(0);
                wait_secs = args.first().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage());
                args.remove(0);
            }
            "--batch" => {
                args.remove(0);
                batch = true;
            }
            "--agent-guide" => {
                print!("{AGENT_GUIDE}");
                std::process::exit(0);
            }
            _ => break,
        }
    }

    if args.first().map(String::as_str) == Some("diff") {
        let res = match args.as_slice() {
            [_, a, b] => run_diff(a, b, None),
            [_, a, b, h] => run_diff(a, b, Some(h)),
            _ => Err("usage: ps2ctl diff <a.png> <b.png> [heatmap.png]".into()),
        };
        match res {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("ps2ctl diff: {e}");
                std::process::exit(1);
            }
        }
    }

    let stream = connect(port, wait_secs);
    let mut writer = stream.try_clone().expect("clone stream");
    let mut reader = BufReader::new(stream);

    if batch {
        let stdin = io::stdin();
        for raw in stdin.lock().lines() {
            let line = raw.expect("read stdin");
            let cmd = line.trim();
            if cmd.is_empty() || cmd.starts_with('#') {
                continue;
            }
            if !send_one(&mut writer, &mut reader, cmd) {
                std::process::exit(1);
            }
        }
        std::process::exit(0);
    }

    let cmd = args.join(" ");
    if cmd.is_empty() {
        usage();
    }
    let ok = send_one(&mut writer, &mut reader, &cmd);
    std::process::exit(if ok { 0 } else { 1 });
}

fn usage() -> ! {
    eprintln!(
        "usage: ps2ctl [--port N] [--wait SECS] [--batch | --agent-guide | diff <a.png> <b.png> [heatmap.png] | <command...>]\n       (try: ps2ctl help  or  ps2ctl --agent-guide)"
    );
    std::process::exit(2);
}
