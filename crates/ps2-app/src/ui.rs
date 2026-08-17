//! egui shell: a thin client over the emulator worker thread.
//!
//! All emulation (and audio) lives in [`crate::emu`]; this module only sends
//! commands, reads published snapshots and draws. Keeping it presentation-only
//! is deliberate — a wasm frontend could reuse the same snapshot types.

use crate::config::Config;
use crate::emu::{Command, DebuggerState, Emu, FrameSnapshot};
use crate::pad;
use eframe::egui;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// Keyboard -> digital pad mapping. egui's `Key` enum has no Shift variant
/// (shift is tracked as a modifier, not a key), so SELECT binds to Backspace
/// only.
const KEYMAP: [(egui::Key, u16); 16] = [
    (egui::Key::ArrowUp, pad::UP),
    (egui::Key::ArrowDown, pad::DOWN),
    (egui::Key::ArrowLeft, pad::LEFT),
    (egui::Key::ArrowRight, pad::RIGHT),
    (egui::Key::Z, pad::CROSS),
    (egui::Key::X, pad::CIRCLE),
    (egui::Key::C, pad::SQUARE),
    (egui::Key::V, pad::TRIANGLE),
    (egui::Key::Q, pad::L1),
    (egui::Key::W, pad::R1),
    (egui::Key::E, pad::L2),
    (egui::Key::R, pad::R2),
    (egui::Key::Num1, pad::L3),
    (egui::Key::Num3, pad::R3),
    (egui::Key::Enter, pad::START),
    (egui::Key::Backspace, pad::SELECT),
];

/// MIPS GPR names, index-aligned with `Cpu::gpr` (shared by EE and IOP).
const REG_NAMES: [&str; 32] = [
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", //
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", //
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", //
    "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra",
];

// Standard MIPS COP0 register numbers; identical on the EE (see
// ps2_core::ee::cop0) and the IOP's private constants of the same values.
const COP0_STATUS: usize = 12;
const COP0_CAUSE: usize = 13;
const COP0_EPC: usize = 14;

pub struct App {
    emu: Emu,
    display_tex: Option<egui::TextureHandle>,
    linear_filter: bool,
    /// Master volume applied on top of the SPU2 output (0..=1).
    volume: f32,
    config: Config,
    config_path: Option<PathBuf>,
    last_screenshot: Option<String>,
    show_tty: bool,
    show_regs: bool,
}

impl App {
    pub fn new(emu: Emu, config: Config, config_path: Option<PathBuf>) -> Self {
        let volume = config.volume.clamp(0.0, 1.0);
        Self {
            emu,
            display_tex: None,
            linear_filter: false,
            volume,
            config,
            config_path,
            last_screenshot: None,
            show_tty: false,
            show_regs: false,
        }
    }

    /// Dump the currently displayed frame to a timestamped BMP next to the
    /// working directory, mirroring the headless `--screenshot` writer.
    fn take_screenshot(&mut self) {
        let frame = self.emu.shared.frame.lock().unwrap();
        if frame.width == 0 || frame.height == 0 {
            tracing::warn!("no frame to screenshot yet");
            return;
        }
        let epoch = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!("screenshot_{epoch}.bmp");
        match crate::write_bmp(&path, frame.width, frame.height, &frame.rgba) {
            Ok(()) => {
                tracing::info!("screenshot written to {path}");
                self.last_screenshot = Some(path);
            }
            Err(e) => tracing::error!("screenshot failed: {e}"),
        }
    }
}

impl Drop for App {
    /// Persist settings changed from the UI. (The worker flushes the memory
    /// card itself when it stops.)
    fn drop(&mut self) {
        if let Some(path) = &self.config_path
            && (self.config.volume - self.volume).abs() > f32::EPSILON
        {
            self.config.volume = self.volume;
            self.config.save(path);
        }
    }
}

/// Convert a framebuffer snapshot (already RGBA8) to an egui image.
fn frame_image(frame: &FrameSnapshot) -> egui::ColorImage {
    let (w, h) = (frame.width as usize, frame.height as usize);
    if frame.rgba.len() < w * h * 4 {
        return egui::ColorImage::default(); // no frame captured yet
    }
    egui::ColorImage::from_rgba_unmultiplied([w, h], &frame.rgba)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let buttons = ctx.input(|i| {
            KEYMAP
                .iter()
                .filter(|(k, _)| i.key_down(*k))
                .fold(0u16, |acc, (_, b)| acc | b)
        });
        self.emu.shared.buttons.store(buttons, Ordering::Relaxed);
        self.emu
            .shared
            .volume
            .store(self.volume.to_bits(), Ordering::Relaxed);

        let status = self.emu.shared.status.lock().unwrap().clone();
        let debugger_active = self.emu.shared.debugger_active.load(Ordering::Relaxed);

        if ctx.input(|i| i.key_pressed(egui::Key::F12)) {
            self.take_screenshot();
        }

        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // The debugger owns run control while attached
                ui.add_enabled_ui(!debugger_active, |ui| {
                    let label = if status.running {
                        "\u{23f8} Pause"
                    } else {
                        "\u{25b6} Run"
                    };
                    if ui.button(label).clicked() {
                        self.emu.send(Command::SetRunning(!status.running));
                    }
                    if ui.button("Step").clicked() {
                        self.emu.send(Command::Step);
                    }
                    if ui.button("Reset").clicked() {
                        self.emu.send(Command::Reset);
                    }
                });
                ui.separator();
                if ui.button("Screenshot (F12)").clicked() {
                    self.take_screenshot();
                }
                if status.debugger != DebuggerState::None {
                    ui.separator();
                    ui.label(match status.debugger {
                        DebuggerState::Halted => "debugger: halted",
                        DebuggerState::Running => "debugger: running",
                        DebuggerState::Waiting => "waiting for debugger",
                        _ => "debugger: listening",
                    });
                }
                ui.separator();
                ui.checkbox(&mut self.linear_filter, "Linear filter");
                ui.separator();
                ui.checkbox(&mut self.show_tty, "TTY");
                ui.checkbox(&mut self.show_regs, "Registers");
                ui.separator();
                ui.label("volume");
                ui.add(
                    egui::Slider::new(&mut self.volume, 0.0..=1.0)
                        .show_value(false)
                        .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                );
                ui.separator();
                ui.monospace(format!(
                    "cycles {}   speed {:3.0}% ({:.0} fps)   audio {:3}ms{}",
                    status.cycles,
                    status.speed * 100.0,
                    status.speed * 60.0,
                    status.audio_buffered * 1000 / 48_000,
                    if status.audio_underruns > 0 {
                        format!("   underruns {}", status.audio_underruns)
                    } else {
                        String::new()
                    }
                ));
            });
        });

        if self.show_regs {
            egui::SidePanel::right("registers")
                .default_width(280.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("EE");
                        ui.monospace(format!("pc {:08x}", status.ee_pc));
                        egui::Grid::new("ee_regs").striped(true).show(ui, |ui| {
                            for (i, name) in REG_NAMES.iter().enumerate() {
                                let [lo, hi] = status.ee_gpr[i];
                                ui.monospace(format!("{name:>4}"));
                                ui.monospace(if hi != 0 {
                                    format!("{lo:016x}\n  hi:{hi:016x}")
                                } else {
                                    format!("{lo:016x}")
                                });
                                ui.end_row();
                            }
                            ui.monospace("  hi");
                            ui.monospace(format!("{:016x}:{:016x}", status.ee_hi[1], status.ee_hi[0]));
                            ui.end_row();
                            ui.monospace("  lo");
                            ui.monospace(format!("{:016x}:{:016x}", status.ee_lo[1], status.ee_lo[0]));
                            ui.end_row();
                            ui.monospace("status");
                            ui.monospace(format!("{:08x}", status.ee_cop0[COP0_STATUS]));
                            ui.end_row();
                            ui.monospace(" cause");
                            ui.monospace(format!("{:08x}", status.ee_cop0[COP0_CAUSE]));
                            ui.end_row();
                            ui.monospace("   epc");
                            ui.monospace(format!("{:08x}", status.ee_cop0[COP0_EPC]));
                            ui.end_row();
                        });

                        ui.separator();
                        ui.heading("IOP");
                        ui.monospace(format!("pc {:08x}", status.iop_pc));
                        egui::Grid::new("iop_regs").striped(true).show(ui, |ui| {
                            for (i, name) in REG_NAMES.iter().enumerate() {
                                ui.monospace(format!("{name:>4}"));
                                ui.monospace(format!("{:08x}", status.iop_gpr[i]));
                                if i % 2 == 1 {
                                    ui.end_row();
                                }
                            }
                            ui.monospace("  hi");
                            ui.monospace(format!("{:08x}", status.iop_hi));
                            ui.monospace("  lo");
                            ui.monospace(format!("{:08x}", status.iop_lo));
                            ui.end_row();
                            ui.monospace("status");
                            ui.monospace(format!("{:08x}", status.iop_cop0[COP0_STATUS]));
                            ui.monospace(" cause");
                            ui.monospace(format!("{:08x}", status.iop_cop0[COP0_CAUSE]));
                            ui.end_row();
                            ui.monospace("   epc");
                            ui.monospace(format!("{:08x}", status.iop_cop0[COP0_EPC]));
                            ui.end_row();
                        });
                    });
                });
        }

        if self.show_tty {
            egui::TopBottomPanel::bottom("tty")
                .resizable(true)
                .default_height(160.0)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("TTY");
                        if ui.button("Clear").clicked() {
                            self.emu.shared.tty.lock().unwrap().clear();
                        }
                    });
                    egui::ScrollArea::vertical()
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            let tty = self.emu.shared.tty.lock().unwrap().clone();
                            ui.add(
                                egui::TextEdit::multiline(&mut tty.as_str())
                                    .font(egui::TextStyle::Monospace)
                                    .desired_width(f32::INFINITY)
                                    .interactive(false),
                            );
                        });
                });
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let (width, height, image) = {
                let frame = self.emu.shared.frame.lock().unwrap();
                (frame.width, frame.height, frame_image(&frame))
            };
            if width == 0 || height == 0 {
                ui.centered_and_justified(|ui| ui.label("waiting for a frame..."));
                return;
            }
            let filter = if self.linear_filter {
                egui::TextureOptions::LINEAR
            } else {
                egui::TextureOptions::NEAREST
            };
            let tex = match &mut self.display_tex {
                Some(t) => {
                    t.set(image, filter);
                    t.clone()
                }
                None => {
                    let t = ui.ctx().load_texture("display", image, filter);
                    self.display_tex = Some(t.clone());
                    t
                }
            };
            // Fit the panel while keeping the framebuffer's own aspect ratio.
            let avail = ui.available_size();
            let aspect = width as f32 / height as f32;
            let scale = (avail.x / aspect).min(avail.y);
            let size = egui::Vec2::new(scale * aspect, scale);
            ui.centered_and_justified(|ui| {
                ui.add(egui::Image::new(&tex).fit_to_exact_size(size));
            });
        });
    }
}
