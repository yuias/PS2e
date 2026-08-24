//! egui shell: a thin client over the emulator worker thread.
//!
//! All emulation (and audio) lives in [`crate::emu`]; this module only sends
//! commands, reads published snapshots and draws. Keeping it presentation-only
//! is deliberate — a wasm frontend could reuse the same snapshot types.

use crate::config::Config;
use crate::emu::{Command, DebuggerState, Emu};
use crate::pad;
use eframe::egui;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// Keyboard -> digital pad mapping: face buttons on S/D/X/Z laid out like
/// the pad (square left, triangle up, circle right, cross down), shoulders
/// on the row above, and the d-pad on the arrow keys.
const KEYMAP: [(egui::Key, u16); 16] = [
    (egui::Key::ArrowUp, pad::UP),
    (egui::Key::ArrowDown, pad::DOWN),
    (egui::Key::ArrowLeft, pad::LEFT),
    (egui::Key::ArrowRight, pad::RIGHT),
    (egui::Key::Z, pad::CROSS),
    (egui::Key::X, pad::CIRCLE),
    (egui::Key::S, pad::SQUARE),
    (egui::Key::D, pad::TRIANGLE),
    (egui::Key::W, pad::L1),
    (egui::Key::E, pad::L2),
    (egui::Key::R, pad::R1),
    (egui::Key::U, pad::R2),
    (egui::Key::Num1, pad::L3),
    (egui::Key::Num3, pad::R3),
    (egui::Key::V, pad::START),
    (egui::Key::C, pad::SELECT),
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
    scale_mode: crate::display::ScaleMode,
    deinterlace: crate::config::DeinterlaceSetting,
    swap_fields: bool,
    internal_2x: bool,
    /// Master volume applied on top of the SPU2 output (0..=1).
    volume: f32,
    config: Config,
    config_path: Option<PathBuf>,
    last_screenshot: Option<String>,
    show_tty: bool,
    show_regs: bool,
    fullscreen: bool,
}

impl App {
    pub fn new(emu: Emu, config: Config, config_path: Option<PathBuf>) -> Self {
        let volume = config.volume.clamp(0.0, 1.0);
        Self {
            emu,
            scale_mode: config.scaler,
            deinterlace: config.deinterlace,
            swap_fields: config.swap_fields,
            internal_2x: config.internal_2x,
            volume,
            config,
            config_path,
            last_screenshot: None,
            show_tty: false,
            show_regs: false,
            fullscreen: false,
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
            && ((self.config.volume - self.volume).abs() > f32::EPSILON
                || self.config.scaler != self.scale_mode
                || self.config.deinterlace != self.deinterlace
                || self.config.swap_fields != self.swap_fields
                || self.config.internal_2x != self.internal_2x)
        {
            self.config.volume = self.volume;
            self.config.scaler = self.scale_mode;
            self.config.deinterlace = self.deinterlace;
            self.config.swap_fields = self.swap_fields;
            self.config.internal_2x = self.internal_2x;
            self.config.save(path);
        }
    }
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
        self.emu.shared.deinterlace.store(self.deinterlace.index(), Ordering::Relaxed);
        self.emu.shared.swap_fields.store(self.swap_fields, Ordering::Relaxed);
        self.emu.shared.internal_2x.store(self.internal_2x, Ordering::Relaxed);

        let status = self.emu.shared.status.lock().unwrap().clone();
        let debugger_active = self.emu.shared.debugger_active.load(Ordering::Relaxed);

        if ctx.input(|i| i.key_pressed(egui::Key::F12)) {
            self.take_screenshot();
        }

        // F11 toggles fullscreen; the chrome (menu, status bar, panels)
        // hides while fullscreen so only the display shows.
        if ctx.input(|i| i.key_pressed(egui::Key::F11)) {
            self.fullscreen = !self.fullscreen;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
        }
        if self.fullscreen && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.fullscreen = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        }
        let chrome = !self.fullscreen;

        if chrome {
            egui::TopBottomPanel::top("menu").show(ctx, |ui| {
                egui::MenuBar::new().ui(ui, |ui| {
                    ui.menu_button("Emulation", |ui| {
                        // The debugger owns run control while attached.
                        ui.add_enabled_ui(!debugger_active, |ui| {
                            let label = if status.running { "Pause" } else { "Run" };
                            if ui.button(label).clicked() {
                                self.emu.send(Command::SetRunning(!status.running));
                                ui.close();
                            }
                            if ui.button("Step").clicked() {
                                self.emu.send(Command::Step);
                                ui.close();
                            }
                            if ui.button("Reset").clicked() {
                                self.emu.send(Command::Reset);
                                ui.close();
                            }
                        });
                        ui.separator();
                        if ui.button("Screenshot	F12").clicked() {
                            self.take_screenshot();
                            ui.close();
                        }
                    });
                    ui.menu_button("View", |ui| {
                        if ui.button("Fullscreen	F11").clicked() {
                            self.fullscreen = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(true));
                            ui.close();
                        }
                        ui.separator();
                        ui.label("Scaler");
                        for mode in crate::display::ScaleMode::ALL {
                            ui.radio_value(&mut self.scale_mode, mode, mode.label());
                        }
                        ui.separator();
                        ui.label("Deinterlace");
                        for mode in crate::config::DeinterlaceSetting::ALL {
                            ui.radio_value(&mut self.deinterlace, mode, mode.label());
                        }
                        ui.checkbox(&mut self.swap_fields, "Swap field order");
                        ui.separator();
                        ui.checkbox(&mut self.internal_2x, "Internal 2x resolution");
                        ui.separator();
                        ui.checkbox(&mut self.show_tty, "TTY panel");
                        ui.checkbox(&mut self.show_regs, "Registers panel");
                    });
                    ui.menu_button("Audio", |ui| {
                        ui.add(
                            egui::Slider::new(&mut self.volume, 0.0..=1.0)
                                .text("volume")
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                        );
                    });
                    ui.menu_button("Help", |ui| {
                        ui.label("Pad: arrows = d-pad, Z/X/C/V = cross/circle/square/triangle,");
                        ui.label("Q/W/E/R = L1/R1/L2/R2, Enter = start, Backspace = select.");
                        ui.label("F11 fullscreen (Esc leaves), F12 screenshot.");
                    });
                });
            });

            egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let state = match status.debugger {
                        DebuggerState::Halted => "debugger: halted",
                        DebuggerState::Running => "debugger: running",
                        DebuggerState::Waiting => "waiting for debugger",
                        DebuggerState::None if status.running => "running",
                        DebuggerState::None => "paused",
                        _ => "debugger: listening",
                    };
                    ui.monospace(state);
                    ui.separator();
                    ui.monospace(format!(
                        "speed {:3.0}% ({:.0} fps)   audio {:3} ms{}",
                        status.speed * 100.0,
                        status.speed * 60.0,
                        status.audio_buffered * 1000 / 48_000,
                        if status.audio_underruns > 0 {
                            format!("   underruns {}", status.audio_underruns)
                        } else {
                            String::new()
                        }
                    ));
                    ui.separator();
                    ui.monospace(format!("cycles {}", status.cycles));
                    if let Some(path) = &self.last_screenshot {
                        ui.separator();
                        ui.monospace(format!("saved {path}"));
                    }
                });
            });
        }

        if chrome && self.show_regs {
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

        if chrome && self.show_tty {
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

        let central = if self.fullscreen {
            egui::CentralPanel::default().frame(egui::Frame::NONE.fill(egui::Color32::BLACK))
        } else {
            egui::CentralPanel::default()
        };
        central.show(ctx, |ui| {
            let (width, height, rgba, seq) = {
                let frame = self.emu.shared.frame.lock().unwrap();
                (frame.width, frame.height, frame.rgba.clone(), frame.seq)
            };
            if width == 0 || height == 0 {
                ui.centered_and_justified(|ui| ui.label("waiting for a frame..."));
                return;
            }
            // Fit the panel while keeping the framebuffer's own aspect ratio.
            let avail = ui.available_size();
            let aspect = width as f32 / height as f32;
            let scale = (avail.x / aspect).min(avail.y);
            let size = egui::Vec2::new(scale * aspect, scale);
            let rect = egui::Rect::from_center_size(ui.available_rect_before_wrap().center(), size);
            let ppp = ui.ctx().pixels_per_point();
            ui.painter().add(eframe::egui_wgpu::Callback::new_paint_callback(
                rect,
                crate::display::DisplayCallback {
                    rgba,
                    width,
                    height,
                    seq,
                    mode: self.scale_mode,
                    dst_size: [size.x * ppp, size.y * ppp],
                },
            ));
        });
    }
}
