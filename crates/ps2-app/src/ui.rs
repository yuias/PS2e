//! egui shell: a thin client over the emulator worker thread.
//!
//! All emulation (and audio) lives in [`crate::emu`]; this module only sends
//! commands, reads published snapshots and draws. Keeping it presentation-only
//! is deliberate — a wasm frontend could reuse the same snapshot types.

use crate::config::Config;
use crate::emu::{Command, DebuggerState, Disc, Emu, MemoryView, PANEL_MEMORY, PANEL_REGS};
use crate::scan;
use eframe::egui;
use crate::gamepad::Gamepad;
use std::path::PathBuf;
use std::sync::atomic::Ordering;

/// Resolve the configured key names to egui keys once at startup. An
/// unknown name falls back to the default for that button rather than
/// leaving it unbound.
fn resolve_keymap(keys: &crate::config::KeyBindings) -> Vec<(egui::Key, u16)> {
    let fallback = crate::config::KeyBindings::default();
    keys.pairs()
        .into_iter()
        .zip(fallback.pairs())
        .filter_map(|((name, bit), (default_name, _))| match egui::Key::from_name(name) {
            Some(key) => Some((key, bit)),
            None => {
                tracing::warn!("unknown key name '{name}'; using '{default_name}'");
                egui::Key::from_name(default_name).map(|key| (key, bit))
            }
        })
        .collect()
}

/// Window title with no disc in the drive; an inserted one is appended.
pub const WINDOW_TITLE: &str = "PS2e";

/// Pad button names, index-aligned with [`crate::config::KeyBindings::pairs`],
/// for listing the bindings in the Help menu.
const BUTTON_NAMES: [&str; 16] = [
    "up", "down", "left", "right", "cross", "circle", "square", "triangle", "L1", "L2", "R1", "R2",
    "L3", "R3", "start", "select",
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
/// An address as typed into the memory panel: always hex, with or
/// without a `0x` prefix.
fn parse_addr(text: &str) -> Option<u32> {
    let t = text.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    u32::from_str_radix(t, 16).ok()
}

/// A value as typed into the scanner: hex with a `0x` prefix, otherwise
/// decimal, which is how a score or a hit-point count is read off the
/// screen.
fn parse_value(text: &str) -> Option<u64> {
    let t = text.trim();
    match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => t.parse().ok(),
    }
}

#[cfg(test)]
mod number_tests {
    use super::*;

    #[test]
    fn addresses_are_hex_and_values_are_decimal_unless_prefixed() {
        assert_eq!(parse_addr("1000"), Some(0x1000));
        assert_eq!(parse_addr("0x00100000"), Some(0x10_0000));
        assert_eq!(parse_addr("zz"), None);
        assert_eq!(parse_value("12345"), Some(12345));
        assert_eq!(parse_value("65535"), Some(65535));
        assert_eq!(parse_value("0x10"), Some(16));
        assert_eq!(parse_value(""), None);
    }
}

const COP0_STATUS: usize = 12;
const COP0_CAUSE: usize = 13;
const COP0_EPC: usize = 14;

pub struct App {
    emu: Emu,
    scale_mode: crate::display::ScaleMode,
    deinterlace: crate::config::DeinterlaceSetting,
    swap_fields: bool,
    cheats: bool,
    internal_2x: bool,
    /// Master volume applied on top of the SPU2 output (0..=1).
    volume: f32,
    config: Config,
    config_path: Option<PathBuf>,
    last_screenshot: Option<String>,
    show_tty: bool,
    show_regs: bool,
    show_mem: bool,
    /// Memory panel state: the viewer's target and address (as typed),
    /// the scanner's width and value (as typed).
    mem_target: scan::Target,
    mem_addr: String,
    scan_width: u8,
    scan_value: String,
    fullscreen: bool,
    /// Key -> pad bit, resolved from the config once at startup.
    keymap: Vec<(egui::Key, u16)>,
    /// Absent when no gamepad backend is available.
    gamepad: Option<Gamepad>,
    /// Last failed disc pick, shown in the status bar until the next one.
    disc_error: Option<String>,
    hotkey_save: Option<egui::Key>,
    hotkey_load: Option<egui::Key>,
    /// Window title last sent to the viewport, so it is only pushed on a
    /// change rather than every frame.
    title: String,
}

impl App {
    pub fn new(emu: Emu, config: Config, config_path: Option<PathBuf>) -> Self {
        let volume = config.volume.clamp(0.0, 1.0);
        let keymap = resolve_keymap(&config.keys);
        let gamepad = Gamepad::new(&config.pad);
        let hotkey_save = egui::Key::from_name(&config.hotkeys.save_state);
        let hotkey_load = egui::Key::from_name(&config.hotkeys.load_state);
        Self {
            emu,
            scale_mode: config.scaler,
            deinterlace: config.deinterlace,
            swap_fields: config.swap_fields,
            cheats: config.cheats,
            internal_2x: config.internal_2x,
            volume,
            config,
            config_path,
            last_screenshot: None,
            show_tty: false,
            show_regs: false,
            show_mem: false,
            mem_target: scan::Target::Ee,
            mem_addr: "00100000".into(),
            scan_width: 4,
            scan_value: String::new(),
            fullscreen: false,
            keymap,
            gamepad,
            disc_error: None,
            hotkey_save,
            hotkey_load,
            title: WINDOW_TITLE.to_string(),
        }
    }

    /// Swap the disc the way the console does: the drive opens, the file
    /// picker comes up, and the drive closes on whatever was picked —
    /// cancelling puts the old disc back. Emulation never stops, so a
    /// multi-disc title can change discs where it asks you to.
    fn insert_disc(&mut self) {
        self.emu.send(Command::OpenTray);
        let disc = self.pick_disc();
        self.emu.send(Command::CloseTray(disc));
    }

    /// Put a disc in and power-cycle onto it, the way the console boots one
    /// that is already in the drive.
    fn boot_disc(&mut self) {
        if let Some(disc) = self.pick_disc() {
            self.emu.send(Command::BootDisc(Some(disc)));
        }
    }

    /// The memory viewer (one window of RAM, refreshed with the frame)
    /// and the scanner (find a value, narrow it down as it moves).
    fn memory_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.mem_target, scan::Target::Ee, "EE");
            ui.radio_value(&mut self.mem_target, scan::Target::Iop, "IOP");
            ui.label("address");
            ui.add(egui::TextEdit::singleline(&mut self.mem_addr).desired_width(90.0).font(egui::TextStyle::Monospace));
        });
        let view: MemoryView = self.emu.shared.memory.lock().unwrap().clone();
        if view.target == Some(self.mem_target) && !view.bytes.is_empty() {
            let mut text = String::new();
            for (row, chunk) in view.bytes.chunks(16).enumerate() {
                let addr = view.base + row as u32 * 16;
                text.push_str(&format!("{addr:08x} "));
                for (i, b) in chunk.iter().enumerate() {
                    text.push_str(&format!(" {b:02x}"));
                    if i == 7 {
                        text.push(' ');
                    }
                }
                text.push_str("  ");
                text.extend(chunk.iter().map(|&b| if (0x20..0x7F).contains(&b) { b as char } else { '.' }));
                text.push('\n');
            }
            ui.add(egui::Label::new(egui::RichText::new(text).monospace()).wrap_mode(egui::TextWrapMode::Extend));
        } else {
            ui.label("waiting for the worker");
        }

        ui.separator();
        ui.heading("Scan");
        ui.horizontal(|ui| {
            ui.label("width");
            for w in [1u8, 2, 4] {
                ui.radio_value(&mut self.scan_width, w, w.to_string());
            }
            ui.label("value");
            ui.add(egui::TextEdit::singleline(&mut self.scan_value).desired_width(110.0).font(egui::TextStyle::Monospace))
                .on_hover_text("decimal, or hex with a 0x prefix");
        });
        let value = parse_value(&self.scan_value);
        let mut request = None;
        ui.horizontal(|ui| {
            ui.label("new scan");
            if ui.add_enabled(value.is_some(), egui::Button::new("= value")).clicked() {
                request = Some((scan::Filter::Exact(value.unwrap()), true));
            }
            if ui.button("unknown value").clicked() {
                request = Some((scan::Filter::Unknown, true));
            }
        });
        let result = self.emu.shared.scan.lock().unwrap().clone();
        let live = result.target == Some(self.mem_target) && result.width == self.scan_width;
        ui.horizontal(|ui| {
            ui.label("narrow");
            ui.add_enabled_ui(live, |ui| {
                if ui.add_enabled(value.is_some(), egui::Button::new("= value")).clicked() {
                    request = Some((scan::Filter::Exact(value.unwrap()), false));
                }
                for (label, filter) in [
                    ("changed", scan::Filter::Changed),
                    ("unchanged", scan::Filter::Unchanged),
                    ("increased", scan::Filter::Increased),
                    ("decreased", scan::Filter::Decreased),
                ] {
                    if ui.button(label).clicked() {
                        request = Some((filter, false));
                    }
                }
            });
        });
        if let Some((filter, restart)) = request {
            self.emu.send(Command::Scan(scan::Request {
                target: self.mem_target,
                width: self.scan_width,
                filter,
                restart,
            }));
        }
        if live {
            ui.label(format!("{} candidate{}", result.count, if result.count == 1 { "" } else { "s" }));
            let digits = usize::from(result.width) * 2;
            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                for &(addr, value) in &result.hits {
                    let line = format!("{addr:08x}  {value:0digits$x}  {value}");
                    if ui
                        .add(egui::Label::new(egui::RichText::new(line).monospace()).sense(egui::Sense::click()))
                        .on_hover_text("show in the viewer")
                        .clicked()
                    {
                        self.mem_addr = format!("{:08x}", addr & !0xF);
                    }
                }
                if result.count > result.hits.len() {
                    ui.label(format!("... and {} more", result.count - result.hits.len()));
                }
            });
        }
    }

    /// File picker; `None` on cancel or on a file that will not open.
    fn pick_disc(&mut self) -> Option<Disc> {
        let path = rfd::FileDialog::new()
            .add_filter("PlayStation 2 disc image", &["iso", "img", "bin"])
            .pick_file()?;
        match Disc::open(&path) {
            Ok(disc) => {
                self.disc_error = None;
                tracing::info!(path = %path.display(), "disc loaded");
                Some(disc)
            }
            Err(e) => {
                let msg = format!("cannot open {}: {e}", path.display());
                tracing::error!("{msg}");
                self.disc_error = Some(msg);
                None
            }
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
                || self.config.cheats != self.cheats
                || self.config.internal_2x != self.internal_2x)
        {
            self.config.volume = self.volume;
            self.config.scaler = self.scale_mode;
            self.config.deinterlace = self.deinterlace;
            self.config.swap_fields = self.swap_fields;
            self.config.cheats = self.cheats;
            self.config.internal_2x = self.internal_2x;
            self.config.save(path);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let buttons = ctx.input(|i| {
            self.keymap
                .iter()
                .filter(|(k, _)| i.key_down(*k))
                .fold(0u16, |acc, (_, b)| acc | b)
        });
        let (pad_buttons, sticks) = self
            .gamepad
            .as_mut()
            .map_or((0, crate::emu::STICKS_CENTRED), Gamepad::poll);
        self.emu
            .shared
            .pad
            .store(crate::emu::pack_pad(buttons | pad_buttons, sticks), Ordering::Relaxed);
        self.emu
            .shared
            .volume
            .store(self.volume.to_bits(), Ordering::Relaxed);
        self.emu.shared.deinterlace.store(self.deinterlace.index(), Ordering::Relaxed);
        self.emu.shared.swap_fields.store(self.swap_fields, Ordering::Relaxed);
        self.emu.shared.cheats.store(self.cheats, Ordering::Relaxed);
        // The worker does the work behind a debug panel only while it is
        // open; a closed one costs nothing but this store.
        let chrome_now = !self.fullscreen;
        let mut panels = 0;
        if chrome_now && self.show_regs {
            panels |= PANEL_REGS;
        }
        if chrome_now && self.show_mem {
            panels |= PANEL_MEMORY;
        }
        self.emu.shared.panels.store(panels, Ordering::Relaxed);
        if panels & PANEL_MEMORY != 0 {
            let base = parse_addr(&self.mem_addr).unwrap_or(0);
            self.emu.shared.view.store(crate::emu::pack_view(self.mem_target, base), Ordering::Relaxed);
        }
        self.emu.shared.internal_2x.store(self.internal_2x, Ordering::Relaxed);

        let status = self.emu.shared.status.lock().unwrap().clone();
        let disc = self.emu.shared.disc.lock().unwrap().clone();
        let debugger_active = self.emu.shared.debugger_active.load(Ordering::Relaxed);

        // A PS2 disc carries no printable title, so the boot serial stands in
        // for one; the file name covers images the serial cannot be read from.
        let title = match &disc {
            Some(d) => format!("{WINDOW_TITLE} - {}", d.serial.as_deref().unwrap_or(&d.name)),
            None => WINDOW_TITLE.to_string(),
        };
        if title != self.title {
            self.title = title.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        }

        if ctx.input(|i| i.key_pressed(egui::Key::F12)) {
            self.take_screenshot();
        }

        // Save/load shortcuts; the debugger owns execution while attached.
        if !debugger_active {
            if self.hotkey_save.is_some_and(|k| ctx.input(|i| i.key_pressed(k))) {
                self.emu.send(Command::SaveState);
            }
            if self.hotkey_load.is_some_and(|k| ctx.input(|i| i.key_pressed(k))) {
                self.emu.send(Command::LoadState);
            }
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
                            if ui
                                .button("Reset")
                                .on_hover_text(
                                    "power-cycle the console; the disc and memory card stay in",
                                )
                                .clicked()
                            {
                                self.emu.send(Command::Reset);
                                ui.close();
                            }
                            ui.separator();
                            if ui
                                .button("Insert disc...")
                                .on_hover_text(
                                    "opens the drive and closes it on the new image;                                      swapping mid-game works, no reset needed",
                                )
                                .clicked()
                            {
                                self.insert_disc();
                                ui.close();
                            }
                            let count = disc.as_ref().map_or(0, |d| d.cheats);
                            ui.add_enabled(count > 0, egui::Checkbox::new(&mut self.cheats, format!("Cheats ({count})")))
                                .on_hover_text("apply the patches in <image>.pnach next to the disc image");
                            ui.separator();
                            let save = &self.config.hotkeys.save_state;
                            if ui.button(format!("Save state	{save}")).clicked() {
                                self.emu.send(Command::SaveState);
                                ui.close();
                            }
                            let load = &self.config.hotkeys.load_state;
                            if ui.button(format!("Load state	{load}")).clicked() {
                                self.emu.send(Command::LoadState);
                                ui.close();
                            }
                            ui.separator();
                            if ui
                                .button("Boot disc...")
                                .on_hover_text(
                                    "power-cycle onto a disc, the way the console starts                                      one that is already in the drive",
                                )
                                .clicked()
                            {
                                self.boot_disc();
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
                        ui.checkbox(&mut self.show_mem, "Memory panel");
                    });
                    ui.menu_button("Audio", |ui| {
                        ui.add(
                            egui::Slider::new(&mut self.volume, 0.0..=1.0)
                                .text("volume")
                                .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
                        );
                    });
                    ui.menu_button("Help", |ui| {
                        ui.label("Pad, as bound in the config file:");
                        for ((name, (key, _)), (btn, _)) in BUTTON_NAMES
                            .iter()
                            .zip(self.config.keys.pairs())
                            .zip(self.config.pad.pairs())
                        {
                            ui.monospace(format!("{name:>8} = {key} / {btn}"));
                        }
                        ui.separator();
                        ui.monospace(format!("    save = {}", self.config.hotkeys.save_state));
                        ui.monospace(format!("    load = {}", self.config.hotkeys.load_state));
                        ui.separator();
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
                    // The image name sits next to the run state: it is what
                    // identifies the session at a glance.
                    if let Some(d) = &disc {
                        ui.separator();
                        ui.monospace(&d.name);
                    }
                    ui.separator();
                    ui.monospace(format!(
                        "speed {:3.0}% ({:.0} fps)   audio {:3} ms{}",
                        status.speed * 100.0,
                        status.speed * status.region.refresh_hz(),
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
                    if let Some(err) = &self.disc_error {
                        ui.separator();
                        ui.colored_label(egui::Color32::LIGHT_RED, err);
                    }
                    if let Some((text, failed)) = &*self.emu.shared.notice.lock().unwrap() {
                        ui.separator();
                        if *failed {
                            ui.colored_label(egui::Color32::LIGHT_RED, text);
                        } else {
                            ui.monospace(text);
                        }
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

        if chrome && self.show_mem {
            egui::SidePanel::right("memory")
                .default_width(460.0)
                .show(ctx, |ui| self.memory_panel(ui));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_default_binding_resolves() {
        let keys = crate::config::KeyBindings::default();
        let map = resolve_keymap(&keys);
        assert_eq!(map.len(), BUTTON_NAMES.len());
        // Distinct keys, and every pad bit covered exactly once.
        let bits = map.iter().fold(0u16, |acc, (_, b)| acc | b);
        assert_eq!(bits, u16::MAX);
    }
}
