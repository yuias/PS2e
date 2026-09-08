//! egui shell: a thin client over the emulator worker thread.
//!
//! All emulation (and audio) lives in [`crate::emu`]; this module only sends
//! commands, reads published snapshots and draws. Keeping it presentation-only
//! is deliberate — a wasm frontend could reuse the same snapshot types.

use crate::cheatfile;
use crate::config::Config;
use crate::emu::{Command, DebuggerState, Disc, DiscInfo, Emu, MemoryView, PANEL_MEMORY, PANEL_REGS, Status};
use crate::keymap::{self, BUTTON_NAMES};
use crate::scan;
use ps2_core::cheats::Group;
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

/// MIPS GPR names, index-aligned with `Cpu::gpr` (shared by EE and IOP).
const REG_NAMES: [&str; 32] = [
    "zero", "at", "v0", "v1", "a0", "a1", "a2", "a3", //
    "t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", //
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", //
    "t8", "t9", "k0", "k1", "gp", "sp", "fp", "ra",
];

// Standard MIPS COP0 register numbers; identical on the EE (see
// ps2_core::ee::cop0) and the IOP's private constants of the same values.
/// An address as typed into the memory page: always hex, with or
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

/// One page of the side pane. Exactly one is drawn at a time, and only
/// that one's data is published by the worker, so an inactive page costs
/// as little as a closed panel used to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Settings,
    Cheats,
    Memory,
    Registers,
}

impl Page {
    const ALL: [Page; 4] = [Page::Settings, Page::Cheats, Page::Memory, Page::Registers];

    fn label(self) -> &'static str {
        match self {
            Page::Settings => "Settings",
            Page::Cheats => "Cheats",
            Page::Memory => "Memory",
            Page::Registers => "Registers",
        }
    }

    /// What the worker has to publish for this page, as `PANEL_*` bits.
    fn panels(self) -> u8 {
        match self {
            // The cheat list is the UI's own, so the worker owes it nothing.
            Page::Settings | Page::Cheats => 0,
            Page::Memory => PANEL_MEMORY,
            Page::Registers => PANEL_REGS,
        }
    }
}

/// A settings dropdown over the fixed set of values a setting can take.
fn combo<T: Copy + PartialEq>(
    ui: &mut egui::Ui,
    id: &str,
    current: &mut T,
    all: &[T],
    label: impl Fn(T) -> &'static str,
) {
    egui::ComboBox::from_id_salt(id)
        .selected_text(label(*current))
        .width(ui.available_width())
        .show_ui(ui, |ui| {
            for &value in all {
                ui.selectable_value(current, value, label(value));
            }
        });
}

/// Both register files, as last published by the worker. The snapshot is
/// only refreshed while this page is the one showing.
fn registers_page(ui: &mut egui::Ui, status: &Status) {
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
}

pub struct App {
    emu: Emu,
    scale_mode: crate::display::ScaleMode,
    aspect: crate::config::AspectSetting,
    deinterlace: crate::config::DeinterlaceSetting,
    swap_fields: bool,
    /// Master cheat switch, mirrored into [`crate::emu::Shared::cheats`].
    cheats_on: bool,
    /// Cheats for the disc in the drive and the pnach they were read
    /// from. The list is the UI's: the worker gets it through
    /// [`Command::SetCheats`].
    cheats: Vec<Group>,
    cheat_file: Option<PathBuf>,
    /// The `cheats.toml` key the current list is keyed under. It starts
    /// as the pnach's file stem and becomes the boot serial as soon as
    /// the worker has read one off the disc, which is a frame or two
    /// after the disc goes in.
    cheat_key: Option<String>,
    cheat_store: cheatfile::Store,
    cheats_toml: PathBuf,
    internal_2x: bool,
    /// Master volume applied on top of the SPU2 output (0..=1).
    volume: f32,
    config: Config,
    config_path: Option<PathBuf>,
    last_screenshot: Option<String>,
    show_tty: bool,
    /// The side pane, and which of its pages is showing. The width is kept
    /// here rather than left to egui: the persistence feature is not
    /// compiled in, so egui would forget it at exit.
    show_pane: bool,
    page: Page,
    pane_width: f32,
    /// The display rect and its aspect, as the central panel last laid
    /// them out. The window's chrome is whatever the window has that this
    /// rect does not, which is how [`Self::resize_to`] is honoured without
    /// having to know any panel's height.
    central_size: egui::Vec2,
    display_aspect: f32,
    /// Window size as of the last windowed frame, for the config. Read
    /// back from egui rather than tracked through resize events, and only
    /// while not fullscreen, where it would be the whole monitor.
    window_size: egui::Vec2,
    /// Pending "display size" request: the height in physical pixels the
    /// display should be resized to.
    resize_to: Option<u32>,
    /// Memory page state: the viewer's target and address (as typed),
    /// the scanner's width and value (as typed).
    mem_target: scan::Target,
    mem_addr: String,
    scan_width: u8,
    scan_value: String,
    fullscreen: bool,
    /// Key -> pad bit, resolved from `keys` whenever it changes.
    keymap: Vec<(egui::Key, u16)>,
    /// The live keyboard bindings. Kept out of `config` so `Drop` can see
    /// that they changed, the way every other UI-owned setting works.
    keys: crate::config::KeyBindings,
    /// The binding dialog, while it is open.
    binder: Option<keymap::Binder>,
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
    /// `disc` is the image `--disc` put in the drive, if any: the Cheats
    /// page needs the pnach beside it the same way an inserted disc's does.
    pub fn new(emu: Emu, config: Config, config_path: Option<PathBuf>, disc: Option<&std::path::Path>) -> Self {
        let volume = config.volume.clamp(0.0, 1.0);
        let keys = config.keys.clone();
        let keymap = resolve_keymap(&keys);
        let gamepad = Gamepad::new(&config.pad);
        let hotkey_save = egui::Key::from_name(&config.hotkeys.save_state);
        let hotkey_load = egui::Key::from_name(&config.hotkeys.load_state);
        let show_pane = config.pane;
        let pane_width = config.pane_width;
        let window_size = egui::vec2(config.window_width, config.window_height);
        let cheats_toml = Config::cheats_path(config_path.as_ref());
        let cheat_store = cheatfile::Store::load(&cheats_toml);
        let cheat_file = disc.map(cheatfile::path_for);
        let cheats = cheat_file.as_deref().map(cheatfile::load).unwrap_or_default();
        Self {
            emu,
            scale_mode: config.scaler,
            aspect: config.aspect,
            deinterlace: config.deinterlace,
            swap_fields: config.swap_fields,
            cheats_on: config.cheats,
            cheats,
            cheat_file,
            cheat_key: None,
            cheat_store,
            cheats_toml,
            internal_2x: config.internal_2x,
            volume,
            config,
            config_path,
            last_screenshot: None,
            show_tty: false,
            show_pane,
            page: Page::Settings,
            pane_width,
            central_size: egui::Vec2::ZERO,
            display_aspect: 4.0 / 3.0,
            window_size,
            resize_to: None,
            mem_target: scan::Target::Ee,
            mem_addr: "00100000".into(),
            scan_width: 4,
            scan_value: String::new(),
            fullscreen: false,
            keymap,
            keys,
            binder: None,
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

    /// Settle which `cheats.toml` entry the current disc uses. The boot
    /// serial only arrives once the worker has read the disc, so the list
    /// runs under the pnach's file stem for the frame or two before that,
    /// then switches over and is re-sent.
    fn sync_cheat_key(&mut self, disc: Option<&DiscInfo>) {
        let (Some(path), Some(disc)) = (&self.cheat_file, disc) else { return };
        let key = cheatfile::key(disc.serial.as_deref(), path);
        if self.cheat_key.as_deref() == Some(key.as_str()) {
            return;
        }
        self.cheat_store.apply(&key, &mut self.cheats);
        self.cheat_key = Some(key);
        self.emu.send(Command::SetCheats(self.cheats.clone()));
    }

    /// Cheats page: the master switch, then one checkbox per section of the
    /// disc's pnach.
    fn cheats_page(&mut self, ui: &mut egui::Ui, disc: Option<&DiscInfo>) {
        ui.checkbox(&mut self.cheats_on, "Apply cheats")
            .on_hover_text("nothing below does anything until this is on");
        ui.separator();
        let (Some(path), true) = (self.cheat_file.clone(), disc.is_some()) else {
            ui.label("No disc in the drive.");
            return;
        };
        if self.cheats.is_empty() {
            ui.label("No cheats for this disc.");
            self.cheat_footer(ui, &path);
            return;
        }

        // The per-cheat boxes stay usable while the master switch is off:
        // setting a list up before switching it on is the normal order.
        let mut changed = None;
        egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
            for cheat in &mut self.cheats {
                let name = if cheat.name.is_empty() { "(unnamed)" } else { cheat.name.as_str() };
                if ui.checkbox(&mut cheat.enabled, name).changed() {
                    changed = Some((cheat.name.clone(), cheat.enabled));
                }
                for w in &cheat.warnings {
                    ui.label(egui::RichText::new(format!("      skipped {w}")).weak().small());
                }
            }
        });
        if let Some((name, on)) = changed {
            // A pnach may open the same section twice. The core switches
            // every group of that name together, so the list has to as
            // well, or what gets written out is half on and half off.
            for g in self.cheats.iter_mut().filter(|g| g.name == name) {
                g.enabled = on;
            }
            self.emu.send(Command::SetCheatEnabled(name, on));
            self.save_cheat_state();
        }
        ui.separator();
        self.cheat_footer(ui, &path);
    }

    fn cheat_footer(&mut self, ui: &mut egui::Ui, path: &std::path::Path) {
        ui.horizontal(|ui| {
            if ui.button("Reload").on_hover_text("re-read the pnach from disk").clicked() {
                self.reload_cheats();
            }
            ui.label(egui::RichText::new(path.display().to_string()).weak().monospace());
        });
    }

    /// Re-read the pnach and install it. This re-arms every one-shot
    /// command, which is what a reload is for.
    fn reload_cheats(&mut self) {
        let Some(path) = self.cheat_file.clone() else { return };
        self.cheats = cheatfile::load(&path);
        if let Some(key) = &self.cheat_key {
            self.cheat_store.apply(key, &mut self.cheats);
        }
        self.emu.send(Command::SetCheats(self.cheats.clone()));
    }

    fn save_cheat_state(&mut self) {
        let Some(key) = &self.cheat_key else { return };
        self.cheat_store.update(key, &self.cheats);
        self.cheat_store.save(&self.cheats_toml);
    }

    /// Turn a scanner hit into a cheat that writes the value back every
    /// frame. The section is named after the address and width, so a second
    /// press on the same hit rewrites it with the current value rather than
    /// stacking another section: the list has no delete, and pressing again
    /// is what a user does when the value has moved on.
    fn keep_value(&mut self, addr: u32, value: u64, width: u8) {
        let Some(path) = self.cheat_file.clone() else { return };
        let target = self.mem_target.into();
        let name = cheatfile::scan_name(target, addr, width);
        let span = self.cheats.iter().find(|g| g.name == name).map(|g| g.span);
        let body = cheatfile::constant_write(&name, target, addr, value, width);
        if let Err(e) = cheatfile::write_section(&path, span, &body) {
            self.notify(format!("cannot write {}: {e}", path.display()), true);
            return;
        }
        // Re-read rather than patch the model: the edit moved the spans of
        // everything after it.
        self.reload_cheats();
        let note = match self.cheats_on {
            true => format!("added '{name}'"),
            // The new cheat is in the file and doing nothing, which is easy
            // to mistake for the write having failed.
            false => format!("added '{name}' - 'Apply cheats' is off"),
        };
        self.notify(note, false);
    }

    fn notify(&mut self, text: String, failed: bool) {
        *self.emu.shared.notice.lock().unwrap() = Some((text, failed));
    }

    /// Video and audio settings: what the View and Audio menus used to
    /// carry. Dropdowns rather than radio lists so the seven deinterlacers
    /// cost one row instead of seven.
    fn settings_page(&mut self, ui: &mut egui::Ui) {
        use crate::config::{AspectSetting, DeinterlaceSetting};
        use crate::display::ScaleMode;

        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Video");
            egui::Grid::new("video").num_columns(2).show(ui, |ui| {
                ui.label("Scaler");
                combo(ui, "scaler", &mut self.scale_mode, &ScaleMode::ALL, ScaleMode::label);
                ui.end_row();
                ui.label("Aspect ratio");
                combo(ui, "aspect", &mut self.aspect, &AspectSetting::ALL, AspectSetting::label);
                ui.end_row();
                ui.label("Deinterlace");
                combo(ui, "deinterlace", &mut self.deinterlace, &DeinterlaceSetting::ALL, DeinterlaceSetting::label);
                ui.end_row();
            });
            ui.checkbox(&mut self.swap_fields, "Swap field order")
                .on_hover_text("for output that looks line-swapped, or bobs by a whole line");
            ui.checkbox(&mut self.internal_2x, "Internal 2x resolution")
                .on_hover_text("true 2x edges on 3D geometry; roughly 5x the GS pixel work");

            ui.add_space(8.0);
            ui.separator();
            ui.heading("Input");
            if ui
                .button("Keyboard...")
                .on_hover_text("bind a key to each pad button on a controller diagram")
                .clicked()
                && self.binder.is_none()
            {
                self.binder = Some(keymap::Binder::new(&self.keys));
            }

            ui.add_space(8.0);
            ui.separator();
            ui.heading("Audio");
            ui.add(
                egui::Slider::new(&mut self.volume, 0.0..=1.0)
                    .text("volume")
                    .custom_formatter(|v, _| format!("{:.0}%", v * 100.0)),
            );
        });
    }

    /// The memory viewer (one window of RAM, refreshed with the frame)
    /// and the scanner (find a value, narrow it down as it moves).
    fn memory_page(&mut self, ui: &mut egui::Ui) {
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
            // A 16-byte row is wider than the pane's default width, which
            // is sized for the Settings page; let it scroll rather than clip.
            egui::ScrollArea::horizontal().id_salt("hex").show(ui, |ui| {
                ui.add(
                    egui::Label::new(egui::RichText::new(text).monospace())
                        .wrap_mode(egui::TextWrapMode::Extend),
                );
            });
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
                let can_keep = self.cheat_file.is_some();
                let mut keep = None;
                for &(addr, value) in &result.hits {
                    let line = format!("{addr:08x}  {value:0digits$x}  {value}");
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(can_keep, egui::Button::new("+").small())
                            .on_hover_text("keep this value: adds a cheat to the disc's .pnach")
                            .clicked()
                        {
                            keep = Some((addr, value));
                        }
                        if ui
                            .add(egui::Label::new(egui::RichText::new(line).monospace()).sense(egui::Sense::click()))
                            .on_hover_text("show in the viewer")
                            .clicked()
                        {
                            self.mem_addr = format!("{:08x}", addr & !0xF);
                        }
                    });
                }
                if let Some((addr, value)) = keep {
                    self.keep_value(addr, value, result.width);
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
        let pnach = cheatfile::path_for(&path);
        let cheats = cheatfile::load(&pnach);
        match Disc::open(&path, cheats.clone()) {
            Ok(disc) => {
                self.disc_error = None;
                tracing::info!(path = %path.display(), "disc loaded");
                // The disc's own list replaces whatever was showing. It stays
                // keyed by the pnach stem until the worker reports a boot
                // serial; `sync_cheat_key` takes it from there.
                self.cheats = cheats;
                self.cheat_file = Some(pnach);
                self.cheat_key = None;
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

    /// Dump the currently displayed frame to a timestamped BMP in the
    /// working directory. Headless `--screenshot` picks BMP or PNG from
    /// the path's extension; this one has no path to read.
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
    /// card itself when it stops.) Comparing whole configs rather than
    /// field by field means a new setting only has to be copied in here,
    /// not also added to a condition that is easy to forget.
    fn drop(&mut self) {
        let Some(path) = &self.config_path else { return };
        let mut cfg = self.config.clone();
        cfg.volume = self.volume;
        cfg.scaler = self.scale_mode;
        cfg.aspect = self.aspect;
        cfg.deinterlace = self.deinterlace;
        cfg.swap_fields = self.swap_fields;
        cfg.cheats = self.cheats_on;
        cfg.internal_2x = self.internal_2x;
        cfg.pane = self.show_pane;
        cfg.pane_width = self.pane_width;
        cfg.window_width = self.window_size.x;
        cfg.window_height = self.window_size.y;
        cfg.keys = self.keys.clone();
        if cfg != self.config {
            cfg.save(path);
        }
    }
}

impl eframe::App for App {
    /// Everything the emulator thread reads. eframe calls this before every
    /// `ui`, and also while the window is hidden, so a minimised window keeps
    /// feeding the pad and the volume.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // A text field in the pane (an address, a scan value) owns the
        // keyboard while it has focus: without this, typing "1000" also
        // presses whatever pad buttons those keys are bound to. The
        // frontend's own function keys below stay live either way.
        let capturing = self.binder.as_ref().is_some_and(keymap::Binder::capturing);
        let typing = capturing || ctx.egui_wants_keyboard_input();
        let buttons = if typing {
            0
        } else {
            ctx.input(|i| {
                self.keymap
                    .iter()
                    .filter(|(k, _)| i.key_down(*k))
                    .fold(0u16, |acc, (_, b)| acc | b)
            })
        };
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
        self.emu.shared.cheats.store(self.cheats_on, Ordering::Relaxed);
        // The worker does the work behind a page only while that page is
        // the one showing; every other page costs nothing but this store.
        // A tab switch reaches the worker on the next frame, so the new
        // page draws one frame of stale data before it catches up.
        let chrome_now = !self.fullscreen;
        let panels = if chrome_now && self.show_pane { self.page.panels() } else { 0 };
        self.emu.shared.panels.store(panels, Ordering::Relaxed);
        if panels & PANEL_MEMORY != 0 {
            let base = parse_addr(&self.mem_addr).unwrap_or(0);
            self.emu.shared.view.store(crate::emu::pack_view(self.mem_target, base), Ordering::Relaxed);
        }
        self.emu.shared.internal_2x.store(self.internal_2x, Ordering::Relaxed);
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // The panels below need `ui` mutably, so take an owned handle to the
        // context (an Arc clone) rather than borrowing it out of `ui`.
        let ctx = &ui.ctx().clone();

        let status = self.emu.shared.status.lock().unwrap().clone();
        let disc = self.emu.shared.disc.lock().unwrap().clone();
        let debugger_active = self.emu.shared.debugger_active.load(Ordering::Relaxed);
        self.sync_cheat_key(disc.as_ref());

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

        // A pad button waiting for its key swallows the press, so the
        // frontend's own shortcuts stand down: otherwise binding F12 would
        // also take a screenshot on the way in.
        let capturing = match self.binder.as_mut().map(|b| b.show(ctx)) {
            Some(keymap::Outcome::Accept(keys)) => {
                self.keys = keys;
                self.keymap = resolve_keymap(&self.keys);
                self.binder = None;
                false
            }
            Some(keymap::Outcome::Cancel) => {
                self.binder = None;
                false
            }
            _ => self.binder.as_ref().is_some_and(keymap::Binder::capturing),
        };

        if !capturing && ctx.input(|i| i.key_pressed(egui::Key::F12)) {
            self.take_screenshot();
        }

        // Save/load shortcuts; the debugger owns execution while attached.
        if !debugger_active && !capturing {
            if self.hotkey_save.is_some_and(|k| ctx.input(|i| i.key_pressed(k))) {
                self.emu.send(Command::SaveState);
            }
            if self.hotkey_load.is_some_and(|k| ctx.input(|i| i.key_pressed(k))) {
                self.emu.send(Command::LoadState);
            }
        }

        // F11 toggles fullscreen; the chrome (menu, status bar, panels)
        // hides while fullscreen so only the display shows.
        if !capturing && ctx.input(|i| i.key_pressed(egui::Key::F11)) {
            self.fullscreen = !self.fullscreen;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(self.fullscreen));
        }
        // Esc leaves fullscreen. It also gives the keyboard back to the pad
        // when a pane text field holds it, but that needs no code here:
        // egui drops focus on Esc in its own begin_pass, so `typing` above
        // is already false by the time this runs.
        if self.fullscreen && !capturing && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.fullscreen = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
        }
        let chrome = !self.fullscreen;

        if chrome {
            self.window_size = ctx.viewport_rect().size();
        }

        // Size the window so the display comes out exactly this tall,
        // measured from the last frame: the pane and the TTY panel keep
        // their own size, so the difference lands on the display. A window
        // the desktop cannot fit is clamped by the window manager. The
        // request is only taken once a frame has been laid out, or it
        // would be consumed with nothing to measure against.
        if self.central_size.x > 0.0
            && let Some(height) = self.resize_to.take()
        {
            let ppp = ctx.pixels_per_point();
            let display = egui::vec2(height as f32 * self.display_aspect, height as f32) / ppp;
            let window_chrome = ctx.viewport_rect().size() - self.central_size;
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(window_chrome + display));
        }

        if chrome {
            egui::Panel::top("menu").show(ui, |ui| {
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
                    // Nothing but visibility lives here: the settings the
                    // menu used to carry are pages of the side pane now.
                    ui.menu_button("View", |ui| {
                        if ui.button("Fullscreen	F11").clicked() {
                            self.fullscreen = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(true));
                            ui.close();
                        }
                        ui.add_enabled_ui(!self.fullscreen, |ui| {
                            ui.menu_button("Display size", |ui| {
                                for (label, height) in [("720p", 720u32), ("1080p", 1080)] {
                                    if ui
                                        .button(label)
                                        .on_hover_text("the pane and the TTY panel keep their size")
                                        .clicked()
                                    {
                                        self.resize_to = Some(height);
                                        ui.close();
                                    }
                                }
                            });
                        });
                        ui.separator();
                        ui.checkbox(&mut self.show_pane, "Side pane");
                        ui.checkbox(&mut self.show_tty, "TTY panel");
                    });
                    ui.menu_button("Help", |ui| {
                        ui.label("Pad (Settings > Input rebinds the keyboard):");
                        for ((name, (key, _)), (btn, _)) in BUTTON_NAMES
                            .iter()
                            .zip(self.keys.pairs())
                            .zip(self.config.pad.pairs())
                        {
                            ui.monospace(format!("{name:>8} = {key} / {btn}"));
                        }
                        ui.separator();
                        ui.monospace(format!("    save = {}", self.config.hotkeys.save_state));
                        ui.monospace(format!("    load = {}", self.config.hotkeys.load_state));
                        ui.separator();
                        ui.label("F11 fullscreen (Esc leaves), F12 screenshot.");
                        ui.label("A pane text field holds the keyboard while focused;");
                        ui.label("Esc, or a click on empty pane, gives it back to the pad.");
                    });
                });
            });

            egui::Panel::bottom("status").show(ui, |ui| {
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

        if chrome && self.show_pane {
            let pane = egui::Panel::right("pane")
                .resizable(true)
                .min_size(240.0)
                .default_size(self.pane_width)
                .show(ui, |ui| {
                    // Claimed before the content so the pages' own widgets
                    // sit on top of it: a click that lands here is a click
                    // on empty pane, which drops text focus and gives the
                    // keyboard back to the pad.
                    let background =
                        ui.interact(ui.max_rect(), ui.id().with("background"), egui::Sense::click());
                    ui.horizontal(|ui| {
                        for page in Page::ALL {
                            ui.selectable_value(&mut self.page, page, page.label());
                        }
                    });
                    ui.separator();
                    match self.page {
                        Page::Settings => self.settings_page(ui),
                        Page::Cheats => self.cheats_page(ui, disc.as_ref()),
                        Page::Memory => self.memory_page(ui),
                        Page::Registers => registers_page(ui, &status),
                    }
                    if background.clicked() {
                        ui.memory_mut(|m| m.stop_text_input());
                    }
                });
            // Follow the drag handle so the width survives to the config.
            self.pane_width = pane.response.rect.width();
        }

        if chrome && self.show_tty {
            egui::Panel::bottom("tty")
                .resizable(true)
                .default_size(160.0)
                .show(ui, |ui| {
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
        central.show(ui, |ui| {
            self.central_size = ui.max_rect().size();
            let (width, height, rgba, seq) = {
                let frame = self.emu.shared.frame.lock().unwrap();
                (frame.width, frame.height, frame.rgba.clone(), frame.seq)
            };
            if width == 0 || height == 0 {
                ui.centered_and_justified(|ui| ui.label("waiting for a frame..."));
                return;
            }
            // Fit the panel to the display aspect ratio. It is not the
            // framebuffer's: PS2 pixels are non-square, so a 512x448 buffer
            // and a 640x448 one both fill the same 4:3 raster.
            let avail = ui.available_size();
            let aspect = self.aspect.ratio(width, height);
            self.display_aspect = aspect;
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
