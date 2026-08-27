//! TOML configuration for the windowed front-end.
//!
//! Search order:
//! 1. `<exe_dir>/config/config.toml` (portable installs)
//! 2. `~/.config/PS2e/config.toml`
//!
//! When neither exists, a commented default is generated at the user
//! location (it survives `cargo clean`, unlike the target directory).
//! Headless (`--cycles`) runs never touch this file's memcard default —
//! only `--memcard` opts a headless run into a persisted card.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const DEFAULT_TEMPLATE: &str = r#"# PS2e configuration

# Path to a 4 MiB PS2 BIOS image. Absolute, or relative to the directory
# this file is in. Falls back to assets/SCPH-50000.bin when unset.
#bios = "path/to/bios.bin"

# Master volume, 0.0 .. 1.0
volume = 0.5

# Display scaler: "nearest", "linear", "sharp" (nearest to an integer
# multiple, then linear) or "lanczos".
scaler = "sharp"

# Interlaced output: "weave" (both fields, combs on motion), "bob" (latest
# field only, bobs by nature), "blend" (weave softened vertically),
# "adaptive" (weave where still, bob where moving), "adaptivedebug"
# (adaptive with the rebuilt pixels tinted) or "bwdif" (ffmpeg's filter:
# a cubic vertical rebuild clamped by the neighbouring fields, one field
# of display latency). "yadif" also parses (bwdif's predecessor, not in
# the menu).
deinterlace = "bwdif"

# Flip the field order (which rows the odd field lands on) if interlaced
# output looks line-swapped or bobs by a whole line.
swap_fields = false

# Render internally at 2x resolution: 3D geometry gets true 2x edges,
# textures and video stay at their native size. Costs roughly 5x the GS
# pixel work.
internal_2x = false

# Memory card image (created and formatted automatically).
# Defaults to memcard0.ps2 next to this file.
#memcard = "memcard0.ps2"

# Digital pad bindings. Names are the ones egui reports: letters and
# digits as themselves ("X", "1"), arrows as "Up"/"Down"/"Left"/"Right",
# plus "Enter", "Backspace", "Space" and "F1".."F35". Omitted buttons keep
# the defaults shown here; an unknown name falls back with a warning.
#[keys]
#up = "Up"
#down = "Down"
#left = "Left"
#right = "Right"
#cross = "Z"
#circle = "X"
#square = "S"
#triangle = "D"
#l1 = "W"
#l2 = "E"
#r1 = "R"
#r2 = "U"
#l3 = "1"
#r3 = "3"
#start = "V"
#select = "C"

# Gamepad bindings for the same pad. Values name gilrs buttons:
# "South"/"East"/"North"/"West" for the action pad, "DPadUp".."DPadRight",
# "LeftTrigger"/"LeftTrigger2"/"RightTrigger"/"RightTrigger2", "Start",
# "Select", "LeftThumb", "RightThumb", "Mode", "C", "Z". Gamepad input is
# merged with the keyboard, so either can drive any button.
#[pad]
#up = "DPadUp"
#down = "DPadDown"
#left = "DPadLeft"
#right = "DPadRight"
#cross = "South"
#circle = "East"
#square = "West"
#triangle = "North"
#l1 = "LeftTrigger"
#l2 = "LeftTrigger2"
#r1 = "RightTrigger"
#r2 = "RightTrigger2"
#l3 = "LeftThumb"
#r3 = "RightThumb"
#start = "Start"
#select = "Select"

# Frontend shortcuts, same key names as [keys].
#[hotkeys]
#save_state = "F5"
#load_state = "F9"
"#;

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Config {
    pub bios: Option<PathBuf>,
    pub volume: f32,
    pub memcard: Option<PathBuf>,
    pub scaler: crate::display::ScaleMode,
    pub deinterlace: DeinterlaceSetting,
    pub swap_fields: bool,
    pub internal_2x: bool,
    pub keys: KeyBindings,
    pub pad: PadBindings,
    pub hotkeys: HotKeys,
    /// Save-state file. Defaults to state0.sst next to this file.
    pub state: Option<PathBuf>,
}

/// Frontend shortcuts, same key names as [`KeyBindings`].
#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct HotKeys {
    pub save_state: String,
    pub load_state: String,
}

impl Default for HotKeys {
    fn default() -> Self {
        Self { save_state: "F5".into(), load_state: "F9".into() }
    }
}

/// One egui key name per digital-pad button, as written in the config
/// file. Resolved to [`egui::Key`] once at startup by the UI.
#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct KeyBindings {
    pub up: String,
    pub down: String,
    pub left: String,
    pub right: String,
    pub cross: String,
    pub circle: String,
    pub square: String,
    pub triangle: String,
    pub l1: String,
    pub l2: String,
    pub r1: String,
    pub r2: String,
    pub l3: String,
    pub r3: String,
    pub start: String,
    pub select: String,
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self {
            up: "Up".into(),
            down: "Down".into(),
            left: "Left".into(),
            right: "Right".into(),
            cross: "Z".into(),
            circle: "X".into(),
            square: "S".into(),
            triangle: "D".into(),
            l1: "W".into(),
            l2: "E".into(),
            r1: "R".into(),
            r2: "U".into(),
            l3: "1".into(),
            r3: "3".into(),
            start: "V".into(),
            select: "C".into(),
        }
    }
}

impl KeyBindings {
    /// Each binding paired with the pad bit it drives, in a fixed order
    /// (the order the Help menu lists them in).
    pub fn pairs(&self) -> [(&str, u16); 16] {
        use crate::pad;
        [
            (&self.up, pad::UP),
            (&self.down, pad::DOWN),
            (&self.left, pad::LEFT),
            (&self.right, pad::RIGHT),
            (&self.cross, pad::CROSS),
            (&self.circle, pad::CIRCLE),
            (&self.square, pad::SQUARE),
            (&self.triangle, pad::TRIANGLE),
            (&self.l1, pad::L1),
            (&self.l2, pad::L2),
            (&self.r1, pad::R1),
            (&self.r2, pad::R2),
            (&self.l3, pad::L3),
            (&self.r3, pad::R3),
            (&self.start, pad::START),
            (&self.select, pad::SELECT),
        ]
    }
}

/// One gilrs button name per digital-pad button, as written in the config
/// file. Resolved to [`gilrs::Button`] once at startup by
/// [`crate::gamepad::Gamepad`].
#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct PadBindings {
    pub up: String,
    pub down: String,
    pub left: String,
    pub right: String,
    pub cross: String,
    pub circle: String,
    pub square: String,
    pub triangle: String,
    pub l1: String,
    pub l2: String,
    pub r1: String,
    pub r2: String,
    pub l3: String,
    pub r3: String,
    pub start: String,
    pub select: String,
}

impl Default for PadBindings {
    fn default() -> Self {
        Self {
            up: "DPadUp".into(),
            down: "DPadDown".into(),
            left: "DPadLeft".into(),
            right: "DPadRight".into(),
            cross: "South".into(),
            circle: "East".into(),
            square: "West".into(),
            triangle: "North".into(),
            l1: "LeftTrigger".into(),
            l2: "LeftTrigger2".into(),
            r1: "RightTrigger".into(),
            r2: "RightTrigger2".into(),
            l3: "LeftThumb".into(),
            r3: "RightThumb".into(),
            start: "Start".into(),
            select: "Select".into(),
        }
    }
}

impl PadBindings {
    /// Each binding paired with the pad bit it drives, in the same order as
    /// [`KeyBindings::pairs`].
    pub fn pairs(&self) -> [(&str, u16); 16] {
        use crate::pad;
        [
            (&self.up, pad::UP),
            (&self.down, pad::DOWN),
            (&self.left, pad::LEFT),
            (&self.right, pad::RIGHT),
            (&self.cross, pad::CROSS),
            (&self.circle, pad::CIRCLE),
            (&self.square, pad::SQUARE),
            (&self.triangle, pad::TRIANGLE),
            (&self.l1, pad::L1),
            (&self.l2, pad::L2),
            (&self.r1, pad::R1),
            (&self.r2, pad::R2),
            (&self.l3, pad::L3),
            (&self.r3, pad::R3),
            (&self.start, pad::START),
            (&self.select, pad::SELECT),
        ]
    }
}

/// Config/UI form of [`ps2_core::gs::Deinterlace`].
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum DeinterlaceSetting {
    Weave,
    Bob,
    Blend,
    Adaptive,
    /// Adaptive with rebuilt pixels tinted, to see where it kicks in.
    AdaptiveDebug,
    Yadif,
    Bwdif,
}

impl DeinterlaceSetting {
    /// Modes offered in the UI. `Yadif` is left out (superseded by bwdif)
    /// but still parses from the config file for hand-tuned setups.
    pub const ALL: [DeinterlaceSetting; 6] = [
        DeinterlaceSetting::Weave,
        DeinterlaceSetting::Bob,
        DeinterlaceSetting::Blend,
        DeinterlaceSetting::Adaptive,
        DeinterlaceSetting::AdaptiveDebug,
        DeinterlaceSetting::Bwdif,
    ];

    pub fn label(self) -> &'static str {
        match self {
            DeinterlaceSetting::Weave => "Weave",
            DeinterlaceSetting::Bob => "Bob",
            DeinterlaceSetting::Blend => "Blend",
            DeinterlaceSetting::Adaptive => "Adaptive",
            DeinterlaceSetting::AdaptiveDebug => "Adaptive (show motion)",
            DeinterlaceSetting::Yadif => "Yadif (1 field late)",
            DeinterlaceSetting::Bwdif => "Bwdif (1 field late)",
        }
    }

    /// Index into [`crate::emu::DEINTERLACE_MODES`].
    pub fn index(self) -> u8 {
        match self {
            DeinterlaceSetting::Weave => 0,
            DeinterlaceSetting::Bob => 1,
            DeinterlaceSetting::Blend => 2,
            DeinterlaceSetting::Adaptive => 3,
            DeinterlaceSetting::AdaptiveDebug => 4,
            DeinterlaceSetting::Yadif => 5,
            DeinterlaceSetting::Bwdif => 6,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bios: None,
            volume: 0.5,
            memcard: None,
            scaler: crate::display::ScaleMode::Sharp,
            deinterlace: DeinterlaceSetting::Bwdif,
            swap_fields: false,
            internal_2x: false,
            keys: KeyBindings::default(),
            pad: PadBindings::default(),
            hotkeys: HotKeys::default(),
            state: None,
        }
    }
}

fn exe_local_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("config").join("config.toml"))
}

fn user_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)?;
    Some(home.join(".config").join("PS2e").join("config.toml"))
}

impl Config {
    /// Load the configuration, generating a default file on first run.
    /// Returns the config and the path it lives at.
    pub fn load() -> (Self, Option<PathBuf>) {
        for path in [exe_local_path(), user_path()].into_iter().flatten() {
            match std::fs::read_to_string(&path) {
                Ok(text) => match toml::from_str::<Config>(&text) {
                    Ok(mut cfg) => {
                        // Relative paths resolve against the config dir
                        if let Some(dir) = path.parent() {
                            for p in [&mut cfg.bios, &mut cfg.memcard, &mut cfg.state] {
                                if let Some(v) = p
                                    && v.is_relative()
                                {
                                    *v = dir.join(&v);
                                }
                            }
                        }
                        tracing::info!("loaded config from {}", path.display());
                        return (cfg, Some(path));
                    }
                    Err(e) => {
                        tracing::error!("ignoring malformed {}: {e}", path.display());
                        return (Config::default(), Some(path));
                    }
                },
                Err(_) => continue,
            }
        }

        // First run: write a commented template to the user location
        let path = user_path();
        if let Some(path) = &path {
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            match std::fs::write(path, DEFAULT_TEMPLATE) {
                Ok(()) => tracing::info!("created default config at {}", path.display()),
                Err(e) => tracing::warn!("could not write {}: {e}", path.display()),
            }
        }
        (Config::default(), path)
    }

    /// Save-state location for windowed sessions: configured path, or
    /// state0.sst next to the config file.
    pub fn state_path(&self, cfg_path: Option<&PathBuf>) -> PathBuf {
        self.state.clone().unwrap_or_else(|| {
            cfg_path
                .and_then(|p| p.parent())
                .map(|d| d.join("state0.sst"))
                .unwrap_or_else(|| PathBuf::from("state0.sst"))
        })
    }

    /// Memory card image location for windowed sessions: configured path,
    /// or memcard0.ps2 next to the config file.
    pub fn memcard_path(&self, cfg_path: Option<&PathBuf>) -> PathBuf {
        self.memcard.clone().unwrap_or_else(|| {
            cfg_path
                .and_then(|p| p.parent())
                .map(|d| d.join("memcard0.ps2"))
                .unwrap_or_else(|| PathBuf::from("memcard0.ps2"))
        })
    }

    /// Persist current settings (e.g. volume changed in the UI), keeping it
    /// simple: full re-serialize, comments in the template are lost once
    /// the user's settings are saved over it.
    pub fn save(&self, path: &PathBuf) {
        // Store the BIOS path as given; no attempt to re-relativize
        match toml::to_string_pretty(self) {
            Ok(text) => {
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                if let Err(e) = std::fs::write(path, text) {
                    tracing::warn!("failed to save config: {e}");
                }
            }
            Err(e) => tracing::warn!("failed to serialize config: {e}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TOML cannot emit a scalar after a table, so the order of `Config`'s
    /// fields is load-bearing for [`Config::save`].
    #[test]
    fn a_saved_config_round_trips() {
        let mut cfg = Config::default();
        cfg.bios = Some("bios.bin".into());
        cfg.memcard = Some("memcard0.ps2".into());
        cfg.state = Some("state0.sst".into());
        let text = toml::to_string_pretty(&cfg).expect("serializes");
        let back: Config = toml::from_str(&text).expect("parses back");
        assert_eq!(back.pad.circle, cfg.pad.circle);
        assert_eq!(back.state, cfg.state);
    }

    #[test]
    fn the_generated_template_parses() {
        toml::from_str::<Config>(DEFAULT_TEMPLATE).expect("template is valid TOML");
    }
}
