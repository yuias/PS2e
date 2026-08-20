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
# (adaptive with the rebuilt pixels tinted), "yadif" (ffmpeg's filter:
# edge-directed rebuild clamped by the neighbouring fields, one field of
# display latency) or "bwdif" (yadif's clamp with a cubic vertical
# rebuild; also one field late).
deinterlace = "bwdif"

# Flip the field order (which rows the odd field lands on) if interlaced
# output looks line-swapped or bobs by a whole line.
swap_fields = false

# Memory card image (created and formatted automatically).
# Defaults to memcard0.ps2 next to this file.
#memcard = "memcard0.ps2"
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
    pub const ALL: [DeinterlaceSetting; 7] = [
        DeinterlaceSetting::Weave,
        DeinterlaceSetting::Bob,
        DeinterlaceSetting::Blend,
        DeinterlaceSetting::Adaptive,
        DeinterlaceSetting::AdaptiveDebug,
        DeinterlaceSetting::Yadif,
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
                            for p in [&mut cfg.bios, &mut cfg.memcard] {
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
