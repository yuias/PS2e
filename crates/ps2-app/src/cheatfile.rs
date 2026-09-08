//! The front-end's side of cheats: the `.pnach` beside a disc image, and
//! `cheats.toml`, which remembers per disc which of that file's sections
//! are switched off.
//!
//! The switched-off list is kept out of `config.toml` on purpose.
//! [`crate::config::Config::save`] re-serializes that file whole, losing
//! the commented template it ships with, and a per-disc map would grow
//! without bound in a file meant to be read and edited by hand.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ps2_core::cheats::{Group, Target};
use serde::{Deserialize, Serialize};

/// The pnach that goes with a disc image: the image's path with its
/// extension replaced.
pub fn path_for(disc: &Path) -> PathBuf {
    disc.with_extension("pnach")
}

/// Read and parse a pnach file. Absent is normal and silent; unreadable
/// is logged. Lines the parser rejects stay on the group they sit in, so
/// the Cheats page can show what the file lost rather than only the log.
pub fn load(path: &Path) -> Vec<Group> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(e) => {
            tracing::warn!(path = %path.display(), "cannot read the cheat file: {e}");
            return Vec::new();
        }
    };
    let groups = ps2_core::cheats::parse(&text);
    let skipped: usize = groups.iter().map(|g| g.warnings.len()).sum();
    for w in groups.iter().flat_map(|g| &g.warnings) {
        tracing::warn!(path = %path.display(), "cheat file: {w}");
    }
    tracing::info!(
        path = %path.display(),
        groups = groups.len(),
        count = groups.iter().map(|g| g.cheats.len()).sum::<usize>(),
        skipped,
        "cheat file loaded"
    );
    groups
}

/// How a disc is keyed in `cheats.toml`: its boot serial, so the entry
/// survives renaming or moving the image, falling back to the pnach's
/// file stem only when the disc has no readable SYSTEM.CNF.
pub fn key(serial: Option<&str>, pnach: &Path) -> String {
    match serial {
        Some(s) => s.to_string(),
        None => pnach.file_stem().unwrap_or(pnach.as_os_str()).to_string_lossy().into_owned(),
    }
}

#[derive(Default, Serialize, Deserialize)]
struct Entry {
    /// Section names the user has switched off.
    #[serde(default)]
    disabled: Vec<String>,
}

/// `cheats.toml`: one table per disc, holding the sections that are off.
///
/// Off rather than on, so that a disc with no entry runs everything its
/// pnach holds -- what the master switch alone used to do -- and a
/// section added to the file later, by hand or by the memory scanner, is
/// live without an edit here.
#[derive(Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Store {
    discs: BTreeMap<String, Entry>,
}

impl Store {
    /// Read the file, treating absent and malformed alike: cheats are a
    /// convenience, and refusing to start over a broken list would not be.
    pub fn load(path: &Path) -> Store {
        match std::fs::read_to_string(path) {
            Ok(text) => match toml::from_str(&text) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(path = %path.display(), "ignoring malformed cheat state: {e}");
                    Store::default()
                }
            },
            Err(_) => Store::default(),
        }
    }

    /// Write the file back. Failure is logged and swallowed: losing a
    /// cheat toggle is not worth interrupting a session for.
    pub fn save(&self, path: &Path) {
        match toml::to_string_pretty(self) {
            Ok(text) => {
                if let Some(dir) = path.parent() {
                    let _ = std::fs::create_dir_all(dir);
                }
                if let Err(e) = std::fs::write(path, text) {
                    tracing::warn!(path = %path.display(), "failed to save cheat state: {e}");
                }
            }
            Err(e) => tracing::warn!("failed to serialize cheat state: {e}"),
        }
    }

    /// Apply the stored list to a pnach. Every group is set, not only the
    /// ones named: the caller re-keys a list it has already applied once
    /// (the pnach stem, then the boot serial), and the flags of the key it
    /// is leaving must not survive into the one it is arriving at.
    pub fn apply(&self, key: &str, groups: &mut [Group]) {
        let disabled = self.discs.get(key).map(|e| e.disabled.as_slice()).unwrap_or_default();
        for g in groups {
            g.enabled = !disabled.contains(&g.name);
        }
    }

    /// Record what is off for this disc. A disc with nothing off keeps no
    /// entry, so the file holds only what the user has changed.
    pub fn update(&mut self, key: &str, groups: &[Group]) {
        let disabled: Vec<String> =
            groups.iter().filter(|g| !g.enabled).map(|g| g.name.clone()).collect();
        if disabled.is_empty() {
            self.discs.remove(key);
        } else {
            self.discs.insert(key.to_string(), Entry { disabled });
        }
    }
}

/// Write one section into a pnach: replace the lines `span` covers, or
/// append it when the section is new.
///
/// The parser keeps no `comment=` lines and no formatting, so rewriting
/// the file from the parsed model would destroy a hand-written one. Only
/// the section's own lines are touched.
pub fn write_section(path: &Path, span: Option<(usize, usize)>, body: &str) -> std::io::Result<()> {
    let old = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(e),
    };
    // `lines()` drops the line endings, so the file is rejoined with the
    // one it already used: a pnach written on Windows must not come back
    // with every line changed by an edit to one section of it.
    let eol = if old.contains("\r\n") { "\r\n" } else { "\n" };
    let mut lines: Vec<&str> = old.lines().collect();
    let body: Vec<&str> = body.lines().collect();
    match span {
        // The span is 1-based and inclusive, and came from a parse of this
        // same file; anything else is treated as a new section.
        Some((first, last)) if first >= 1 && last <= lines.len() => {
            lines.splice(first - 1..last, body);
        }
        _ => {
            if lines.last().is_some_and(|l| !l.trim().is_empty()) {
                lines.push("");
            }
            lines.extend(body);
        }
    }
    let mut text = lines.join(eol);
    text.push_str(eol);
    std::fs::write(path, text)
}

/// A pnach section holding one constant write, as the memory scanner's
/// "keep this value" makes it. `place` 1 is "every frame", which is what
/// holds a value against the game writing its own.
pub fn constant_write(name: &str, target: Target, addr: u32, value: u64, width: u8) -> String {
    let cpu = match target {
        Target::Ee => "EE",
        Target::Iop => "IOP",
    };
    let (kind, digits) = match width {
        1 => ("byte", 2),
        2 => ("short", 4),
        _ => ("word", 8),
    };
    format!("[{name}]\npatch=1,{cpu},{addr:08X},{kind},{value:0digits$X}\n")
}

/// The section name a scanner hit gets. Naming it after the address and
/// width means a second "keep" on the same hit rewrites that section
/// instead of stacking another one: the list has no delete, and pressing
/// it again is what a user does when the value has moved on.
pub fn scan_name(target: Target, addr: u32, width: u8) -> String {
    let cpu = match target {
        Target::Ee => "EE",
        Target::Iop => "IOP",
    };
    let kind = match width {
        1 => "byte",
        2 => "short",
        _ => "word",
    };
    format!("Scan {cpu} {addr:08X} {kind}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("game.pnach")
    }

    #[test]
    fn a_replaced_section_leaves_the_rest_of_the_file_alone() {
        let path = temp("ps2e-cheatfile-replace");
        let name = scan_name(Target::Ee, 0x0010_0000, 4);
        std::fs::write(
            &path,
            format!(
                "gametitle=Some Game\n\n[{name}]\npatch=1,EE,00100000,word,00000001\n\n[Kept]\npatch=1,EE,00200000,byte,ff\n"
            ),
        )
        .unwrap();
        let span = load(&path).iter().find(|g| g.name == name).unwrap().span;
        write_section(&path, Some(span), &constant_write(&name, Target::Ee, 0x0010_0000, 99, 4)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("gametitle=Some Game"), "{text}");
        assert!(text.contains("00000063") && !text.contains("00000001"), "{text}");
        assert!(text.contains("[Kept]") && text.contains("00200000"), "{text}");
        // Still one section of each name after the rewrite.
        assert_eq!(load(&path).len(), 2);
    }

    /// A pnach written on Windows must not come back with every line
    /// changed by an edit to one section of it.
    #[test]
    fn a_crlf_file_keeps_its_line_endings() {
        let path = temp("ps2e-cheatfile-crlf");
        std::fs::write(&path, "[Kept]\r\npatch=1,EE,00200000,byte,ff\r\n").unwrap();
        let name = scan_name(Target::Ee, 0x0010_0000, 2);
        write_section(&path, None, &constant_write(&name, Target::Ee, 0x0010_0000, 5, 2)).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.matches('\n').count(), text.matches("\r\n").count(), "{text:?}");
        assert_eq!(load(&path).len(), 2);
    }

    #[test]
    fn a_new_section_is_appended() {
        let path = temp("ps2e-cheatfile-append");
        std::fs::write(&path, "[Kept]\npatch=1,EE,00200000,byte,ff\n").unwrap();
        let name = scan_name(Target::Iop, 0x0010_0000, 1);
        write_section(&path, None, &constant_write(&name, Target::Iop, 0x0010_0000, 7, 1)).unwrap();
        let groups = load(&path);
        assert_eq!(groups.iter().map(|g| g.name.as_str()).collect::<Vec<_>>(), ["Kept", name.as_str()]);
        assert_eq!(groups[1].cheats.len(), 1);
    }

    #[test]
    fn only_the_switched_off_sections_are_stored() {
        let mut groups = vec![
            Group { name: "A".into(), enabled: false, ..Group::of(Vec::new()) },
            Group { name: "B".into(), enabled: true, ..Group::of(Vec::new()) },
        ];
        let mut store = Store::default();
        store.update("SLPS-25418", &groups);
        let text = toml::to_string_pretty(&store).unwrap();
        assert!(text.contains("SLPS-25418") && text.contains("\"A\""), "{text}");
        assert!(!text.contains("\"B\""), "{text}");

        // A fresh parse comes back all-on; the store puts A back off.
        for g in &mut groups {
            g.enabled = true;
        }
        store.apply("SLPS-25418", &mut groups);
        assert_eq!(groups.iter().map(|g| g.enabled).collect::<Vec<_>>(), [false, true]);

        // Re-keying to a disc with no entry clears what the old key set,
        // rather than leaving its flags to be written out under the new one.
        store.apply("SLPS-99999", &mut groups);
        assert_eq!(groups.iter().map(|g| g.enabled).collect::<Vec<_>>(), [true, true]);
        store.apply("SLPS-25418", &mut groups);

        // Nothing off leaves no entry behind.
        for g in &mut groups {
            g.enabled = true;
        }
        store.update("SLPS-25418", &groups);
        assert_eq!(toml::to_string_pretty(&store).unwrap().trim(), "");
    }
}
