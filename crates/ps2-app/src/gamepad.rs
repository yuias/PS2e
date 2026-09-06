//! Gamepad input via gilrs: the buttons the config binds, and the two
//! sticks. Button bits are OR-ed with the keyboard, letting either drive
//! any button; the sticks have no keyboard equivalent and rest centred
//! when no gamepad is connected.

use crate::config::PadBindings;
use crate::emu::STICKS_CENTRED;
use gilrs::{Axis, Button, Gilrs};

/// Stick travel below this fraction of full deflection reads as centred,
/// so a resting stick does not creep.
const DEADZONE: f32 = 0.12;

pub struct Gamepad {
    gilrs: Gilrs,
    /// Gamepad button -> pad bit, resolved from the config once at startup.
    map: Vec<(Button, u16)>,
}

impl Gamepad {
    /// Open the gamepad subsystem. Returns `None` when it is unavailable
    /// (no driver, no permission) — the frontend then runs keyboard-only.
    pub fn new(bindings: &PadBindings) -> Option<Self> {
        let gilrs = match Gilrs::new() {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("gamepad support unavailable: {e}");
                return None;
            }
        };
        let fallback = PadBindings::default();
        let map = bindings
            .pairs()
            .into_iter()
            .zip(fallback.pairs())
            .filter_map(
                |((name, bit), (default_name, _))| match parse_button(name) {
                    Some(btn) => Some((btn, bit)),
                    None => {
                        tracing::warn!("unknown gamepad button '{name}'; using '{default_name}'");
                        parse_button(default_name).map(|btn| (btn, bit))
                    }
                },
            )
            .collect();
        Some(Self { gilrs, map })
    }

    /// Pad bits held on any connected gamepad, and the sticks of the first
    /// gamepad that has moved one off centre.
    pub fn poll(&mut self) -> (u16, [u8; 4]) {
        // gilrs refreshes its cached button state from the event queue, so
        // the queue has to be drained before the state is worth reading.
        while self.gilrs.next_event().is_some() {}
        let mut bits = 0;
        let mut sticks = STICKS_CENTRED;
        for (_id, pad) in self.gilrs.gamepads() {
            for (btn, bit) in &self.map {
                if pad.is_pressed(*btn) {
                    bits |= bit;
                }
            }
            if sticks == STICKS_CENTRED {
                let axis = |a: Axis| pad.axis_data(a).map_or(0.0, |d| d.value());
                // The pad reports right stick first, and counts down from
                // the top where gilrs counts up.
                sticks = [
                    stick_byte(axis(Axis::RightStickX)),
                    stick_byte(-axis(Axis::RightStickY)),
                    stick_byte(axis(Axis::LeftStickX)),
                    stick_byte(-axis(Axis::LeftStickY)),
                ];
            }
        }
        (bits, sticks)
    }
}

/// One axis, -1.0..1.0 from gilrs, to the pad's 0x00..0xFF with 0x7F at
/// rest. The dead zone is removed and the remainder rescaled, so travel
/// past it starts from centre rather than jumping.
fn stick_byte(v: f32) -> u8 {
    let mag = v.abs();
    if mag <= DEADZONE {
        return 0x7F;
    }
    let scaled = ((mag - DEADZONE) / (1.0 - DEADZONE)).min(1.0);
    if v < 0.0 {
        (0x7F as f32 * (1.0 - scaled)).round() as u8
    } else {
        (0x7F as f32 + 0x80 as f32 * scaled).round() as u8
    }
}

/// Parse a `gilrs::Button` variant name. `Unknown` is rejected: it names no
/// physical button and would silently swallow the binding.
fn parse_button(name: &str) -> Option<Button> {
    Some(match name {
        "South" => Button::South,
        "East" => Button::East,
        "North" => Button::North,
        "West" => Button::West,
        "C" => Button::C,
        "Z" => Button::Z,
        "LeftTrigger" => Button::LeftTrigger,
        "LeftTrigger2" => Button::LeftTrigger2,
        "RightTrigger" => Button::RightTrigger,
        "RightTrigger2" => Button::RightTrigger2,
        "Select" => Button::Select,
        "Start" => Button::Start,
        "Mode" => Button::Mode,
        "LeftThumb" => Button::LeftThumb,
        "RightThumb" => Button::RightThumb,
        "DPadUp" => Button::DPadUp,
        "DPadDown" => Button::DPadDown,
        "DPadLeft" => Button::DPadLeft,
        "DPadRight" => Button::DPadRight,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bindings_all_parse() {
        let defaults = PadBindings::default();
        for (name, _) in defaults.pairs() {
            assert!(parse_button(name).is_some(), "'{name}' should parse");
        }
    }

    #[test]
    fn default_bindings_cover_every_button() {
        let defaults = PadBindings::default();
        let bits = defaults.pairs().iter().fold(0u16, |acc, (_, b)| acc | b);
        assert_eq!(bits, u16::MAX);
    }

    #[test]
    fn a_stick_rests_at_centre_and_reaches_both_ends() {
        assert_eq!(stick_byte(0.0), 0x7F);
        assert_eq!(stick_byte(DEADZONE), 0x7F);
        assert_eq!(stick_byte(-DEADZONE), 0x7F);
        assert_eq!(stick_byte(1.0), 0xFF);
        assert_eq!(stick_byte(-1.0), 0x00);
        assert_eq!(stick_byte(2.0), 0xFF);
        // Just past the dead zone: barely off centre, no jump.
        assert!((0x7F - 4..=0x7F).contains(&stick_byte(-DEADZONE - 0.01)));
        assert!((0x7F..=0x7F + 4).contains(&stick_byte(DEADZONE + 0.01)));
        // Monotonic across the range.
        let bytes: Vec<u8> = (-20..=20).map(|i| stick_byte(i as f32 / 20.0)).collect();
        assert!(bytes.windows(2).all(|w| w[0] <= w[1]), "{bytes:?}");
    }

    #[test]
    fn unknown_is_not_a_valid_binding() {
        assert!(parse_button("Unknown").is_none());
        assert!(parse_button("south").is_none());
    }
}
