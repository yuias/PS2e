//! Digital pad button bits, shared by `--press` parsing and the windowed
//! keyboard mapping (see [`crate::ui`]).
//!
//! Bit order matches the SIO2 pad reply layout (see `Sio2::buttons` in
//! `ps2-core`): SELECT=0, L3, R3, START, UP, RIGHT, DOWN, LEFT, L2, R2, L1,
//! R1, TRIANGLE, CIRCLE, CROSS, SQUARE=15. Bits set = held.

pub const SELECT: u16 = 1 << 0;
pub const L3: u16 = 1 << 1;
pub const R3: u16 = 1 << 2;
pub const START: u16 = 1 << 3;
pub const UP: u16 = 1 << 4;
pub const RIGHT: u16 = 1 << 5;
pub const DOWN: u16 = 1 << 6;
pub const LEFT: u16 = 1 << 7;
pub const L2: u16 = 1 << 8;
pub const R2: u16 = 1 << 9;
pub const L1: u16 = 1 << 10;
pub const R1: u16 = 1 << 11;
pub const TRIANGLE: u16 = 1 << 12;
pub const CIRCLE: u16 = 1 << 13;
pub const CROSS: u16 = 1 << 14;
pub const SQUARE: u16 = 1 << 15;

/// Button bit by `--press` script name (lowercase).
pub fn bit_by_name(name: &str) -> Result<u16, String> {
    Ok(match name {
        "select" => SELECT,
        "l3" => L3,
        "r3" => R3,
        "start" => START,
        "up" => UP,
        "right" => RIGHT,
        "down" => DOWN,
        "left" => LEFT,
        "l2" => L2,
        "r2" => R2,
        "l1" => L1,
        "r1" => R1,
        "triangle" => TRIANGLE,
        "circle" => CIRCLE,
        "cross" => CROSS,
        "square" => SQUARE,
        _ => return Err(format!("unknown button '{name}'")),
    })
}
