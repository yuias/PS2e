//! Save-state files: a zstd frame wrapping the core's own serialization.
//!
//! The core has no compressor (it stays wasm-safe and dependency-light) and
//! hands out raw bytes; the machine image is ~40 MiB of mostly-zero RAM and
//! VRAM, which zstd reduces by a large factor at its default level.

use std::path::Path;

/// Compression level. The default: the image is highly compressible, and
/// anything stronger costs wall time on every quick save.
const LEVEL: i32 = 3;

pub fn write(path: &Path, data: &[u8]) -> std::io::Result<u64> {
    let file = std::fs::File::create(path)?;
    zstd::stream::copy_encode(data, &file, LEVEL)?;
    file.sync_all()?;
    Ok(file.metadata()?.len())
}

pub fn read(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut out = Vec::new();
    zstd::stream::copy_decode(file, &mut out)?;
    Ok(out)
}
