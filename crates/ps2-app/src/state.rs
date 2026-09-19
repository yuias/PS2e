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

/// zstd-compress a raw state for an in-memory slot (same level as files):
/// 16 slots of ~6-9 MB each beats 16 of the ~40 MiB raw state.
pub fn encode(data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    zstd::stream::copy_encode(data, &mut out, LEVEL)?;
    Ok(out)
}

/// Inverse of [`encode`].
pub fn decode(blob: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut out = Vec::new();
    zstd::stream::copy_decode(blob, &mut out)?;
    Ok(out)
}
