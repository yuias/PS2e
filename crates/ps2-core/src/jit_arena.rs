//! Executable code arena: one read/write/execute mapping, bump-allocated.
//! Blocks are never freed individually; when the arena fills the whole
//! block cache is dropped and allocation restarts from the top.

use std::io;

pub struct Arena {
    base: *mut u8,
    size: usize,
    used: usize,
}

// The mapping is process memory owned by this struct; nothing else aliases it.
unsafe impl Send for Arena {}

impl Arena {
    pub fn new(size: usize) -> io::Result<Self> {
        let base = alloc_rwx(size)?;
        Ok(Self { base, size, used: 0 })
    }

    /// Address the next block will be placed at (for position-dependent
    /// relocations while assembling).
    pub fn next_addr(&self) -> usize {
        self.base as usize + self.used
    }

    pub fn remaining(&self) -> usize {
        self.size - self.used
    }

    /// Copy assembled code in and return its entry address. Callers check
    /// [`Arena::remaining`] first; the arena keeps 16-byte alignment.
    pub fn place(&mut self, code: &[u8]) -> *const u8 {
        assert!(code.len() <= self.remaining(), "JIT arena overflow");
        let dst = self.next_addr() as *mut u8;
        // SAFETY: dst..dst+len lies inside our own RWX mapping.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), dst, code.len()) };
        self.used += (code.len() + 15) & !15;
        dst
    }

    /// Forget every block (the memory is simply reused).
    pub fn reset(&mut self) {
        self.used = 0;
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        free(self.base, self.size);
    }
}

#[cfg(windows)]
fn alloc_rwx(size: usize) -> io::Result<*mut u8> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READWRITE, VirtualAlloc,
    };
    // SAFETY: plain allocation request; a null return is checked below.
    let p = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_EXECUTE_READWRITE,
        )
    };
    if p.is_null() {
        return Err(io::Error::last_os_error());
    }
    Ok(p as *mut u8)
}

#[cfg(windows)]
fn free(base: *mut u8, _size: usize) {
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
    // SAFETY: base came from VirtualAlloc with MEM_RESERVE.
    unsafe { VirtualFree(base as _, 0, MEM_RELEASE) };
}

#[cfg(unix)]
fn alloc_rwx(size: usize) -> io::Result<*mut u8> {
    // SAFETY: anonymous private mapping; MAP_FAILED is checked below.
    let p = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if p == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok(p as *mut u8)
}

#[cfg(unix)]
fn free(base: *mut u8, size: usize) {
    // SAFETY: base/size describe the mapping made in alloc_rwx.
    unsafe { libc::munmap(base as _, size) };
}
