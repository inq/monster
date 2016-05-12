mod ffi;
mod memory;

pub use self::memory::Memory;

use libc::c_void;
use std::ptr::null_mut;

pub fn cuda_malloc(bytes: usize)
               -> Result<*mut c_void, &'static str> {
    let mut data: *mut c_void = null_mut();
    match unsafe { ffi::cudaMalloc(&mut data, bytes as u64) } {
        ffi::Error::Success => Ok(data),
        e => Err(e.to_str())
    }
}
