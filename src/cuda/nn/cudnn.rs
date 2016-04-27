use super::ffi;
use std::ptr;

pub struct Cudnn {
    handle: ffi::Handle
}

impl Cudnn {
    pub unsafe fn new() -> Cudnn {
        let mut handle: ffi::Handle = ptr::null_mut();
        let status = ffi::cudnnCreate(&mut handle);
        Cudnn {
            handle : handle
        }
    }
}
