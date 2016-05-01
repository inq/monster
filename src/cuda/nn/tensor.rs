use super::ffi;

pub struct Tensor {
    descriptor: ffi::TensorDescriptor
}

impl Tensor {
    pub unsafe fn new() -> Tensor {
        let mut descriptor: ffi::Handle = ::std::ptr::null_mut();
        let status = ffi::cudnnCreate(&mut descriptor);
        Tensor {
            descriptor : descriptor
        }
    }
}
