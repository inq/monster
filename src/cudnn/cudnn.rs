use cudnn::ffi;
use cudart;
use std::ptr;

pub struct Cudnn {
    handle: ffi::Handle
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroy(self.handle) };
    }
}

impl Cudnn {
    pub fn new() -> Result<Cudnn, &'static str> {
        let mut handle: ffi::Handle = ptr::null_mut();
        match unsafe { ffi::cudnnCreate(&mut handle) } {
            ffi::Status::Success => Ok(Cudnn { handle : handle }),
            e => Err(e.to_str())
        }
    }

    pub fn sigmoid_forward(self,
                           src_desc: super::Tensor,
                           src: &cudart::Memory<f32>,
                           dst_desc: super::Tensor,
                           dst: &mut cudart::Memory<f32>) -> Result<(), &'static str> {
        match unsafe { ffi::cudnnActivationForward(self.handle,
                                                   ffi::ActivationDescriptor::Sigmoid,
                                                   *&[1.0f32].as_ptr() as *const ::libc::c_void,
                                                   src_desc.desc,
                                                   src.data,
                                                   *&[0.0f32].as_ptr() as *const ::libc::c_void,
                                                   dst_desc.desc,
                                                   dst.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }
}
