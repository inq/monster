use super::ffi;
use cuda;
use std::ptr;

pub struct Cudnn {
    handle: ffi::Handle
}

impl Cudnn {
    pub fn new() -> Result<Cudnn, &'static str> {
        let mut handle: ffi::Handle = ptr::null_mut();
        match unsafe { ffi::cudnnCreate(&mut handle) } {
            ffi::Status::Success => Ok(Cudnn { handle : handle }),
            ffi::Status::NotInitialized => Err("CUDA Runtime API initialization failed."),
            ffi::Status::AllocFailed => Err("The resources could not be allocated."),
            _ => Err("Unknown Error")
        }
    }

    pub fn sigmoid_forward(self,
                           src_desc: super::Tensor,
                           src: &cuda::Memory<f32>,
                           dst_desc: super::Tensor,
                           dst: &mut cuda::Memory<f32>) -> Result<(), &'static str> {
        match unsafe { ffi::cudnnActivationForward(self.handle,
                                                   ffi::ActivationDescriptor::Sigmoid,
                                                   *&[1.0f32].as_ptr() as *const ::libc::c_void,
                                                   src_desc.descriptor,
                                                   src.data,
                                                   *&[0.0f32].as_ptr() as *const ::libc::c_void,
                                                   dst_desc.descriptor,
                                                   dst.data) } {
            ffi::Status::Success => Ok(()),
            ffi::Status::BadParam => Err("Bad Parameters."),
            ffi::Status::ExecutionFailed => Err("The function failed to launch on the GPU."),
            _ => Err("Unknown Error")
        }
    }
}
