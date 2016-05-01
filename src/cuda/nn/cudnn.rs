
use super::ffi;
use cuda;
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

    pub fn sigmoid_forward(self,
                           src_desc: super::Tensor,
                           src: &cuda::Memory<f32>,
                           dst_desc: super::Tensor,
                           dst: &mut cuda::Memory<f32>) -> Cudnn {
        let status = unsafe {ffi::cudnnActivationForward(self.handle,
                                                         ffi::ActivationDescriptor::Sigmoid,
                                                         *&[1.0f32].as_ptr() as *const ::libc::c_void,
                                                         src_desc.descriptor,
                                                         src.data,
                                                         *&[0.0f32].as_ptr() as *const ::libc::c_void,
                                                         dst_desc.descriptor,
                                                         dst.data) };
        println!("{:?}", status);

        self
    }
}
