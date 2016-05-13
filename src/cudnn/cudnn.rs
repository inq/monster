use cudnn::{ffi, Filter4d, Convolution2d, Pooling};
use nn::Tensor;
use cudart::Memory;
use std::ptr;

pub struct Cudnn {
    pub handle: ffi::Handle
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

    pub fn softmax_forward(&self,
                           src_tensor: &Tensor,
                           src_memory: &Tensor,
                           dst_tensor: &Tensor,
                           dst_memory: &Tensor)
                           -> Result<(), &'static str> {
        let alpha = 1f32;
        let beta = 0f32;
        match unsafe { ffi::cudnnSoftmaxForward(self.handle,
                                                ffi::SoftmaxAlgorithm::Fast,
                                                ffi::SoftmaxMode::Channel,
                                                &alpha as *const _ as *const ::libc::c_void,
                                                src_tensor.desc,
                                                src_memory.data,
                                                &beta as *const _ as *const ::libc::c_void,
                                                dst_tensor.desc,
                                                dst_memory.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn softmax_backward(&self,
                            y_tensor: &Tensor,
                            y_memory: &Tensor,
                            dy_tensor: &Tensor,
                            dy_memory: &Tensor,
                            dx_tensor: &mut Tensor,
                            dx_memory: &Tensor)
                            -> Result<(), &'static str>{
        let alpha = 1f32;
        let beta = 0f32;
        match unsafe { ffi::cudnnSoftmaxBackward(self.handle,
                                                 ffi::SoftmaxAlgorithm::Fast,
                                                 ffi::SoftmaxMode::Channel,
                                                 &alpha as *const _ as *const ::libc::c_void,
                                                 y_tensor.desc,
                                                 y_memory.data,
                                                 dy_tensor.desc,
                                                 dy_memory.data,
                                                 &beta as *const _ as *const ::libc::c_void,
                                                 dx_tensor.desc,
                                                 dx_memory.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }
}
