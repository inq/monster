use cudnn::{ffi, Cudnn, Filter4d, Convolution2d, Pooling};
use nn::Tensor;

impl Cudnn {
    pub fn activation_forward(&self,
                              mode: ffi::ActivationDescriptor,
                              alpha: f32, x: &Tensor,
                              beta: f32, y: &Tensor)
                              -> Result<(), &'static str> {
        unsafe {
            ffi::cudnnActivationForward(self.handle,
                                        mode,
                                        &alpha as *const _ as *const ::libc::c_void,
                                        x.desc, x.data,
                                        &beta as *const _ as *const ::libc::c_void,
                                        y.desc, y.data)
        }.to_result()
    }

    pub fn activation_forward_inplace(&self,
                                      mode: ffi::ActivationDescriptor,
                                      alpha: f32, x: &Tensor, beta: f32)
                                      -> Result<(), &'static str> {
        unsafe {
            ffi::cudnnActivationForward(self.handle,
                                        mode,
                                        &alpha as *const _ as *const ::libc::c_void,
                                        x.desc, x.data,
                                        &beta as *const _ as *const ::libc::c_void,
                                        x.desc, x.data)
        }.to_result()
    }
}
