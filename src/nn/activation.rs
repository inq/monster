use cudnn::{ffi, Cudnn, Filter4d, Convolution2d, Pooling};
use nn::Tensor;

pub trait Activation {
    fn activation_forward(&self,
                          mode: ffi::ActivationDescriptor::ReLU,
                          alpha: f32, x: &Tensor,
                          beta: f32, y: &mut Tensor)
                          -> Result<(), &'static str>;
}

impl Cudnn {
    pub fn activation_forward(&self,
                              mode: ffi::ActivationDescriptor,
                              alpha: f32, x: &Tensor,
                              beta: f32, y: &mut Tensor)
                              -> Result<(), &'static str> {
        unsafe {
            ffi::cudnnActivationForward(self.handle,
                                        ffi::ActivationDescriptor::ReLU,
                                        &alpha as *const _ as *const ::libc::c_void,
                                        x.desc, x.data,
                                        &alpha as *const _ as *const ::libc::c_void,
                                        y.desc, y.data)
        }.to_result()
    }
}
