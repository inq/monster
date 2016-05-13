use cudnn::{ffi, Cudnn, Filter4d, Convolution2d, Pooling};
use nn::Tensor;

impl Cudnn {
    pub fn relu_forward_inplace(&self,
                                x: &mut Tensor)
                                -> Result<(), &'static str> {
        unsafe {
            ffi::cudnnActivationForward(self.handle,
                                        ffi::ActivationDescriptor::ReLU,
                                        *&[1.0f32].as_ptr() as *const ::libc::c_void,
                                        x.desc,
                                        x.data,
                                        *&[0.0f32].as_ptr() as *const ::libc::c_void,
                                        x.desc,
                                        x.data)
        }.to_result() 
    }
}

