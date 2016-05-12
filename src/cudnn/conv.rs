use Tensor;
use cudnn::Filter4d;
use cudnn::ffi;
use std::ptr;

pub struct Convolution2d {
    pub desc: ffi::ConvolutionDescriptor
}

impl Drop for Convolution2d {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyConvolutionDescriptor(self.desc) };
    }
}

impl Convolution2d {
    fn new_desc() -> Result<Convolution2d, &'static str> {
        let mut desc: ffi::ConvolutionDescriptor = ptr::null_mut();
        match unsafe { ffi::cudnnCreateConvolutionDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Convolution2d { desc : desc }),
            e => Err(e.to_str())
        }
    }

    pub fn new(pad_h: i32,
               pad_w: i32,
               u: i32,
               v: i32,
               upscalex: i32,
               upscaley: i32) -> Result<Convolution2d, &'static str> {
        let conv = try! { Convolution2d::new_desc() };
        match unsafe { ffi::cudnnSetConvolution2dDescriptor(conv.desc, 
                                                            pad_h,
                                                            pad_w,
                                                            u, v,
                                                            upscalex,
                                                            upscaley,
                                                            ffi::ConvolutionMode::Convolution) } {
            ffi::Status::Success => Ok(conv),
            e => Err(e.to_str())
        }
    }

    pub fn get_forward_output_dim(&self,
                                  input_tensor: &Tensor,
                                  filter: &Filter4d) -> Result<(i32, i32, i32, i32), &'static str> {
        let (mut n, mut c, mut h, mut w) = (0i32, 0i32, 0i32, 0i32);
        match unsafe { ffi::cudnnGetConvolution2dForwardOutputDim(self.desc,
                                                                  input_tensor.desc,
                                                                  filter.desc,
                                                                  &mut n,
                                                                  &mut c,
                                                                  &mut h,
                                                                  &mut w) } {
            ffi::Status::Success => Ok((n, c, h, w)),
            e => Err(e.to_str())
        }
    }
}
