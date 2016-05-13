use nn::Tensor;
use cudnn::{ffi, Cudnn, Filter};
use cudnn::ffi::{ConvolutionMode};
use cudart::Memory;
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
                                                      ConvolutionMode::Convolution) } {
            ffi::Status::Success => Ok(conv),
            e => Err(e.to_str())
        }
    }

    pub fn get_forward_output_dim(&self,
                                  input_tensor: &Tensor,
                                  filter: &Filter) -> Result<(i32, i32, i32, i32), &'static str> {
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

impl Cudnn {
    pub fn convolution_forward(&self,
                               alpha: f32, x: &Tensor, w: &Tensor,
                               conv: &Convolution2d,
                               algo: ffi::ConvolutionFwdAlgo,
                               workspace: &Memory<f32>, workspace_size: usize,
                               beta: f32, y: &Tensor)
                               -> Result<(), &'static str> {
        unsafe {
            ffi::cudnnConvolutionForward(self.handle,
                                         &alpha as *const _ as *const ::libc::c_void,
                                         x.desc, x.data, w.desc, w.data,
                                         conv.desc,
                                         algo,
                                         workspace.data, workspace_size,
                                         &beta as *const _ as *const ::libc::c_void,
                                         y.desc, y.data)
        }.to_result()
    }
    
    pub fn get_conv_forward_algo(&self,
                                 x: &Tensor, w: &Filter,
                                 conv: &Convolution2d, y: &Tensor,
                                 preference: ffi::ConvolutionFwdPreference)
                                 -> Result<ffi::ConvolutionFwdAlgo, &'static str> {
        let mut res = ffi::ConvolutionFwdAlgo::ImplicitGemm;
        match unsafe { ffi::cudnnGetConvolutionForwardAlgorithm(self.handle,
                                                                x.desc, w.desc, conv.desc, y.desc,
                                                                preference, 0,
                                                                &mut res) } {
            ffi::Status::Success => Ok(res),
            e => Err(e.to_str())
        }
    }

    pub fn get_conv_forward_workspace_size(&self,
                                           x: &Tensor,
                                           filter: &Filter,
                                           conv: &Convolution2d,
                                           y: &Tensor,
                                           algo: ffi::ConvolutionFwdAlgo)
                                           -> Result<usize, &'static str> {

        let mut res = 0usize;
        match unsafe { ffi::cudnnGetConvolutionForwardWorkspaceSize(self.handle,
                                                                      x.desc,
                                                                      filter.desc,
                                                                      conv.desc,
                                                                      y.desc,
                                                                      algo,
                                                                      &mut res) } {
            ffi::Status::Success => Ok(res),
            e => Err(e.to_str())
        }
    }
}
