use cudnn::{ffi, Cudnn, Filter, Tensor};
use cudart::Memory;
use std::ptr;
use nn::Res;

pub struct Convolution {
    pub desc: ffi::ConvolutionDescriptor
}

impl Drop for Convolution {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyConvolutionDescriptor(self.desc) };
    }
}

impl Convolution {
    pub fn new() -> Res<Convolution> {
        let mut desc: ffi::ConvolutionDescriptor = ptr::null_mut();
        match unsafe { ffi::cudnnCreateConvolutionDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Convolution { desc : desc }),
            e => Err(e.to_str())
        }
    }

    pub fn set_2d_desc(&self,
                       pad_h: i32, pad_w: i32,
                       u: i32, v: i32,
                       upscalex: i32, upscaley: i32,
                       mode: ffi::ConvolutionMode)
                       -> Res<()> {
        unsafe {
            ffi::cudnnSetConvolution2dDescriptor(self.desc, 
                                                 pad_h, pad_w,
                                                 u, v,
                                                 upscalex, upscaley,
                                                 mode)
        }.to_result()
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
                               conv: &Convolution,
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
                                 conv: &Convolution, y: &Tensor,
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
                                           conv: &Convolution,
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
