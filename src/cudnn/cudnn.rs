use cudnn::ffi;
use cudnn::{Tensor, Filter4d, Convolution2d, Pooling};
use cudart::Memory;
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

    pub fn sigmoid_forward(&self,
                           src_desc: Tensor,
                           src: &Memory<f32>,
                           dst_desc: Tensor,
                           dst: &mut Memory<f32>)
                           -> Result<(), &'static str> {
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

    pub fn relu_forward_inplace(&self,
                                src_desc: &Tensor,
                                src: &mut Memory<f32>)
                                -> Result<(), &'static str> {
        match unsafe { ffi::cudnnActivationForward(self.handle,
                                                   ffi::ActivationDescriptor::ReLU,
                                                   *&[1.0f32].as_ptr() as *const ::libc::c_void,
                                                   src_desc.desc,
                                                   src.data,
                                                   *&[0.0f32].as_ptr() as *const ::libc::c_void,
                                                   src_desc.desc,
                                                   src.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn get_conv_forward_algo(&self,
                                 x_desc: &Tensor,
                                 w_desc: &Filter4d,
                                 conv_desc: &Convolution2d,
                                 y_desc: &Tensor)
                                 -> Result<ffi::ConvolutionFwdAlgo, &'static str> {
        let mut res = ffi::ConvolutionFwdAlgo::ImplicitGemm;
        match unsafe { ffi::cudnnGetConvolutionForwardAlgorithm(self.handle,
                                                                x_desc.desc,
                                                                w_desc.desc,
                                                                conv_desc.desc,
                                                                y_desc.desc,
                                                                ffi::ConvolutionFwdPreference::NoWorkspace,
                                                                0,
                                                                &mut res) } {
            ffi::Status::Success => Ok(res),
            e => Err(e.to_str())
        }
    }

    pub fn get_conv_forward_workspace_size(&self,
                                           x_desc: &Tensor,
                                           w_desc: &Filter4d,
                                           conv_desc: &Convolution2d,
                                           y_desc: &Tensor,
                                           algo: ffi::ConvolutionFwdAlgo)
                                           -> Result<usize, &'static str> {

        let mut res = 0usize;
        match unsafe { ffi::cudnnGetConvolutionForwardWorkspaceSize(self.handle,
                                                                    x_desc.desc,
                                                                    w_desc.desc,
                                                                    conv_desc.desc,
                                                                    y_desc.desc,
                                                                    algo,
                                                                    &mut res) } {
            ffi::Status::Success => Ok(res),
            e => Err(e.to_str())
        }
    }

    pub fn conv_forward(&self,
                        x_desc: &Tensor,
                        x: &Memory<f32>,
                        w_desc: &Filter4d,
                        w: &Memory<f32>,
                        conv_desc: &Convolution2d,
                        y_desc: &Tensor,
                        y: &Memory<f32>)
                        -> Result<(), &'static str> {
        let alpha = 1f32;
        let beta = 0f32;
        let algo = try!(self.get_conv_forward_algo(&x_desc,
                                                   &w_desc,
                                                   &conv_desc,
                                                   &y_desc));
        let workspace_size = try!(self.get_conv_forward_workspace_size(&x_desc,
                                                                       &w_desc,
                                                                       &conv_desc,
                                                                       &y_desc,
                                                                       algo));
        let workspace = try!(Memory::<f32>::new(workspace_size / 4));
        match unsafe { ffi::cudnnConvolutionForward(self.handle,
                                                    &alpha as *const _ as *const ::libc::c_void,
                                                    x_desc.desc,
                                                    x.data,
                                                    w_desc.desc,
                                                    w.data,
                                                    conv_desc.desc,
                                                    algo,
                                                    workspace.data,
                                                    workspace_size,
                                                    &beta as *const _ as *const ::libc::c_void,
                                                    y_desc.desc,
                                                    y.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn conv_forward_src(&self,
                            x_desc: &Tensor,
                            x: &Memory<f32>,
                            w_desc: &Filter4d,
                            w: &Memory<f32>,
                            conv_desc: &Convolution2d,
                            y_desc: &Tensor)
                            -> Result<(), &'static str> {
        let alpha = 1f32;
        let beta = 0f32;
        let algo = try!(self.get_conv_forward_algo(&x_desc,
                                                   &w_desc,
                                                   &conv_desc,
                                                   &y_desc));
        let workspace_size = try!(self.get_conv_forward_workspace_size(&x_desc,
                                                                       &w_desc,
                                                                       &conv_desc,
                                                                       &y_desc,
                                                                       algo));
        let workspace = try!(Memory::<f32>::new(workspace_size / 4));
        match unsafe { ffi::cudnnConvolutionForward(self.handle,
                                                    &alpha as *const _ as *const ::libc::c_void,
                                                    x_desc.desc,
                                                    x.data,
                                                    w_desc.desc,
                                                    w.data,
                                                    conv_desc.desc,
                                                    algo,
                                                    workspace.data,
                                                    workspace_size,
                                                    &beta as *const _ as *const ::libc::c_void,
                                                    y_desc.desc,
                                                    x.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn max_pooling_forward(&self,
                               pooling: &Pooling,
                               src_tensor: &Tensor,
                               src_memory: &Memory<f32>,
                               dst_tensor: &Tensor,
                               dst_memory: &Memory<f32>) 
                               -> Result<(), &'static str> {
        let alpha = 1f32;
        let beta = 0f32;
        match unsafe { ffi::cudnnPoolingForward(self.handle,
                                                pooling.desc,
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
}
