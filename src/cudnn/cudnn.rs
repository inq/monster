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
                        x: &Tensor,
                        w_desc: &Filter4d,
                        w: &Tensor,
                        conv_desc: &Convolution2d,
                        y_desc: &Tensor,
                        y: &Tensor)
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
                            x: &Tensor,
                            w_desc: &Filter4d,
                            w: &Tensor,
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
                               src_memory: &Tensor,
                               dst_tensor: &Tensor,
                               dst_memory: &Tensor) 
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

    pub fn max_pooling_backward(&self,
                                pooling: &Pooling,
                                y_tensor: &Tensor,
                                y_memory: &Tensor,
                                dy_tensor: &Tensor,
                                dy_memory: &Tensor,
                                x_tensor: &Tensor,
                                x_memory: &Tensor,
                                dx_tensor: &mut Tensor,
                                dx_memory: &mut Tensor) 
                                -> Result<(), &'static str> {
        let alpha = 1f32;
        let beta = 0f32;
        match unsafe { ffi::cudnnPoolingBackward(self.handle,
                                                 pooling.desc,
                                                 &alpha as *const _ as *const ::libc::c_void,
                                                 y_tensor.desc,
                                                 y_memory.data,
                                                 dy_tensor.desc,
                                                 dy_memory.data,
                                                 x_tensor.desc,
                                                 x_memory.data,
                                                 &beta as *const _ as *const ::libc::c_void,
                                                 dx_tensor.desc,
                                                 dx_memory.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn add_bias(&self,
                    bias_tensor: &Tensor,
                    bias_memory: &Tensor,
                    dst_tensor: &Tensor,
                    dst_memory: &Tensor)
                    -> Result<(), &'static str> {
        let alpha = 1f32;
        let beta = 1f32;
        match unsafe { ffi::cudnnAddTensor(self.handle,
                                           &alpha as *const _ as *const ::libc::c_void,
                                           bias_tensor.desc,
                                           bias_memory.data,
                                           &beta as *const _ as *const ::libc::c_void,
                                           dst_tensor.desc,
                                           dst_memory.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }                                           
    }

    pub fn bias_backward(&self,
                         scale: f32,
                         bias_tensor: &mut Tensor,
                         bias_memory: &Tensor,
                         dy_tensor: &Tensor,
                         dy_memory: &Tensor)
                         -> Result<(), &'static str> {
        let beta = 1f32;
        match unsafe { ffi::cudnnAddTensor(self.handle,
                                           &scale as *const _ as *const ::libc::c_void,
                                           dy_tensor.desc,
                                           dy_memory.data,
                                           &beta as *const _ as *const ::libc::c_void,
                                           bias_tensor.desc,
                                           bias_memory.data) } {
            ffi::Status::Success => Ok(()),
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
