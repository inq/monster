use cudnn::ffi::*;

use libc::{c_int, c_void, size_t};

#[link(name = "cudnn")]
extern "C" {
    // Convolution2d
    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut ConvolutionDescriptor)
                                            -> Status;
    pub fn cudnnSetConvolution2dDescriptor(convDesc: ConvolutionDescriptor,
                                           pad_h: c_int,
                                           pad_w: c_int,
                                           u: c_int,
                                           v: c_int,
                                           upscalex: c_int,
                                           upscaley: c_int,
                                           mode: ConvolutionMode)
                                           -> Status;
    pub fn cudnnGetConvolution2dForwardOutputDim(convDesc: ConvolutionDescriptor,
                                                 inputTensorDesc: TensorDescriptor,
                                                 filterDesc: FilterDescriptor,
                                                 n: *mut c_int,
                                                 c: *mut c_int,
                                                 h: *mut c_int,
                                                 w: *mut c_int)
                                                 -> Status;
    pub fn cudnnGetConvolutionForwardAlgorithm(handle: Handle,
                                               xDesc: TensorDescriptor,
                                               wDesc: FilterDescriptor,
                                               convDesc: ConvolutionDescriptor,
                                               yDesc: TensorDescriptor,
                                               preference: ConvolutionFwdPreference,
                                               memoryLimitInBytes: size_t,
                                               algo: *mut ConvolutionFwdAlgo)
                                               -> Status;
    pub fn cudnnGetConvolutionForwardWorkspaceSize(handle: Handle,
                                                   xDesc: TensorDescriptor,
                                                   wDesc: FilterDescriptor,
                                                   convDesc: ConvolutionDescriptor,
                                                   yDesc: TensorDescriptor,
                                                   algo: ConvolutionFwdAlgo,
                                                   sizeInBytes: *mut size_t)
                                                   -> Status;
    pub fn cudnnConvolutionForward(handle: Handle,
                                   alpha: *const c_void,
                                   xDesc: TensorDescriptor,
                                   x: *const c_void,
                                   wDesc: FilterDescriptor,
                                   w: *const c_void,
                                   convDesc: ConvolutionDescriptor,
                                   algo: ConvolutionFwdAlgo,
                                   workspace: *mut c_void,
                                   workSpaceSizeInBytes: size_t,
                                   beta: *const c_void,
                                   y_desc: TensorDescriptor,
                                   y: *mut c_void)
                                   -> Status;
    pub fn cudnnDestroyConvolutionDescriptor(convDesc: ConvolutionDescriptor)
                                             -> Status;
}
