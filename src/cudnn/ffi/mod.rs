use std::ffi::CStr;
use std::str;

pub enum Context {}
pub type Handle = *mut Context;
pub type TensorDescriptor = *mut Context;
pub type FilterDescriptor = *mut Context;
pub type ConvolutionDescriptor = *mut Context;

#[allow(dead_code)]
#[repr(C)]
pub enum Status {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 2,
    BadParam = 3,
    InternalError = 4,
    InvalidValue = 5,
    ArchMismatch = 6,
    MappingError = 7,
    ExecutionFailed = 8,
    NotSupported = 9,
    LicenseError = 10,
}

impl Status {
    pub fn to_str(self) -> &'static str{
        let buf = unsafe { CStr::from_ptr(cudnnGetErrorString(self) ) }.to_bytes();
        str::from_utf8(buf).unwrap()
    }
}

#[allow(dead_code)]
#[repr(C)]
pub enum Format {
    NCHW = 0,
    NWHC = 1
}

#[allow(dead_code)]
#[repr(C)]
pub enum DataType {
    Float = 0,
    Double = 1,
    Half = 2
}

#[allow(dead_code)]
#[repr(C)]
pub enum ActivationDescriptor {
    Sigmoid = 0,
    ReLU = 1,
    Tanh = 2,
    ClippedReLU = 3
}

#[allow(dead_code)]
#[repr(C)]
pub enum ConvolutionMode {
    Convolution = 0,
    CrossCorrelation = 1
}

#[allow(dead_code)]
#[repr(C)]
pub enum ConvolutionFwdPreference {
    NoWorkspace = 0,
    PreferFastest = 1,
    SpecifyWorkspaceLimit = 2
}

#[derive(Debug)]
#[derive(PartialEq)]
#[allow(dead_code)]
#[repr(C)]
pub enum ConvolutionFwdAlgo {
    ImplicitGemm = 0,
    ImplicitPrecompGemm = 1,
    Gemm = 2,
    Direct = 3,
    Fft = 4,
    FftTiling = 5,
    Winograd = 6
}
    
#[link(name = "cudnn")]
extern "C" {
    // Cudnn
    pub fn cudnnCreate(handle: *mut Handle) -> Status;

    pub fn cudnnDestroy(handle: Handle) -> Status;

    pub fn cudnnGetErrorString(status: Status) -> *const i8;

    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut TensorDescriptor) -> Status;

    // Filter
    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut FilterDescriptor) -> Status;

    pub fn cudnnSetFilter4dDescriptor(filterDesc: FilterDescriptor,
                                      dataType: DataType,
                                      k: ::libc::c_int,
                                      c: ::libc::c_int,
                                      h: ::libc::c_int,
                                      w: ::libc::c_int) -> Status;

    pub fn cudnnDestroyFilterDescriptor(filterDesc: FilterDescriptor) -> Status;

    // Convolution2d
    pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut ConvolutionDescriptor) -> Status;

    pub fn cudnnSetConvolution2dDescriptor(convDesc: ConvolutionDescriptor,
                                           pad_h: ::libc::c_int,
                                           pad_w: ::libc::c_int,
                                           u: ::libc::c_int,
                                           v: ::libc::c_int,
                                           upscalex: ::libc::c_int,
                                           upscaley: ::libc::c_int,
                                           mode: ConvolutionMode) -> Status;

    pub fn cudnnGetConvolution2dForwardOutputDim(convDesc: ConvolutionDescriptor,
                                                 inputTensorDesc: TensorDescriptor,
                                                 filterDesc: FilterDescriptor,
                                                 n: *mut ::libc::c_int,
                                                 c: *mut ::libc::c_int,
                                                 h: *mut ::libc::c_int,
                                                 w: *mut ::libc::c_int) -> Status;

    pub fn cudnnGetConvolutionForwardAlgorithm(handle: Handle,
                                               xDesc: TensorDescriptor,
                                               wDesc: FilterDescriptor,
                                               convDesc: ConvolutionDescriptor,
                                               yDesc: TensorDescriptor,
                                               preference: ConvolutionFwdPreference,
                                               memoryLimitInBytes: ::libc::size_t,
                                               algo: *mut ConvolutionFwdAlgo) -> Status;

    pub fn cudnnDestroyConvolutionDescriptor(convDesc: ConvolutionDescriptor) -> Status;

    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: TensorDescriptor,
        format: Format,
        dataType: DataType,
        n: ::libc::c_int,
        c: ::libc::c_int,
        h: ::libc::c_int,
        w: ::libc::c_int
    ) -> Status;

    pub fn cudnnActivationForward(
        handle: Handle,
        activationDesc: ActivationDescriptor,
        alpha: *const ::libc::c_void,
        srcDesc: TensorDescriptor,
        srcData: *const ::libc::c_void,
        beta: *const ::libc::c_void,
        destDesc: TensorDescriptor,
        destData: *const ::libc::c_void
    ) -> Status;
}
