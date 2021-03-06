mod conv;

pub use self::conv::*;
use std::ffi::CStr;
use std::str;
use libc::{c_int, c_void};

pub enum Context {}
pub type Handle = *mut Context;
pub type TensorDescriptor = *mut Context;
pub type FilterDescriptor = *mut Context;
pub type PoolingDescriptor = *mut Context;
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

    pub fn to_result(self) -> Result<(), &'static str> {
        match self {
            Status::Success => Ok(()),
            e => Err(e.to_str())
        }
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

#[allow(dead_code)]
#[repr(C)]
pub enum PoolingMode {
    Max = 0,
    AverageCountIncludePadding = 1,
    AverageCountExcludePadding = 2
}

#[allow(dead_code)]
#[repr(C)]
pub enum SoftmaxAlgorithm {
    Fast = 0,
    Accurate = 1,
    Log = 2
}

#[allow(dead_code)]
#[repr(C)]
pub enum SoftmaxMode {
    Instance = 0,
    Channel = 1
}

#[derive(Debug, Copy, Clone, PartialEq)]
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
    pub fn cudnnCreate(handle: *mut Handle)
                       -> Status;
    pub fn cudnnDestroy(handle: Handle)
                        -> Status;
    pub fn cudnnGetErrorString(status: Status)
                               -> *const i8;
    // Filter
    pub fn cudnnCreateFilterDescriptor(filterDesc: *mut FilterDescriptor)
                                       -> Status;
    pub fn cudnnSetFilter4dDescriptor(filterDesc: FilterDescriptor,
                                      dataType: DataType,
                                      k: c_int,
                                      c: c_int,
                                      h: c_int,
                                      w: c_int)
                                      -> Status;
    pub fn cudnnDestroyFilterDescriptor(filterDesc: FilterDescriptor)
                                        -> Status;

    // Tensor
    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut TensorDescriptor)
                                       -> Status;
    pub fn cudnnSetTensor4dDescriptor(tensorDesc: TensorDescriptor,
                                      format: Format,
                                      dataType: DataType,
                                      n: c_int,
                                      c: c_int,
                                      h: c_int,
                                      w: c_int)
                                      -> Status;
    pub fn cudnnAddTensor(handle: Handle,
                          alpha: *const c_void,
                          bDesc: TensorDescriptor,
                          b: *const c_void,
                          beta: *const c_void,
                          yDesc: TensorDescriptor,
                          y: *mut c_void)
                          -> Status;
    pub fn cudnnActivationForward(handle: Handle,
                                  activationDesc: ActivationDescriptor,
                                  alpha: *const c_void,
                                  srcDesc: TensorDescriptor,
                                  srcData: *const c_void,
                                  beta: *const c_void,
                                  destDesc: TensorDescriptor,
                                  destData: *const c_void)
                                  -> Status;
    // Pooling
    pub fn cudnnCreatePoolingDescriptor(handle: *mut PoolingDescriptor)
                                        -> Status;
    pub fn cudnnSetPooling2dDescriptor(poolingDesc: PoolingDescriptor,
                                       mode: PoolingMode,
                                       windowHeight: c_int,
                                       windowWidth: c_int,
                                       verticalPadding: c_int,
                                       horizontalPadding: c_int,
                                       verticalStride: c_int,
                                       horizontalStride: c_int)
                                       -> Status;
    pub fn cudnnPoolingForward(handle: Handle,
                               poolingDesc: PoolingDescriptor,
                               alpha: *const c_void,
                               xDesc: TensorDescriptor,
                               x: *const c_void,
                               beta: *const c_void,
                               yDesc: TensorDescriptor,
                               y: *const c_void)
                               -> Status;
    pub fn cudnnPoolingBackward(handle: Handle,
                                poolingDesc: PoolingDescriptor,
                                alpha: *const c_void,
                                yDesc: TensorDescriptor,
                                y: *const c_void,
                                dyDesc: TensorDescriptor,
                                dy: *const c_void,
                                xDesc: TensorDescriptor,
                                x: *const c_void,
                                beta: *const c_void,
                                dxDesc: TensorDescriptor,
                                dx: *mut c_void)
                                -> Status;
    pub fn cudnnGetPooling2dForwardOutputDim(poolingDesc: PoolingDescriptor,
                                             inputDesc: TensorDescriptor,
                                             outN: *mut c_int,
                                             outC: *mut c_int,
                                             outH: *mut c_int,
                                             outW: *mut c_int)
                                             -> Status;
    pub fn cudnnDestroyPoolingDescriptor(poolingDesc: PoolingDescriptor)
                                         -> Status;
    // Etc.
    pub fn cudnnSoftmaxForward(handle: Handle,
                               algorithm: SoftmaxAlgorithm,
                               mode: SoftmaxMode,
                               alpha: *const c_void,
                               xDesc: TensorDescriptor,
                               x: *const c_void,
                               beta: *const c_void,
                               yDesc: TensorDescriptor,
                               y: *const c_void)
                               -> Status;
    pub fn cudnnSoftmaxBackward(handle: Handle,
                                algorigthm: SoftmaxAlgorithm,
                                mode: SoftmaxMode,
                                alpha: *const c_void,
                                yDesc: TensorDescriptor,
                                y: *const c_void,
                                dyDesc: TensorDescriptor,
                                dy: *const c_void,
                                beta: *const c_void,
                                dxDesc: TensorDescriptor,
                                dx: *mut c_void)
                                -> Status;
}
