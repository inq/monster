use libc::{c_int, c_float};

pub enum Context {}
pub type Handle = *mut Context;

#[allow(dead_code)]
#[repr(C)]
pub enum Status {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16
}

impl Status {
    pub fn to_str(self) -> &'static str {
        match self {
            Status::Success => "CUBLAS_STATUS_SUCCESS",
            Status::NotInitialized => "CUBLAS_STATUS_NOT_INITIALIZED",
            Status::AllocFailed => "CUBLAS_STATUS_ALLOC_FAILED",
            Status::InvalidValue => "CUBLAS_STATUS_INVALID_VALUE",
            Status::ArchMismatch => "CUBLAS_STATUS_ARCH_MISMATCH",
            Status::MappingError => "CUBLAS_STATUS_MAPPING_ERROR",
            Status::ExecutionFailed => "CUBLAS_STATUS_EXECUTION_FAILED",
            Status::InternalError => "CUBLAS_STATUS_INTERNAL_ERROR",
            Status::NotSupported => "CUBLAS_STATUS_NOT_SUPPORTED",
            Status::LicenseError => "CUBLAS_STATUS_LICENSE_ERROR"
        }
    }
}

#[allow(dead_code)]
#[repr(C)]
pub enum Operation {
    N = 0,
    T = 1,
    C = 2
}

#[link(name = "cublas")]
extern "C" {
    pub fn cublasCreate_v2(handle: *mut Handle)
                           -> Status;
    pub fn cublasDestroy_v2(handle: Handle)
                            -> Status;
    pub fn cublasSgemv_v2(handle: Handle,
                          trans: Operation,
                          m: c_int,
                          n: c_int,
                          alpha: *const c_float, // host or device pointer
                          A: *const c_float,
                          lda: c_int,
                          x: *const c_float,
                          incx: c_int,
                          beta: *const c_float, // host or device pointer
                          y: *const c_float,
                          incy: c_int)
                          -> Status;

    pub fn cublasSger_v2(handle: Handle,
                         m: c_int,
                         n: c_int,
                         alpha: *const c_float,
                         x: *const c_float,
                         incx: c_int,
                         y: *const c_float,
                         incy: c_int,
                         A: *mut c_float,
                         lda: c_int)
                         -> Status;
                         
}
