enum Context {}
pub type Handle = *mut Context;
pub type TensorDescriptor = *mut Context;

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

extern "C" {
    pub fn cudnnCreate(handle: *mut Handle) -> Status;

    pub fn cudnnDestroy(handle: Handle) -> Status;

    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut TensorDescriptor) -> Status;
    
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
