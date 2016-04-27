enum Context {}
pub type Handle = *mut Context;
pub type TensorDescriptor = *mut Context;
pub type TensorFormat = *mut Context;
pub type DataType = *mut Context;

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

extern "C" {
    pub fn cudnnCreate(handle: *mut Handle) -> Status;

    pub fn cudnnDestroy(handle: Handle) -> Status;

    pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut TensorDescriptor) -> Status;
    
    pub fn cudnnSetTensor4dDescriptor(
        tensorDesc: TensorDescriptor,
        format: TensorFormat,
        dataType: DataType,
        n: ::libc::c_int,
        c: ::libc::c_int,
        h: ::libc::c_int,
        w: ::libc::c_int
    ) -> Status;
}
