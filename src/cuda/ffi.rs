#[allow(dead_code)]
#[repr(C)]
pub enum Error {
    Success = 0,
    MemoryAllocation = 2,
    InvalidValue = 11,
    InvalidDevicePointer = 17,
    InvalidMemcpyDirection = 21
}

#[allow(dead_code)]
#[repr(C)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4
}

extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut ::libc::c_void,
                      size: ::libc::c_ulong)
        -> Error;

    pub fn cudaMemcpy(dst: *mut ::libc::c_void,
                      src: *const ::libc::c_void,
                      count: ::libc::c_ulong,
                      kind: MemcpyKind)
        -> Error;
}
