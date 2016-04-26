use std::mem;

pub enum Error {
    Success = 0,
    ErrorInvalidValue = 1,
}

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

pub struct Memory<T> {
    pub data: *mut ::libc::c_void,
    size: usize,
    dummy: [T; 0]
}

impl<T> Memory<T> {
    pub fn new(size: usize) -> Memory<T> {
        let mut data: *mut ::libc::c_void = ::std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut data, (size * ::std::mem::size_of::<T>()) as u64) };
        Memory::<T> {
            data: data,
            size: size,
            dummy: []
        }
    }

    pub fn write(&self, data: *const ::libc::c_void, size: usize) {
        unsafe { cudaMemcpy(self.data, data, (size * ::std::mem::size_of::<T>()) as u64, MemcpyKind::HostToDevice) };
    }

    pub fn read(&self, data: *mut ::libc::c_void, size: usize) {
        unsafe { cudaMemcpy(data, self.data, (size * ::std::mem::size_of::<T>()) as u64, MemcpyKind::DeviceToHost) };
    }
}

