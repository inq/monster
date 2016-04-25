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

pub struct Memory {
    pub data: *mut ::libc::c_void,
    size: u64
}

impl Memory {
    pub fn new(size: u64) -> Memory {
        let mut data: *mut ::libc::c_void = ::std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut data, size) };
        Memory {
            data: data,
            size: size
        }
    }

    pub fn write(&self, data: *const ::libc::c_void, size: u64) {
        unsafe { cudaMemcpy(self.data, data, size, MemcpyKind::HostToDevice) };
    }

    pub fn read(&self, data: *mut ::libc::c_void, size: u64) {
        unsafe { cudaMemcpy(data, self.data, size, MemcpyKind::DeviceToHost) };
    }
}

