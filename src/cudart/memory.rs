use cudart::ffi;
use libc::c_void;
use std::ptr::null_mut;
use std::mem::size_of;

#[allow(dead_code)]
pub struct Memory<T> {
    pub data: *mut c_void,
    size: usize,
    dummy: [T; 0]
}

impl<T> Memory<T> {
    pub fn new(size: usize) -> Result<Memory<T>, &'static str> {
        let mut data: *mut c_void = null_mut();
        match unsafe { ffi::cudaMalloc(&mut data, (size * size_of::<T>()) as u64) } {
            ffi::Error::Success => Ok(Memory::<T> {
                data: data,
                size: size,
                dummy: []
            }),
            e => Err(e.to_str())
        }
    }

    pub fn write(&self, data: &Vec<T>) -> Result<(), &'static str> {
        match unsafe { ffi::cudaMemcpy(self.data,
                                       data.as_ptr() as *mut c_void,
                                       (data.len() * size_of::<T>()) as u64,
                                       ffi::MemcpyKind::HostToDevice) } {
            ffi::Error::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn read(&self, data: &Vec<T>) -> Result<(), &'static str> {
        match unsafe { ffi::cudaMemcpy(data.as_ptr() as *mut c_void,
                                       self.data,
                                       (data.len() * size_of::<T>()) as u64,
                                       ffi::MemcpyKind::DeviceToHost) } {
            ffi::Error::Success => Ok(()),
            e => Err(e.to_str())
        }
    }
}
