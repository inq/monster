use cuda::ffi;

#[allow(dead_code)]
pub struct Memory<T> {
    pub data: *mut ::libc::c_void,
    size: usize,
    dummy: [T; 0]
}

impl<T> Memory<T> {
    pub fn new(size: usize) -> Result<Memory<T>, &'static str> {
        let mut data: *mut ::libc::c_void = ::std::ptr::null_mut();
        match unsafe { ffi::cudaMalloc(&mut data, (size * ::std::mem::size_of::<T>()) as u64) } {
            ffi::Error::Success => Ok(Memory::<T> {
                data: data,
                size: size,
                dummy: []
            }),
            ffi::Error::MemoryAllocation => Err("Unable to allocate enough memory to perform the requested operation."),
            _ => Err("Unknown error")
        }
    }

    pub fn write(&self, data: &Vec<T>) -> Result<(), &'static str> {
        match unsafe { ffi::cudaMemcpy(self.data,
                                       data.as_ptr() as *mut ::libc::c_void,
                                       (data.len() * ::std::mem::size_of::<T>()) as u64,
                                       ffi::MemcpyKind::HostToDevice) } {
            ffi::Error::Success => Ok(()),
            ffi::Error::InvalidValue => Err("one or more of the parameters passed to the API call is not within an acceptable range of values."),
            ffi::Error::InvalidDevicePointer => Err("At least one device pointer passed to the API call is not a valid device pointer."),
            ffi::Error::InvalidMemcpyDirection => Err("The direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind."),
            _ => Err("Unknown error")
        }
    }

    pub fn read(&self, data: &Vec<T>) -> Result<(), &'static str> {
        match unsafe { ffi::cudaMemcpy(data.as_ptr() as *mut ::libc::c_void,
                                       self.data,
                                       (data.len() * ::std::mem::size_of::<T>()) as u64,
                                       ffi::MemcpyKind::DeviceToHost) } {
            ffi::Error::Success => Ok(()),
            ffi::Error::InvalidValue => Err("one or more of the parameters passed to the API call is not within an acceptable range of values."),
            ffi::Error::InvalidDevicePointer => Err("At least one device pointer passed to the API call is not a valid device pointer."),
            ffi::Error::InvalidMemcpyDirection => Err("The direction of the memcpy passed to the API call is not one of the types specified by cudaMemcpyKind."),
            _ => Err("Unknown error")
        }
    }
}
