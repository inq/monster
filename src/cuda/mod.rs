mod ffi;

#[allow(dead_code)]
pub struct Memory<T> {
    pub data: *mut ::libc::c_void,
    size: usize,
    dummy: [T; 0]
}

impl<T> Memory<T> {
    pub fn new(size: usize) -> Memory<T> {
        let mut data: *mut ::libc::c_void = ::std::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut data, (size * ::std::mem::size_of::<T>()) as u64) };
        Memory::<T> {
            data: data,
            size: size,
            dummy: []
        }
    }

    pub fn write(&self, data: &Vec<T>) {
        unsafe {
            ffi::cudaMemcpy(self.data,
                       data.as_ptr() as *mut ::libc::c_void,
                       (data.len() * ::std::mem::size_of::<T>()) as u64,
                       ffi::MemcpyKind::HostToDevice);
        }
    }

    pub fn read(&self, data: &Vec<T>) {
        unsafe {
            ffi::cudaMemcpy(data.as_ptr() as *mut ::libc::c_void,
                       self.data,
                       (data.len() * ::std::mem::size_of::<T>()) as u64,
                       ffi::MemcpyKind::DeviceToHost);
        }
    }
}

