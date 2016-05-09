pub struct Filter4d {
    desc: ffi::FilterDescriptor
}

impl Drop for Filter4d {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyFilterDescriptor(self.desc) };
    }
}

impl Filter4d {
    /* CUDNN_TENSOR_NCHW */
    fn new_desc() -> Result<Filter4d, &static str> {
        let mut desc: ffi::FilterDescriptor = ptr::null_mut();
        match unsafe { ffi::createFilterDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Filter4d { desc: desc }),
            e => Err(e.to_str())
        }
    }

    pub fn new() -> Result<Filter4d, &static str> {
        let filter = try! { Filter4d::new_desc() };
        match unsafe { ffi::cudnnSetFilter4dDescriptor(filter,
                                                       ffi::DataType::Float,
                                                       k, c, h, w) } {
            ffi::Status::Success => Ok(filter),
            e => Err(e.to_str())
        }
    }
}
