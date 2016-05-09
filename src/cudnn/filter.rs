use cudnn::ffi;
use std::ptr;

pub struct Filter4d {
    pub desc: ffi::FilterDescriptor
}

impl Drop for Filter4d {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyFilterDescriptor(self.desc) };
    }
}

impl Filter4d {
    /* CUDNN_TENSOR_NCHW */
    fn new_desc() -> Result<Filter4d, &'static str> {
        let mut desc: ffi::FilterDescriptor = ptr::null_mut();
        match unsafe { ffi::cudnnCreateFilterDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Filter4d { desc: desc }),
            e => Err(e.to_str())
        }
    }

    pub fn new(k: i32, c: i32, h: i32, w: i32) -> Result<Filter4d, &'static str> {
        let filter = try! { Filter4d::new_desc() };
        match unsafe { ffi::cudnnSetFilter4dDescriptor(filter.desc,
                                                       ffi::DataType::Float,
                                                       k, c, h, w) } {
            ffi::Status::Success => Ok(filter),
            e => Err(e.to_str())
        }
    }
}
