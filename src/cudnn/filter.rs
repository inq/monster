use cudnn::ffi;
use std::ptr;
use nn::Res;

pub struct Filter {
    pub desc: ffi::FilterDescriptor
}

impl Drop for Filter {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyFilterDescriptor(self.desc) };
    }
}

impl Filter {
    /* CUDNN_TENSOR_NCHW */
    pub fn new() -> Result<Filter, &'static str> {
        let mut desc: ffi::FilterDescriptor = ptr::null_mut();
        match unsafe { ffi::cudnnCreateFilterDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Filter { desc: desc }),
            e => Err(e.to_str())
        }
    }

    pub fn set_filter_desc(&self, data_type: ffi::DataType,
                           k: i32, c: i32, h: i32, w: i32)
                           -> Res<()> {
        unsafe {
            ffi::cudnnSetFilter4dDescriptor(self.desc, data_type,
                                            k, c, h, w)
        }.to_result()
    }
}
