pub struct Convolution2d {
    desc: ffi::ConvolutionDescriptor
}

impl Drop for Convolution2d {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyConvolutionDescriptor(self.desc) };
    }
}

impl Convolution2d {
    fn new_desc() -> Result<Convolution2D, &static str> {
        let mut desc: ffi::ConvolutionDescriptor = ptr::null_mut();
        match unsafe { ffi::createConvolutionDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Convolution2D { desc : desc }),
            e => Err(e.to_str())
        }
    }

    pub fn new(pad_h: i32,
               pad_w: i32,
               u: i32,
               v: i32,
               upscalex: i32,
               upscaley: i32) -> Result<Convolution2D, &static str> {
        let conv = try! { Convolution2D::new_desc() };
        match unsafe { ffi::cudnnSetConvolution2dDescriptor(conv, 
                                                            pad_h,
                                                            pad_w,
                                                            u, v,
                                                            upscalex,
                                                            upscaley,
                                                            ffi::ConvolutionMode::Convolution) } {
            ffi::Status::Success => Ok(conv),
            e => Err(e.to_str())
        }
    }

    pub fn 
}
