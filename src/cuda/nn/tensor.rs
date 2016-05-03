use super::ffi;

pub struct Tensor {
    pub descriptor: ffi::TensorDescriptor
}

impl Tensor {
    pub fn new() -> Result<Tensor, &'static str> {
        let mut descriptor: ffi::Handle = ::std::ptr::null_mut();
        match unsafe { ffi::cudnnCreateTensorDescriptor(&mut descriptor) } {
            ffi::Status::Success => Ok(Tensor { descriptor: descriptor }),
            e => Err(e.to_str())
        }
    }

    pub fn new_4d(n: i32, c: i32, h: i32, w: i32) -> Result<Tensor, &'static str> {
        let tensor = try! { Tensor::new() };
        match unsafe { ffi::cudnnSetTensor4dDescriptor(tensor.descriptor,
                                                       ffi::Format::NCHW,
                                                       ffi::DataType::Float,
                                                       n, c, h, w) } {
            ffi::Status::Success => Ok(tensor),
            e => Err(e.to_str())
        }
    }
}
