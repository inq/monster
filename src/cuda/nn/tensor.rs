use super::ffi;

pub struct Tensor {
    pub descriptor: ffi::TensorDescriptor
}

impl Tensor {
    pub fn new() -> Result<Tensor, &'static str> {
        let mut descriptor: ffi::Handle = ::std::ptr::null_mut();
        match unsafe { ffi::cudnnCreateTensorDescriptor(&mut descriptor) } {
            ffi::Status::Success => Ok(Tensor { descriptor: descriptor }),
            ffi::Status::AllocFailed => Err("The resources could not be allocated."),
            _ => Err("Unknown Error")
        }
    }

    pub fn new_4d(n: i32, c: i32, h: i32, w: i32) -> Result<Tensor, &'static str> {
        let tensor = try! { Tensor::new() };
        match unsafe { ffi::cudnnSetTensor4dDescriptor(tensor.descriptor,
                                                       ffi::Format::NCHW,
                                                       ffi::DataType::Float,
                                                       n, c, h, w) } {
            ffi::Status::Success => Ok(tensor),
            ffi::Status::BadParam => Err("At least one of the parameters n,c,h,w was negative or format has an invalid enumerant value or dataType has an invalid enumerant value."),
            ffi::Status::NotSupported => Err("The total size of the tensor descriptor exceeds the maximim limit of 2 Giga-elements."),
            _ => Err("Unknown Error")
        }
    }
}
