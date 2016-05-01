use super::ffi;

pub struct Tensor {
    pub descriptor: ffi::TensorDescriptor
}

impl Tensor {
    pub unsafe fn new() -> Tensor {
        let mut descriptor: ffi::Handle = ::std::ptr::null_mut();
        let status = ffi::cudnnCreate(&mut descriptor);
        Tensor {
            descriptor : descriptor
        }
    }

    pub unsafe fn new_4d(n: i32, c: i32, h: i32, w: i32) -> Tensor {
        let tensor = Tensor::new();
        let status = ffi::cudnnSetTensor4dDescriptor(
            tensor.descriptor,
            ffi::Format::NCHW,
            ffi::DataType::Float,
            n, c, h, w);

        tensor
    }
}
