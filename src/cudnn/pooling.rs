use cudnn::ffi;
use Tensor;
use std::ptr::null_mut;

pub struct Pooling {
    pub desc: ffi::PoolingDescriptor
}

impl Drop for Pooling {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyPoolingDescriptor(self.desc) };
    }
}

impl Pooling {
    pub fn new()
               -> Result<Pooling, &'static str> {
        let mut desc: ffi::PoolingDescriptor = null_mut();
        match unsafe { ffi::cudnnCreatePoolingDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Pooling { desc: desc }),
            e => Err(e.to_str())
        }
    }

    pub fn new_2d_max(size: i32, padding: i32, stride: i32)
                      -> Result<Pooling, &'static str> {
        let pooling = try! { Pooling::new() };
        match unsafe { ffi::cudnnSetPooling2dDescriptor(pooling.desc,
                                                        ffi::PoolingMode::Max,
                                                        size, size,
                                                        padding, padding,
                                                        stride, stride) } {
            ffi::Status::Success => Ok(pooling),
            e => Err(e.to_str())
        }
    }

    pub fn output_dim(&self, tensor: &Tensor)
                      -> Result<(i32, i32, i32, i32), &'static str> {
        let (mut n, mut c, mut h, mut w) = (0i32, 0i32, 0i32, 0i32);
        match unsafe { ffi::cudnnGetPooling2dForwardOutputDim(self.desc,
                                                              tensor.desc,
                                                              &mut n,
                                                              &mut c,
                                                              &mut h,
                                                              &mut w) } {
            ffi::Status::Success => Ok((n, c, h, w)),
            e => Err(e.to_str())
        }
    }
}

#[test]
pub fn test_output_dim() {
    let tensor = Tensor::new_4d(128, 256, 64, 64).unwrap();
    let pooling = Pooling::new_2d_max(2, 0, 2).unwrap();
    assert_eq!((128, 256, 32, 32), pooling.output_dim(&tensor).unwrap());
}
