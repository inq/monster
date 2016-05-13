use cudnn::{ffi, Tensor, Cudnn};
use std::ptr::null_mut;
use nn::Res;

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
               -> Res<Pooling> {
        let mut desc: ffi::PoolingDescriptor = null_mut();
        match unsafe { ffi::cudnnCreatePoolingDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(Pooling { desc: desc }),
            e => Err(e.to_str())
        }
    }

    pub fn set_2d_desc(&self, mode: ffi::PoolingMode,
                       w: i32, h: i32,
                       padding_w: i32, padding_h: i32,
                       stride_w: i32, stride_h: i32)
                       -> Res<()> {
        unsafe {
            ffi::cudnnSetPooling2dDescriptor(self.desc, mode,
                                             w, h,
                                             padding_w, padding_h,
                                             stride_w, stride_h)
        }.to_result()
    }

    pub fn get_2d_forward_output_dim(&self, tensor: &Tensor)
                                     -> Res<(i32, i32, i32, i32)> {
        let (mut n, mut c, mut h, mut w) = (0i32, 0i32, 0i32, 0i32);
        match unsafe { ffi::cudnnGetPooling2dForwardOutputDim(self.desc, tensor.desc,
                                                              &mut n, &mut c, &mut h, &mut w) } {
            ffi::Status::Success => Ok((n, c, h, w)),
            e => Err(e.to_str())
        }
    }
}

impl Cudnn {
    pub fn pooling_forward(&self, pooling: &Pooling,
                           alpha: f32, x: &Tensor,
                           beta: f32, y: &Tensor)
                           -> Res<()> {
        unsafe {
            ffi::cudnnPoolingForward(self.handle, pooling.desc,
                                     &alpha as *const _ as *const ::libc::c_void,
                                     x.desc, x.data,
                                     &beta as *const _ as *const ::libc::c_void,
                                     y.desc, y.data)
        }.to_result()
    }

    pub fn pooling_backward(&self, pooling: &Pooling,
                            alpha: f32, y: &Tensor, dy: &Tensor, x: &Tensor,
                            beta: f32, dx: &Tensor)
                            -> Res<()> {
        unsafe {
            ffi::cudnnPoolingBackward(self.handle,
                                      pooling.desc,
                                      &alpha as *const _ as *const ::libc::c_void,
                                      y.desc, y.data,
                                      dy.desc, dy.data,
                                      x.desc, x.data,
                                      &beta as *const _ as *const ::libc::c_void,
                                      dx.desc, dx.data)
        }.to_result()
    }
}

#[test]
pub fn test_output_dim() {
    let tensor = Tensor::new(128, 256, 64, 64).unwrap();
    let pooling = Pooling::new().unwrap();
    pooling.set_2d_desc(ffi::PoolingMode::Max, 2, 2, 0, 0, 2, 2).unwrap();
    assert_eq!((128, 256, 32, 32), pooling.get_2d_forward_output_dim(&tensor).unwrap());
}
