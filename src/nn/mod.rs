mod tensor;

use cudnn::{ffi, Cudnn};
use cublas::Cublas;
pub use nn::tensor::Tensor;

pub struct Nn {
    pub cudnn: Cudnn,
    pub cublas: Cublas
}

impl Nn {
    pub fn new() -> Result<Nn, &'static str> {
        let cudnn = try!(Cudnn::new());
        let cublas = try!(Cublas::new());

        Ok(Nn{ cudnn: cudnn, cublas: cublas })
    }

    pub fn sigmoid_forward(&self,
                           x: &Tensor,
                           y: &mut Tensor)
                           -> Result<(), &'static str> {
        match unsafe { ffi::cudnnActivationForward(self.cudnn.handle,
                                                   ffi::ActivationDescriptor::Sigmoid,
                                                   *&[1.0f32].as_ptr() as *const ::libc::c_void,
                                                   x.desc,
                                                   x.data,
                                                   *&[0.0f32].as_ptr() as *const ::libc::c_void,
                                                   y.desc,
                                                   y.data) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn fcn_forward(&self,
                       m: usize,
                       n: usize,
                       src: &Tensor,
                       dst: &mut Tensor,
                       params: &Tensor)
                       -> Result<(), &'static str> {
        self.cublas.s_gemv(m as i32,
                           n as i32,
                           params,
                           src,
                           dst) 
    }

    pub fn fcn_backward(&self,
                        scale: f32,
                        m: usize,
                        n: usize,
                        x: &Tensor,
                        dy: &Tensor,
                        dx: &mut Tensor,
                        params: &mut Tensor)
                        -> Result<(), &'static str> {
        try!(self.cublas.s_gemv_n(m as i32,
                                  n as i32,
                                  params,
                                  dy,
                                  dx));
        self.cublas.s_ger(m as i32,
                          n as i32,
                          scale,
                          x,
                          dy,
                          params)
    }
}
