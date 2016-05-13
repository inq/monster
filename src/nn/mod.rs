use cudnn::{Cudnn};
use cublas::Cublas;
pub use cudnn::Tensor;

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

    /*
    pub fn sigmoid_forward(&self, x: &Tensor, y: &mut Tensor)
                           -> Result<(), &'static str> {
        self.cudnn.activation_forward(ffi::ActivationDescriptor::Sigmoid,
                                      1.0f32, x, 0.0f32, y)
    }*/

    pub fn fcn_forward(&self, m: usize, n: usize, src: &Tensor, dst: &mut Tensor,
                       params: &Tensor)
                       -> Result<(), &'static str> {
        self.cublas.s_gemv(m as i32, n as i32, params, src, dst) 
    }

    pub fn fcn_backward(&self, scale: f32,
                        m: usize, n: usize,
                        x: &Tensor, dy: &Tensor, dx: &mut Tensor,
                        params: &mut Tensor)
                        -> Result<(), &'static str> {
        try!(self.cublas.s_gemv_n(m as i32, n as i32,
                                  params, dy, dx));
        self.cublas.s_ger(m as i32, n as i32, scale, x, dy, params)
    }
}
