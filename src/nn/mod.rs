use cudnn::{Cudnn, Tensor};
use cublas::{Cublas};
use cudart::{Memory};

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

    pub fn fcn_forward(&self,
                       m: usize,
                       n: usize,
                       src: &Memory<f32>,
                       dst: &mut Memory<f32>,
                       params: &Memory<f32>)
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
                        x: &Memory<f32>,
                        dy: &Memory<f32>,
                        dx: &mut Memory<f32>,
                        params: &mut Memory<f32>)
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
