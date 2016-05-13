use cudnn::{Tensor, ActivationDescriptor};
use nn::{Nn, Res};

impl Nn {
    pub fn fcn_forward(&self, x: &Tensor, y: &Tensor,
                       params: &Tensor)
                       -> Result<(), &'static str> {
        self.cublas.s_gemv(x.channel_size(),
                           y.channel_size(),
                           params, x, y) 
    }

    pub fn fcn_backward(&self, scale: f32,
                        x: &Tensor, dy: &Tensor, dx: &Tensor,
                        params: &mut Tensor)
                        -> Result<(), &'static str> {
        try!(self.cublas.s_gemv_n(x.channel_size(),
                                  dy.channel_size(),
                                  params, dy, dx));
        self.cublas.s_ger(x.channel_size(),
                          dy.channel_size(),
                          scale, x, dy, params)
    }
}
