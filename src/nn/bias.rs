use cudnn::Tensor;
use nn::Nn;

impl Nn {
    pub fn bias_forward(&self,
                        x: &Tensor,
                        b: &Tensor)
                        -> Result<(), &'static str> {
        self.cudnn.add_tensor(1f32, b, 1f32, x)
    }

    pub fn bias_backward(&self,
                         scale: f32,
                         b: &Tensor,
                         dy: &Tensor)
                         -> Result<(), &'static str> {
        self.cudnn.add_tensor(scale, dy, 1f32, b)
    }
}
