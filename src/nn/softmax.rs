use cudnn::{Tensor, Filter, SoftmaxAlgorithm, SoftmaxMode};
use cudart::Memory;
use cudnn;
use nn::{Nn, Res};

impl Nn {
    pub fn softmax_forward(&self, x: &Tensor, y: &Tensor)
                           -> Res<()> {
        self.cudnn.softmax_forward(SoftmaxAlgorithm::Fast,
                                   SoftmaxMode::Channel,
                                   1f32, x, 0f32, y)
    }

    pub fn softmax_backward(&self, y: &Tensor, dy: &Tensor, dx: &Tensor)
                            -> Result<(), &'static str>{
        self.cudnn.softmax_backward(SoftmaxAlgorithm::Fast,
                                    SoftmaxMode::Channel,
                                    1f32, y, dy, 0f32, dx)
    }
}
