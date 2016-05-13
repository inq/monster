use cudnn::{Tensor, ActivationDescriptor};
use nn::Nn;

impl Nn {
    pub fn relu_forward(&self,
                        x: &Tensor,
                        y: &Tensor)
                        -> Result<(), &'static str> {
        self.cudnn.activation_forward(ActivationDescriptor::ReLU,
                                      1f32, x, 0f32, y)
    }

    pub fn relu_forward_inplace(&self,
                                x: &mut Tensor)
                                -> Result<(), &'static str> {
        self.cudnn.activation_forward_inplace(ActivationDescriptor::ReLU,
                                              1f32, x, 0f32)
    }
}
