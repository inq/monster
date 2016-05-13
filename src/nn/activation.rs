use cudnn::{Tensor, ActivationDescriptor};
use nn::{Nn, Res};

impl Nn {
    pub fn relu_forward(&self,
                        x: &Tensor,
                        y: &Tensor)
                        -> Res<()> {
        self.cudnn.activation_forward(ActivationDescriptor::ReLU,
                                      1f32, x, 0f32, y)
    }

    pub fn relu_forward_inplace(&self,
                                x: &mut Tensor)
                                -> Res<()> {
        self.cudnn.activation_forward_inplace(ActivationDescriptor::ReLU,
                                              1f32, x, 0f32)
    }

    pub fn sigmoid_forward(&self,
                           x: &Tensor,
                           y: &Tensor)
                           -> Res<()> {
        self.cudnn.activation_forward(ActivationDescriptor::Sigmoid,
                                      1f32, x, 0f32, y)
    }

}
