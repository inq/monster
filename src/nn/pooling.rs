use cudnn::{Tensor, Pooling, PoolingMode};
use nn::{Nn, Res};

impl Nn {
    pub fn new_max_pooling(size: i32, padding: i32, stride: i32)
                       -> Res<Pooling> {
        let pooling = try! { Pooling::new() };
        try! { pooling.set_2d_desc(PoolingMode::Max,
                                   size, size,
                                   padding, padding,
                                   stride, stride) };
        Ok(pooling)
    }
    
    pub fn pooling_forward(&self, pooling: &Pooling,
                           x: &Tensor, y: &Tensor)
                           -> Res<()> {
        self.cudnn.pooling_forward(pooling,
                                   1f32, x, 0f32, y)
    }

    pub fn pooling_backward(&self, pooling: &Pooling,
                            y: &Tensor, dy: &Tensor,
                            x: &Tensor, dx: &Tensor)
                            -> Result<(), &'static str> {
        self.cudnn.pooling_backward(pooling, 1f32, y, dy, x,
                                    0f32, dx)
    }
}
