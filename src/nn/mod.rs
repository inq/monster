mod nn;
mod conv;
mod activation;
mod pooling;
mod softmax;

use cudnn::{Cudnn};
use cublas::Cublas;
pub use cudnn::Tensor;
pub use self::nn::Nn;

pub type Res<T> = Result<T, &'static str>;
