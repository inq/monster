use cudnn::{Cudnn, Tensor};
use cublas::{Cublas};

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
}
