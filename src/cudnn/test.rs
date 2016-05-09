use cudnn::Cudnn;
use cudnn::Filter4d;

#[test]
fn initialize() {
    let cudnn = Cudnn::new().unwrap();
    let filter = Filter4d::new(30, 30, 240, 240).unwrap();
}
