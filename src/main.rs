extern crate libc;
extern crate image;

mod cudart;
mod cudnn;
mod cifar;
mod util;

use std::process::exit;
use std::path::Path;
use std::env;

fn run(args: Vec<String>) -> Result<(), &'static str> {
    let cudnn = try!{ cudnn::Cudnn::new() };

    let cifar = cifar::Cifar::new(&args[1]);
    let image = cifar.images.iter().nth(9999).unwrap();

    // alloc device memory
    let _src_desc = try! { cudnn::Tensor::new_4d(1, 3, 32, 32) };
    let _dst_desc = try! { cudnn::Tensor::new_4d(1, 3, 32, 32) };
    let mut dst = try! { cudart::Memory::<f32>::new(32 * 32 * 3) };
    let src = try! { image.to_device() };
    try! { cudnn.sigmoid_forward(_src_desc, &src, _dst_desc, &mut dst) };
    let img = try! { util::Image::from_device(dst, 1u8, 32, 32) };

    // write png image
    img.save("images/cifar.png")
}

fn main() {
    match run(env::args().collect()) {
        Ok(_) => exit(0),
        Err(e) => {
            println!("{}", e);
            exit(1)
        }
    }
}
