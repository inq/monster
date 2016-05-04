extern crate libc;
extern crate image;

mod cudart;
mod cudnn;
mod cifar;

use std::process::exit;
use std::path::Path;
use std::env;

fn run(args: Vec<String>) -> Result<(), &'static str> {
    let cudnn = try!{ cudnn::Cudnn::new() };

    let cifar = cifar::Cifar::new(args[1].clone());

    // read png image
    let img = image::open(&Path::new("images/hr.png")).unwrap();
    let buf = img.raw_pixels();
    let buf_float = buf.into_iter().map(|x: u8| (x as f32) / 255.0).collect::<Vec<_>>();

    // alloc device memory
    let _src_desc = try! { cudnn::Tensor::new_4d(1, 3, 240, 240) };
    let _dst_desc = try! { cudnn::Tensor::new_4d(1, 3, 240, 240) };
    let src = try! { cudart::Memory::<f32>::new(buf_float.len()) };
    let mut dst = try! { cudart::Memory::<f32>::new(buf_float.len()) };

    try! { src.write(&buf_float) };
    try! { cudnn.sigmoid_forward(_src_desc, &src, _dst_desc, &mut dst) };
    try! { dst.read(&buf_float) };

    // write png image
    let buf_output = buf_float.into_iter().map(|x: f32| (x * 255.0) as u8).collect::<Vec<_>>();
    match image::save_buffer(&Path::new("images/output.png"), &buf_output, 240, 240, image::RGB(8)) {
        Ok(()) => Ok(()),
        Err(_) => Err("Failed to save the result image file.")
    }
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
