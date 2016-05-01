extern crate libc;
extern crate image;

mod cuda;

use std::process::exit;
use std::path::Path;
use cuda::nn;


fn run() -> Result<(), &'static str> {
    let mut cudnn = try!{ nn::Cudnn::new() };

    // read png image
    let img = image::open(&Path::new("images/hr.png")).unwrap();
    let buf = img.raw_pixels();
    let buf_float = buf.into_iter().map(|x: u8| (x as f32) / 255.0).collect::<Vec<_>>();

    // alloc device memory
    let _src_desc = try! { nn::Tensor::new_4d(1, 3, 240, 240) };
    let _dst_desc = try! { nn::Tensor::new_4d(1, 3, 240, 240) };
    let mut src = try! { cuda::Memory::<f32>::new(buf_float.len()) };
    let mut dst = try! { cuda::Memory::<f32>::new(buf_float.len()) };

    try! { src.write(&buf_float) };
    try! { cudnn.sigmoid_forward(_src_desc, &src, _dst_desc, &mut dst) };
    try! { dst.read(&buf_float) };

    // write png image
    let buf_output = buf_float.into_iter().map(|x: f32| (x * 255.0) as u8).collect::<Vec<_>>();
    image::save_buffer(&Path::new("images/output.png"), &buf_output, 240, 240, image::RGB(8));
    Ok(())
}

fn main() {
    match run() {
        Ok(_) => exit(0),
        Err(e) => {
            println!("{}", e);
            exit(1)
        }
    }
}
