extern crate libc;
extern crate image;

mod cuda;

use std::path::Path;
use cuda::nn;

fn main() {
    let cudnn = unsafe {nn::Cudnn::new() };

    // read png image
    let img = image::open(&Path::new("images/hr.png")).unwrap();
    let buf = img.raw_pixels();
    let buf_float = buf.into_iter().map(|x: u8| (x as f32) / 255.0).collect::<Vec<_>>();

    // alloc device memory
    let _src_desc = unsafe { nn::Tensor::new_4d(1, 3, 240, 240) };
    let _dst_desc = unsafe { nn::Tensor::new_4d(1, 3, 240, 240) };

    let mut src = cuda::Memory::<f32>::new(buf_float.len());
    let mut dst = cuda::Memory::<f32>::new(buf_float.len());

    src.write(&buf_float);
    cudnn.sigmoid_forward(_src_desc, &src, _dst_desc, &mut dst);
    dst.read(&buf_float);

    // write png image
    let buf_output = buf_float.into_iter().map(|x: f32| (x * 255.0) as u8).collect::<Vec<_>>();
    image::save_buffer(&Path::new("images/output.png"), &buf_output, 240, 240, image::RGB(8));

}
