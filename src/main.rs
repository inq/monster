extern crate cudnn;
extern crate libc;
extern crate image;


mod cuda;

use std::path::Path;
use cudnn::{Cudnn, TensorDescriptor};
use cudnn::utils::{ScalParams, DataType};

fn main() {
    let cudnn = Cudnn::new().unwrap();

    // read png image
    let img = image::open(&Path::new("images/hr.png")).unwrap();
    let buf = img.raw_pixels();
    let buf_float = buf.into_iter().map(|x: u8| (x as f32) / 255.0).collect::<Vec<_>>();

    // alloc device memory
    let src_desc = TensorDescriptor::new(&[240, 240, 3], &[3 * 240, 3, 1], DataType::Float).unwrap();
    let dst_desc = TensorDescriptor::new(&[240, 240, 3], &[3 * 240, 3, 1], DataType::Float).unwrap();

    let mut src = cuda::Memory::<f32>::new(buf_float.len());
    let mut dst = cuda::Memory::<f32>::new(buf_float.len());

    src.write(buf_float.as_ptr() as *mut ::libc::c_void, buf_float.len());
    let res = cudnn.sigmoid_forward::<f32>(&src_desc, src.data, &dst_desc, dst.data, ScalParams::default());
    dst.read(buf_float.as_ptr() as *mut ::libc::c_void, buf_float.len());

    // write png image
    let buf_output = buf_float.into_iter().map(|x: f32| (x * 255.0) as u8).collect::<Vec<_>>();
    image::save_buffer(&Path::new("images/output.png"), &buf_output, 240, 240, image::RGB(8));

}
