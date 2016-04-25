extern crate cudnn;
extern crate libc;

mod cuda;

use cudnn::{Cudnn, TensorDescriptor};
use cudnn::utils::{ScalParams, DataType};

fn main() {
    let cudnn = Cudnn::new().unwrap();

    let src_desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();
    let dest_desc = TensorDescriptor::new(&[2, 2, 2], &[4, 2, 1], DataType::Float).unwrap();

    let mut nums: [f32; 10] = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let mut src = cuda::Memory::new(10 * 4);
    let mut dst = cuda::Memory::new(10 * 4);
    
    src.write(nums.as_ptr() as *mut ::libc::c_void, 10 * 4);
    cudnn.sigmoid_forward::<f32>(&src_desc, src.data, &dest_desc, dst.data, ScalParams::default());
    dst.read(nums.as_ptr() as *mut ::libc::c_void, 10 * 4);
    for i in nums.iter() {
        println!("{}", i);
    }
}
