use cudnn::{ffi, Cudnn};
use libc::c_void;
use cudart;
use std::ptr::null_mut;
use std::mem::size_of;

#[warn(dead_code)]
pub struct Tensor {
    pub desc: ffi::TensorDescriptor,
    pub data: *mut c_void,
    n: i32,
    c: i32,
    h: i32,
    w: i32
}

impl Tensor {
    pub fn debug(&self) {
        println!("{:?}", (self.n, self.c, self.h, self.w));
    }
    
    fn create_desc()
                   -> Result<ffi::TensorDescriptor, &'static str> {
        let mut desc: ffi::Handle = null_mut();
        match unsafe { ffi::cudnnCreateTensorDescriptor(&mut desc) } {
            ffi::Status::Success => Ok(desc),
            e => Err(e.to_str())
        }
    }
    
    pub fn new(n: i32, c: i32, h: i32, w: i32)
               -> Result<Tensor, &'static str> {
        let desc = try!(Tensor::create_desc());
        let data = try!(cudart::cuda_malloc(size_of::<f32>() * (n * c * h * w) as usize));

        match unsafe { ffi::cudnnSetTensor4dDescriptor(desc,
                                                       ffi::Format::NCHW,
                                                       ffi::DataType::Float,
                                                       n, c, h, w) } {
            ffi::Status::Success => Ok(Tensor {
                desc: desc,
                data: data,
                n: n, c: c, h: h, w: w
            }),
            e => Err(e.to_str())
        }
    }

    pub fn channel_size(&self) -> i32 {
        self.c * self.h * self.w
    }
    
    pub fn write(&self, data: &Vec<f32>) -> Result<(), &'static str> {
        match unsafe { cudart::ffi::cudaMemcpy(self.data,
                                               data.as_ptr() as *mut c_void,
                                               (data.len() * size_of::<f32>()) as u64,
                                               cudart::ffi::MemcpyKind::HostToDevice) } {
            cudart::ffi::Error::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn read(&self, data: &Vec<f32>) -> Result<(), &'static str> {
        match unsafe { cudart::ffi::cudaMemcpy(data.as_ptr() as *mut c_void,
                                               self.data,
                                               (data.len() * size_of::<f32>()) as u64,
                                               cudart::ffi::MemcpyKind::DeviceToHost) } {
            cudart::ffi::Error::Success => Ok(()),
            e => Err(e.to_str())
        }
    }
}

impl Cudnn {
    pub fn add_tensor(&self,
                      alpha: f32, x: &Tensor,
                      beta: f32, y: &Tensor)
                      -> Result<(), &'static str> {
        unsafe {
            ffi::cudnnAddTensor(self.handle,
                                &alpha as *const _ as *const ::libc::c_void,
                                x.desc,
                                x.data,
                                &beta as *const _ as *const ::libc::c_void,
                                y.desc,
                                y.data)
        }.to_result()
    }
}
