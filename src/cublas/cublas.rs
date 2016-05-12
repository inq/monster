use nn::Tensor;
use cublas::ffi;
use std::ptr;

pub struct Cublas {
    handle: ffi::Handle
}

impl Drop for Cublas {
    fn drop(&mut self) {
        unsafe { ffi::cublasDestroy_v2(self.handle) };
    }
}

impl Cublas {
    pub fn new() -> Result<Cublas, &'static str> {
        let mut handle: ffi::Handle = ptr::null_mut();
        match unsafe { ffi::cublasCreate_v2(&mut handle) } {
            ffi::Status::Success => Ok(Cublas { handle : handle }),
            e => Err(e.to_str())
        }
    }

    pub fn s_gemv(&self,
                  m: i32,
                  n: i32,
                  a: &Tensor,
                  x: &Tensor,
                  y: &Tensor)
                  -> Result<(), &'static str> {
        match unsafe { ffi::cublasSgemv_v2(self.handle,
                                           ffi::Operation::T,
                                           m, n,
                                           *&[1.0f32].as_ptr() as *const ::libc::c_float,
                                           a.data as *const f32, m,
                                           x.data as *const f32, 1,
                                           *&[0.0f32].as_ptr() as *const ::libc::c_float,
                                           y.data as *const f32, 1) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    pub fn s_gemv_n(&self,
                    m: i32,
                    n: i32,
                    a: &Tensor,
                    x: &Tensor,
                    y: &mut Tensor)
                    -> Result<(), &'static str> {
        match unsafe { ffi::cublasSgemv_v2(self.handle,
                                           ffi::Operation::N,
                                           m, n,
                                           *&[1.0f32].as_ptr() as *const ::libc::c_float,
                                           a.data as *const f32, m,
                                           x.data as *const f32, 1,
                                           *&[0.0f32].as_ptr() as *const ::libc::c_float,
                                           y.data as *const f32, 1) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }

    // adding outer product
    pub fn s_ger(&self,
                 m: i32,
                 n: i32,
                 alpha: f32,
                 x: &Tensor,
                 y: &Tensor,
                 a: &mut Tensor)
                 -> Result<(), &'static str> {
        match unsafe { ffi::cublasSger_v2(self.handle,
                                          m, n,
                                          *&[alpha].as_ptr() as *const ::libc::c_float,
                                          x.data as *const f32, 1,
                                          y.data as *const f32, 1,
                                          a.data as *mut f32, m) } {
            ffi::Status::Success => Ok(()),
            e => Err(e.to_str())
        }
    }
}

#[test]
fn test_s_gemv() {
    let x = Memory::<f32>::new(2).unwrap();
    let x_data = vec![3f32, 4f32];
    x.write(&x_data).unwrap();
    let mut y = Memory::<f32>::new(3).unwrap();
    let mut y_data = vec![0f32, 0f32, 0f32];
    let a = Memory::<f32>::new(2 * 3).unwrap();
    let a_data = vec![1f32, 0f32, 0f32, 1f32, 2f32, 3f32];
    a.write(&a_data).unwrap();
    let cublas = Cublas::new().unwrap();
    cublas.s_gemv(2, 3, &a, &x, &mut y).unwrap();
    y.read(&mut y_data).unwrap();
    assert_eq!(y_data[0], 3f32);
    assert_eq!(y_data[1], 4f32);
    assert_eq!(y_data[2], 18f32);
}

#[test]
fn test_s_ger() {
    let x = Memory::<f32>::new(2).unwrap();
    let x_data = vec![3f32, 4f32];
    x.write(&x_data).unwrap();
    let y = Memory::<f32>::new(3).unwrap();
    let y_data = vec![-1f32, 2f32, -3f32];
    y.write(&y_data).unwrap();
    let mut a = Memory::<f32>::new(2 * 3).unwrap();
    let mut a_data = vec![1f32, 2f32, 3f32, 4f32, 5f32, 6f32];
    a.write(&a_data).unwrap();
    let cublas = Cublas::new().unwrap();
    cublas.s_ger(2, 3, 1f32, &x, &y, &mut a).unwrap();
    a.read(&mut a_data).unwrap();
    assert_eq!(a_data[0], -2f32);
    assert_eq!(a_data[1], -2f32);
    assert_eq!(a_data[2], 9f32);
    assert_eq!(a_data[3], 12f32);
    assert_eq!(a_data[4], -4f32);
    assert_eq!(a_data[5], -6f32);
}
