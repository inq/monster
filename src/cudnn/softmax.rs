use cudnn::ffi;

impl Cudnn {
    pub fn softmax_forward(&self, algo: ffi::SoftmaxAlgorithm, mode: ffi::SoftmaxMode,
                           alpha: f32, x: Tensor, beta: f32, y: Tensor)
                           -> Res<()> {
        unsafe {
            ffi::cudnnSoftmaxForward(self.handle, algo, mode,
                                     &alpha as *const _ as *const ::libc::c_void,
                                     x.desc, x.data,
                                     &beta as *const _ as *const ::libc::c_void,
                                     y.desc, y.data)
        }.to_result()
    }

    pub fn softmax_backward(&self, algo: ffi::SoftmaxAlgorithm, mode: ffi::SoftmaxMode,
                            alpha: f32, y: Tensor, dy: Tensor, beta: f32, dx: Tensor) {
        unsafe {
            ffi::cudnnSoftmaxBackward(self.handle, algo, mode,
                                      &alpha as *const _ as *const ::libc::c_void,
                                      y.desc, y.data, dy.desc, dy.data,
                                      &beta as *const _ as *const ::libc::c_void,
                                      dx.desc, dx.data)
        }.to_result()
    }
}
