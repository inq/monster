extern crate monster;


#[cfg(test)]
mod test {
    use monster::cudnn::{Cudnn, Filter4d, Convolution2d, ConvolutionFwdAlgo};
    use monster::nn::{Tensor};

    #[test]
    fn convolution_test() {
        let cudnn = Cudnn::new().unwrap();
        let filter = Filter4d::new(10, 10, 3, 3).unwrap();
        let src_tensor = Tensor::new(1, 10, 256, 256).unwrap();
        let dst_tensor = Tensor::new(1, 10, 256, 256).unwrap();
        let conv = Convolution2d::new(1, 1, 1, 1, 1, 1).unwrap();
        let res = conv.get_forward_output_dim(&src_tensor, &filter).unwrap();
        assert_eq!(res, (1, 10, 256, 256));
        let algo = cudnn.get_conv_forward_algo(&src_tensor,
                                               &filter,
                                               &conv,
                                               &dst_tensor).unwrap();
        assert_eq!(algo, ConvolutionFwdAlgo::ImplicitGemm);
        let mem_size = cudnn.get_conv_forward_workspace_size(&src_tensor,
                                                             &filter,
                                                             &conv,
                                                             &dst_tensor,
                                                             algo).unwrap();
        assert_eq!(mem_size, 0);
    }
}
