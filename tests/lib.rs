extern crate monster;


#[cfg(test)]
mod test {
    use monster::cudnn::{Cudnn, Filter4d, Tensor, Convolution2d, ConvolutionFwdAlgo};

    #[test]
    fn conv_forward_output_dim() {
        let filter = Filter4d::new(3, 3, 3, 3).unwrap();
    }

    #[test]
    fn conv_get_forward_algo() {
        let cudnn = Cudnn::new().unwrap();
        let filter = Filter4d::new(10, 10, 3, 3).unwrap();
        let src_tensor = Tensor::new_4d(1, 10, 256, 256).unwrap();
        let dst_tensor = Tensor::new_4d(1, 10, 256, 256).unwrap();
        let conv = Convolution2d::new(1, 1, 1, 1, 1, 1).unwrap();
        let res = conv.get_forward_output_dim(&src_tensor, &filter).unwrap();
        assert_eq!(res, (1, 10, 256, 256));
        let algo = cudnn.get_conv_forward_algo(&src_tensor,
                                               &filter,
                                               &conv,
                                               &dst_tensor).unwrap();
        assert_eq!(algo, ConvolutionFwdAlgo::ImplicitPrecompGemm);
        let mem_size = cudnn.get_conv_forward_workspace_size(&src_tensor,
                                                             &filter,
                                                             &conv,
                                                             &dst_tensor,
                                                             algo).unwrap();
        assert_eq!(mem_size, 393224); // TODO: why?
    }
}
