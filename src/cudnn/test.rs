use cudnn::{ffi, Filter4d, Convolution2d};
use nn::{Nn, Tensor};

#[test]
fn convolution_test() {
    let nn = Nn::new().unwrap();
    let filter = Filter4d::new(10, 10, 3, 3).unwrap();
    let src = Tensor::new(1, 10, 256, 256).unwrap();
    let dst = Tensor::new(1, 10, 256, 256).unwrap();
    let conv = Convolution2d::new(1, 1, 1, 1, 1, 1).unwrap();
    let res = conv.get_forward_output_dim(&src, &filter).unwrap();
    assert_eq!(res, (1, 10, 256, 256));
    let algo = nn.cudnn.get_conv_forward_algo(&src,
                                              &filter,
                                              &conv,
                                              &dst,
                                              ffi::ConvolutionFwdPreference::NoWorkspace).unwrap();
    assert_eq!(algo, ffi::ConvolutionFwdAlgo::ImplicitGemm);
    let mem_size = nn.cudnn.get_conv_forward_workspace_size(&src,
                                                            &filter,
                                                            &conv,
                                                            &dst,
                                                            algo).unwrap();
    assert_eq!(mem_size, 0);
}
