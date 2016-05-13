use cudnn::{Tensor, Filter4d, Convolution2d};
use cudart::Memory;
use cudnn;
use nn::Nn;

impl Nn {
    pub fn conv_forward(&self,
                        x: &Tensor,
                        filter: &Filter4d,
                        w: &Tensor,
                        conv: &Convolution2d,
                        y: &Tensor)
                        -> Result<(), &'static str> {
        let algo = try!(self.cudnn.get_conv_forward_algo(&x,
                                                         &filter,
                                                         &conv,
                                                         &y,
                                                         cudnn::ConvolutionFwdPreference::NoWorkspace));
        let workspace_size = try!(self.cudnn.get_conv_forward_workspace_size(&x,
                                                                             &filter,
                                                                             &conv,
                                                                             &y,
                                                                             algo));
        let workspace = try!(Memory::<f32>::new(workspace_size / 4));
        self.cudnn.convolution_forward(1f32, x, w, conv, algo, &workspace, workspace_size, 0f32, y)
    }
}
