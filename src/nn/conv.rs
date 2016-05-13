use cudnn::{Tensor, Filter, Convolution, ConvolutionMode, ConvolutionFwdPreference, DataType};
use cudart::Memory;
use nn::{Nn, Res};

impl Nn {
    pub fn new_filter(k: i32, c: i32, h: i32, w: i32)
                      -> Res<Filter> {
        let filter = try!{ Filter::new() };
        try! { filter.set_filter_desc(DataType::Float, k, c, h, w) }
        Ok(filter)
    }

    pub fn new_conv(pad: i32, size: i32, upscale: i32)
                    -> Res<Convolution> {
        let conv = try!{ Convolution::new() };
        try! {
            conv.set_2d_desc(pad, pad,
                             size, size,
                             upscale, upscale,
                             ConvolutionMode::Convolution)
        };
        Ok(conv)
    }
    
    pub fn conv_forward(&self,
                        x: &Tensor,
                        filter: &Filter,
                        w: &Tensor,
                        conv: &Convolution,
                        y: &Tensor)
                        -> Result<(), &'static str> {
        let algo = try!(self.cudnn.get_conv_forward_algo(&x,
                                                         &filter,
                                                         &conv,
                                                         &y,
                                                         ConvolutionFwdPreference::NoWorkspace));
        let workspace_size = try!(self.cudnn.get_conv_forward_workspace_size(&x,
                                                                             &filter,
                                                                             &conv,
                                                                             &y,
                                                                             algo));
        let workspace = try!(Memory::<f32>::new(workspace_size / 4));
        self.cudnn.convolution_forward(1f32, x, filter, w, conv, algo, &workspace, workspace_size, 0f32, y)
    }
}
