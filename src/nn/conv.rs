use cudnn::{Tensor, Filter, Convolution2d, ConvolutionFwdPreference, DataType};
use cudart::Memory;
use nn::{Nn, Res};

impl Nn {
    pub fn new_filter(&self, k: i32, c: i32, h: i32, w: i32)
                      -> Res<Filter> {
        let filter = try!{ Filter::new() };
        try! { filter.set_filter_desc(DataType::Float, k, c, h, w) }
        Ok(filter)
    }
    
    pub fn conv_forward(&self,
                        x: &Tensor,
                        filter: &Filter,
                        w: &Tensor,
                        conv: &Convolution2d,
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
        self.cudnn.convolution_forward(1f32, x, w, conv, algo, &workspace, workspace_size, 0f32, y)
    }
}
