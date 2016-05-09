extern crate monster;


#[cfg(test)]
mod test {
    use monster::cudnn::Filter4d;

    #[test]
    fn conv_forward_output_dim() {
        let filter = Filter4d::new(3, 3, 3, 3).unwrap();
    }
}
