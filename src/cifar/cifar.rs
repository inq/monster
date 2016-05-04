use std::collections::LinkedList;
use std::fs::File;
use std::path::Path;
use std::io::Read;
use std::io::Error;

pub struct Cifar {
    path: String,
    pub images: LinkedList<Img>
}

pub struct Img {
    label: u8,
    pub data: [u8; 3072]
}

impl Cifar {
    fn read_img(file: &mut File) -> Result<Img, &'static str> {
        let mut img_buf = [0u8; 3072];
        let mut img = Img { label : 0, data : [0; 3072] };
        let mut buf : [u8; 1] = [0];

        try!(match file.read(&mut buf) {
            Ok(len) if len == buf.len() => Ok(()),
            _ => Err("EOF")
        });
        try!(match file.read(&mut img_buf) {
            Ok(len) if len == img_buf.len() => Ok(()),
            _ => Err("EOF")
        });
        for i in 0..img.data.len() {
            img.data[i] = img_buf[(i % 3) * (32 * 32) + (i / 3)];
        }
        Ok(img)
    }
    
    pub fn new(loc: &String) -> Cifar {
        let path = Path::new(loc).join("data_batch_1.bin");
        let mut file = File::open(path).unwrap();
        let mut imgs = LinkedList::new();
        loop {
            match Cifar::read_img(&mut file) {
                Ok(img) => imgs.push_back(img),
                _ => break
            }
        }
        Cifar { path: loc.clone(), images: imgs }
    }
}
