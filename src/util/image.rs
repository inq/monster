use std::fs::File;
use std::io::Read;

pub struct Image<T> {
    info: T,
    width: usize,
    height: usize,
    pub data: Vec<u8>
}

const Channels:usize = 3;

impl<T> Image<T> {
    pub fn from_file(file: &mut File,
                     width: usize,
                     height: usize,
                     info: T) -> Result<Image<T>, &'static str> {
        let mut buf = vec![0u8; width * height * Channels];
        try!(match file.read(&mut buf) {
            Ok(n) if(n == width * height * Channels) => Ok(()),
            _ => Err("EOF")
        });

        Ok(Image::<T> {
            info: info,
            width: width,
            height: height,
            data: buf.to_vec()
        })
    }

    pub fn size(&self) -> usize {
        self.width * self.height * Channels
    }

    pub fn data_whc(&self) -> Vec<u8> {
        let mut res = vec![0u8; self.size()];
        for i in 0..res.len() {
            res[i] = self.data[(i % 3) * (32 * 32) + (i / 3)];
        }
        res
    }
}
