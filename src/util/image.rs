use std::fs::File;
use std::io::Read;
use cudart::Memory;
use std::path::Path;
use image;

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

    pub fn from_device(memory: Memory<f32>,
                       info: T,
                       width: usize,
                       height: usize) -> Result<Image<T>, &'static str> {
        let mut buf_f32 = vec![0f32; width * height * Channels];
        try!(memory.read(&mut buf_f32));
        let buf_u8 = buf_f32.into_iter().map(|x: f32| (x * 255.0) as u8).collect::<Vec<_>>();

        Ok(Image::<T> {
            info: info, 
            width: width,
            height: height,
            data: buf_u8
        })
    }

    pub fn save(self, path: &str) -> Result<(), &'static str> {
        match image::save_buffer(&Path::new(&path),
                                 &self.data_whc(),
                                 self.width as u32,
                                 self.height as u32,
                                 image::RGB(8)) {
            Ok(()) => Ok(()),
            Err(_) => Err("Failed to save the result image file.")
        }
    }

    pub fn to_device(&self) -> Result<Memory<f32>, &'static str> {
        let buf = self.data.clone().into_iter().map(|x: u8| (x as f32) / 255.0).collect::<Vec<_>>();
        let mut mem = try!(Memory::new(self.size()));
        try!(mem.write(&buf));
        Ok(mem)
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
