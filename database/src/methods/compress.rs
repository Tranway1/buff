use crate::kernel::Kernel;
use crate::segment::Segment;
use ndarray::Array2;
extern crate flate2;

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io;
use std::io::prelude::*;
use flate2::write::ZlibEncoder;
use serde::{Serialize, Deserialize};
use self::flate2::read::{ZlibDecoder, DeflateDecoder};
use self::flate2::write::DeflateEncoder;
use parity_snappy as snappy;
use parity_snappy::{compress, decompress};

pub trait CompressionMethod<T> {


    fn get_segments(&self);

	fn get_batch(&self) -> usize;

    fn run_compress<'a>(&self, segs: &mut Vec<Segment<T>>);

	fn run_decompress(&self, segs: &mut Vec<Segment<T>>);
}


#[derive(Clone)]
pub struct GZipCompress {
    chunksize: usize,
    batchsize: usize
}

impl GZipCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        GZipCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let mut e = GzEncoder::new(Vec::new(), Compression::fast());
        e.write_all(seg.convert_to_bytes().unwrap().as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a Gz Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        let mut gz = GzDecoder::new(&bytes[..]);
        let mut s = String::new();
        gz.read_to_string(&mut s)?;
        Ok(s)
    }
}

impl<'a, T> CompressionMethod<T> for GZipCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}


#[derive(Clone)]
pub struct DeflateCompress {
    chunksize: usize,
    batchsize: usize
}

impl DeflateCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        DeflateCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let mut e = DeflateEncoder::new(Vec::new(), Compression::fast());
        e.write_all(seg.convert_to_bytes().unwrap().as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a deflte Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        let mut df = DeflateDecoder::new(&bytes[..]);
        let mut s = String::new();
        df.read_to_string(&mut s)?;
        Ok(s)
    }
}

impl<'a, T> CompressionMethod<T> for DeflateCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}



#[derive(Clone)]
pub struct ZlibCompress {
    chunksize: usize,
    batchsize: usize
}

impl ZlibCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        ZlibCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let mut e = ZlibEncoder::new(Vec::new(), Compression::fast());
        e.write_all(seg.convert_to_bytes().unwrap().as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a zl Encoded vector of bytes and returns a string or error
    // todo: fix later for decompression
    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        let mut zl = ZlibDecoder::new(&bytes[..]);
        let mut s = String::new();
        zl.read_to_string(&mut s)?;
        Ok(s)
    }
}

impl<'a, T> CompressionMethod<T> for ZlibCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}




#[derive(Clone)]
pub struct SnappyCompress {
    chunksize: usize,
    batchsize: usize
}

impl SnappyCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        SnappyCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a snappy Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode_reader(bytes: Vec<u8>) -> Vec<u8> {
        let mut snappy = decompress(bytes.as_slice());
        let mut s = snappy.unwrap();
        s
    }
}

impl<'a, T> CompressionMethod<T> for SnappyCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[test]
fn test_zlib_compress() {
	let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
	e.write_all(b"foo");
	e.write_all(b"bar");
	let compressed_bytes = e.finish();
}