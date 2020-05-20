use crate::kernel::Kernel;
use crate::segment::{Segment, PAACompress, FourierCompress, fourier_compress, paa_compress};
use ndarray::Array2;
extern crate flate2;
extern crate tsz;
use log::{info, trace, warn};

extern crate bitpacking;
use bitpacking::{BitPacker4x, BitPacker};
use std::vec::Vec;
use croaring::Bitmap;
use tsz::{DataPoint, Encode, Decode, StdEncoder, StdDecoder};
use tsz::stream::{BufferedReader, BufferedWriter};
use tsz::decode::Error;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::{io, env, mem};
use std::io::prelude::*;
use flate2::write::ZlibEncoder;
use serde::{Serialize, Deserialize};
use self::flate2::read::{ZlibDecoder, DeflateDecoder};
use self::flate2::write::DeflateEncoder;
use parity_snappy as snappy;
use parity_snappy::{compress, decompress};
use std::time::{SystemTime, Instant};
use crate::client::{construct_file_client_skip_newline, construct_file_iterator_skip_newline, construct_file_iterator_int, construct_file_iterator_int_signed};
use crate::methods::Methods::Fourier;
use self::bitpacking::BitPacker1x;
use crate::methods::bit_packing::{BP_encoder, deltaBP_encoder, delta_offset, delta_num_bits, split_double_encoder, BitPack, bp_double_encoder};
use std::str::FromStr;
use num::{FromPrimitive, Num};
use rustfft::FFTnum;
use crate::methods::fcm_encoder::FCMCompressor;
use std::hash::Hash;
use std::borrow::Borrow;
use crate::methods::prec_double::PrecisionBound;
use crate::methods::gorilla_encoder::{GorillaEncoder, SepEncode};
use crate::methods::gorilla_decoder::{GorillaDecoder, SepDecode};
use std::any::Any;
use rustfft::num_traits::float::FloatCore;
use std::collections::HashMap;
use rustfft::num_traits::real::Real;

pub const SCALE: f64 = 1.0f64;
pub const PRED: f64 = 39.75f64;
pub const PRECISION:i32 = 5;
pub const PREC_DELTA:f64 = 0.000005f64;
// pub const TEST_FILE:&str = "../taxi/dropoff_latitude-fulltaxi-1k.csv";
pub const TEST_FILE:&str = "../UCRArchive2018/Kernel/randomwalkdatasample1k-40k";


lazy_static! {
    static ref PRECISION_MAP: HashMap<i32, i32> =[(1, 5),
        (2, 8),
        (3, 11),
        (4, 15),
        (5, 18),
        (6, 21),
        (7, 25),
        (8, 28),
        (9, 31),
        (10, 35),
        (11, 38),
        (12, 50),
        (13, 10),
        (14, 10),
        (15, 10)]
        .iter().cloned().collect();
}


pub trait CompressionMethod<T> {


    fn get_segments(&self);

	fn get_batch(&self) -> usize;

    fn run_compress<'a>(&self, segs: &mut Vec<Segment<T>>);

	fn run_decompress(&self, segs: &mut Vec<Segment<T>>);
}

#[derive(Clone)]
pub struct FCMCompress {
    chunksize: usize,
    batchsize: usize
}

impl FCMCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        FCMCompress { chunksize, batchsize }
    }

    fn encode<T> (&self, seg: &mut Segment<T>)
        where T: Clone + Eq + Hash +Copy + Num + Into<i32>{
        let mut fcm = FCMCompressor::new(seg.get_data().to_vec(), 3, false);
        fcm.delta_compress();
    }

    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        unimplemented!()
    }
}

impl<T> CompressionMethod<T> for FCMCompress
    where T:Clone+Eq +Hash+Copy+Num+Into<i32>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress(&self, segs: &mut Vec<Segment<T>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }

        let duration = start.elapsed();
//        println!("Time elapsed in BP function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct DFCMCompress {
    chunksize: usize,
    batchsize: usize
}

impl DFCMCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        DFCMCompress { chunksize, batchsize }
    }

    fn encode<T> (&self, seg: &mut Segment<T>)
        where T: Clone + Eq + Hash +Copy + Num + Into<i32>{
        let mut fcm = FCMCompressor::new(seg.get_data().to_vec(), 3, true);
        fcm.delta_compress();
    }

    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        unimplemented!()
    }
}

impl<T> CompressionMethod<T> for DFCMCompress
    where T:Clone+Eq +Hash+Copy+Num+Into<i32>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress(&self, segs: &mut Vec<Segment<T>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }

        let duration = start.elapsed();
//        println!("Time elapsed in BP function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct BPCompress {
    chunksize: usize,
    batchsize: usize
}

impl BPCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        BPCompress { chunksize, batchsize }
    }

    fn encode (&self, seg: &mut Segment<i32>) -> Vec<u8>{
        let comp = BP_encoder(seg.get_data().as_slice());
        comp
    }

    fn decode(&self, bytes: Vec<u8>) {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        println!("bit packing length:{}",ilen);
        let target = PRED;
        let adjust_target = (target -base_int as f64).ceil() as u32;
        let mut cur;
        let mut isqualify = true;
        let mut res = Bitmap::create();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            isqualify = cur>adjust_target;
            if isqualify{
                res.add(i)
            }
        }
        res.run_optimize();
    }
}

impl CompressionMethod<i32> for BPCompress
    {
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress(&self, segs: &mut Vec<Segment<i32>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }

        let duration = start.elapsed();
//        println!("Time elapsed in BP function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<i32>>) {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct DeltaBPCompress {
    chunksize: usize,
    batchsize: usize
}

impl DeltaBPCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        DeltaBPCompress { chunksize, batchsize }
    }

    fn encode (&self, seg: &mut Segment<i32>){
        let comp = deltaBP_encoder(seg.get_data().as_slice());
    }

    // Uncompresses a Gz Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        unimplemented!()
    }
}

impl CompressionMethod<i32> for DeltaBPCompress
{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress(&self, segs: &mut Vec<Segment<i32>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }

        let duration = start.elapsed();
//        println!("Time elapsed in BP function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<i32>>) {
        unimplemented!()
    }
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
        let mut e = GzEncoder::new(Vec::new(), Compression::best());
        let origin_bin =seg.convert_to_bytes().unwrap();
        info!("original size:{}", origin_bin.len());
        e.write_all(origin_bin.as_slice()).unwrap();
        //println!("orignal file size:{}", seg.get_byte_size().unwrap());
        let bytes = e.finish().unwrap();
        info!("compressed size:{}", bytes.len());
        //println!("{}", decode_reader(bytes).unwrap());
        let ratio = bytes.len() as f64 /origin_bin.len() as f64;
        print!("{}",ratio);
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
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }

        let duration = start.elapsed();
        info!("Time elapsed in Gzip function() is: {:?}", duration);
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
        let mut e = DeflateEncoder::new(Vec::new(), Compression::default());
        let origin_bin =seg.convert_to_bytes().unwrap();
        info!("original size:{}", origin_bin.len());
        e.write_all(origin_bin.as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        info!("compressed size:{}", bytes.len());
        //println!("{}", decode_reader(bytes).unwrap());
        let ratio = bytes.len() as f64 /origin_bin.len() as f64;
        print!("{}",ratio);

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
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }
        let duration = start.elapsed();
        info!("Time elapsed in Deflate function() is: {:?}", duration);
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
        let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
        let origin_bin =seg.convert_to_bytes().unwrap();
        info!("original size:{}", origin_bin.len());
        e.write_all(origin_bin.as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        info!("compressed size:{}", bytes.len());
        //println!("{}", decode_reader(bytes).unwrap());
        let ratio = bytes.len() as f64 /origin_bin.len() as f64;
        print!("{}",ratio);
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
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }
        let duration = start.elapsed();
        println!("Time elapsed in Zlib function() is: {:?}", duration);
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
        let origin_bin =seg.convert_to_bytes().unwrap();
        info!("original size:{}", origin_bin.len());
        let bytes = compress(origin_bin.as_slice());
        info!("compressed size:{}", bytes.len());
        //println!("{}", decode_reader(bytes).unwrap());
        let ratio = bytes.len() as f64 /origin_bin.len() as f64;
        print!("{}",ratio);
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
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }
        let duration = start.elapsed();
        info!("Time elapsed in Snappy function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct GorillaCompress {
    chunksize: usize,
    batchsize: usize
}

impl GorillaCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        GorillaCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        // 1482892260 is the Unix timestamp of the start of the stream
        let mut encoder = GorillaEncoder::new(0, w);
        let mut t =0;
        let mut bound = PrecisionBound::new(PREC_DELTA);
        let start = Instant::now();
        for val in seg.get_data(){
//            let v = bound.precision_bound((*val).into());
//            println!("{}=>{}",(*val).into(),v);
//            let preu =  unsafe { mem::transmute::<f64, u64>((*val).into()) };
//            let bdu =  unsafe { mem::transmute::<f64, u64>(v) };
//            println!("{:#066b} => {:#066b}", preu, bdu);
            encoder.encode_float((*val).into());
            t+=1;
        }
        let duration = start.elapsed();
        println!("Time elapsed in gorilla function() is: {:?}", duration);
        let origin = t * ((mem::size_of::<T>()) as u64);
        info!("original size:{}", origin);
        let bytes = encoder.close();
        let byte_vec = bytes.to_vec();
        info!("compressed size:{}", byte_vec.len());
        let ratio = (byte_vec.len() as usize) as f64 /origin as f64;
        print!("{}",ratio);
        byte_vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    fn decode(&self, bytes: Vec<u8>) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);

        let mut expected_datapoints:Vec<f64> = Vec::new();
        let mut i = 0;
        let mut done = false;
        loop {
            if done {
                break;
            }

            match decoder.next_val() {
                Ok(dp) => {
                    if i<10 {
                        println!("{}",dp);
                    }
                    i += 1;
                    expected_datapoints.push(dp);
                },
                Err(err) => {
                    if err == Error::EndOfStream {
                        done = true;
                    } else {
                        panic!("Received an error from decoder: {:?}", err);
                    }
                }
            };
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    fn range_filter(&self, bytes: Vec<u8>) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);

        let mut expected_datapoints:Vec<f64> = Vec::new();

        let mut done = false;
        let mut i=0;
        let mut isqualify = true;
        let mut res = Bitmap::create();
        loop {
            if done {
                break;
            }

            match decoder.next_val() {
                Ok(dp) => {
                    // if i<10 {
                    //     println!("{}",dp);
                    // }
                    if (dp>PRED){
                        res.add(i);
                    }
                    i+=1;
                    //expected_datapoints.push(dp);
                },
                Err(err) => {
                    if err == Error::EndOfStream {
                        done = true;
                    } else {
                        panic!("Received an error from decoder: {:?}", err);
                    }
                }
            };
        }
        res.run_optimize();
        println!("Number of qualified items:{}", res.cardinality());
        expected_datapoints
    }

}

impl<'a, T> CompressionMethod<T> for GorillaCompress
    where T: Serialize + Clone+ Copy+Into<f64>+ Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }
        let duration = start.elapsed();
        info!("Time elapsed in Gorilla function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct GorillaBDCompress {
    chunksize: usize,
    batchsize: usize
}

impl GorillaBDCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        GorillaBDCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        // 1482892260 is the Unix timestamp of the start of the stream
        let mut encoder = GorillaEncoder::new(0, w);

        let mut actual_datapoints:Vec<f64> = Vec::new();

        let mut t =0;
        let mut bound = PrecisionBound::new(PREC_DELTA);

        let start = Instant::now();
        for val in seg.get_data(){
            let v = bound.precision_bound((*val).into());
//            println!("{}=>{}",(*val).into(),v);
//            let preu =  unsafe { mem::transmute::<f64, u64>((*val).into()) };
//            let bdu =  unsafe { mem::transmute::<f64, u64>(v) };
//            println!("{:#066b} => {:#066b}", preu, bdu);
            encoder.encode_float(v);
            t+=1;
        }
        let duration = start.elapsed();
        println!("Time elapsed in gorillaBD function() is: {:?}", duration);
        let origin = t * ((mem::size_of::<T>()) as u64);
        info!("original size:{}", origin);
        let bytes = encoder.close();
        let byte_vec = bytes.to_vec();
        info!("compressed size:{}", byte_vec.len() as usize);
        let ratio = (byte_vec.len() as usize) as f64 /origin as f64;
        print!("{}",ratio);
        byte_vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }


    fn decode(&self, bytes: Vec<u8>) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);

        let mut expected_datapoints:Vec<f64> = Vec::new();
        let mut i = 0;
        let mut done = false;
        loop {
            if done {
                break;
            }

            match decoder.next_val() {
                Ok(dp) => {
                    if i<10 {
                        println!("{}",dp);
                    }
                    i += 1;
                    expected_datapoints.push(dp);
                },
                Err(err) => {
                    if err == Error::EndOfStream {
                        done = true;
                    } else {
                        panic!("Received an error from decoder: {:?}", err);
                    }
                }
            };
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    fn range_filter(&self, bytes: Vec<u8>) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);

        let mut expected_datapoints:Vec<f64> = Vec::new();

        let mut done = false;
        let mut i=0;
        let mut isqualify = true;
        let mut res = Bitmap::create();
        loop {
            if done {
                break;
            }

            match decoder.next_val() {
                Ok(dp) => {
                    // if i<10 {
                    //     println!("{}",dp);
                    // }
                    if (dp>PRED){
                        res.add(i);
                    }
                    i+=1;
                    //expected_datapoints.push(dp);
                },
                Err(err) => {
                    if err == Error::EndOfStream {
                        done = true;
                    } else {
                        panic!("Received an error from decoder: {:?}", err);
                    }
                }
            };
        }
        res.run_optimize();
        println!("Number of qualified items:{}", res.cardinality());
        expected_datapoints
    }
}

impl<'a, T> CompressionMethod<T> for GorillaBDCompress
    where T: Serialize + Clone+ Copy+Into<f64>+ Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }
        let duration = start.elapsed();
        info!("Time elapsed in Gorilla function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}





#[derive(Clone)]
pub struct SplitDoubleCompress {
    chunksize: usize,
    batchsize: usize,
    scale: usize
}

impl SplitDoubleCompress {
    pub fn new(chunksize: usize, batchsize: usize, scale: usize) -> Self {
        SplitDoubleCompress { chunksize, batchsize, scale }
    }

    fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
        let comp = split_double_encoder(seg.get_data().as_slice(),self.scale);
        comp
    }

    fn decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let mut expected_datapoints:Vec<f64> = Vec::new();
        let scl = self.scale as f64;
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(8).unwrap();
        // check integer part and update bitmap;
        let target = PRED;
        let scl_target = ((target - base_int as f64)* self.scale as f64).ceil() as i64;
        let int_target = (scl_target/self.scale as i64) as u32;
        let dec_target = (scl_target%self.scale as i64) as u32;
        let mut cur;
        let mut int_vec:Vec<i32> = Vec::new();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            int_vec.push(cur as i32 + base_int);
        }

        let mut dec = 0;
        let dec_scl:f64 = self.scale as f64;
        println!("Scale for decimal:{}", dec_scl);
        let mut j = 0;
        let mut cur_intf = 0f64;
        for int_comp in int_vec{
            cur_intf = int_comp as f64;
            dec = bitpack.read(dlen as usize).unwrap();
            if j<10{
                println!("{}th item {}, decimal:{}",j, cur_intf + (dec as f64) / dec_scl,dec);
            }
            j += 1;
            expected_datapoints.push(cur_intf + (dec as f64) / dec_scl);
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    fn range_filter(&self, bytes: Vec<u8>) {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(8).unwrap();
        // check integer part and update bitmap;
        let target = PRED;
        let scl_target = ((target - base_int as f64)* self.scale as f64).ceil() as i64;
        let int_target = (scl_target/self.scale as i64) as u32;
        let dec_target = (scl_target%self.scale as i64) as u32;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            if cur>int_target{
                res.add(i);
            }
            else if cur == int_target {
                rb1.add(i);
            };
        }
        rb1.run_optimize();
        res.run_optimize();
        println!("Number of qualified int items:{}", res.cardinality());
        let mut iterator = rb1.iter();
        // check the decimal part
        let mut it = iterator.next();
        let mut dec_cur = 0;
        let mut dec_pre:u32 = 0;
        let mut dec = 0;
        let mut delta = 0;
        if it!=None{
            dec_cur = it.unwrap();
            if dec_cur!=0{
                bitpack.skip(((dec_cur) * dlen) as usize);
            }
            dec = bitpack.read(dlen as usize).unwrap();
            if dec>dec_target{
                res.add(dec_cur);
            }
            // println!("index qualified {}, decimal:{}",dec_cur,dec);
            it = iterator.next();
            dec_pre = dec_cur;
        }
        while it!=None{
            dec_cur = it.unwrap();
            //println!("index qualified {}",dec_cur);
            delta = dec_cur-dec_pre;
            if delta != 1 {
                bitpack.skip(((delta-1) * dlen) as usize);
            }
            dec = bitpack.read(dlen as usize).unwrap();
            // if dec_cur<10{
            //     println!("index qualified {}, decimal:{}",dec_cur,dec);
            // }
            if dec>dec_target{
                res.add(dec_cur);
            }
            it = iterator.next();
            dec_pre=dec_cur;
        }
        println!("Number of qualified items:{}", res.cardinality());
    }

}

impl<'a, T> CompressionMethod<T> for SplitDoubleCompress
    where T: Serialize + Clone+ Copy+Into<f64>+ Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }

        let duration = start.elapsed();
//        println!("Time elapsed in BP function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}


#[derive(Clone)]
pub struct BPDoubleCompress {
    chunksize: usize,
    batchsize: usize,
    scale: usize
}

impl BPDoubleCompress {
    pub fn new(chunksize: usize, batchsize: usize, scale: usize) -> Self {
        BPDoubleCompress { chunksize, batchsize, scale }
    }

    fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
        let comp = bp_double_encoder(seg.get_data().as_slice(),self.scale);
        comp
    }

    fn decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let mut expected_datapoints:Vec<f64> = Vec::new();
        let scl = self.scale as f64;
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        let target = PRED;
        let adjust_target = ((target*scl).ceil() as i32 -base_int) as u32;
        let base_f64 = base_int as f64;
        // check integer part and update bitmap;
        let mut cur;
        let mut res = Bitmap::create();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,(cur as f64 + base_f64)/scl);
            // }
            expected_datapoints.push((cur as f64 + base_f64)/scl);

        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    fn range_filter(&self, bytes: Vec<u8>) {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        let target = PRED;
        let adjust_target = ((target*self.scale as f64).ceil() as i32 -base_int) as u32;
        // check integer part and update bitmap;
        let mut cur;
        let mut res = Bitmap::create();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            if cur>adjust_target{
                res.add(i);
            }

        }
        res.run_optimize();
        println!("Number of qualified items:{}", res.cardinality());
    }
}

impl<'a, T> CompressionMethod<T> for BPDoubleCompress
    where T: Serialize + Clone+ Copy+Into<f64>+ Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        let start = Instant::now();
        for seg in segs {
            self.encode(seg);
        }

        let duration = start.elapsed();
//        println!("Time elapsed in BP function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct SplitBDDoubleCompress {
    chunksize: usize,
    batchsize: usize
}

impl SplitBDDoubleCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        SplitBDDoubleCompress { chunksize, batchsize}
    }
    fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        let mut bd_vec = Vec::new();
        let mut dec_vec = Vec::new();

        let mut t:u32 =0;
        let mut bound = PrecisionBound::new(PREC_DELTA);
        let start = Instant::now();
        for val in seg.get_data(){
            let v = bound.precision_bound((*val).into());
            bd_vec.push(v);
            t+=1;
            bound.cal_length(v);
//            println!("{}=>{}",(*val).into(),v);
//            let preu =  unsafe { mem::transmute::<f64, u64>((*val).into()) };
//            let bdu =  unsafe { mem::transmute::<f64, u64>(v) };
//            println!("{:#066b} => {:#066b}", preu, bdu);
        }
        let duration = start.elapsed();
        println!("Time elapsed in bound double function() is: {:?}", duration);
        let start1 = Instant::now();
        let (int_len,dec_len) = bound.get_length();
        // let (int_len,dec_len) = (4u64,19u64);
        let ilen = int_len as usize;
        let dlen = dec_len as usize;
        println!("int_len:{},dec_len:{}",int_len,dec_len);
        //let cap = (int_len+dec_len)* t as u64 /8;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);
        let mut i= 0;
        for bd in bd_vec{
            let (int_part, dec_part) = bound.fetch_components(bd);
            // if i<10{
            //     println!("cur: {}",bd);
            //     println!("{}th, int: {}, decimal: {} in form:{}",i,int_part,dec_part as f64*1.0f64/(2i64.pow(dec_len as u32)) as f64, dec_part);
            // }
            // i += 1;
            bitpack_vec.write(int_part as u32, ilen).unwrap();
            dec_vec.push(dec_part);
        }
        let duration1 = start1.elapsed();
        println!("Time elapsed in dividing double function() is: {:?}", duration1);

        for d in dec_vec {
            bitpack_vec.write(d as u32, dlen).unwrap();
        }
        let vec = bitpack_vec.into_vec();

        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    fn offset_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut bd_vec = Vec::new();
        let mut dec_vec = Vec::new();

        let mut t:u32 = seg.get_data().len() as u32;
        let mut bound = PrecisionBound::new(PREC_DELTA);
        let start = Instant::now();
        let mut min = std::f64::MAX;
        let mut max = std::f64::MIN;


        for val in seg.get_data(){
            let v = (*val).into();
            if (v<min){
                min = v;
            }
            if (v>max){
                max = v;
            }
            bd_vec.push(v);
            // bound.cal_length(v);
//            println!("{}=>{}",(*val).into(),v);
//            let preu =  unsafe { mem::transmute::<f64, u64>((*val).into()) };
//            let bdu =  unsafe { mem::transmute::<f64, u64>(v) };
//            println!("{:#066b} => {:#066b}", preu, bdu);
        }
        let min_int = min.trunc();
        let max_int = max.trunc();
        let delta:f64 = max_int-min_int;
        let base_int = min_int as i32;
        println!("base integer: {}, max:{}",base_int,max_int);
        let ubase_int = unsafe { mem::transmute::<i32, u32>(base_int) };
        let base_int64:i64 = base_int as i64;
        let mut single_int = false;
        let mut cal_int_length = 0.0;
        if delta == 0.0 {
            single_int = true;
        }else {
            cal_int_length = delta.log2().ceil();
        }
        let duration = start.elapsed();
        println!("Time elapsed in bound double function() is: {:?}", duration);
        let start1 = Instant::now();
        let (int_len,dec_len) = (cal_int_length as u64,*(PRECISION_MAP.get(&PRECISION).unwrap()) as u64);
        bound.set_length(int_len,dec_len);
        let ilen = int_len as usize;
        let dlen = dec_len as usize;
        println!("int_len:{},dec_len:{}",int_len,dec_len);
        //let cap = (int_len+dec_len)* t as u64 /8;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(ubase_int,32);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);
        let mut i= 0;
        // for val in seg.get_data(){
        for bd in bd_vec{
            let (int_part, dec_part) = bound.fetch_components(bd);
            // if i<10{
            //     println!("cur: {}",bd);
            //     println!("{}th, int: {}, decimal: {} in form:{}",i,int_part,dec_part as f64*1.0f64/(2i64.pow(dec_len as u32)) as f64, dec_part);
            // }
            // i += 1;
            bitpack_vec.write((int_part-base_int64) as u32, ilen).unwrap();
            dec_vec.push(dec_part as u32);
        }
        let duration1 = start1.elapsed();
        println!("Time elapsed in dividing double function() is: {:?}", duration1);

        let mut j= 0;
        for d in dec_vec {
            // j += 1;
            bitpack_vec.write(d, dlen).unwrap();
        }
        println!("total number of dec is: {}", j);
        let vec = bitpack_vec.into_vec();

        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    fn fast_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        // 1482892260 is the Unix timestamp of the start of the stream
        let mut encoder = StdEncoder::new(0, w);

        let mut dec_vec = Vec::new();

        let mut t:u32 =0;
        let mut bound = PrecisionBound::new(PREC_DELTA);
        let start = Instant::now();
        t = seg.get_data().len() as u32;
        let duration = start.elapsed();
        println!("Time elapsed in bound double function() is: {:?}", duration);
        let start1 = Instant::now();
        // let (int_len,dec_len) = bound.get_length();
        let (int_len,dec_len) = (4u64,18u64);
        bound.set_length(int_len,dec_len);
        let ilen = int_len as usize;
        let dlen = dec_len as usize;
        println!("int_len:{},dec_len:{}",int_len,dec_len);
        //let cap = (int_len+dec_len)* t as u64 /8;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);
        let mut i =0;
        for bd in seg.get_data(){
            let (int_part, dec_part) = bound.fetch_components((*bd).into());
            // if i<10{
            //     println!("cur: {}",(*bd).into());
            //     println!("{}th, int: {}, decimal: {} in form:{}",i,int_part,dec_part as f64*1.0f64/(2i64.pow(dec_len as u32)) as f64, dec_part);
            // }
            // i+=1;
            bitpack_vec.write(int_part as u32, ilen).unwrap();
            dec_vec.push(dec_part);
        }
        let duration1 = start1.elapsed();
        println!("Time elapsed in dividing double function() is: {:?}", duration1);

        for d in dec_vec {
            bitpack_vec.write(d as u32, dlen).unwrap();
        }
        let vec = bitpack_vec.into_vec();

        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
    }

    fn decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(PREC_DELTA);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let target = PRED;
        let (int_part, dec_part) = bound.fetch_components(target);
        let mut expected_datapoints:Vec<f64> = Vec::new();
        let mut int_vec:Vec<i32> = Vec::new();

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            int_vec.push(cur as i32 + base_int);
        }

        let mut dec = 0;
        let dec_scl:f64 = 2.0f64.powi((dlen as i32));
        println!("Scale for decimal:{}", dec_scl);
        let mut j = 0;
        let mut cur_intf = 0f64;
        for int_comp in int_vec{
            cur_intf = int_comp as f64;
            dec = bitpack.read(dlen as usize).unwrap();
            // if j<10{
            //     println!("{}th item {}, decimal:{}",j, cur_intf + cur_intf.signum() *(dec as f64) / dec_scl,dec);
            // }
            // j += 1;
            expected_datapoints.push(cur_intf + cur_intf.signum() *(dec as f64) / dec_scl);
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    fn range_filter(&self, bytes: Vec<u8>) {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(PREC_DELTA);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        let target = PRED;
        let (int_part, dec_part) = bound.fetch_components(target);
        let int_target = (int_part-base_int as i64) as u32;
        let dec_target = dec_part as u32;
        println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            if i<10{
                println!("{}th value: {}",i,cur);
            }
            if cur>int_target{
                res.add(i);
            }
            else if cur == int_target {
                rb1.add(i);
            };
        }
        rb1.run_optimize();
        res.run_optimize();
        println!("Number of qualified int items:{}", res.cardinality());
        let mut iterator = rb1.iter();
        // check the decimal part
        let mut it = iterator.next();
        let mut dec_cur = 0;
        let mut dec_pre:u32 = 0;
        let mut dec = 0;
        let mut delta = 0;
        if it!=None{
            dec_cur = it.unwrap();
            if dec_cur!=0{
                bitpack.skip(((dec_cur) * dlen) as usize);
            }
            dec = bitpack.read(dlen as usize).unwrap();
            if dec>dec_target{
                res.add(dec_cur);
            }
            // println!("index qualified {}, decimal:{}",dec_cur,dec);
            it = iterator.next();
            dec_pre = dec_cur;
        }
        while it!=None{
            dec_cur = it.unwrap();
            //println!("index qualified {}",dec_cur);
            delta = dec_cur-dec_pre;
            if delta != 1 {
                bitpack.skip(((delta-1) * dlen) as usize);
            }
            dec = bitpack.read(dlen as usize).unwrap();
            // if dec_cur<10{
            //     println!("index qualified {}, decimal:{}",dec_cur,dec);
            // }
            if dec>dec_target{
                res.add(dec_cur);
            }
            it = iterator.next();
            dec_pre=dec_cur;
        }
        println!("Number of qualified items:{}", res.cardinality());
    }
}

impl<'a, T> CompressionMethod<T> for SplitBDDoubleCompress
    where T: Serialize + Clone+ Copy+Into<f64>+ Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        let start = Instant::now();
        for seg in segs {
            self.offset_encode(seg);
        }
        let duration = start.elapsed();
        info!("Time elapsed in Gorilla function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

pub fn test_paa_compress_on_file<'a,T>(file:&str)
    where T: FromStr + Clone + Copy + FromPrimitive+Num{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = PAACompress::new(10,10);
    let window = 100;
    let compressed = paa_compress(&mut seg, window);
    let duration = start.elapsed();
    info!("Time elapsed in PAA compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<T>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!("{},    {}", 1.0/window as f32, throughput);
}

pub fn test_paa_compress_on_int_file(file:&str,scl:i32){
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);;
    let file_vec: Vec<i32> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = PAACompress::new(10,10);
    let window = 100;
    let compressed = paa_compress(&mut seg, window);
    let duration = start.elapsed();
    info!("Time elapsed in PAA compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<u32>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!("{},    {}", 1.0/window as f32, throughput);
}

pub fn test_fourier_compress_on_file<'a,T>(file:&str)
    where T: FromStr + Clone + FFTnum +Serialize+ Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = FourierCompress::new(10,10);
    let compressed = fourier_compress(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in Fourier compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<T>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!("1,    {}", throughput);
}

//pub fn test_fourier_compress_on_int_file(file:&str){
//    let file_iter = construct_file_iterator_int(file, 1, ',');
//    let file_vec: Vec<u32> = file_iter.unwrap().collect();
//    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
//    let start = Instant::now();
//    let comp = FourierCompress::new(10,10);
//    let compressed = fourier_compress(&mut seg);
//    let duration = start.elapsed();
//    info!("Time elapsed in Fourier compress function() is: {:?}", duration);
//    //let decompress = comp.decode(compressed);
//    //println!("expected datapoints: {:?}", decompress);
//    let org_size = file_vec.len() * mem::size_of::<u32>();
//    let throughput = 1000.0 * org_size as f64 / duration.as_millis() as f64;
//    println!("1,    {}", throughput);
//}

pub fn test_snappy_compress_on_file<'a,T>(file:&str)
    where T: FromStr + Clone+Serialize + Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = SnappyCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in Snappy compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<T>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_snappy_compress_on_int_file(file:&str,scl:i32){
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = SnappyCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in Snappy compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<u32>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}


pub fn test_deflate_compress_on_file<'a,T>(file:&str)
    where T: FromStr + Clone+Serialize + Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = DeflateCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in Deflate compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<T>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_deflate_compress_on_int_file(file:&str,scl:i32){
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = DeflateCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in Deflate compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<u32>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}


pub fn test_gzip_compress_on_file<'a,T>(file:&str)
    where T: FromStr + Clone+Serialize + Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = GZipCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in Gzip compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<T>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}



pub fn test_gzip_compress_on_int_file(file:&str,scl:i32){
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = GZipCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in Gzip compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<u32>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}




pub fn test_zlib_compress_on_file<'a,T>(file:&str)
    where T: FromStr + Clone+Serialize + Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = ZlibCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in zlib compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<T>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_zlib_compress_on_int_file(file:&str,scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = ZlibCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in zlib compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
    let org_size = file_vec.len() * mem::size_of::<u32>();
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_splitbd_compress_on_file<'a,T>(file: &str)
    where T: FromStr + Serialize + Clone +Copy+Into<f64> + Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = SplitBDDoubleCompress::new(10,10);
    let compressed = comp.offset_encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in splitbd compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
//    let org_size = file_vec.len() * mem::size_of::<T>();
//    println!("record size: {}", mem::size_of::<T>() + 8);
    let org_size = file_vec.len() * (mem::size_of::<T>());
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_grillabd_compress_on_file<'a,T>(file: &str)
    where T: FromStr + Serialize + Clone +Copy+Into<f64> + Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = GorillaBDCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in grilla compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
//    let org_size = file_vec.len() * mem::size_of::<T>();
//    println!("record size: {}", mem::size_of::<T>() + 8);
    let org_size = file_vec.len() * (mem::size_of::<T>());
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_grilla_compress_on_file<'a,T>(file: &str)
    where T: FromStr + Serialize + Clone +Copy+Into<f64> + Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = GorillaCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in grilla compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
//    let org_size = file_vec.len() * mem::size_of::<T>();
//    println!("record size: {}", mem::size_of::<T>() + 8);
    let org_size = file_vec.len() * (mem::size_of::<T>());
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_grilla_compress_on_int_file(file: &str, scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = GorillaCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in grilla compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
//    let org_size = file_vec.len()*4;
    let org_size = file_vec.len() * (mem::size_of::<i32>());;
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}



pub fn test_offsetgrilla_compress_on_file<'a,T>(file: &str)
    where T: FromStr + Serialize + Clone +Copy+Into<f64> + Num+PartialOrd+ Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_v: Vec<T> = file_iter.unwrap().collect();
    let start = Instant::now();
    let file_vec = delta_offset(&file_v);
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let comp = GorillaBDCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in grilla compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
    //println!("expected datapoints: {:?}", decompress);
//    let org_size = file_vec.len() * mem::size_of::<T>();
//    println!("record size: {}", mem::size_of::<T>() + 8);
    let org_size = file_vec.len() * (mem::size_of::<T>());
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_offsetgrilla_compress_on_int_file(file: &str, scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_v: Vec<i32> = file_iter.unwrap().collect();
    let start = Instant::now();
    let file_vec = delta_offset(&file_v);
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let comp = GorillaCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in grilla compress function() is: {:?}", duration);
    //let decompress = comp.decode(compressed);
//    let org_size = file_vec.len()*4;
    let org_size = file_vec.len() * (mem::size_of::<i32>());;
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}



pub fn test_BP_compress_on_int(file:&str,scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();

    //println!("integer vector: {:?}", file_vec);
    info!("integer vector size: {}", file_vec.len());
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = BPCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in BP compress function() is: {:?}", duration);
    let org_size = file_vec.len()*4;
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_BP_double_compress_on_file<'a,T>(file: &str,scl:i32)
    where T: FromStr + Serialize + Clone +Copy+Into<f64> + Num+PartialOrd+ Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    //println!("integer vector: {:?}", file_vec);
    info!("integer vector size: {}", file_vec.len());
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = BPDoubleCompress::new(10, 10, scl as usize);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in split compress function() is: {:?}", duration);
    let org_size = file_vec.len()*((mem::size_of::<T>()) as usize);
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_split_compress_on_file<'a,T>(file: &str,scl:i32)
    where T: FromStr + Serialize + Clone +Copy+Into<f64> + Num+PartialOrd+ Deserialize<'a>{
    let file_iter = construct_file_iterator_skip_newline::<T>(file, 1, ',');
    let file_vec: Vec<T> = file_iter.unwrap().collect();
    //println!("integer vector: {:?}", file_vec);
    info!("integer vector size: {}", file_vec.len());
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = SplitDoubleCompress::new(10, 10, scl as usize);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in split compress function() is: {:?}", duration);
    let org_size = file_vec.len()*((mem::size_of::<T>()) as usize);
    //println!("orginal size {}", org_size);
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
    }
pub fn test_split_compress_on_int(file:&str,scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();

    //println!("integer vector: {:?}", file_vec);
    info!("integer vector size: {}", file_vec.len());
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = SplitDoubleCompress::new(10, 10, scl as usize);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in split compress function() is: {:?}", duration);
    let org_size = file_vec.len()*4;
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_deltaBP_compress_on_int(file:&str,scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();

    //println!("integer vector: {:?}", file_vec);
    info!("integer vector size: {}", file_vec.len());
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = DeltaBPCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in BP compress function() is: {:?}", duration);
    let org_size = file_vec.len()*4;
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_DFCM_compress_on_int(file:&str,scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();

    //println!("integer vector: {:?}", file_vec);
    info!("integer vector size: {}", file_vec.len());
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = DFCMCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in DFCM compress function() is: {:?}", duration);
    let org_size = file_vec.len()*4;
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}

pub fn test_FCM_compress_on_int(file:&str,scl:i32) {
    let file_iter = construct_file_iterator_int_signed(file, 1, ',',scl);
    let file_vec: Vec<i32> = file_iter.unwrap().collect();

    //println!("integer vector: {:?}", file_vec);
    info!("integer vector size: {}", file_vec.len());
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let start = Instant::now();
    let comp = FCMCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    info!("Time elapsed in FCM compress function() is: {:?}", duration);
    let org_size = file_vec.len()*4;
    let throughput = 1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0;
    println!(",    {}", throughput);
}


#[test]
fn test_bit_packing_on_int() {

// Detects if `SSE3` is available on the current computed
// and uses the best available implementation accordingly.
    let bitpacker = BitPacker1x::new();

    let my_data = vec![1391, 1756, 532, 1912, 569, 382, 1520, 1238, 795, 1742, 1867, 665, 1972, 1342, 1205, 1717,
                       722, 1182, 961, 1623, 470, 8, 321, 789, 1448, 1022, 783, 941, 1204, 196, 866, 1555];
    println!("Time elapsed in zlib compress fu");
// Computes the number of bits used for each integers in the blocks.
// my_data is assumed to have a len of 128 for `BitPacker4x`.
    let num_bits: u8 = bitpacker.num_bits(&my_data);

// The compressed array will take exactly `num_bits * BitPacker4x::BLOCK_LEN / 8`.
// But it is ok to have an output with a different len as long as it is larger
// than this.
    let mut compressed = vec![0u8; 4 * BitPacker1x::BLOCK_LEN];

// Compress returns the len.
    let compressed_len = bitpacker.compress(&my_data, &mut compressed[..], num_bits);
    assert_eq!((num_bits as usize) * BitPacker1x::BLOCK_LEN / 8, compressed_len);

// Decompressing
    let mut decompressed = vec![0u32; BitPacker1x::BLOCK_LEN];

    bitpacker.decompress(&compressed[..compressed_len], &mut decompressed[..], num_bits);
    println!("decompressed: {:?}", decompressed.as_slice());
    assert_eq!(&my_data, &decompressed);
}

#[test]
fn test_zlib_compress() {
    let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
    e.write_all(b"foo");
    e.write_all(b"bar");
    let compressed_bytes = e.finish();
}

#[test]
fn test_Grilla_seg_compress() {
    //let mut data = vec![-100,-1,1,-2,3,-4,5,9876,-234,384,-3365,23,-34];
//    data = delta_offset(&data);
//    let mut data = vec![0,2,3,4,5,6,9877];
    let mut data = vec![0.53,0.53,0.53,0.53,0.53,0.53,0.53,-51.2,-51.0,-49.1,-52.8,-0.10501,3.4734,6.2428,4.8929,7.9279,8.6533,8.5902,9.305,9.1,8.9758,10.466,11.875,13.292,13.963,12.756,13.473,15.103,15.592,16.627,17.354,17.05,17.344,16.557,17.445,16.298,15.229,14.42,11.476,12.914,13.239,12.484,13.854,12.143,12.041,11.799,12.118,12.431,11.566,11.536,11.372,11.999,13.093,14.202,13.338,13.415,12.201,11.088,11.081,12.614,11.844,12.215,11.99,13.107,12.018,12.051,12.603,13.704,15.248,15.334,13.842,13.1,12.038,14.389,13.773,14.521,14.329,15.218,14.453,13.05,11.628,12.116,11.939,11.743,13.162,13.454,13.652,15.239,14.435,15.131,15.966,15.723,15.938,14.773,13.625,13.73,14.452,17.037,16.37,16.558,16.475,14.542,14.103,12.309,13.149,12.261,12.361,11.816,12.12,11.52,12.01,12.749,14.461,14.267,12.128,11.289,12.643,11.571,12.532,12.656,14.093,12.132,11.934,10.726,13.634,14.46,15.839,14.78,14.312,14.039,15.138,14.86,15.562,13.51,13.156,12.332,10.755,11.263,11.545,11.579,10.245,11.372,11.723,11.424,11.446,11.184,9.4342,9.1486,8.3172,7.338,6.1816,5.6481,3.6454,4.6097,5.1297,5.1097,5.0749,4.2768,5.2954,5.1622,4.4477,5.7991,5.5743,4.9853,4.6915,3.8436,2.7235,5.2495,6.905,7.2125,5.9554,5.0899,4.9134,5.7048,4.3728,2.0429,0.59383,0.92734,1.3187,1.7704,1.6401,1.8238,1.3476,2.2096,0.84795,1.303,0.45427,0.11938,0.67217,1.7113,0.59362,1.8543,2.5144,2.4466,2.2513,2.0337,1.7306,1.7537,1.805,2.631,4.158,4.6249,4.4152,5.0404,5.2236,4.1938,5.1431,5.4501,5.5853,6.1006,6.362,5.4205,5.2581,5.1121,4.5801,6.2622,5.3864,4.9026,4.1906,3.0164,2.8242,2.5501,4.0802,3.8311,2.7669,4.3704,5.6051,5.3754,3.8693,3.4247,3.2687,3.5448,3.2836,3.727,4.1189,2.8683,1.9203,1.1792,0.67138,0.3508,0.36327,-2.6659,-3.1229,-1.8805,-2.9472,-2.0134,-1.6631,-1.6921,-1.5097,-3.0747,-3.1593,-1.5553,-1.457,-1.4156,-2.1498,-2.1806,-1.9482,-1.5219,-1.8947,-2.1311,-0.10743,-2.3658,-0.13634,0.20123,1.2013,-0.46288,-1.0529,-1.331,-0.90826,-2.5785,-2.1068,-3.3197,-3.2535,-2.6011,-2.2741,-1.1914,-0.18536,-0.83627,-0.57921,-1.5236,-2.8454,-1.9205,-1.9205,-1.9754,-1.0643,-0.46971,-0.11951,1.1307,2.0605,2.3003,1.6099,0.95838,2.1505,0.53865,0.51419,-1.4347,-0.41416,0.44756,0.44872,0.37788,-2.1084,-1.5272,-3.7197,-6.0389,-5.959,-6.9075,-6.496,-5.819,-4.9613,-5.6524,-5.2031,-5.1024,-4.2764,-3.7402,-2.8423,-2.9743,-3.1215,-2.1137,-4.2373,-4.7419,-6.0125,-6.3951,-5.7464,-4.9207,-5.9356,-6.4067,-6.2697,-6.5616,-6.2597,-5.8598,-6.7898,-6.9666,-9.0987,-7.9533,-8.5824,-9.7863,-10.04,-11.469,-11.49,-12.05,-9.8726,-8.7341,-11.231,-10.79,-12.188,-12.443,-12.278,-11.531,-11.804,-10.228,-10.708,-10.381,-9.7162,-9.631,-8.7501,-8.4268,-9.211,-11.016,-9.1578,-9.7623,-9.6589,-9.0958,-8.9822,-9.8869,-10.355,-10.48,-9.0005,-9.8614,-9.0767,-8.7681,-9.0019,-10.059,-10.343,-10.43,-11.899,-11.707,-12.529,-12.623,-12.287,-13.192,-13.48,-13.13,-14.966,-13.93,-11.506,-10.546,-10.862,-10.433,-11.469,-9.5914,-8.6507,-7.8634,-8.7392,-8.4193,-8.9776,-9.289,-9.859,-10.885,-11.793,-12.003,-13.702,-13.095,-13.212,-12.513,-12.244,-11.749,-13.232,-14.253,-14.7,-14.59,-13.461,-13.751,-12.49,-12.014,-10.84,-10.713,-11.37,-12.851,-12.696,-11.877,-12.17,-12.711,-13.019,-14.116,-14.609,-14.79,-14.744,-14.808,-14.196,-14.087,-12.273,-11.961,-10.157,-10.88,-10.353,-10.613,-10.013,-9.4193,-11.605,-12.932,-14.373,-13.972,-12.501,-12.828,-12.016,-11.47,-12.522,-12.124,-12.876,-11.36,-11.393,-9.7566,-10.182,-9.5923,-9.6551,-11.677,-12.659,-12.047,-12.102,-13.22,-13.847,-13.597,-14.59,-13.615,-14.256,-12.447,-13.527,-13.328,-14.849,-15.572,-16.166,-15.764,-14.822,-14.522,-14.895,-14.079,-13.28,-13.16,-12.589,-12.176,-13.163,-12.404,-13.061,-13.665,-13.488,-13.795,-13.927,-13.332,-12.285,-12.483,-12.155,-12.393,-12.164,-11.724,-12.341,-12.066,-11.465,-11.372,-9.6426,-10.251,-10.988,-12.738,-11.828,-10.961,-11.04,-10.142,-9.9582,-9.6674,-9.5545,-9.1145,-9.0129,-6.2255,-7.3922,-9.2465,-10.387,-11.481,-11.914,-12.083,-12.301,-11.76,-11.371,-10.619,-8.8411,-7.618,-8.9012,-11.23,-10.328,-12.164,-12.097,-12.062,-9.8345,-9.9037,-10.411,-10.175,-9.9294,-9.8594,-10.468,-11.691,-11.374,-12.717,-13.749,-12.418,-12.837,-12.977,-12.077,-12.377,-11.348,-11.693,-10.68,-10.051,-10.264,-11.13,-12.173,-12.443,-12.881,-13.29,-12.306,-12.604,-11.46,-11.992,-11.019,-11.541,-11.365,-10.394,-10.808,-11.246,-9.243,-8.292,-8.724,-8.0751,-8.4351,-7.7293,-6.3134,-7.9179,-6.8891,-5.4311,-5.3836,-3.6374,-3.482,-4.7191,-6.9126,-7.246,-6.5325,-6.2151,-5.8014,-6.3785,-6.2345,-7.8732,-8.6333,-9.4521,-8.9323,-8.9465,-10.102,-10.112,-10.801,-11.468,-10.604,-10.491,-10.092,-9.2082,-9.0279,-8.4771,-7.7941,-6.6235,-6.1476,-4.7354,-4.7128,-4.7607,-3.0593,-3.569,-3.5719,-2.652,-2.5022,-1.0973,-0.063151,0.22842,-0.54928,0.017417,-1.3652,-1.1207,-0.31229,-0.099249,0.78043,2.8193,3.7432,4.0102,4.6518,5.0773,3.7626,3.3462,4.5709,4.5273,5.1097,4.1032,4.1677,4.768,3.4065,3.7541,3.5722,2.6327,2.5952,0.69886,-1.4291,-2.606,-3.5966,-4.7696,-6.495,-6.2068,-7.801,-7.6908,-6.9037,-6.9059,-6.8128,-7.191,-8.6737,-8.7175,-7.7566,-6.0184,-6.4486,-8.0759,-7.9096,-7.5333,-7.7603,-8.9092,-6.8848,-9.2444,-9.7543,-11.076,-11.712,-11.394,-11.256,-11.967,-11.19,-10.568,-9.9202,-10.346,-9.2972,-8.6365,-6.1277,-5.0643,-3.9073,-3.8544,-5.1427,-5.514,-6.2718,-6.8357,-6.2806,-6.8374,-7.7325,-8.1418,-8.3027,-7.8934,-8.846,-8.5287,-8.4507,-7.1263,-7.3394,-7.4739,-8.6453,-10.031,-9.72,-9.9695,-9.4658,-10.358,-8.4499,-8.3277,-7.2807,-7.5076,-7.6701,-6.98,-6.4243,-7.5445,-9.0772,-10.175,-11.591,-11.531,-11.943,-12.311,-13.672,-12.892,-12.453,-12.542,-11.521,-12.395,-11.98,-11.632,-11.283,-12.012,-11.685,-12.2,-13.096,-14.3,-13.262,-14.108,-14.281,-15.489,-15.786,-19.018,-20.105,-21.532,-22.546,-22.76,-23.085,-21.14,-21.712,-21.962,-23.532,-24.009,-25.347,-25.317,-24.464,-24.059,-24.76,-26.39,-24.93,-22.88,-22.76,-23.75,-22.552,-23.145,-23.615,-22.728,-24.113,-26.07,-25.649,-25.249,-25.154,-24.657,-23.575,-22.604,-23.173,-22.363,-22.19,-22.695,-23.888,-23.241,-23.595,-23.549,-24.342,-25.892,-25.72,-25.783,-24.584,-23.782,-22.729,-23.477,-24.414,-25.683,-25.185,-22.396,-21.668,-22.441,-21.605,-22.733,-24.157,-23.44,-24.218,-23.902,-22.495,-22.094,-21.165,-22.77,-22.109,-19.97,-19.429,-20.97,-21.173,-21.673,-21.29,-20.878,-20.473,-20.836,-21.436,-22.025,-21.172,-23.025,-23.232,-22.962,-23.614,-23.137,-23.209,-24.147,-23.986,-24.254,-24.664,-25.375,-25.313,-27.16,-27.558,-28.101,-29.013,-28.361,-29.095,-28.554,-27.578,-27.735,-27.458,-26.818,-26.899,-26.358,-27.621,-26.51,-27.5,-29.329,-27.944,-28.007,-27.558,-27.921,-28.942,-32.015,-31.389,-31.675,-31.873,-31.467,-32.886,-33.616,-32.468,-31.871,-33.152,-35.355,-35.926,-35.712,-34.77,-34.676,-35.799,-35.492,-36.665,-37.626,-38.279,-39.509,-39.78,-40.68,-40.965,-41.428,-41.838,-42.341,-41.108,-40.498,-40.439,-41.905,-43.531,-45.496,-42.891,-41.918,-41.661,-42.636,-43.782,-43.234,-41.669,-43.363,-43.812,-43.896,-45.888,-45.047,-45.462,-43.55,-43.94,-43.531,-44.674,-45.299,-46.467,-46.075,-44.773,-45.367,-44.93,-45.435,-45.332,-44.136,-44.016,-45.053,-45.91,-46.08,-46.271,-47.137,-46.957,-45.69,-45.941,-46.146,-48.347,-49.122,-50.515,-50.901,-50.376,-48.852,-47.054,-47.171,-47.491,-46.674,-46.183,-45.418,-44.64,-46.12,-45.58,-45.671,-46.432,-47.125,-45.844,-46.653,-47.89,-47.676,-45.665,-45.639,-45.331,-46.269,-44.595,-44.47,-43.94,-44.892,-44.038,-43.649,-44.805,-44.765,-45.216,-45.106,-45.357,-45.547,-46.58,-46.903,-46.136,-44.392,-45.552,-43.175,-41.649,-41.48,-41.782,-42.48,-41.647,-42.342,-42.804,-41.92,-41.484,-40.588,-40.083,-40.484,-40.998,-40.201,-40.872,-39.686,-38.895,-38.607,-38.604,-38.239,-34.712,-34.824,-36.381,-34.466,-33.856,-34.504,-31.887,-31.336,-31.041,-31.819,-32.884,-34.653,-35.075,-36.129,-35.481,-35.798,-34.029,-32.519,-32.355,-32.638,-31.485];
    data  = delta_offset(&data);
    let mut seg1 = Segment::new(None,SystemTime::now(),0,data.clone(),None,None);
    let start = Instant::now();
    let comp = GorillaCompress::new(10,10);
    let compressed = comp.encode(&mut seg1);
    println!("Compressed size: {}", compressed.len());
    let duration = start.elapsed();
    println!("Time elapsed in compress function() is: {:?}", duration);
    let decompress = comp.decode(compressed);
    println!("expected datapoints: {:?}", decompress);

}

#[test]
fn test_Grilla_compress() {
    const DATA: &'static str = "1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892280,7.78
1482892288,7.95
1482892292,5.53
1482892310,4.41
1482892323,5.30
1482892334,5.30
1482892341,2.92
1482892350,0.73
1482892360,-1.33
1482892370,-1.78
1482892390,-12.45
1482892401,-34.76
1482892490,78.9
1482892500,335.67
1482892800,12908.12
";
    let w = BufferedWriter::new();

    // 1482892260 is the Unix timestamp of the start of the stream
    let mut encoder = StdEncoder::new(1482892260, w);

    let mut actual_datapoints = Vec::new();

    for line in DATA.lines() {
        let substrings: Vec<&str> = line.split(",").collect();
        let t = substrings[0].parse::<u64>().unwrap();
        let v = substrings[1].parse::<f64>().unwrap();
        let dp = DataPoint::new(t, v);
        actual_datapoints.push(dp);
    }

    for dp in &actual_datapoints {
        encoder.encode(*dp);
    }

    let bytes = encoder.close();
    let r = BufferedReader::new(bytes);
    let mut decoder = StdDecoder::new(r);

    let mut expected_datapoints = Vec::new();

    let mut done = false;
    loop {
        if done {
            break;
        }

        match decoder.next() {
            Ok(dp) => expected_datapoints.push(dp),
            Err(err) => {
                if err == Error::EndOfStream {
                    done = true;
                } else {
                    panic!("Received an error from decoder: {:?}", err);
                }
            }
        };
    }

    println!("actual datapoints: {:?}", actual_datapoints);
    println!("expected datapoints: {:?}", expected_datapoints);
}

#[test]
fn test_My_Grilla_compress() {
    const DATA: &'static str = "1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892270,1.76
1482892280,7.78
1482892288,7.95
1482892292,5.53
1482892310,4.41
1482892323,5.30
1482892334,5.30
1482892341,2.92
1482892350,0.73
1482892360,-1.33
1482892370,-1.78
1482892390,-12.45
1482892401,-34.76
1482892490,78.9
1482892500,335.67
1482892800,12908.12
";
    let w = BufferedWriter::new();

    // 1482892260 is the Unix timestamp of the start of the stream
    let mut encoder = GorillaEncoder::new(1482892260, w);

    let mut actual_datapoints = Vec::new();

    for line in DATA.lines() {
        let substrings: Vec<&str> = line.split(",").collect();
        let t = substrings[0].parse::<u64>().unwrap();
        let v = substrings[1].parse::<f64>().unwrap();
        encoder.encode_float(v);
        actual_datapoints.push(v);
    }

    let bytes = encoder.close();
    let r = BufferedReader::new(bytes);
    let mut decoder = GorillaDecoder::new(r);

    let mut expected_datapoints = Vec::new();

    let mut done = false;
    loop {
        if done {
            break;
        }

        match decoder.next_val() {
            Ok(dp) => expected_datapoints.push(dp),
            Err(err) => {
                if err == Error::EndOfStream {
                    done = true;
                } else {
                    panic!("Received an error from decoder: {:?}", err);
                }
            }
        };
    }

    println!("actual datapoints: {:?}", actual_datapoints);
    println!("expected datapoints: {:?}", expected_datapoints);
}


#[test]
fn run_bpsplit_encoding_decoding() {
    let args: Vec<String> = env::args().collect();
    println!("Arguments: {:?}", args);
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let comp = SplitDoubleCompress::new(10,10,100000);
    let start = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} bpsplit compress function() is: {:?}",comp.type_id(), duration);
    test_split_compress_on_file::<f64>(TEST_FILE,100000);
    let start = Instant::now();
    comp.decode(compressed);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} decompress function() is: {:?}",comp.type_id(), duration);
}

#[test]
fn run_bp_double_encoding_decoding() {
    let args: Vec<String> = env::args().collect();
    println!("Arguments: {:?}", args);
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 1, ',');
    let file_vec: Vec<f64>= file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let comp = BPDoubleCompress::new(10,10,100000);
    let start = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} bp_double compress function() is: {:?}",comp.type_id(), duration);
    test_BP_double_compress_on_file::<f64>(TEST_FILE,100000);
    let start = Instant::now();
    comp.decode(compressed);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} decompress function() is: {:?}",comp.type_id(), duration);
}

#[test]
fn run_splitbd_encoding_decoding() {
    let args: Vec<String> = env::args().collect();
    println!("Arguments: {:?}", args);
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 1, ',');
//    let file_iter = construct_file_iterator_skip_newline::<f64>("../UCRArchive2018/ECGFiveDays/ECGFiveDays_TEST", 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let comp = SplitBDDoubleCompress::new(10,10);
    let start = Instant::now();
    let compressed = comp.offset_encode(&mut seg);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} splitbd compress function() is: {:?}",comp.type_id(), duration);
    test_splitbd_compress_on_file::<f64>(TEST_FILE);
    let start = Instant::now();
    comp.decode(compressed);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} decompress function() is: {:?}",comp.type_id(), duration);
}


#[test]
fn run_gorillabd_encoding_decoding() {
    let args: Vec<String> = env::args().collect();
    println!("Arguments: {:?}", args);
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let comp = GorillaBDCompress::new(10,10);
    let start = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} gorillabd compress function() is: {:?}",comp.type_id(), duration);
    test_grillabd_compress_on_file::<f64>(TEST_FILE);
    let start = Instant::now();
    comp.decode(compressed);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} decompress function() is: {:?}",comp.type_id(), duration);
}

#[test]
fn run_gorilla_encoding_decoding() {
    let args: Vec<String> = env::args().collect();
    println!("Arguments: {:?}", args);
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let comp = GorillaCompress::new(10,10);
    let start = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} gorilla compress function() is: {:?}",comp.type_id(), duration);
    test_grilla_compress_on_file::<f64>(TEST_FILE);
    let start = Instant::now();
    comp.decode(compressed);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} decompress function() is: {:?}",comp.type_id(), duration);
}

