use crate::segment::Segment;
use serde::{Deserialize, Serialize};
use tsz::stream::{BufferedWriter, BufferedReader};
use crate::methods::gorilla_encoder::{GorillaEncoder, SepEncode};
use std::time::Instant;
use std::mem;
use log::info;
use crate::methods::gorilla_decoder::{GorillaDecoder, SepDecode};
use tsz::decode::Error;
use croaring::Bitmap;
use crate::methods::compress::CompressionMethod;
use crate::methods::prec_double::{get_precision_bound, PrecisionBound};

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
    pub(crate) fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        // 1482892260 is the Unix timestamp of the start of the stream
        let mut encoder = GorillaEncoder::new(0, w);
        let mut t =0;
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

    pub(crate) fn decode(&self, bytes: Vec<u8>) -> Vec<f64> {
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
                    // if i<10 {
                    //     println!("{}",dp);
                    // }
                    // i += 1;
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

    pub(crate) fn range_filter(&self, bytes: Vec<u8>,pred:f64) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);

        let mut expected_datapoints:Vec<f64> = Vec::new();
        println!("predicate:{} ", pred);
        let mut done = false;
        let mut i=0;
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
                    if dp>pred {
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
        // res.run_optimize();
        println!("Number of qualified items:{}", res.cardinality());
        expected_datapoints
    }

    pub(crate) fn equal_filter(&self, bytes: Vec<u8>,pred:f64) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);

        let mut expected_datapoints:Vec<f64> = Vec::new();
        println!("predicate:{} ", pred);
        let mut done = false;
        let mut i=0;
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
                    if dp==pred {
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
        // res.run_optimize();
        println!("Number of qualified items for equal:{}", res.cardinality());
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
    batchsize: usize,
    scale:usize
}

impl GorillaBDCompress {
    pub fn new(chunksize: usize, batchsize: usize, scale:usize) -> Self {
        GorillaBDCompress { chunksize, batchsize, scale }
    }

    // Compress a sample string and print it after transformation.
    pub(crate) fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        // 1482892260 is the Unix timestamp of the start of the stream
        let mut encoder = GorillaEncoder::new(0, w);

        let mut actual_datapoints:Vec<f64> = Vec::new();

        let mut t =0;
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        let mut bound = PrecisionBound::new(prec_delta);

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


    pub(crate) fn decode(&self, bytes: Vec<u8>) -> Vec<f64> {
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
                    // if i<10 {
                    //     println!("{}",dp);
                    // }
                    // i += 1;
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

    pub(crate) fn range_filter(&self, bytes: Vec<u8>, pred:f64) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        let mut bound = PrecisionBound::new(prec_delta);
        let target = bound.precision_bound(pred);

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
                    if dp>target {
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
        // res.run_optimize();
        println!("Number of qualified items:{}", res.cardinality());
        expected_datapoints
    }

    pub(crate) fn equal_filter(&self, bytes: Vec<u8>, pred:f64) -> Vec<f64> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = GorillaDecoder::new(r);
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        let mut bound = PrecisionBound::new(prec_delta);
        let target = bound.precision_bound(pred);

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
                    if dp==target {
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
        // res.run_optimize();
        println!("Number of qualified items for equal:{}", res.cardinality());
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
        info!("Time elapsed in GorillaBD function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}


