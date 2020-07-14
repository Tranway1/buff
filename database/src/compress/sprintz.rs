use crate::segment::Segment;
use serde::{Serialize, Deserialize};
use crate::methods::bit_packing::{sprintz_double_encoder, BitPack, unzigzag};
use std::mem;
use croaring::Bitmap;
use std::time::Instant;
use crate::methods::compress::CompressionMethod;

#[derive(Clone)]
pub struct SprintzDoubleCompress {
    chunksize: usize,
    batchsize: usize,
    scale: usize
}

impl SprintzDoubleCompress {
    pub fn new(chunksize: usize, batchsize: usize, scale: usize) -> Self {
        SprintzDoubleCompress { chunksize, batchsize, scale }
    }

    pub(crate) fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
        let comp = sprintz_double_encoder(seg.get_data().as_slice(),self.scale);
        comp
    }

    pub(crate) fn decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let mut expected_datapoints:Vec<f64> = Vec::new();
        let scl = self.scale as f64;
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();

        // check integer part and update bitmap;
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            // if i<10{
            //     println!("{}th value: {}",i,(cur_int as f64)/scl);
            // }
            expected_datapoints.push((cur_int as f64)/scl);
            pre = cur_int;
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    pub(crate) fn sum(&self, bytes: Vec<u8>) -> Vec<f64>{
        let mut expected_datapoints:Vec<f64> = Vec::new();
        let scl = self.scale as f64;
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();

        // check integer part and update bitmap;
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            // if i<10{
            //     println!("{}th value: {}",i,(cur_int as f64)/scl);
            // }
            expected_datapoints.push((cur_int as f64)/scl);
            pre = cur_int;
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    pub fn range_filter(&self, bytes: Vec<u8>,pred:f64) {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        let target = pred;
        let adjust_target = (target*self.scale as f64).ceil() as i32;
        // check integer part and update bitmap;
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        let mut res = Bitmap::create();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            if cur_int>adjust_target{
                res.add(i);
            }
            pre = cur_int;

        }
        // res.run_optimize();
        println!("Number of qualified items:{}", res.cardinality());
    }

    pub fn equal_filter(&self, bytes: Vec<u8>,pred:f64) {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        let target = pred;
        let adjust_target = (target*self.scale as f64).ceil() as i32;
        // check integer part and update bitmap;
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        let mut res = Bitmap::create();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            if cur_int==adjust_target{
                res.add(i);
            }
            pre = cur_int;

        }
        //res.run_optimize();
        println!("Number of qualified items for equal:{}", res.cardinality());
    }
}

impl<'a, T> CompressionMethod<T> for SprintzDoubleCompress
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
//        println!("Time elapsed in sprintz function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}
