use crate::segment::Segment;
use serde::{Serialize, Deserialize};
use crate::methods::bit_packing::{sprintz_double_encoder, BitPack, unzigzag};
use std::mem;
use croaring::Bitmap;
use std::time::Instant;
use crate::methods::compress::CompressionMethod;
use std::slice::Iter;
use my_bit_vec::BitVec;

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


    pub(crate) fn decode_condition(&self, bytes: Vec<u8>,cond:Iter<usize>) -> Vec<f64>{
        let mut expected_datapoints:Vec<f64> = Vec::new();
        let scl = self.scale as f64;
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();

        // check integer part and update bitmap;
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        let mut iter = cond.clone();
        let mut it = iter.next();
        let mut point = *it.unwrap();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            if i==point{
                expected_datapoints.push((cur_int as f64)/scl);
                it = iter.next();
                if it==None{
                    break;
                }
                point = *it.unwrap();
            }

            pre = cur_int;
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }


    pub(crate) fn sum(&self, bytes: Vec<u8>) -> f64{
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
        let mut sum_int = 0i64;
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            sum_int+=(cur_int as i64);
            // if i<10{
            //     println!("{}th value: {}",i,(cur_int as f64)/scl);
            // }
            pre = cur_int;
        }
        let sum = sum_int as f64/scl;
        println!("sum is: {:?}",sum);
        sum

    }

    pub(crate) fn max(&self, bytes: Vec<u8>) {
        let scl = self.scale as f64;
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        let mut res = Bitmap::create();
        // check integer part and update bitmap;
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        let mut max_int = i64::min_value();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            if cur_int as i64 > max_int{
                max_int =  cur_int as i64;
                res.clear();
                res.add(i);
            }
            else if cur_int as i64 == max_int {
                res.add(i);
            }
            pre = cur_int;
        }
        let max_f = max_int as f64/scl;
        println!("Max: {:?}",max_f);
        println!("Number of qualified items for max:{}", res.cardinality());
    }


    pub(crate) fn max_range(&self, bytes: Vec<u8>,s:u32, e:u32, window:u32) {
        let scl = self.scale as f64;
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(8).unwrap();
        let mut res = Bitmap::create();
        // check integer part and update bitmap;
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        let mut max_int = i64::min_value();
        let mut max_vec = Vec::new();
        let mut cur_s= s;

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            pre = cur_int;
            if i<s {
                continue;
            }else if i>=e {
                break;
            }
            if i==cur_s+window{
                max_vec.push(max_int);
                // println!("{}",max);
                max_int =i64::min_value();
                cur_s=i;
            }

            if cur_int as i64 > max_int{
                max_int =  cur_int as i64;
                res.remove_range(cur_s as u64 .. i as u64);
                res.add(i);
            }
            else if cur_int as i64 == max_int {
                res.add(i);
            }

        }
        assert_eq!((cur_s-s)/window+1, (e-s)/window);
        /// set max for last window
        max_vec.push(max_int);

        let max_vec_f64 : Vec<f64> = max_vec.iter().map(|&x| x as f64/scl).collect();

        println!("Max: {}",max_vec_f64.len());
        // println!("Number of qualified items for max_groupby:{}", res.cardinality());
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

    pub fn range_filter_condition(&self, bytes: Vec<u8>, pred:f64, mut iter: Iter<usize>) -> BitVec<u32> {
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base int:{}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let mut res = BitVec::from_elem(len as usize, false);
        let ilen = bitpack.read(8).unwrap();
        let target = pred;
        let adjust_target = (target*self.scale as f64).ceil() as i32;
        // check integer part and update bitmap;
        let mut it = iter.next();
        let mut point = *it.unwrap();
        let mut cur;
        let mut pre = base_int;
        let mut delta = 0i32;
        let mut cur_int = 0i32;
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            delta = unzigzag(cur);
            cur_int = pre+delta;
            if i==point{
                if cur_int>adjust_target{
                    res.set(i,true);
                }
                it = iter.next();
                if it==None{
                    break;
                }
                point = *it.unwrap();
            }

            pre = cur_int;
        }
        // res.run_optimize();
        println!("Number of qualified items:{}", res.cardinality());
        return res;
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
