use crate::segment::Segment;
use serde::{Serialize, Deserialize};
use std::mem;
use croaring::Bitmap;
use std::time::Instant;
use crate::methods::compress::CompressionMethod;
use num::Float;
use crate::stats::{Stats, merge_adjacent};
use std::collections::HashMap;
use crate::avl::btrarr::update_stats;

pub const NODE_SIZE: usize = 10000;

#[derive(Clone)]
pub struct BtrArrayIndex {
    chunksize: usize,
    batchsize: usize,
    scale: usize
}

impl BtrArrayIndex {
    pub fn new(chunksize: usize, batchsize: usize, scale: usize) -> Self {
        BtrArrayIndex { chunksize, batchsize, scale }
    }

    pub(crate) fn encode<'a,T>(&self, seg: &mut Segment<T>) -> (Vec<u8>, HashMap<i32, Vec<Stats<f64>>>)
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
        return btrindexing(seg.get_data().as_slice());
    }

    // pub(crate) fn decode(&self, bytes: Vec<u8>) -> Vec<f64>{
    //     let mut expected_datapoints:Vec<f64> = Vec::new();
    //     let scl = self.scale as f64;
    //     let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
    //     let ubase_int = bitpack.read(32).unwrap();
    //     let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
    //     println!("base int:{}",base_int);
    //     let len = bitpack.read(32).unwrap();
    //     println!("total vector size:{}",len);
    //     let ilen = bitpack.read(8).unwrap();
    //
    //     // check integer part and update bitmap;
    //     let mut cur;
    //     let mut pre = base_int;
    //     let mut delta = 0i32;
    //     let mut cur_int = 0i32;
    //     for i in 0..len {
    //         cur = bitpack.read(ilen as usize).unwrap();
    //         delta = unzigzag(cur);
    //         cur_int = pre+delta;
    //         // if i<10{
    //         //     println!("{}th value: {}",i,(cur_int as f64)/scl);
    //         // }
    //         expected_datapoints.push((cur_int as f64)/scl);
    //         pre = cur_int;
    //     }
    //     println!("Number of scan items:{}", expected_datapoints.len());
    //     expected_datapoints
    // }


    // pub(crate) fn sum(&self, bytes: Vec<u8>) -> f64{
    //     let mut expected_datapoints:Vec<f64> = Vec::new();
    //     let scl = self.scale as f64;
    //     let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
    //     let ubase_int = bitpack.read(32).unwrap();
    //     let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
    //     println!("base int:{}",base_int);
    //     let len = bitpack.read(32).unwrap();
    //     println!("total vector size:{}",len);
    //     let ilen = bitpack.read(8).unwrap();
    //
    //     // check integer part and update bitmap;
    //     let mut cur;
    //     let mut pre = base_int;
    //     let mut delta = 0i32;
    //     let mut cur_int = 0i32;
    //     let mut sum_int = 0i64;
    //     for i in 0..len {
    //         cur = bitpack.read(ilen as usize).unwrap();
    //         delta = unzigzag(cur);
    //         cur_int = pre+delta;
    //         sum_int+=(cur_int as i64);
    //         // if i<10{
    //         //     println!("{}th value: {}",i,(cur_int as f64)/scl);
    //         // }
    //         pre = cur_int;
    //     }
    //     let sum = sum_int as f64/scl;
    //     println!("sum is: {:?}",sum);
    //     sum
    //
    // }

    pub(crate) fn max_no_partial(&self, data: &[f64], ind: HashMap<i32, Vec<Stats<f64>>>) {
        let max_f = ind.get(&0).unwrap()[0].get_max();
        println!("Max: {:?}",max_f);
    }

    pub(crate) fn max_one_partial(&self, data: &[f64], ind: HashMap<i32, Vec<Stats<f64>>>) {
        // 0-80100
        let mut max_f = ind.get(&2).unwrap()[0].get_max().clone();
        for x in 80000..80100{
            if max_f<data[x]{
                max_f = data[x]
            }
        }

        println!("Max: {:?}",max_f);

    }

    pub(crate) fn max_two_partial(&self, data: &[f64], ind: HashMap<i32, Vec<Stats<f64>>>) {
        // 70100 - 160900
        let mut max_f = ind.get(&2).unwrap()[0].get_max().clone();
        for x in 70100..80000{
            if max_f<data[x]{
                max_f = data[x]
            }
        }

        for x in 160000..160900{
            if max_f<data[x]{
                max_f = data[x]
            }
        }

        println!("Max: {:?}",max_f);

    }

    // pub(crate) fn max(&self, bytes: Vec<u8>, ind: HashMap<i32, Vec<Stats<f64>>>, start: i64, end: i64) {
    //     let scl = self.scale as f64;
    //     let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
    //     let ubase_int = bitpack.read(32).unwrap();
    //     let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
    //     println!("base int:{}",base_int);
    //     let len = bitpack.read(32).unwrap();
    //     println!("total vector size:{}",len);
    //     let ilen = bitpack.read(8).unwrap();
    //     let mut res = Bitmap::create();
    //     // check integer part and update bitmap;
    //     let mut cur;
    //     let mut pre = base_int;
    //     let mut delta = 0i32;
    //     let mut cur_int = 0i32;
    //     let mut max_int = i64::min_value();
    //     for i in 0..len {
    //         cur = bitpack.read(ilen as usize).unwrap();
    //         delta = unzigzag(cur);
    //         cur_int = pre+delta;
    //         if cur_int as i64 > max_int{
    //             max_int =  cur_int as i64;
    //             res.clear();
    //             res.add(i);
    //         }
    //         else if cur_int as i64 == max_int {
    //             res.add(i);
    //         }
    //         pre = cur_int;
    //     }
    //     let max_f = max_int as f64/scl;
    //     println!("Max: {:?}",max_f);
    //     println!("Number of qualified items for max:{}", res.cardinality());
    // }

    // pub fn range_filter(&self, bytes: Vec<u8>,pred:f64) {
    //     let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
    //     let ubase_int = bitpack.read(32).unwrap();
    //     let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
    //     println!("base int:{}",base_int);
    //     let len = bitpack.read(32).unwrap();
    //     println!("total vector size:{}",len);
    //     let ilen = bitpack.read(8).unwrap();
    //     let target = pred;
    //     let adjust_target = (target*self.scale as f64).ceil() as i32;
    //     // check integer part and update bitmap;
    //     let mut cur;
    //     let mut pre = base_int;
    //     let mut delta = 0i32;
    //     let mut cur_int = 0i32;
    //     let mut res = Bitmap::create();
    //     for i in 0..len {
    //         cur = bitpack.read(ilen as usize).unwrap();
    //         delta = unzigzag(cur);
    //         cur_int = pre+delta;
    //         // if i<10{
    //         //     println!("{}th value: {}",i,cur);
    //         // }
    //         if cur_int>adjust_target{
    //             res.add(i);
    //         }
    //         pre = cur_int;
    //
    //     }
    //     // res.run_optimize();
    //     println!("Number of qualified items:{}", res.cardinality());
    // }

    // pub fn equal_filter(&self, bytes: Vec<u8>,pred:f64) {
    //     let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
    //     let ubase_int = bitpack.read(32).unwrap();
    //     let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
    //     println!("base int:{}",base_int);
    //     let len = bitpack.read(32).unwrap();
    //     println!("total vector size:{}",len);
    //     let ilen = bitpack.read(8).unwrap();
    //     let target = pred;
    //     let adjust_target = (target*self.scale as f64).ceil() as i32;
    //     // check integer part and update bitmap;
    //     let mut cur;
    //     let mut pre = base_int;
    //     let mut delta = 0i32;
    //     let mut cur_int = 0i32;
    //     let mut res = Bitmap::create();
    //     for i in 0..len {
    //         cur = bitpack.read(ilen as usize).unwrap();
    //         delta = unzigzag(cur);
    //         cur_int = pre+delta;
    //         // if i<10{
    //         //     println!("{}th value: {}",i,cur);
    //         // }
    //         if cur_int==adjust_target{
    //             res.add(i);
    //         }
    //         pre = cur_int;
    //
    //     }
    //     //res.run_optimize();
    //     println!("Number of qualified items for equal:{}", res.cardinality());
    // }
}

pub fn f_max(a : f64, b : f64) -> f64{
    if a>b{
        return a;
    }else {
        return b;
    }
}

pub fn f_min(a : f64, b : f64) -> f64{
    if a<b{
        return a;
    }else {
        return b;
    }
}

pub(crate) fn btrindexing<'a, T>(mydata: &[T]) -> (Vec<u8>, HashMap<i32, Vec<Stats<f64>>>)
    where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
    let ldata: Vec<f64> = mydata.into_iter().map(|x| (*x).into()).collect::<Vec<f64>>();
    let mut results:HashMap<i32, Vec<Stats<f64>>> = HashMap::new();

    let mut i = 0;
    let mut start = 0;
    let mut end = 0;
    let mut min = f64::max_value();
    let mut max = f64::min_value();
    let mut count = 0;
    let mut sum = 0.0;
    let mut avg = 0.0;
    let mut whole_stats = Stats::new(start,end, max, min,count, avg, sum);

    for &val  in ldata.iter() {
        count+=1;
        min = f_min(min,val);
        max = f_max(max,val);
        sum += val;
        if count==NODE_SIZE{
            end = i;
            avg = sum/(count as f64);
            let c_stat = Stats::new(start,end, max, min,count, avg, sum);
            whole_stats = merge_adjacent(&whole_stats, &c_stat);
            update_stats(&mut results, c_stat , 1);
            start = i+1;
            count = 0;
            min = f64::max_value();
            max = f64::min_value();
            sum = 0.0;
        }
        i+=1;
    }
    end = i;
    avg = sum/(count as f64);
    let c_stat = Stats::new(start,end, max, min,count, avg, sum);
    whole_stats = merge_adjacent(&whole_stats, &c_stat);

    update_stats(&mut results, c_stat , 1);
    update_stats(&mut results, whole_stats , 0);

    let bin = bincode::serialize(&ldata).unwrap();
    // for v in results.keys() {
    //     println!("{}th level vector with length {} : {:?}",v, results.get(v).unwrap().len(),results.get(v).unwrap());
    // }
    return (bin, results);
}

impl<'a, T> CompressionMethod<T> for BtrArrayIndex
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
