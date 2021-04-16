use num::traits::real::Real;
use num::FromPrimitive;
use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{TEST_FILE, SnappyCompress, GZipCompress};
use std::collections::{HashMap, HashSet};
use crate::compress::gorilla::GorillaCompress;
use crate::segment::Segment;
use std::time::SystemTime;
use parquet::basic::Compression::SNAPPY;
use std::mem;
use crate::compress::PRECISION_MAP;



pub fn est_buff_cr(vec:Vec<f64>, sample_count:usize, prec:usize) -> f64{
    let mut sample= &vec[0..sample_count];
    let mut min = f64::max_value();
    let mut max = f64::min_value();

    for &val in sample{
        if val<min{
            min = val;
        }else if val>max {
            max = val;
        }
    }
    let offset = max-min;
    let integral = u64::from_f64(offset).unwrap();
    let bits = 64-u64::leading_zeros(integral);
    let dec_len = *(PRECISION_MAP.get(&(prec as i32)).unwrap()) as u32;
    ((bits+dec_len) as f64)/64.0
}


pub fn est_dict_cr(vec:Vec<f64>, sample_count:usize, prec:usize) -> f64{
    let ahead_ratio = 1.1;
    let mut sample= &vec[0..sample_count];
    let mut ahead = (sample_count as f64 * ahead_ratio) as usize;
    let mut ahead_v = &vec[sample_count..ahead];
    let mut sample_u64 = sample.iter().clone();
    let mut ahead_vu64 = ahead_v.iter().clone();
    let hash_set: HashSet<u64> = sample_u64.map(|&x| unsafe { mem::transmute::<f64, u64>(x) }).collect();
    let ahead_set: HashSet<u64> = ahead_vu64.map(|&x| unsafe { mem::transmute::<f64, u64>(x) }).collect();
    println!("vector size: {}, hash set size: {}", sample.len(),hash_set.len());

    let union: HashSet<_> = hash_set.union(&ahead_set).collect();
    println!("ahead vector size: {}, total hash set size: {}", ahead_v.len(),union.len());

    let dist = hash_set.len() as u64;
    let bits = 64-u64::leading_zeros(dist);
    ((bits as u64*sample.len() as u64+dist*64) as f64)/(64.0* sample.len() as f64)
}

pub fn est_sprintz_cr(vec:Vec<f64>, sample_count:usize, prec:usize) -> f64{
    let sample = &vec[0..sample_count];

    let mut zigzag = 0;
    let mut pre = 0;
    let mut cur = 0;
    let scale  = isize::pow(10, prec as u32) as f64;
    let mut max_z = 0;

    for &val in sample{
        cur = (val*scale) as isize;
        zigzag = cur-pre;
        if zigzag<=0{
            zigzag = -2*zigzag;
        }else {
            zigzag = 2*zigzag-1;
        }
        if (zigzag>max_z){
            max_z = zigzag;
        }
        pre = cur;
    }
    let offset = max_z as u64;
    let bits = 64-u64::leading_zeros(offset);
    (bits as f64)/64.0
}


pub fn est_snappy_cr(vec:Vec<f64>, sample_count:usize, prec:usize) -> f64{
    let sample = &vec[0..sample_count].to_vec();
    let mut seg = Segment::new(None,SystemTime::now(),0,sample.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SnappyCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    compressed.len() as f64/ org_size as f64
}

pub fn est_gzip_cr(vec:Vec<f64>, sample_count:usize, prec:usize) -> f64{
    let sample = &vec[0..sample_count].to_vec();
    let mut seg = Segment::new(None,SystemTime::now(),0,sample.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GZipCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    compressed.len() as f64/ org_size as f64
}

pub fn est_gorilla_cr(vec:Vec<f64>, sample_count:usize, prec:usize) -> f64{
    let sample = &vec[0..sample_count].to_vec();
    let mut seg = Segment::new(None,SystemTime::now(),0,sample.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GorillaCompress::new(10,10);
    let compressed = comp.encode(&mut seg);
    compressed.len() as f64/ org_size as f64
}

#[test]
fn run_predict() {
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap().collect();
    let len = file_vec.len();
    let sample = (len as f64 * 0.01) as usize;
    let p_ratio = est_dict_cr(file_vec,sample, 4);
    println!("\npredicted ratio {}",p_ratio);

}