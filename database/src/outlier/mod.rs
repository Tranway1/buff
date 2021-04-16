use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::SCALE;
use crate::segment::Segment;
use std::time::{SystemTime, Instant};
use crate::compress::split_double::SplitBDDoubleCompress;
use rand::distributions::{Distribution, Uniform};
use std::collections::HashMap;
use crate::compress::FILE_MIN_MAX;


pub const MAJOR:f32 = 69.0;

pub fn gen_u8_with_outlier(ratio:f64, size: u64) -> Result<Vec<u8>, usize>{
    let major = MAJOR as u8;
    let threshold = (ratio * 100.0) as i32;
    let mut vec = Vec::new();
    let mut rng = rand::thread_rng();
    let unif = Uniform::from(0..100);
    let mut ran = unif.sample(&mut rng);

    let mut otherrng = rand::thread_rng();
    let otherunif = Uniform::from(0..256);
    let mut rare = otherunif.sample(&mut otherrng);

    let mut array_rng = rand::thread_rng();
    let array_unif = Uniform::from(0..size);
    let index0 = array_unif.sample(&mut array_rng);
    let index255 = array_unif.sample(&mut array_rng);

    for i in 0..size{
        ran = unif.sample(&mut rng);
        if i==index0{
            vec.push(0u8);
            println!("zero index: {}",i);
        }else if i==index255 {
            vec.push(255u8);
            println!("255 index: {}",i);
        }
        else if ran >= threshold {
            rare = otherunif.sample(&mut otherrng);
            vec.push(rare as u8);
        }
        else {
            vec.push(major)
        }

    }
    let mut major_c = 0;
    let mut min=255;
    let mut max = 0;
    for &x in vec.iter() {
        if x == major{
            major_c += 1;
        }
        if x>max{
            max = x;
        }
        else if x<min {
            min = x;
        }
    }

    println!("min: {}, max: {}, size: {}", min, max,vec.len());
    println!("major ratio: {}", (major_c as f32)/(vec.len() as f32));

    Ok(vec)

}


pub fn outlier_byte_majority_encoding_decoding(vec_u8:Vec<u8>, size:u64, ratio:f64, scl:usize,pred: f64) {

    let mut file_vec: Vec<f64> = Vec::new();
    for x in vec_u8.into_iter(){
        file_vec.push(x as f64)
    }

    let test_file = "outlier_gen";
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.byte_residue_encode_outlieru8(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in RAPG-major byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_residue_decode_majority(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in RAPG-major byte decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.byte_residue_range_filter_majority(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in RAPG-major byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.byte_residue_equal_filter_majority(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in RAPG-major byte equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.byte_residue_sum_majority(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in RAPG-major sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.byte_residue_max_majority(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in RAPG-major max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", ratio, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0


    )
}

pub fn outlier_byteall_encoding_decoding(vec_u8:Vec<u8>, size:u64, ratio:f64, scl:usize,pred: f64) {

    let mut file_vec: Vec<f64> = Vec::new();
    for x in vec_u8.into_iter(){
        file_vec.push(x as f64)
    }

    let test_file = "outlier_gen";
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let mut compressed;
    if FILE_MIN_MAX.contains_key(test_file){
        let (min,max ) = *(FILE_MIN_MAX.get(test_file).unwrap());
        compressed = comp.single_pass_byte_fixed_encode(&mut seg,min,max);
    }
    else {
        println!("no min/max");
        compressed = comp.byte_fixed_encode(&mut seg);
    }
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_fixed_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd byte decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.byte_fixed_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.byte_fixed_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in splitbd byte equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.byte_fixed_sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in byte_splitbd sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.byte_fixed_max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in byte_splitbd max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", ratio, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0


    )
}