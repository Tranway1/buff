use time_series_start::client::construct_file_iterator_skip_newline;
use time_series_start::methods::compress::SCALE;
use time_series_start::segment::Segment;
use std::time::{SystemTime, Instant};
use time_series_start::compress::split_double::SplitBDDoubleCompress;
use std::env;
use log::{ info};


fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let test_file = &args[1];
    let scl = args[2].parse::<usize>().unwrap();
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.byte_encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_de = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_decode(comp_de);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd byte full decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.decode_with_precision(comp_cp,2);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte 2 precision decompress function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.decode_with_precision(comp_eq,0);
    let duration4 = start4.elapsed();
    println!("Time elapsed in splitbd byte 0 precision filter function() is: {:?}", duration4);

    let comp_5 = compressed.clone();
    let comp_6 = compressed.clone();
    let comp_7 = compressed.clone();

    let start5 = Instant::now();
    comp.byte_sum(comp_5);
    let duration5 = start5.elapsed();
    println!("Time elapsed in splitbd byte full sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.sum_with_precision(comp_6,2);
    let duration6 = start6.elapsed();
    println!("Time elapsed in splitbd byte 2 precision sum function() is: {:?}", duration6);

    let start7 = Instant::now();
    comp.sum_with_precision(comp_7,0);
    let duration7 = start7.elapsed();
    println!("Time elapsed in splitbd byte 0 precision sum function() is: {:?}", duration7);

    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration7.as_nanos() as f64 / 1024.0/1024.0
    )
}