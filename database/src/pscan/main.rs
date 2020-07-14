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
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd byte full decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.decode_with_precision(comp_cp,0);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte 0 precision decompress function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.decode_with_precision(comp_eq,2);
    let duration4 = start4.elapsed();
    println!("Time elapsed in splitbd byte 2 precision filter function() is: {:?}", duration4);

    println!("Performance:{},{},{},{},{},{},{}", test_file, scl,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}