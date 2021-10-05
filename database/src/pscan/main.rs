use buff::client::construct_file_iterator_skip_newline;
use buff::methods::compress::SCALE;
use buff::segment::Segment;
use std::time::{SystemTime, Instant};
use buff::compress::split_double::SplitBDDoubleCompress;
use std::env;
use log::{ info};


fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let test_file = &args[1];
    let scl = args[2].parse::<usize>().unwrap();
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.byte_residue_encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_d6 = compressed.clone();
    let comp_de = compressed.clone();
    let comp_d5 = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_residue_decode(comp_de);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd byte full decompress function() is: {:?}", duration2);

    let comp_d0 = compressed.clone();
    let comp_d1 = compressed.clone();
    let comp_d2 = compressed.clone();
    let comp_d3 = compressed.clone();
    let comp_d4 = compressed.clone();

    // let startd6 = Instant::now();
    // comp.byte_residue_decode_with_precision(comp_d6,6);
    // let durationd6 = startd6.elapsed();
    // println!("Time elapsed in splitbd byte 6 precision decompress function() is: {:?}", durationd6);


    let startd5 = Instant::now();
    comp.byte_residue_decode_with_precision(comp_d5,5);
    let durationd5 = startd5.elapsed();
    println!("Time elapsed in splitbd byte 5 precision decompress function() is: {:?}", durationd5);

    let startd4 = Instant::now();
    comp.byte_residue_decode_with_precision(comp_d4,4);
    let durationd4 = startd4.elapsed();
    println!("Time elapsed in splitbd byte 4 precision filter function() is: {:?}", durationd4);

    let startd3 = Instant::now();
    comp.byte_residue_decode_with_precision(comp_d3,3);
    let durationd3 = startd3.elapsed();
    println!("Time elapsed in splitbd byte 3 precision filter function() is: {:?}", durationd3);

    let startd2 = Instant::now();
    comp.byte_residue_decode_with_precision(comp_d2,2);
    let durationd2 = startd2.elapsed();
    println!("Time elapsed in splitbd byte 2 precision filter function() is: {:?}", durationd2);

    let startd1 = Instant::now();
    comp.byte_residue_decode_with_precision(comp_d1,1);
    let durationd1 = startd1.elapsed();
    println!("Time elapsed in splitbd byte 1 precision filter function() is: {:?}", durationd1);

    let startd0 = Instant::now();
    comp.byte_residue_decode_with_precision(comp_d0,0);
    let durationd0 = startd0.elapsed();
    println!("Time elapsed in splitbd byte 0 precision filter function() is: {:?}", durationd0);

    let comp_0 = compressed.clone();
    let comp_s6 = compressed.clone();
    let comp_s5 = compressed.clone();
    let comp_s4 = compressed.clone();
    let comp_s3 = compressed.clone();
    let comp_s2 = compressed.clone();
    let comp_s1 = compressed.clone();
    let comp_s0 = compressed.clone();

    let start3 = Instant::now();
    comp.byte_residue_sum(comp_0);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte full sum function() is: {:?}", duration3);

    // let starts6 = Instant::now();
    // comp.byte_residue_sum_with_precision(comp_s6,6);
    // let durations6 = starts6.elapsed();
    // println!("Time elapsed in splitbd byte 6 precision sum function() is: {:?}", durations6);

    let starts5 = Instant::now();
    comp.byte_residue_sum_with_precision(comp_s5,5);
    let durations5 = starts5.elapsed();
    println!("Time elapsed in splitbd byte 5 precision sum function() is: {:?}", durations5);

    let starts4 = Instant::now();
    comp.byte_residue_sum_with_precision(comp_s4,4);
    let durations4 = starts4.elapsed();
    println!("Time elapsed in splitbd byte 4 precision sum function() is: {:?}", durations4);

    let starts3 = Instant::now();
    comp.byte_residue_sum_with_precision(comp_s3,3);
    let durations3 = starts3.elapsed();
    println!("Time elapsed in splitbd byte 3 precision sum function() is: {:?}", durations3);

    let starts2 = Instant::now();
    comp.byte_residue_sum_with_precision(comp_s2,2);
    let durations2 = starts2.elapsed();
    println!("Time elapsed in splitbd byte 2 precision sum function() is: {:?}", durations2);

    let starts1 = Instant::now();
    comp.byte_residue_sum_with_precision(comp_s1,1);
    let durations1 = starts1.elapsed();
    println!("Time elapsed in splitbd byte 1 precision sum function() is: {:?}", durations1);

    let starts0 = Instant::now();
    comp.byte_residue_sum_with_precision(comp_s0,0);
    let durations0 = starts0.elapsed();
    println!("Time elapsed in splitbd byte 0 precision sum function() is: {:?}", durations0);

    println!("Performance:{},{},{},{},full mat:{},{},{},{},{},{},{},full sum: {},{},{},{},{},{},{}", test_file, scl,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             // 1000000000.0 * org_size as f64 / durationd6.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durationd5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durationd4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durationd3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durationd2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durationd1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durationd0.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             // 1000000000.0 * org_size as f64 / durations6.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durations5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durations4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durations3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durations2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durations1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / durations0.as_nanos() as f64 / 1024.0/1024.0
    );
    println!("Performance:{},{},{},{},full mat:{},{},{},{},{},{},{},full sum: {},{},{},{},{},{},{}", test_file, scl,
             comp_size as f64/ org_size as f64,
             duration1.as_millis(),
             duration2.as_millis(),
             // durationd6.as_millis(),
             durationd5.as_millis(),
             durationd4.as_millis(),
             durationd3.as_millis(),
             durationd2.as_millis(),
             durationd1.as_millis(),
             durationd0.as_millis(),
             duration3.as_millis(),
             // durations6.as_millis(),
             durations5.as_millis(),
             durations4.as_millis(),
             durations3.as_millis(),
             durations2.as_millis(),
             durations1.as_millis(),
             durations0.as_millis()
    )
}