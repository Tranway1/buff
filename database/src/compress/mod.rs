pub mod split_double;
pub mod sprintz;
pub mod gorilla;

use std::{env, fs};
use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{SCALE, SplitDoubleCompress, test_split_compress_on_file, BPDoubleCompress, test_BP_double_compress_on_file, test_sprintz_double_compress_on_file, test_splitbd_compress_on_file, test_grillabd_compress_on_file, test_grilla_compress_on_file, GZipCompress, SnappyCompress, PRED, TEST_FILE};
use std::time::{SystemTime, Instant};
use crate::segment::Segment;
use std::path::Path;
use std::rc::Rc;
use parquet::file::properties::WriterProperties;
use parquet::schema::parser::parse_message_type;
use parquet::basic::{Encoding, Compression};
use crate::methods::parquet::{DICTPAGE_LIM, USE_DICT};
use parquet::file::writer::{SerializedFileWriter, FileWriter};
use parquet::column::writer::ColumnWriter;
use std::fs::File;
use croaring::Bitmap;
use parquet::file::reader::{SerializedFileReader, FileReader};
use parquet::record::RowAccessor;
use parity_snappy::compress;
use crate::compress::split_double::SplitBDDoubleCompress;
use crate::compress::sprintz::SprintzDoubleCompress;
use crate::compress::gorilla::{GorillaBDCompress, GorillaCompress};

pub fn run_bpsplit_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq= compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in bpsplit compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in bpsplit decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in bpsplit range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in bpsplit equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in bpsplit sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in bpsplit max function() is: {:?}", duration6);



    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0
    )
}


pub fn run_bp_double_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64>= file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = BPDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in bp_double compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in bp_double decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in bp_double range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(&comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in bp_double equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in bp_double sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in bp_double max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0

    )
}

pub fn run_sprintz_double_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64>= file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SprintzDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in sprintz_double compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in sprintz_double decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in sprintz_double range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in sprintz equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in sprintz sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in sprintz max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_splitbd_byte_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
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
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd byte decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.byte_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.byte_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in splitbd byte equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.byte_sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in byte_splitbd sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.byte_max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in byte_splitbd max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0


    )
}

pub fn run_splitdouble_byte_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.byte_fixed_encode(&mut seg);
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


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0


    )
}

pub fn run_splitdouble_byte_residue_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
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
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_residue_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd byte decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.byte_residue_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.byte_residue_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in splitbd byte equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.byte_residue_sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in byte_splitbd sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.byte_residue_max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in byte_splitbd max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0


    )
}

pub fn run_splitdouble_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.offset_encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in splitbd equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in splitbd sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in splitbd max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_gorillabd_encoding_decoding(test_file:&str, scl:usize,pred: f64 ) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GorillaBDCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in gorillabd compress function() is: {:?}", duration1);
    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in gorillabd decompress function() is: {:?}", duration2);
    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in gorillabd range filter function() is: {:?}", duration3);
    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in gorillabd equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in gorillabd sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in gorillabd max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_gorilla_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GorillaCompress::new(10,10);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in gorilla compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in gorilla decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in gorilla range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in gorilla equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in gorilla sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in gorilla max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_gzip_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GZipCompress::new(10,10);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in gzip compress function() is: {:?}", duration1);
    //test_grilla_compress_on_file::<f64>(TEST_FILE);
    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in gzip decompress function() is: {:?}", duration2);
    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in gzip range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in gzip equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in gzip sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in gzip max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_snappy_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SnappyCompress::new(10,10);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in snappy compress function() is: {:?}", duration1);
    //test_grilla_compress_on_file::<f64>(TEST_FILE);
    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in snappy decompress function() is: {:?}", duration2);
    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in snappy range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in snappy equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in snappy sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in snappy max function() is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0
    )
}


pub fn run_parquet_write_filter(test_file:&str, scl:usize,pred: f64, enc:&str){
    let path = Path::new("target/debug/examples/sample.parquet");
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let org_size=file_vec.len()*8;
    let mut comp = Compression::UNCOMPRESSED;
    let mut dictpg_lim:usize = 20;
    let mut use_dict = false;
    match enc {
        "dict" => {
            use_dict = true;
            dictpg_lim = 200000000;
        },
        "plain" => {},
        "pqgzip" => {comp=Compression::GZIP},
        "pqsnappy" => {comp=Compression::SNAPPY},
        _ => {panic!("Compression not supported by parquet.")}
    }
    // profile encoding
    let start = Instant::now();
    let message_type = "
      message schema {
        REQUIRED DOUBLE b;
      }
    ";
    let schema = Rc::new(parse_message_type(message_type).unwrap());
    let props = Rc::new(WriterProperties::builder()
        .set_encoding(Encoding::PLAIN)
        .set_compression(comp)
        .set_dictionary_pagesize_limit(dictpg_lim) // change max page size to avoid fallback to plain and make sure dict is used.
        .set_dictionary_enabled(use_dict)
        .build());
    let file = fs::File::create(&path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group_writer = writer.next_row_group().unwrap();
    while let Some(mut col_writer) = row_group_writer.next_column().unwrap() {
        // ... write values to a column writer
        match col_writer {
            ColumnWriter::DoubleColumnWriter(ref mut typed) => {
                typed.write_batch(&file_vec, None, None).unwrap();
            }
            _ => {
                unimplemented!();
            }
        }
        row_group_writer.close_column(col_writer).unwrap();
    }
    writer.close_row_group(row_group_writer).unwrap();
    writer.close().unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in parquet compress function() is: {:?}", duration);


    let bytes = fs::read(&path).unwrap();
    let comp_size= bytes.len();
    println!("file size: {}",comp_size);
    //println!("read: {:?}",str::from_utf8(&bytes[0..4]).unwrap());
    let file = File::open(&path).unwrap();

    // profile decoding
    let start1 = Instant::now();
    let mut expected_datapoints:Vec<f64> = Vec::new();
    let reader = SerializedFileReader::new(file).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    while let Some(record) = iter.next() {
        expected_datapoints.push( record.get_double(0).unwrap());
    }

    let duration1 = start1.elapsed();
    let num = expected_datapoints.len();
    println!("Time elapsed in parquet scan {} items is: {:?}",num, duration1);

    let file2 = File::open(&path).unwrap();

    // profile range filtering
    let start2 = Instant::now();
    let reader = SerializedFileReader::new(file2).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    let mut i = 0;
    let mut res = Bitmap::create();
    while let Some(record) = iter.next() {
        if (record.get_double(0).unwrap()>pred){
            res.add(i);
        }
        i+=1;
    }
    res.run_optimize();
    let duration2 = start2.elapsed();
    println!("Number of qualified items:{}", res.cardinality());
    println!("Time elapsed in parquet filter is: {:?}",duration2);

    let file3 = File::open(&path).unwrap();

    // profile equality filtering
    let start3 = Instant::now();
    let reader = SerializedFileReader::new(file3).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    let mut i = 0;
    let mut res = Bitmap::create();
    while let Some(record) = iter.next() {
        if (record.get_double(0).unwrap()==pred){
            res.add(i);
        }
        i+=1;
    }
    res.run_optimize();
    let duration3 = start3.elapsed();
    println!("Number of qualified items for equal:{}", res.cardinality());
    println!("Time elapsed in parquet filter is: {:?}",duration3);

    // profile agg
    let file4 = File::open(&path).unwrap();
    let start4 = Instant::now();
    let reader = SerializedFileReader::new(file4).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    let mut sum = 0f64;
    while let Some(record) = iter.next() {
        sum += record.get_double(0).unwrap()
    }

    let duration4 = start4.elapsed();
    println!("sum is: {:?}",sum);
    println!("Time elapsed in parquet sum is: {:?}",duration4);

    // profile max
    let file5 = File::open(&path).unwrap();
    let start5 = Instant::now();
    let mut res = Bitmap::create();
    let reader = SerializedFileReader::new(file5).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    let mut max_f = std::f64::MIN;
    let mut cur = 0.0;
    let mut i = 0;
    while let Some(record) = iter.next() {
        cur = record.get_double(0).unwrap();
        if max_f < cur{
            max_f = cur;
            res.clear();
            res.add(i)
        }
        else if max_f ==cur{
            res.add(i);
        }
        i+=1;
    }

    let duration5 = start5.elapsed();
    println!("Max: {:?}",max_f);
    println!("Number of qualified items for max:{}", res.cardinality());
    println!("Time elapsed in parquet max is: {:?}",duration5);

    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0
    )
}


