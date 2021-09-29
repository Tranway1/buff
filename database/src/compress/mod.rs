pub mod split_double;
pub mod sprintz;
pub mod gorilla;
pub mod btr_array;
pub mod buff_simd;
pub mod buff_slice;
pub mod scaled_slice;

use std::{env, fs};
use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{SCALE, SplitDoubleCompress, test_split_compress_on_file, BPDoubleCompress, test_BP_double_compress_on_file, test_sprintz_double_compress_on_file, test_splitbd_compress_on_file, test_grillabd_compress_on_file, test_grilla_compress_on_file, GZipCompress, SnappyCompress, PRED, TEST_FILE};
use std::time::{SystemTime, Instant};
use crate::segment::{Segment, FourierCompress, PAACompress};
use std::path::Path;
use std::rc::Rc;
use parquet::file::properties::WriterProperties;
use parquet::schema::parser::parse_message_type;
use parquet::basic::{Encoding, Compression};
use crate::methods::parquet::{DICTPAGE_LIM, USE_DICT};
use parquet::file::writer::{SerializedFileWriter, FileWriter};
use parquet::column::writer::ColumnWriter;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use croaring::Bitmap;
use parquet::file::reader::{SerializedFileReader, FileReader};
use parquet::record::RowAccessor;
use parity_snappy::compress;
use crate::compress::split_double::{SplitBDDoubleCompress, OUTLIER_R};
use crate::compress::sprintz::SprintzDoubleCompress;
use crate::compress::gorilla::{GorillaBDCompress, GorillaCompress};
use std::collections::HashMap;
use crate::methods::bit_packing::{BitPack, BYTE_BITS};
use core::mem;
use crate::methods::prec_double::{PrecisionBound, get_precision_bound};
use std::sync::Arc;

lazy_static! {
    pub static ref PRECISION_MAP: HashMap<i32, i32> =[(1, 5),
        (2, 8),
        (3, 11),
        (4, 15),
        (5, 18),
        (6, 21),
        (7, 25),
        (8, 28),
        (9, 31),
        (10, 35),
        (11, 38),
        (12, 50),
        (13, 10),
        (14, 10),
        (15, 10)]
        .iter().cloned().collect();
}


lazy_static! {
    pub static ref FILE_MIN_MAX: HashMap<&'static str, (i64,i64)> =[("/home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k", (-166032i64,415662i64)),
        ("/home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv", (67193104i64,134089487i64)),
        ("/home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv", (9544233i64,9721774i64)),
        ("/home/cc/float_comp/signal/time_series_120rpm-c2-current.csv", (-78188537i64,80072697i64)),
        ("/home/cc/float_comp/signal/pmu_p1_L1MAG", (15029721087i64,15814987775i64)),
        ("/home/cc/float_comp/signal/vm_cpu_readings-file-19-20_c4-avg.csv", (0i64,209715200i64)),
        ("/home/cc/float_comp/signal/Stocks-c1-open.csv", (0i64,245760000i64)),
        ("/home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCR-all.csv", (-4317773i64,15728640i64)),
        ("/home/cc/float_comp/signal/gas-array-all-c2-Humidity.txt", (4183i64, 21455i64)),
        ("/home/cc/float_comp/signal/city_temperature_c8.csv", (-3168i64,3520i64)),
        ("/home/cc/float_comp/signal/Household_power_consumption_c3_voltage.csv", (57139i64,65062i64)),
        ("/home/cc/float_comp/signal/Household_power_consumption_c4_global_intensity.csv", (6i64,1548i64)),
        ("/home/cc/float_comp/benchmark/dt_cur_load.csv", (0i64,6825i64)),
        ("/home/cc/float_comp/benchmark/d_fuel_state.csv", (0i64,32i64)),
        ("/home/cc/float_comp/benchmark/r_longitude.csv", (-131931i64,47185920i64)),
        ("/home/cc/float_comp/benchmark/r_latitude.csv", (-227399i64,23592960i64)),
        ("/home/cc/float_comp/benchmark/acr_temperature.csv", (491520i64,917504i64))
        ]

        .iter().cloned().collect();
}

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

pub fn run_splitdouble_byte_residue_majority_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.byte_residue_encode_majority(&mut seg);
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

pub fn run_fixed_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.fixed_encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in fixed bp compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.fixed_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in fixed bp decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.fixed_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in fixed bp range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.fixed_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in fixed bp equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.fixed_sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in fixed bp sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.fixed_max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in fixed bp max function() is: {:?}", duration6);


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
    println!("Time elapsed in RAPG byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.byte_residue_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in RAPG byte decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.byte_residue_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in RAPG byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.byte_residue_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in RAPG byte equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.byte_residue_sum(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in RAPG sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.byte_residue_max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in RAPG max function() is: {:?}", duration6);


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
    let schema = Arc::new(parse_message_type(message_type).unwrap());
    let props = Arc::new(WriterProperties::builder()
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

#[test]
fn test_given_min_max() {
    let (min,max ) = *(FILE_MIN_MAX.get(&"../UCRArchive2018/Kernel/randomwalkdatasample1k-40k").unwrap());
    println!("min:{}, max{}", min,max);
}

#[test]
fn test_common_leading_bits() {
    let test_file:&str = &"/home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv";
    let scl:usize= 1000000;
    let SAMPLE = 100000;
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);

    let mut fixed_vec = Vec::new();

    let mut t:u32 = seg.get_data().len() as u32;
    let prec = (scl as f32).log10() as i32;
    let prec_delta = get_precision_bound(prec);
    println!("precision {}, precision delta:{}", prec, prec_delta);

    let mut bound = PrecisionBound::new(prec_delta);
    // let start1 = Instant::now();
    let dec_len = *(PRECISION_MAP.get(&prec).unwrap()) as u64;
    bound.set_length(0,dec_len);
    let mut min = i64::max_value();
    let mut max = i64::min_value();

    for bd in seg.get_data(){
        let fixed = bound.fetch_fixed_aligned((*bd).into());
        if fixed<min {
            min = fixed;
            println!("min: {}, value: {}", min, *bd)
        }
        if fixed>max {
            max = fixed;
        }
        fixed_vec.push(fixed);
    }
    let delta = max-min;
    let base_fixed = min as i64;
    println!("base integer: {}, max:{}",base_fixed,max);
    let base_fixed64:i64 = base_fixed;
    let mut single_val = false;
    let mut cal_int_length = 0.0;
    if delta == 0 {
        single_val = true;
    }else {
        cal_int_length = (delta as f64).log2().ceil();
    }


    let fixed_len = cal_int_length as usize;
    bound.set_length((cal_int_length as u64-dec_len), dec_len);
    let ilen = fixed_len -dec_len as usize;
    let dlen = dec_len as usize;
    println!("int_len:{},dec_len:{}",ilen as u64,dec_len);


    // if more than two bytes for integer section, use category byte compression
    if (ilen>1){
        // let duration1 = start1.elapsed();
        // println!("Time elapsed in dividing double function() is: {:?}", duration1);

        // let start1 = Instant::now();
        let mut head_len = 64 - fixed_len as usize;
        let mut remain = fixed_len;
        let mut bytec = 0;
        let mut bits_counter = [0u32; 32];

        let mut sample_u  = 0u64;
        let mut leading_zeros = 0usize;

        for &sample in &fixed_vec[0..SAMPLE]{
            sample_u = (sample-base_fixed64) as u64;
            leading_zeros = sample_u.leading_zeros() as usize - head_len;
            if leading_zeros>=0 && leading_zeros<32{
                bits_counter[leading_zeros] += 1;
            }
        }
        let mut sum  = 0;
        let zero_bytes = ilen/BYTE_BITS;
        let adjusted = zero_bytes*BYTE_BITS;
        let ol_conter = (SAMPLE as f32 * OUTLIER_R) as u32;
        let mut cols_outlier:u32 = 0;
        for i in 0..32 {
            println!("{} leading zeros frequency: {}", i, bits_counter[i]);
        }
        println!("outlier ratio: {}",ol_conter);
        for i in 0..adjusted{
            println!("{}: {}",i, bits_counter[i]);
            sum += bits_counter[i];
            if i%BYTE_BITS == (BYTE_BITS-1) {
                // println!("sum: {}",sum);
                if sum<ol_conter{
                    cols_outlier += 1;
                }
                else {
                    break;
                }
            }
        }
        println!("cols_outlier: {}",cols_outlier);

        // test for finding common prefix
        bits_counter = [0u32; 32];
        let mut pre_u = 0u64;
        let mut xor_u = 0u64;
        for &sample in &fixed_vec[0..SAMPLE]{
            sample_u = (sample-base_fixed64) as u64;
            xor_u = sample_u ^ pre_u;
            leading_zeros = xor_u.leading_zeros() as usize - head_len;
            if leading_zeros>=0 && leading_zeros<32{
                bits_counter[leading_zeros] += 1;
            }
            pre_u=sample_u;
        }
        let mut sum  = 0;
        let ol_conter = (SAMPLE as f32 * OUTLIER_R) as u32;
        let mut bits_prefix:u32 = 0;
        println!("outlier ratio: {}",ol_conter);
        for i in 0..32 {
            println!("{} same prefix frequency: {}", i, bits_counter[i]);
        }
        for i in 0..32{
            println!("{}: {}",i, bits_counter[i]);
            sum += bits_counter[i];
            // println!("sum: {}",sum);
            if sum<ol_conter{
                bits_prefix = i as u32;
            }
            else {
                break;
            }
        }

        println!("bits_prefix: {}",bits_prefix);

        let rsf=fixed_len-bits_prefix as usize;
        let mut frequency = 0i32;
        let mut major = 0u64;
        for &sample in &fixed_vec[0..SAMPLE]{
            sample_u = (sample-base_fixed64) as u64 >> rsf;
            if frequency == 0{
                major = sample_u;
                frequency = 1;
            }
            else if major == sample_u {
                frequency += 1;
            }
            else {
                frequency -= 1;
            }
        }
        println!("majority item {} with frequency {} in sample : {:#066b}", major, frequency, major);

    }
}
