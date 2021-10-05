pub mod split_double;
pub mod buff_simd;
pub mod buff_slice;

use std::{env, fs};
use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{SCALE,  PRED, TEST_FILE};
use std::time::{SystemTime, Instant};
use crate::segment::{Segment, FourierCompress, PAACompress};
use std::path::Path;
use std::rc::Rc;
use parquet::file::properties::WriterProperties;
use parquet::schema::parser::parse_message_type;
use parquet::basic::{Encoding, Compression};
use parquet::file::writer::{SerializedFileWriter, FileWriter};
use parquet::column::writer::ColumnWriter;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use croaring::Bitmap;
use parquet::file::reader::{SerializedFileReader, FileReader};
use parquet::record::RowAccessor;
use parity_snappy::compress;
use crate::compress::split_double::{SplitBDDoubleCompress, OUTLIER_R};
use std::collections::HashMap;
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
