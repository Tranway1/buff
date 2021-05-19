use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::SCALE;
use crate::segment::Segment;
use std::time::{SystemTime, Instant};
use crate::compress::split_double::{SplitBDDoubleCompress, SAMPLE, MAJOR_R};
use crate::methods::prec_double::{get_precision_bound, PrecisionBound};
use crate::methods::bit_packing::{BitPack, BYTE_BITS};
use crate::simd::vectorize_query::{range_simd_myroaring, equal_simd_myroaring, equal_simd_mybitvec, range_simd_mybitvec};
use std::mem;
use log::{info,error};
use serde::{Serialize, Deserialize};
use crate::compress::buff_slice::{flip, floor, set_pred_word, BYTE_WORD, avx_iszero};
use crate::compress::PRECISION_MAP;
use itertools::Itertools;
use std::ops::{BitAnd, BitOr};
use std::arch::x86_64::{_mm256_lddqu_si256, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8, __m256i, _mm256_cmpgt_epi8, _mm256_or_si256};
use my_bit_vec::BitVec;
use crate::query::bit_vec_iter::{BVIter, bit_vec_compress, bit_vec_decompress};
use crate::compress::FILE_MIN_MAX;
use std::iter::FromIterator;
use myroaring::RoaringBitmap;
use croaring::Bitmap;

// pub const SIMD_THRESHOLD:f64 = 0.018;
pub const SIMD_THRESHOLD:f64 = 0.06;

pub fn run_buff_simd_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let mut compressed= comp.buff_simd256_encode(&mut seg);

    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.buff_simd256_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in buff simd decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.buff_simd_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in buff simd range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.buff_simd_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in buff simd equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.buff_simd_range_filter_with_slice(comp_sum,pred);
    let duration5 = start5.elapsed();
    println!("Time elapsed in buff simd range in BS way is: {:?}", duration5);

    let start6 = Instant::now();
    comp.buff_simd_equal_filter_with_slice(comp_max,pred);
    let duration6 = start6.elapsed();
    println!("Time elapsed in buff simd equal in BS way is: {:?}", duration6);


    // println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
    //          comp_size as f64/ org_size as f64,
    //          duration1.as_nanos() as f64 / 1000000.0,
    //          duration2.as_nanos() as f64 / 1000000.0,
    //          duration3.as_nanos() as f64 / 1000000.0,
    //          duration4.as_nanos() as f64 / 1000000.0,
    //          duration5.as_nanos() as f64 / 1000000.0,
    //          duration6.as_nanos() as f64 / 1000000.0
    // )

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


pub fn run_buff_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let mut compressed= comp.buff_simd256_encode(&mut seg);

    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.buff_simd256_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in buff simd decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.buff_simd_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in buff simd range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.buff_simd_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in buff simd equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.buff_simd_range_filter_with_slice(comp_sum,pred);
    let duration5 = start5.elapsed();
    println!("Time elapsed in buff simd range in BS way is: {:?}", duration5);

    let start6 = Instant::now();
    comp.buff_simd_equal_filter_with_slice(comp_max,pred);
    let duration6 = start6.elapsed();
    println!("Time elapsed in buff simd equal in BS way is: {:?}", duration6);


    // println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
    //          comp_size as f64/ org_size as f64,
    //          duration1.as_nanos() as f64 / 1000000.0,
    //          duration2.as_nanos() as f64 / 1000000.0,
    //          duration3.as_nanos() as f64 / 1000000.0,
    //          duration4.as_nanos() as f64 / 1000000.0,
    //          duration5.as_nanos() as f64 / 1000000.0,
    //          duration6.as_nanos() as f64 / 1000000.0
    // )

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

pub fn run_buff_encoding_decoding_mybitvec(test_file:&str, scl:usize,pred: f64) {
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
    comp.buff_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.buff_equal_filter(comp_eq,pred);
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


    // println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
    //          comp_size as f64/ org_size as f64,
    //          duration1.as_nanos() as f64 / 1000000.0,
    //          duration2.as_nanos() as f64 / 1000000.0,
    //          duration3.as_nanos() as f64 / 1000000.0,
    //          duration4.as_nanos() as f64 / 1000000.0,
    //          duration5.as_nanos() as f64 / 1000000.0,
    //          duration6.as_nanos() as f64 / 1000000.0
    //
    //
    // )

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


pub fn run_buff_majority_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.buff_encode_majority(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in buff-major byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.buff_decode_majority(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in buff-major byte decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.buff_range_filter_majority(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in buff-major byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.buff_equal_filter_majority(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in buff-major byte equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.buff_sum_majority(comp_sum);
    let duration5 = start5.elapsed();
    println!("Time elapsed in buff-major sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.buff_max_majority(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in buff-major max function() is: {:?}", duration6);

    // println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
    //          comp_size as f64/ org_size as f64,
    //          duration1.as_nanos() as f64 / 1000000.0,
    //          duration2.as_nanos() as f64 / 1000000.0,
    //          duration3.as_nanos() as f64 / 1000000.0,
    //          duration4.as_nanos() as f64 / 1000000.0,
    //          duration5.as_nanos() as f64 / 1000000.0,
    //          duration6.as_nanos() as f64 / 1000000.0
    // )
    //
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

impl SplitBDDoubleCompress {


    // encoder for buff-outlier with my bit vec
    pub fn buff_encode_majority<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut fixed_vec = Vec::new();

        let mut t:u32 = seg.get_data().len() as u32;
        let mut prec = 0;
        if self.scale == 0{
            prec = 0;
        }
        else{
            prec = (self.scale as f32).log10() as i32;
        }
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
            }
            if fixed>max {
                max = fixed;
            }
            fixed_vec.push(fixed);
        }
        let delta = max-min;
        let base_fixed = min;
        println!("base integer: {}, max:{}",base_fixed,max);
        let ubase_fixed = unsafe { mem::transmute::<i64, u64>(base_fixed) };
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
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(ubase_fixed as u32,32);
        bitpack_vec.write((ubase_fixed>>32) as u32,32);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);

        // let duration1 = start1.elapsed();
        // println!("Time elapsed in dividing double function() is: {:?}", duration1);

        // let start1 = Instant::now();
        let mut remain = fixed_len;
        let mut bytec = 0;

        let mut sample_u  = 0u8;

        let mut freq_vec = Vec::new();
        let mut frequency = 0i32;
        let mut major = 0u8;
        let mut rsf = fixed_len-8;
        let m_limit=(SAMPLE as f32*MAJOR_R) as i32;
        let mut smp_u64 = 0u64;
        let mut smp_u8 = 0u8;

        let mut head_sample_i:Vec<i64> = Vec::from_iter(fixed_vec[0..SAMPLE].iter().cloned());
        let head_sample = head_sample_i.iter().map(|x|{
            smp_u64 = (*x-base_fixed64) as u64;
            smp_u8 = (smp_u64>>rsf) as u8;
            if frequency == 0{
                major = smp_u8;
                frequency = 1;
            }
            else if major == smp_u8 {
                frequency += 1;
            }
            else {
                frequency -= 1;
            }
            smp_u64
        }).collect_vec();
        println!("major item: {} with frequency: {}", major, frequency);

        if frequency>m_limit{
            println!("sparse coding with major item: {} with frequency: {}", major, frequency);
            freq_vec.push(major);

            while (rsf>7){
                rsf -= 8;
                frequency = 0i32;
                major = 0u8;

                for &sample in &head_sample{
                    // for &sample in &fixed_vec[0..SAMPLE]{
                    sample_u = (sample >> rsf) as u8;
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
                if frequency>m_limit{
                    println!("major item: {} with frequency: {}", major, frequency);
                    freq_vec.push(major);
                }
                else {
                    break;
                }
            }
        }


        let mut cols_outlier:u32= freq_vec.len() as u32;
        let mut f_iter = freq_vec.into_iter();
        let mut f_item = 0u8;



        if cols_outlier>0{
            // write number of outlier cols
            f_item = f_iter.next().unwrap();
            let mut outlier = Vec::new();
            println!("sub column for outlier: {}",cols_outlier);
            bitpack_vec.write(cols_outlier, 8);
            let mut cur_u64 = 0u64;
            let mut cur_u8 = 0u8;
            let mut fixed_u64 = Vec::new();
            // write the first outlier column
            remain -= 8;
            bytec += 1;
            let mut ind = 0;
            let mut bm_outlier = BitVec::from_elem(t as usize, false);
            fixed_u64 = fixed_vec.iter().map(|x|{
                cur_u64 = (*x-base_fixed64) as u64;
                cur_u8 = (cur_u64>>remain) as u8;
                if cur_u8 != f_item{
                    outlier.push(cur_u8);
                    bm_outlier.set(ind,true);
                }
                ind += 1;
                cur_u64
            }).collect_vec();

            let mut n_outlier = bm_outlier.cardinality() as u32;
            let mut bm_vec = bit_vec_compress(&bm_outlier.to_binary());
            let mut bm_size = bm_vec.len() as u32;
            println!("bitmap size in {} byte: {}", bytec, bm_size);
            println!("number of outliers in {} byte: {}", bytec, n_outlier);
            // write the majority item
            bitpack_vec.write(f_item as u32,8);
            // write the serialized bitmap size before the serialized bitmap.
            bitpack_vec.write(bm_size,32);
            bitpack_vec.write_bytes(&mut bm_vec);
            bitpack_vec.finish_write_byte();

            // write the number of outlier before the actual outlier values
            bitpack_vec.write(n_outlier,32);
            bitpack_vec.write_bytes(&mut outlier);
            bitpack_vec.finish_write_byte();
            cols_outlier -= 1;

            for round in 0..cols_outlier{
                bm_outlier.clear();
                outlier.clear();
                f_item = f_iter.next().unwrap();
                ind = 0;
                remain -= 8;
                bytec += 1;
                for d in &fixed_u64 {
                    cur_u8 = (*d >>remain) as u8;
                    if cur_u8 != f_item{
                        outlier.push(cur_u8);
                        bm_outlier.set(ind,true);
                    }
                    ind += 1;
                }

                n_outlier = bm_outlier.cardinality() as u32;
                bm_vec = bit_vec_compress(&bm_outlier.to_binary());
                bm_size = bm_vec.len() as u32;
                println!("bitmap size in {} byte: {}", bytec, bm_size);
                println!("number of outliers in {} byte: {}", bytec, n_outlier);
                // write the majority item
                bitpack_vec.write(f_item as u32,8);
                // write the serialized bitmap size before the serialized bitmap.
                bitpack_vec.write(bm_size,32);
                bitpack_vec.write_bytes(&mut bm_vec);
                bitpack_vec.finish_write_byte();

                // write the number of outlier before the actual outlier values
                bitpack_vec.write(n_outlier,32);
                bitpack_vec.write_bytes(&mut outlier);
                bitpack_vec.finish_write_byte();

            }
            while (remain>=8){
                bytec+=1;
                remain -= 8;
                if remain>0{
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte((*d >>remain) as u8).unwrap();
                    }
                }
                else {
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte(*d as u8).unwrap();
                    }
                }


                println!("write the {}th byte of dec",bytec);
            }
            if (remain>0){
                bitpack_vec.finish_write_byte();
                for d in fixed_u64 {
                    bitpack_vec.write_bits(d as u32, remain as usize).unwrap();
                }
                println!("write remaining {} bits of dec",remain);
            }
        }
        else{
            // write number of outlier cols
            bitpack_vec.write_byte(0);
            if remain<8{
                for i in fixed_vec{
                    bitpack_vec.write_bits((i-base_fixed64) as u32, remain).unwrap();
                }
                remain = 0;
            }
            else {
                bytec+=1;
                remain -= 8;
                let mut fixed_u64 = Vec::new();
                let mut cur_u64 = 0u64;
                if remain>0{
                    // let mut k = 0;
                    fixed_u64 = fixed_vec.iter().map(|x|{
                        cur_u64 = (*x-base_fixed64) as u64;
                        bitpack_vec.write_byte((cur_u64>>remain) as u8);
                        cur_u64
                    }).collect_vec();
                }
                else {
                    fixed_u64 = fixed_vec.iter().map(|x|{
                        cur_u64 = (*x-base_fixed64) as u64;
                        bitpack_vec.write_byte((cur_u64) as u8);
                        cur_u64
                    }).collect_vec();
                }
                println!("write the {}th byte of dec",bytec);

                while (remain>=8){
                    bytec+=1;
                    remain -= 8;
                    if remain>0{
                        for d in &fixed_u64 {
                            bitpack_vec.write_byte((*d >>remain) as u8).unwrap();
                        }
                    }
                    else {
                        for d in &fixed_u64 {
                            bitpack_vec.write_byte(*d as u8).unwrap();
                        }
                    }


                    println!("write the {}th byte of dec",bytec);
                }
                if (remain>0){
                    bitpack_vec.finish_write_byte();
                    for d in fixed_u64 {
                        bitpack_vec.write_bits(d as u32, remain as usize).unwrap();
                    }
                    println!("write remaining {} bits of dec",remain);
                }
            }
        }


        // println!("total number of dec is: {}", j);
        let vec = bitpack_vec.into_vec();

        // let duration1 = start1.elapsed();
        // println!("Time elapsed in writing double function() is: {:?}", duration1);

        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // buff decode with majority sparse coding and my_bit_vec
    pub fn buff_decode_majority(&self, bytes: Vec<u8>) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;

        let mut expected_datapoints:Vec<f64> = Vec::new();
        let mut fixed_vec:Vec<u64> = Vec::new();

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk;

        // check if there are potential outiler cols
        let outlier_cols = bitpack.read(8).unwrap();
        println!("number of outlier cols:{}",outlier_cols);

        // if there are outlier cols, then read outlier cols first
        if outlier_cols>0{
            let marjor_item = bitpack.read(8).unwrap();
            let first_major = (marjor_item as u64)<< (remain-8);
            fixed_vec = vec![first_major; len as usize];
            bytec+=1;
            remain -= 8;
            let bm_len = bitpack.read(32).unwrap() as usize;
            println!("bitmap length: {}",bm_len);
            let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
            let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
            bitpack.finish_read_byte();

            let n_outlier = bitpack.read(32).unwrap() as usize;
            let outliers = bitpack.read_n_byte(n_outlier).unwrap();
            let mut outlier_iter = outliers.iter();
            let mut outlier_value = outlier_iter.next().unwrap();

            let mut bm_iterator = BVIter::new(&bitmap_outlier);
            println!("bitmap cardinaliry: {}",bitmap_outlier.cardinality());
            let mut ind = bm_iterator.next().unwrap();

            let mut i = 0;
            let mut counter = 0;
            for x in fixed_vec.iter_mut() {
                if i==ind{
                    *x = (((*outlier_value) as u64)<<remain);
                    counter+=1;
                    if counter==n_outlier{
                        break
                    }
                    ind = bm_iterator.next().unwrap();
                    outlier_value = outlier_iter.next().unwrap();
                }
                i += 1;
            }
            bitpack.finish_read_byte();
            println!("read the {}th byte of outlier",bytec);

            for round in 1..outlier_cols{
                bytec+=1;
                remain -= 8;
                let marjor_item = bitpack.read(8).unwrap();
                let major_u64 = (marjor_item as u64)<< remain;
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}",bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();
                let mut outlier_value = outlier_iter.next().unwrap();

                let mut bm_iterator = BVIter::new(&bitmap_outlier);
                println!("bitmap cardinaliry: {}",bitmap_outlier.cardinality());
                let mut ind = bm_iterator.next().unwrap();

                i = 0;
                counter = 0;
                for x in fixed_vec.iter_mut() {
                    if i==ind{
                        *x = (*x)|(((*outlier_value) as u64)<<remain);
                        counter+=1;
                        if counter==n_outlier{
                            break
                        }
                        ind = bm_iterator.next().unwrap();
                        outlier_value = outlier_iter.next().unwrap();
                    }
                    else{
                        *x = (*x)|(major_u64<<remain);
                    }
                    i += 1;
                }
                bitpack.finish_read_byte();
                println!("read the {}th byte of outlier",bytec);
            }

            while (remain>=8){
                bytec+=1;
                remain -= 8;
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                if remain == 0 {
                    // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                    // let mut iiter = int_vec.iter();
                    // let mut diter = dec_vec.iter();
                    // for cur_chunk in chunk.iter(){
                    //     expected_datapoints.push( *(iiter.next().unwrap()) as f64+ (((diter.next().unwrap())|((*cur_chunk) as u32)) as f64) / dec_scl);
                    // }

                    for (cur_fixed,cur_chunk) in fixed_vec.iter().zip(chunk.iter()){
                        expected_datapoints.push( (base_int + ((*cur_fixed)|((*cur_chunk) as u64)) as i64 ) as f64 / dec_scl);
                    }
                }
                else{
                    let mut it = chunk.into_iter();
                    fixed_vec=fixed_vec.into_iter().map(|x| x|((*(it.next().unwrap()) as u64)<<remain)).collect();
                }
                println!("read the {}th byte of dec",bytec);
            }
            // let duration = start.elapsed();
            // println!("Time elapsed in leading bytes: {:?}", duration);


            // let start5 = Instant::now();
            if (remain>0){
                bitpack.finish_read_byte();
                println!("read remaining {} bits of dec",remain);
                println!("length for fixed:{}", fixed_vec.len());
                for cur_fixed in fixed_vec.into_iter(){
                    expected_datapoints.push( (base_int + ((cur_fixed)|(bitpack.read_bits( remain as usize).unwrap() as u64)) as i64) as f64 / dec_scl);
                }
            }

        }
        else {
            if remain<8{
                for i in 0..len {
                    cur = bitpack.read_bits(remain as usize).unwrap();
                    expected_datapoints.push((base_int + cur as i64) as f64 / dec_scl);
                }
                remain=0
            }
            else {
                bytec+=1;
                remain -= 8;
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                if remain == 0 {
                    for &x in chunk {
                        expected_datapoints.push((base_int + x as i64) as f64 / dec_scl);
                    }
                }
                else{
                    // dec_vec.push((bitpack.read_byte().unwrap() as u32) << remain);
                    // let mut k = 0;
                    for x in chunk{
                        // if k<10{
                        //     println!("write {}th value with first byte {}",k,(*x))
                        // }
                        // k+=1;
                        fixed_vec.push(((*x) as u64)<<remain)
                    }
                }
                println!("read the {}th byte of dec",bytec);

                while (remain>=8){
                    bytec+=1;
                    remain -= 8;
                    chunk = bitpack.read_n_byte(len as usize).unwrap();
                    if remain == 0 {
                        // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                        // let mut iiter = int_vec.iter();
                        // let mut diter = dec_vec.iter();
                        // for cur_chunk in chunk.iter(){
                        //     expected_datapoints.push( *(iiter.next().unwrap()) as f64+ (((diter.next().unwrap())|((*cur_chunk) as u32)) as f64) / dec_scl);
                        // }

                        for (cur_fixed,cur_chunk) in fixed_vec.iter().zip(chunk.iter()){
                            expected_datapoints.push( (base_int + ((*cur_fixed)|((*cur_chunk) as u64)) as i64 ) as f64 / dec_scl);
                        }
                    }
                    else{
                        let mut it = chunk.into_iter();
                        fixed_vec=fixed_vec.into_iter().map(|x| x|((*(it.next().unwrap()) as u64)<<remain)).collect();
                    }


                    println!("read the {}th byte of dec",bytec);
                }
                // let duration = start.elapsed();
                // println!("Time elapsed in leading bytes: {:?}", duration);


                // let start5 = Instant::now();
                if (remain>0){
                    bitpack.finish_read_byte();
                    println!("read remaining {} bits of dec",remain);
                    println!("length for fixed:{}", fixed_vec.len());
                    for cur_fixed in fixed_vec.into_iter(){
                        expected_datapoints.push( (base_int + ((cur_fixed)|(bitpack.read_bits( remain as usize).unwrap() as u64)) as i64) as f64 / dec_scl);
                    }
                }
            }
        }



        // for i in 0..10{
        //     println!("{}th item:{}",i,expected_datapoints.get(i).unwrap())
        // }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    pub(crate) fn buff_range_filter_majority(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain = (dlen+ilen) as usize;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int as i64{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int as i64) as u64;
        let mut byte_count = 0;
        let mut cur_tar = 0u8;
        println!("fixed_target:{}",fixed_target);
        let mut outlier_cols = bitpack.read(8).unwrap();
        println!("number of outlier cols:{}",outlier_cols);

        // if there are outlier cols, then read outlier cols first
        if outlier_cols>0{
            // first col is a little bit different
            byte_count += 1;
            remain -= 8;
            cur_tar = (fixed_target >> remain) as u8;
            let first_major = bitpack.read(8).unwrap() as u8;
            if cur_tar==0 && first_major==0{
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                res.or(&bitmap_outlier);
                bitmap_outlier.negate();
                rb1.or(&bitmap_outlier);
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                bitpack.skip_n_byte(n_outlier).unwrap();

                bitpack.finish_read_byte();
            }
            else{
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                // let mut flip_bitmap = bitmap_outlier.flip(0..(len as u64));
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();

                let mut bm_iterator = BVIter::new(&bitmap_outlier);
                println!("bitmap cardinaliry: {} and hit outliers", bitmap_outlier.cardinality());
                let mut ind = 0;

                for &val in outlier_iter{
                    ind = bm_iterator.next().unwrap();
                    if val==cur_tar{
                        rb1.set(ind,true);
                    }
                    else if val>cur_tar{
                        res.set(ind,true);
                    }
                }
                // check if major is also qualified
                bitmap_outlier.negate();
                if cur_tar<first_major{
                    res.or(&bitmap_outlier);
                }else if cur_tar==first_major {
                    rb1.or(&bitmap_outlier);
                }

                bitpack.finish_read_byte();
            }
            // println!("read the {}th byte of outlier, get {} records need futher check",byte_count,rb1.cardinality());
            outlier_cols -= 1;

            for round in 0..outlier_cols {
                byte_count += 1;
                remain -= 8;
                cur_tar = (fixed_target >> remain) as u8;
                let cur_major = bitpack.read(8).unwrap() as u8;
                if cur_tar==0 && cur_major==0{
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}", bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                    rb1.and(&bitmap_outlier);
                    res.or(&rb1);
                    bitmap_outlier.negate();
                    rb1.and(&bitmap_outlier);
                    bitpack.finish_read_byte();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    bitpack.skip_n_byte(n_outlier).unwrap();

                    bitpack.finish_read_byte();
                }
                else {
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}", bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                    // let mut flip_bitmap = bitmap_outlier.flip(0..(len as u64));
                    bitpack.finish_read_byte();
                    let mut cur_rb = BitVec::from_elem(len as usize, false);

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                    let mut outlier_iter = outliers.iter();

                    let mut bm_iterator = BVIter::new(&bitmap_outlier);
                    println!("bitmap cardinaliry: {}", bitmap_outlier.cardinality());
                    let mut ind = 0;

                    for &val in outlier_iter{
                        ind = bm_iterator.next().unwrap();
                        if val==cur_tar{
                            cur_rb.set(ind,true);
                        }
                        else if val>cur_tar{
                            res.set(ind,true);
                        }
                    }
                    // check if major is also qualified
                    bitmap_outlier.negate();
                    if cur_tar<cur_major{
                        bitmap_outlier.and(&rb1);
                        res.or(&bitmap_outlier);
                    }else if cur_tar==cur_major {
                        cur_rb.or(&bitmap_outlier);
                    }

                    rb1.and(&cur_rb);


                    bitpack.finish_read_byte();
                }
                // println!("read the {}th byte of outlier, get {} records need futher check",byte_count,rb1.cardinality());
            }

            while (remain>0){
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;
                    let mut cur_rb = BitVec::from_elem(len as usize, false);
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;
                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        // shift right to get corresponding byte
                        cur_tar = (fixed_target >> remain as u64) as u8;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec>cur_tar{
                                res.set(dec_cur, true);
                            }
                            else if dec == cur_tar{
                                cur_rb.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec>cur_tar{
                                res.set(dec_cur,true);
                            }
                            else if dec == cur_tar{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    else{
                        bitpack.skip_n_byte((len) as usize);
                        break;
                    }
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    cur_tar =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                    bitpack.finish_read_byte();
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;

                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip(((dec_cur) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec>cur_tar{
                                res.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip(((delta-1) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec>cur_tar{
                                res.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        println!("read the remain {} bits of dec",remain);
                        remain = 0;
                    }
                    else{
                        break;
                    }
                }
            }
        }
        else{
            if remain<8{
                for i in 0..len as usize {
                    cur = bitpack.read_bits(remain as usize).unwrap();
                    if cur as u64>fixed_target{
                        res.set(i, true);
                    }
                }
                remain = 0;
            }else {
                remain-=8;
                byte_count+=1;
                let chunk = bitpack.read_n_byte(len as usize).unwrap();
                cur_tar = (fixed_target >> remain) as u8;
                let mut i =0;
                for &c in chunk {
                    if c >cur_tar{
                        res.set(i, true);
                    }
                    else if c == cur_tar {
                        rb1.set(i, true);
                    };
                    i+=1;
                }
            }


            // println!("Number of qualified items in bitmap:{}", rb1.cardinality());

            while (remain>0){
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;
                    let mut cur_rb = BitVec::from_elem(len as usize, false);
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;
                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        // shift right to get corresponding byte
                        cur_tar = (fixed_target >> remain as u64) as u8;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec>cur_tar{
                                res.set(dec_cur, true);
                            }
                            else if dec == cur_tar{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec>cur_tar{
                                res.set(dec_cur, true);
                            }
                            else if dec == cur_tar{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    else{
                        bitpack.skip_n_byte((len) as usize);
                        break;
                    }
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    cur_tar =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                    bitpack.finish_read_byte();
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;

                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip(((dec_cur) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec>cur_tar{
                                res.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip(((delta-1) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec>cur_tar{
                                res.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        println!("read the remain {} bits of dec",remain);
                        remain = 0;
                    }
                    else{
                        break;
                    }
                }
            }
        }

        println!("Number of qualified items:{}", res.cardinality());
    }

    pub(crate) fn buff_equal_filter_majority(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =(dlen+ilen) as usize;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int as i64{
            println!("Number of qualified items for equal:{}", 0);
            return;
        }
        let fixed_target = (fixed_part-base_int as i64) as u64;
        let mut dec_byte = fixed_target as u8;
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        // let start = Instant::now();

        let mut outlier_cols = bitpack.read(8).unwrap();
        println!("number of outlier cols:{}",outlier_cols);

        // if there are outlier cols, then read outlier cols first
        if outlier_cols>0{
            // first col is a little bit different
            byte_count += 1;
            remain -= 8;
            dec_byte = (fixed_target >> remain) as u8;
            let first_major = bitpack.read(8).unwrap() as u8;
            if dec_byte==first_major{
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                bitmap_outlier.negate();
                rb1.or(&bitmap_outlier);
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                bitpack.skip_n_byte(n_outlier).unwrap();

                bitpack.finish_read_byte();
            }
            else {
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();

                let mut bm_iterator = BVIter::new(&bitmap_outlier);
                println!("bitmap cardinaliry: {} and hit outliers", bitmap_outlier.cardinality());
                let mut ind = 0;

                for &val in outlier_iter{
                    ind = bm_iterator.next().unwrap();
                    if val==dec_byte{
                        rb1.set(ind,true);
                    }
                }


                bitpack.finish_read_byte();
            }
            // println!("read the {}th byte of outlier, get {} records need futher check",byte_count,rb1.cardinality());
            outlier_cols -= 1;

            for round in 0..outlier_cols {
                byte_count += 1;
                remain -= 8;
                dec_byte = (fixed_target >> remain) as u8;
                let major_byte = bitpack.read(8).unwrap() as u8;
                if dec_byte==major_byte{
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}", bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                    bitmap_outlier.negate();
                    rb1.and(&bitmap_outlier);
                    bitpack.finish_read_byte();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    bitpack.skip_n_byte(n_outlier).unwrap();

                    bitpack.finish_read_byte();
                }
                else {
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}", bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                    bitpack.finish_read_byte();
                    let mut cur_rb = BitVec::from_elem(len, false);

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                    let mut outlier_iter = outliers.iter();

                    let mut bm_iterator = BVIter::new(&bitmap_outlier);
                    println!("bitmap cardinaliry: {}", bitmap_outlier.cardinality());
                    let mut ind = 0;

                    for &val in outlier_iter{
                        ind = bm_iterator.next().unwrap();
                        if val==dec_byte{
                            cur_rb.set(ind, true);
                        }
                    }
                    rb1.and(&cur_rb);

                    bitpack.finish_read_byte();
                }
                // println!("read the {}th byte of outlier, get {} records need futher check",byte_count,rb1.cardinality());
            }

            println!("Number of qualified items for equal:{}", rb1.cardinality());

            while (remain>0){
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;
                    let mut cur_rb = BitVec::from_elem(len as usize, false);
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;
                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        // shift right to get corresponding byte
                        dec_byte = (fixed_target >> remain as u64) as u8;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec == dec_byte{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec == dec_byte{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    else{
                        bitpack.skip_n_byte((len) as usize);
                        break;
                    }
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                    bitpack.finish_read_byte();
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;

                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip(((dec_cur) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec==dec_byte{
                                res.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip(((delta-1) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec==dec_byte{
                                res.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        println!("read the remain {} bits of dec",remain);
                        remain = 0;
                    }
                    else{
                        break;
                    }
                }
            }
        }
        else{
            if remain<8{
                for i in 0..len as usize{
                    cur = bitpack.read_bits(remain as usize).unwrap();
                    if cur as u64==fixed_target{
                        res.set(i, true);
                    }
                }
                remain = 0;
            }else {
                remain-=8;
                byte_count+=1;
                let chunk = bitpack.read_n_byte(len as usize).unwrap();
                dec_byte = (fixed_target >> remain) as u8;
                let mut i =0;
                for &c in chunk {
                    if c == dec_byte {
                        rb1.set(i, true);
                    };
                    i+=1;
                }
            }
            // rb1.run_optimize();
            // let duration = start.elapsed();
            // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
            println!("Number of qualified items for equal:{}", rb1.cardinality());

            while (remain>0){
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;
                    let mut cur_rb = BitVec::from_elem(len as usize, false);
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;
                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        // shift right to get corresponding byte
                        dec_byte = (fixed_target >> remain as u64) as u8;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec == dec_byte{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec == dec_byte{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    else{
                        bitpack.skip_n_byte((len) as usize);
                        break;
                    }
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                    bitpack.finish_read_byte();
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;

                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip(((dec_cur) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec==dec_byte{
                                res.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip(((delta-1) * remain) as usize);
                            }
                            dec = bitpack.read_bits(remain as usize).unwrap();
                            if dec==dec_byte{
                                res.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        println!("read the remain {} bits of dec",remain);
                        remain = 0;
                    }
                    else{
                        break;
                    }
                }
            }
        }
        println!("Number of qualified int items:{}", res.cardinality());
    }

    pub fn buff_sum_majority(&self, bytes: Vec<u8>) -> f64{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);

        let mut remain = ilen+dlen;
        let mut processed = 0;
        let mut sum_base:f64 = len as f64 * base_int as f64;
        let mut sum_fixed:u64 = 0;
        let mut sum = 0.0f64;

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        sum_base = sum_base/dec_scl;
        // println!("Scale for decimal:{}", dec_scl);

        let mut bytec = 0;
        let mut chunk;


        let outlier_cols = bitpack.read(8).unwrap();
        println!("number of outlier cols:{}",outlier_cols);

        // if there are outlier cols, then read outlier cols first
        if outlier_cols>0 {
            for round in 0..outlier_cols {
                let major_byte = bitpack.read(8).unwrap() as u8;
                bytec += 1;
                remain -= 8;
                processed += 8;
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();

                println!("outlier number: {}", outliers.len());
                dec_scl = 2.0f64.powi(processed  - ilen as i32);

                for otlier in outliers.iter(){
                    sum_fixed += (*otlier) as u64;
                }
                sum_fixed += major_byte as u64 * (len as u64 - n_outlier as u64);
                sum = sum+(sum_fixed as f64)/dec_scl;
                // println!("sum the {}th byte of fixed number, which is {}",bytec,sum_fixed);
                sum_fixed=0;
                println!("now sum :{}", sum);
                bitpack.finish_read_byte();
            }
            while (remain>=8){
                bytec+=1;
                remain -= 8;
                processed += 8;
                dec_scl = 2.0f64.powi(processed  - ilen as i32);
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                for dec_comp in chunk.iter(){
                    sum_fixed += (*dec_comp) as u64;
                }
                sum = sum+(sum_fixed as f64)/dec_scl;
                // println!("sum the {}th byte of fixed number, which is {}",bytec,sum_fixed);
                sum_fixed=0;
                println!("now sum :{}", sum);
                if remain == 0 {
                    return sum
                }
            }
            if (remain>0){
                // let mut j =0;
                if processed>=8{
                    bitpack.finish_read_byte();
                }
                processed += remain as i32;
                dec_scl = 2.0f64.powi(processed  - ilen as i32);
                for i in 0..len {
                    sum_fixed += (bitpack.read_bits( remain as usize).unwrap() as u64);
                }
                // println!("sum remaining {} bits of fixed number, which is {}",remain,sum_fixed);
                sum = sum+(sum_fixed as f64)/dec_scl;
            }
        }
        else {
            if (remain>=8){
                bytec+=1;
                remain -= 8;
                processed += 8;
                dec_scl = 2.0f64.powi(processed  - ilen as i32);
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                for dec_comp in chunk.iter(){
                    sum_fixed += (*dec_comp) as u64;
                }
                sum = sum+(sum_fixed as f64)/dec_scl;
                // println!("sum the {}th byte of fixed number, which is {}",bytec,sum_fixed);
                sum_fixed=0;
                println!("now sum :{}", sum);
                if remain == 0 {
                    return sum
                }
            }
            while (remain>=8){
                bytec+=1;
                remain -= 8;
                processed += 8;
                dec_scl = 2.0f64.powi(processed  - ilen as i32);
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                for dec_comp in chunk.iter(){
                    sum_fixed += (*dec_comp) as u64;
                }
                sum = sum+(sum_fixed as f64)/dec_scl;
                // println!("sum the {}th byte of fixed number, which is {}",bytec,sum_fixed);
                sum_fixed=0;
                println!("now sum :{}", sum);
                if remain == 0 {
                    return sum
                }
            }
            if (remain>0){
                // let mut j =0;
                if processed>=8{
                    bitpack.finish_read_byte();
                }
                processed += remain as i32;
                dec_scl = 2.0f64.powi(processed  - ilen as i32);
                for i in 0..len {
                    sum_fixed += (bitpack.read_bits( remain as usize).unwrap() as u64);
                }
                // println!("sum remaining {} bits of fixed number, which is {}",remain,sum_fixed);
                sum = sum+(sum_fixed as f64)/dec_scl;
            }
        }


        sum+= sum_base as f64;
        println!("sum is: {:?}",sum);
        sum
    }

    pub(crate) fn buff_max_majority(&self, bytes: Vec<u8>) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        // println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        // println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        // println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =(dlen+ilen) as usize;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let mut max = u64::min_value();

        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        let mut byte_max = u8::min_value();
        // let start = Instant::now();


        let mut outlier_cols = bitpack.read(8).unwrap();
        println!("number of outlier cols:{}", outlier_cols);

        // if there are outlier cols, then read outlier cols first
        if outlier_cols > 0 {
            // handle the first outlier column
            let major_byte = bitpack.read(8).unwrap() as u8;
            byte_max = major_byte;
            byte_count += 1;
            remain -= 8;
            let bm_len = bitpack.read(32).unwrap() as usize;
            println!("bitmap length: {}", bm_len);
            let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
            let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
            rb1 = bitmap_outlier.clone();
            rb1.negate();
            bitpack.finish_read_byte();

            let n_outlier = bitpack.read(32).unwrap() as usize;
            let outliers = bitpack.read_n_byte(n_outlier).unwrap();
            let mut outlier_iter = outliers.iter();

            let mut bm_iterator = BVIter::new(&bitmap_outlier);
            println!("bitmap cardinaliry: {} and hit outliers", bitmap_outlier.cardinality());
            let mut ind = 0;

            // iterate all outlier cols
            for &val in outlier_iter{
                ind = bm_iterator.next().unwrap();
                if val>byte_max{
                    byte_max = val;
                    rb1.clear();
                    rb1.set(ind,true);
                }
                else if val==byte_max{
                    rb1.set(ind,true);
                }
            }

            bitpack.finish_read_byte();
            // println!("read the {}th byte of outlier, get {} records need futher check",byte_count,rb1.cardinality());
            outlier_cols -= 1;
            max = max | ((byte_max as u64)<< remain);

            for round in 0..outlier_cols {
                let major_byte = bitpack.read(8).unwrap() as u8;
                byte_max = major_byte;
                byte_count += 1;
                remain -= 8;

                let mut cur_rb = BitVec::from_elem(len as usize, false);
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = BitVec::from_binary(&bit_vec_decompress(bm_mat));
                let mut bp_non_outlier = bitmap_outlier.clone();
                bp_non_outlier.negate();
                bp_non_outlier.and(&rb1);
                cur_rb = bp_non_outlier;
                let mut bm_iterator = BVIter::new(&bitmap_outlier);
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();

                rb1.and(&bitmap_outlier);
                let max_bitmap = rb1;
                if max_bitmap.cardinality()==0{
                    rb1 = cur_rb;
                    bitpack.finish_read_byte();
                    continue;
                }else{
                    let mut ind = 0;
                    let mut max_iterator = BVIter::new(&max_bitmap);
                    let mut max_it = max_iterator.next();
                    let mut max_cur = max_it.unwrap();
                    for &val in outlier_iter{
                        ind = bm_iterator.next().unwrap();
                        if ind==max_cur{
                            if val>byte_max{
                                byte_max = val;
                                cur_rb.clear();
                                cur_rb.set(ind,true);
                            }
                            else if val==byte_max{
                                cur_rb.set(ind,true);
                            }

                            max_it = max_iterator.next();
                            if max_it==None{
                                break;
                            }
                            else {
                                max_cur = max_it.unwrap();
                            }
                        }
                    }
                }

                rb1 = cur_rb;

                bitpack.finish_read_byte();
                max = max | ((byte_max as u64)<< remain);

                // println!("read the {}th byte of outlier, get {} records need futher check",byte_count,rb1.cardinality());
            }

            while (remain>0){
                byte_max = u8::min_value();
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;

                    let mut dec_cur = 0;
                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    let mut cur_rb = BitVec::from_elem(len as usize, false);

                    {
                        let mut iterator = BVIter::new(&rb1);
                        // let start = Instant::now();
                        // check the decimal part
                        let mut it = iterator.next();
                        // shift right to get corresponding byte
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec > byte_max{
                                byte_max = dec;
                                cur_rb.set(dec_cur,true);
                            }else if dec == byte_max {
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec > byte_max{
                                byte_max = dec;
                                cur_rb.clear();
                                cur_rb.set(dec_cur,true);
                            }else if dec == byte_max {
                                cur_rb.set(dec_cur,true);
                            }

                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    max = max | ((byte_max as u64)<< remain);
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    byte_max = u8::min_value();
                    bitpack.finish_read_byte();
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(((dec_cur) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.set(dec_cur, true);
                        }
                        else if dec == byte_max {
                            res.set(dec_cur, true);
                        }

                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip(((delta-1) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.clear();
                            res.set(dec_cur, true);
                        }
                        else if dec==byte_max{
                            res.set(dec_cur, true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    // println!("read the remain {} bits of dec",remain);
                    remain = 0;
                    max = max | (byte_max as u64);
                }
            }
        }
        else {
            if remain<8{
                for i in 0..len {
                    cur = bitpack.read_bits(remain as usize).unwrap() as u64;
                    if cur >max{
                        max = cur ;
                        res.clear();
                        res.set(i, true);
                    }
                    else if cur == max {
                        res.set(i, true);
                    }
                }
                remain = 0;
            }else {
                remain-=8;
                byte_count+=1;
                let chunk = bitpack.read_n_byte(len as usize).unwrap();
                let mut i =0;
                for &c in chunk {
                    if c > byte_max {
                        byte_max=c;
                        rb1.clear();
                        rb1.set(i, true);
                    }
                    else if c == byte_max {
                        rb1.set(i, true);
                    }
                    i+=1;
                }
                max = (byte_max as u64) << (remain as u64);
            }
            // rb1.run_optimize();
            // let duration = start.elapsed();
            // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
            // println!("Number of qualified items for max:{}", rb1.cardinality());

            while (remain>0){
                byte_max = u8::min_value();
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;

                    let mut dec_cur = 0;
                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    let mut cur_rb = BitVec::from_elem(len as usize, false);

                    {
                        let mut iterator = BVIter::new(&rb1);
                        // let start = Instant::now();
                        // check the decimal part
                        let mut it = iterator.next();
                        // shift right to get corresponding byte
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec > byte_max{
                                byte_max = dec;
                                cur_rb.set(dec_cur,true);
                            }else if dec == byte_max {
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec > byte_max{
                                byte_max = dec;
                                cur_rb.clear();
                                cur_rb.set(dec_cur,true);
                            }else if dec == byte_max {
                                cur_rb.set(dec_cur,true);
                            }

                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    max = max | ((byte_max as u64)<< remain);
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    byte_max = u8::min_value();
                    bitpack.finish_read_byte();
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(((dec_cur) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.set(dec_cur, true);
                        }
                        else if dec == byte_max {
                            res.set(dec_cur, true);
                        }

                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip(((delta-1) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.clear();
                            res.set(dec_cur, true);
                        }
                        else if dec==byte_max{
                            res.set(dec_cur, true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    // println!("read the remain {} bits of dec",remain);
                    remain = 0;
                    max = max | (byte_max as u64);
                }
            }
        }

        let max_f = (max as i64+base_int) as f64 / 2.0f64.powi(dlen as i32);
        println!("Number of qualified max items:{}", res.cardinality());
        println!("Max value:{}", max_f);
    }

    pub(crate) fn buff_max_majority_range(&self, bytes: Vec<u8>,st:u32, ed:u32, wd:u32) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        let s = st as usize;
        let e = ed as usize;
        let window = wd as usize;

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        // println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        // println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        // println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =(dlen+ilen) as usize;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let mut max = u64::min_value();

        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        let mut byte_max = u8::min_value();
        let mut max_vec = Vec::new();
        let mut cur_s = s;
        // let start = Instant::now();


        let mut outlier_cols = bitpack.read(8).unwrap();
        println!("number of outlier cols:{}", outlier_cols);

        // if there are outlier cols, then read outlier cols first
        if outlier_cols > 0 {
            // todo: handle outlier for max range groupby.
            // handle the first outlier column
            error!("handle groupby for outlier is not immplemented")
        }
        else {
            if remain<8{
                bitpack.skip(((s) * remain) as usize);
                for i in s..e {
                    if i==cur_s+window{
                        cur_s = i;
                        max_vec.push(max);
                        max = u64::min_value();
                    }
                    cur = bitpack.read_bits(remain as usize).unwrap() as u64;
                    if cur >max{
                        max = cur ;
                        res.remove_range(cur_s , i );
                        res.set(i, true);
                    }
                    else if cur == max {
                        res.set(i, true);
                    }
                }
                if e==cur_s+window{
                    cur_s = e;
                    max_vec.push(max);
                    max = u64::min_value();
                }
                remain = 0;
            }else {
                remain-=8;
                byte_count+=1;
                bitpack.skip_n_byte(s as usize);
                let chunk = bitpack.read_n_byte((e-s) as usize).unwrap();
                let mut i =s;
                for &c in chunk {
                    if i==cur_s+window{
                        cur_s = i;
                        max_vec.push((byte_max as u64) << (remain as u64));
                        byte_max = u8::min_value();
                    }
                    if c > byte_max {
                        byte_max=c;
                        rb1.remove_range(cur_s , i );
                        rb1.set(i, true);
                    }
                    else if c == byte_max {
                        rb1.set(i, true);
                    }
                    i+=1;
                }
                if i==cur_s+window{
                    cur_s = i;
                    max_vec.push((byte_max as u64) << (remain as u64));
                    byte_max = u8::min_value();
                }
                bitpack.skip_n_byte(len-e );

            }
            // rb1.run_optimize();
            // let duration = start.elapsed();
            // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
            // println!("Number of qualified items for max:{}", rb1.cardinality());

            while (remain>0){
                byte_max = u8::min_value();
                let mut cur_s= s;
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;

                    let mut dec_cur = 0;
                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    let mut cur_rb = BitVec::from_elem(len as usize, false);

                    {
                        let mut iterator = BVIter::new(&rb1);
                        // let start = Instant::now();
                        let mut it = iterator.next();
                        // shift right to get corresponding byte
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec > byte_max{
                                byte_max = dec;
                                cur_rb.set(dec_cur,true);
                            }else if dec == byte_max {
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur >= cur_s+window{
                                let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                                *max_elem = *max_elem | ((byte_max as u64)<< remain);
                                byte_max = u8::min_value();
                                cur_s= cur_s+window;
                            }
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = bitpack.read_byte().unwrap();
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec > byte_max{
                                byte_max = dec;
                                cur_rb.remove_range(cur_s,dec_cur);
                                cur_rb.set(dec_cur,true);
                            }else if dec == byte_max {
                                cur_rb.set(dec_cur,true);
                            }

                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        assert_eq!((cur_s-s)/window+1, (e-s)/window);
                        /// set max for last window
                        let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                        *max_elem = *max_elem | ((byte_max as u64)<< remain);
                        byte_max = u8::min_value();
                        cur_s= cur_s+window;

                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    rb1 = cur_rb;
                    // println!("read the {}th byte of dec",byte_count);
                    // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
                }
                // else we have to read by bits
                else {
                    byte_max = u8::min_value();
                    bitpack.finish_read_byte();
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(((dec_cur) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.set(dec_cur, true);
                        }
                        else if dec == byte_max {
                            res.set(dec_cur, true);
                        }

                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur >= cur_s+window{
                            let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                            *max_elem = *max_elem | (byte_max as u64);
                            byte_max = u8::min_value();
                            cur_s= cur_s+window;
                        }
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip(((delta-1) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.remove_range(cur_s,dec_cur);
                            res.set(dec_cur, true);
                        }
                        else if dec==byte_max{
                            res.set(dec_cur, true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    assert_eq!((cur_s-s)/window+1, (e-s)/window);
                    /// set max for last window
                    let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                    *max_elem = *max_elem | (byte_max as u64);
                    byte_max = u8::min_value();
                    cur_s= cur_s+window;
                    // println!("read the remain {} bits of dec",remain);
                    remain = 0;

                }
            }
        }

        let max_vec_f64 : Vec<f64> = max_vec.iter().map(|&x| (x as i64 +base_int) as f64 / 2.0f64.powi(dlen as i32)).collect();
        println!("Number of qualified max_groupby items:{}", res.cardinality());
        println!("Max value:{:?}", max_vec_f64);
    }

    pub fn buff_simd256_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut fixed_vec = Vec::new();

        let mut t:u32 = seg.get_data().len() as u32;
        let mut prec = 0;
        if self.scale == 0{
            prec = 0;
        }
        else{
            prec = (self.scale as f32).log10() as i32;
        }
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
            }
            if fixed>max {
                max = fixed;
            }
            fixed_vec.push(fixed);
        }
        let delta = max-min;
        let base_fixed = min;
        println!("base integer: {}, max:{}",base_fixed,max);
        let ubase_fixed = unsafe { mem::transmute::<i64, u64>(base_fixed) };
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
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(ubase_fixed as u32,32);
        bitpack_vec.write((ubase_fixed>>32) as u32,32);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);

        // let duration1 = start1.elapsed();
        // println!("Time elapsed in dividing double function() is: {:?}", duration1);

        // let start1 = Instant::now();
        let mut remain = fixed_len;
        let mut bytec = 0;

        if remain<8{
            for i in fixed_vec{
                bitpack_vec.write_bits((i-base_fixed64) as u32, remain).unwrap();
            }
            remain = 0;
        }
        else {
            bytec+=1;
            remain -= 8;
            let mut fixed_u64 = Vec::new();
            let mut cur_u64 = 0u64;
            if remain>0{
                // let mut k = 0;
                fixed_u64 = fixed_vec.iter().map(|x|{
                    cur_u64 = (*x-base_fixed64) as u64;
                    bitpack_vec.write_byte(flip((cur_u64>>remain) as u8));
                    cur_u64
                }).collect_vec();
            }
            else {
                fixed_u64 = fixed_vec.iter().map(|x|{
                    cur_u64 = (*x-base_fixed64) as u64;
                    bitpack_vec.write_byte(flip(cur_u64 as u8));
                    cur_u64
                }).collect_vec();
            }
            println!("write the {}th byte of dec",bytec);

            while (remain>=8){
                bytec+=1;
                remain -= 8;
                if remain>0{
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte(flip((*d >>remain) as u8)).unwrap();
                    }
                }
                else {
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte(flip(*d as u8)).unwrap();
                    }
                }


                println!("write the {}th byte of dec",bytec);
            }
            if (remain>0){
                bitpack_vec.finish_write_byte();
                for d in fixed_u64 {
                    bitpack_vec.write_bits(d as u32, remain as usize).unwrap();
                }
                println!("write remaining {} bits of dec",remain);
            }
        }


        // println!("total number of dec is: {}", j);
        let vec = bitpack_vec.into_vec();

        // let duration1 = start1.elapsed();
        // println!("Time elapsed in writing double function() is: {:?}", duration1);

        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    pub fn buff_simd256_decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;

        let mut expected_datapoints:Vec<f64> = Vec::new();
        let mut fixed_vec:Vec<u64> = Vec::new();

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk;
        let mut f_cur = 0f64;
        let mut cur = 0;

        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                expected_datapoints.push((base_int + cur as i64 ) as f64 / dec_scl);
            }
            remain=0
        }
        else {
            bytec+=1;
            remain -= 8;
            chunk = bitpack.read_n_byte(len as usize).unwrap();

            if remain == 0 {
                for &x in chunk {
                    expected_datapoints.push((base_int + flip(x) as i64) as f64 / dec_scl);
                }
            }
            else{
                // dec_vec.push((bitpack.read_byte().unwrap() as u32) << remain);
                // let mut k = 0;
                for x in chunk{
                    // if k<10{
                    //     println!("write {}th value with first byte {}",k,(*x))
                    // }
                    // k+=1;
                    fixed_vec.push((flip(*x) as u64)<<remain)
                }
            }
            println!("read the {}th byte of dec",bytec);

            while (remain>=8){
                bytec+=1;
                remain -= 8;
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                if remain == 0 {
                    // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                    // let mut iiter = int_vec.iter();
                    // let mut diter = dec_vec.iter();
                    // for cur_chunk in chunk.iter(){
                    //     expected_datapoints.push( *(iiter.next().unwrap()) as f64+ (((diter.next().unwrap())|((*cur_chunk) as u32)) as f64) / dec_scl);
                    // }

                    for (cur_fixed,cur_chunk) in fixed_vec.iter().zip(chunk.iter()){
                        expected_datapoints.push( (base_int + ((*cur_fixed)|(flip(*cur_chunk) as u64)) as i64 ) as f64 / dec_scl);
                    }
                }
                else{
                    let mut it = chunk.into_iter();
                    fixed_vec=fixed_vec.into_iter().map(|x| x|((flip(*(it.next().unwrap())) as u64)<<remain)).collect();
                }


                println!("read the {}th byte of dec",bytec);
            }
            // let duration = start.elapsed();
            // println!("Time elapsed in leading bytes: {:?}", duration);


            // let start5 = Instant::now();
            if (remain>0){
                bitpack.finish_read_byte();
                println!("read remaining {} bits of dec",remain);
                println!("length for fixed:{}", fixed_vec.len());
                for cur_fixed in fixed_vec.into_iter(){
                    f_cur = (base_int + ((cur_fixed)|(bitpack.read_bits( remain as usize).unwrap() as u64)) as i64) as f64 / dec_scl;
                    expected_datapoints.push( f_cur);
                }
            }
        }
        // for i in 0..10{
        //     println!("{}th item:{}",i,expected_datapoints.get(i).unwrap())
        // }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }


    pub(crate) fn buff_simd_range_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain = dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        println!("fixed target:{}", fixed_target);
        let mut byte_count = 0;
        let mut cur_tar = 0u8;
        let mut check_ratio =0.0f64;

        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                if cur as u64>fixed_target{
                    res.set(i,true);
                }
            }
            remain = 0;
        }else {
            remain-=8;
            byte_count+=1;
            let chunk = bitpack.read_n_byte(len as usize).unwrap();
            cur_tar = (fixed_target >> remain) as u8;
            unsafe { rb1 = range_simd_mybitvec(chunk, &mut res, cur_tar); }
            check_ratio = rb1.cardinality() as f64/len as f64;
            println!("to check ratio: {}", check_ratio);
        }


        // println!("Number of qualified items in bitmap:{}", rb1.cardinality());

        while (remain>0){
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;
                if check_ratio>SIMD_THRESHOLD{
                    let mut temp_res = BitVec::from_elem(len, false);
                    let chunk = bitpack.read_n_byte(len as usize).unwrap();
                    cur_tar = (fixed_target >> remain) as u8;
                    let mut eq_bm;
                    unsafe { eq_bm = range_simd_mybitvec(chunk, &mut temp_res, cur_tar); }
                    temp_res.and(&rb1);
                    res.or(&temp_res);

                    rb1.and(&eq_bm);

                    check_ratio = rb1.cardinality() as f64/len as f64;
                    println!("to check ratio: {}",check_ratio );
                }
                else{
                    let mut cur_rb = BitVec::from_elem(len, false);
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;
                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        // shift right to get corresponding byte
                        cur_tar = (fixed_target >> remain as u64) as u8;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = flip(bitpack.read_byte().unwrap());
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec>cur_tar{
                                res.set(dec_cur,true);
                            }
                            else if dec == cur_tar{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = flip(bitpack.read_byte().unwrap());
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec>cur_tar{
                                res.set(dec_cur,true);
                            }
                            else if dec == cur_tar{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    else{
                        bitpack.skip_n_byte((len) as usize);
                        break;
                    }
                    rb1 = cur_rb;
                }

                // println!("read the {}th byte of dec",byte_count);
                // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
            }
            // else we have to read by bits
            else {
                cur_tar =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                bitpack.finish_read_byte();
                if rb1.cardinality()!=0{
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip((dec_cur) * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>cur_tar{
                            res.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip((delta-1) * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>cur_tar{
                            res.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    println!("read the remain {} bits of dec",remain);
                    remain = 0;
                }
                else{
                    break;
                }
            }
        }

        println!("Number of qualified items:{}", res.cardinality());
    }

    pub(crate) fn buff_simd_equal_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        // println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        // println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        // println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int {
            println!("Number of qualified items for equal:{}", 0);
            return;
        }
        let fixed_target = (fixed_part-base_int ) as u64;
        let mut dec_byte = fixed_target as u8;
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        // let start = Instant::now();
        let mut check_ratio =0.0f64;

        if remain<8{
            for i in 0..len{
                cur = bitpack.read_bits(remain as usize).unwrap();
                if cur as u64==fixed_target{
                    res.set(i,true);
                }
            }
            remain = 0;
        }else {
            remain-=8;
            byte_count+=1;
            let start = Instant::now();
            let chunk = bitpack.read_n_byte(len as usize).unwrap();
            dec_byte = (fixed_target >> remain) as u8;
            unsafe { rb1 = equal_simd_mybitvec(chunk, dec_byte); }
            check_ratio =rb1.cardinality() as f64/len as f64;
            println!("to check ratio: {}", check_ratio);
            let duration = start.elapsed();
            println!("Time elapsed in first byte is: {:?}", duration);
            // let mut i =0;
            // for &c in chunk {
            //     if c == dec_byte {
            //         rb1.insert(i);
            //     };
            //     i+=1;
            // }

        }
        // rb1.run_optimize();
        // let duration = start.elapsed();
        // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);

        while (remain>0){
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;
                if check_ratio>SIMD_THRESHOLD{
                    let chunk = bitpack.read_n_byte(len as usize).unwrap();
                    // let start = Instant::now();
                    dec_byte = (fixed_target >> remain) as u8;
                    unsafe { rb1.and(&equal_simd_mybitvec(chunk, dec_byte)); }

                    check_ratio =rb1.cardinality() as f64/len as f64;
                    println!("to check ratio: {}", check_ratio);
                }else {
                    let mut cur_rb = BitVec::from_elem(len, false);
                    if rb1.cardinality()!=0{
                        // let start = Instant::now();
                        let mut iterator = BVIter::new(&rb1);
                        // check the decimal part
                        let mut it = iterator.next();
                        let mut dec_cur = 0;
                        let mut dec_pre = 0;
                        let mut dec = 0;
                        let mut delta = 0;
                        // shift right to get corresponding byte
                        dec_byte = (fixed_target >> remain as u64) as u8;
                        if it!=None{
                            dec_cur = it.unwrap();
                            if dec_cur!=0{
                                bitpack.skip_n_byte((dec_cur) as usize);
                            }
                            dec = flip(bitpack.read_byte().unwrap());
                            // println!("{} first match: {}", dec_cur, dec);
                            if dec == dec_byte{
                                cur_rb.set(dec_cur, true);
                            }
                            it = iterator.next();
                            dec_pre = dec_cur;
                        }
                        while it!=None{
                            dec_cur = it.unwrap();
                            delta = dec_cur-dec_pre;
                            if delta != 1 {
                                bitpack.skip_n_byte((delta-1) as usize);
                            }
                            dec = flip(bitpack.read_byte().unwrap());
                            // if dec_cur<10{
                            //     println!("{} first match: {}", dec_cur, dec);
                            // }
                            if dec == dec_byte{
                                cur_rb.set(dec_cur,true);
                            }
                            it = iterator.next();
                            dec_pre=dec_cur;
                        }
                        if len - dec_pre>1 {
                            bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                        }
                    }
                    else{
                        bitpack.skip_n_byte(len);
                        break;
                    }
                    rb1 = cur_rb;
                }

                // println!("read the {}th byte of dec",byte_count);
                // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
            }
            // else we have to read by bits
            else {
                dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                bitpack.finish_read_byte();
                if rb1.cardinality()!=0{
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(dec_cur * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec==dec_byte{
                            res.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip((delta-1) * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec==dec_byte{
                            res.set(dec_cur, true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    // println!("read the remain {} bits of dec",remain);
                    remain = 0;
                }
                else{
                    break;
                }
            }
        }
        println!("Number of qualified int items:{}", res.cardinality());
    }

    pub(crate) fn buff_simd_range_filter_with_slice(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        // println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        // println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        // println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut bveq = Vec::new();
        let mut bvgt = Vec::new();
        let mut res = BitVec::from_elem(len as usize, false);
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        // let start = Instant::now();
        let num_slice = floor(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        bound.set_length(ilen as u64, dlen as u64);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        let mut dec_byte = fixed_target as u8;

        for cnt in 0..num_slice{
            let chunk = bitpack.read_n_byte_unmut((cnt*len) as usize, len as usize).unwrap();
            slice_ptr.push(chunk.as_ptr());
            data.push(chunk);
            let mut pred_u8= 0;
            if (remain-8*cnt>=8){
                pred_u8 =flip((fixed_target >> (remain - 8 * (1 + cnt))) as u8);
            }
            else {
                let pad = 8*(1+cnt)-remain;
                pred_u8= flip((fixed_target << pad) as u8)
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }

        let mut res_bm = RoaringBitmap::new();
        let mut i = 0;
        unsafe {
            while i <= len - BYTE_WORD {
                let mut word = _mm256_lddqu_si256(slice_ptr.get(0).unwrap().add(i as usize) as *const __m256i);
                let mut equal = _mm256_cmpeq_epi8(word, *target_word.get(0).unwrap());
                let mut greater = _mm256_cmpgt_epi8(word, *target_word.get(0).unwrap());


                if num_slice > 1 && !avx_iszero(equal){
                    let word1 = _mm256_lddqu_si256(slice_ptr.get(1).unwrap().add(i as usize) as *const __m256i);
                    // previous equal and current greater, then append (or) to previous greater
                    greater =_mm256_or_si256( greater,_mm256_and_si256(equal,_mm256_cmpgt_epi8(word1, *target_word.get(1).unwrap())));
                    // current equal and with previous equal
                    equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word1, *target_word.get(1).unwrap()));

                    if num_slice > 2 && !avx_iszero(equal){
                        let word2 = _mm256_lddqu_si256(slice_ptr.get(2).unwrap().add(i as usize) as *const __m256i);
                        // previous equal and current greater, then append (or) to previous greater
                        greater =_mm256_or_si256( greater,_mm256_and_si256(equal,_mm256_cmpgt_epi8(word2, *target_word.get(2).unwrap())));
                        // current equal and with previous equal
                        equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word2, *target_word.get(2).unwrap()));

                        if num_slice > 3 && !avx_iszero(equal){
                            let word3 = _mm256_lddqu_si256(slice_ptr.get(3).unwrap().add(i as usize) as *const __m256i);
                            // previous equal and current greater, then append (or) to previous greater
                            greater =_mm256_or_si256( greater,_mm256_and_si256(equal,_mm256_cmpgt_epi8(word3, *target_word.get(3).unwrap())));
                            // current equal and with previous equal
                            equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word3, *target_word.get(3).unwrap()));
                        }
                    }
                }
                let eq_mask = _mm256_movemask_epi8(equal);
                let gt_mask = _mm256_movemask_epi8(greater);
                bveq.push(mem::transmute::<i32, u32>(eq_mask));
                bvgt.push(mem::transmute::<i32, u32>(gt_mask));
                i += BYTE_WORD;
            }
        }
        let rb1 = BitVec::from_vec(&mut bveq,len as usize);
        res.set_storage(&bvgt);
        let card = rb1.cardinality();
        let mut check_ratio = card as f64/len as f64;

        remain = remain - 8*num_slice;
        println!("remain: {}", remain);

        if remain>0 && card!=0{
            bitpack.skip_n_byte((num_slice * len) as usize);
            dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
            bitpack.finish_read_byte();
            // let start = Instant::now();
            let mut iterator = BVIter::new(&rb1);
            // check the decimal part
            let mut it = iterator.next();
            let mut dec_cur = 0;

            let mut dec_pre = 0;
            let mut dec = 0;
            let mut delta = 0;
            if it!=None{
                dec_cur = it.unwrap();
                if dec_cur!=0{
                    bitpack.skip(dec_cur * remain as usize);
                }
                dec = bitpack.read_bits(remain as usize).unwrap();
                if dec>dec_byte{
                    res.set(dec_cur,true);
                }
                it = iterator.next();
                dec_pre = dec_cur;
            }
            while it!=None{
                dec_cur = it.unwrap();
                delta = dec_cur-dec_pre;
                if delta != 1 {
                    bitpack.skip((delta-1) * remain as usize);
                }
                dec = bitpack.read_bits(remain as usize).unwrap();
                if dec>dec_byte{
                    res.set(dec_cur,true);
                }
                it = iterator.next();
                dec_pre=dec_cur;
            }
            // println!("read the remain {} bits of dec",remain);
            remain = 0;
        }

        println!("Number of qualified int items:{}", res.cardinality());
    }


    pub(crate) fn buff_simd_equal_filter_with_slice(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        // println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        // println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        // println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut bv = Vec::new();
        let mut res = BitVec::from_elem(len as usize, false);
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        // let start = Instant::now();
        let num_slice = floor(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        bound.set_length(ilen as u64, dlen as u64);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        let mut dec_byte = fixed_target as u8;

        for cnt in 0..num_slice{
            let chunk = bitpack.read_n_byte_unmut((cnt*len) as usize, len as usize).unwrap();
            slice_ptr.push(chunk.as_ptr());
            data.push(chunk);
            let mut pred_u8= 0;
            if (remain-8*cnt>=8){
                pred_u8 =flip((fixed_target >> (remain - 8 * (1 + cnt))) as u8);
            }
            else {
                let pad = 8*(1+cnt)-remain;
                pred_u8= flip((fixed_target << pad) as u8)
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }

        let mut res_bm = RoaringBitmap::new();
        let mut i = 0;
        unsafe {
            while i <= len - BYTE_WORD {
                let mut word = _mm256_lddqu_si256(slice_ptr.get(0).unwrap().add(i as usize) as *const __m256i);
                let mut equal = _mm256_cmpeq_epi8(word, *target_word.get(0).unwrap());


                if num_slice > 1 && !avx_iszero(equal){
                    let word1 = _mm256_lddqu_si256(slice_ptr.get(1).unwrap().add(i as usize) as *const __m256i);
                    // current equal and with previous equal
                    equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word1, *target_word.get(1).unwrap()));

                    if num_slice > 2 && !avx_iszero(equal){
                        let word2 = _mm256_lddqu_si256(slice_ptr.get(2).unwrap().add(i as usize) as *const __m256i);
                        // current equal and with previous equal
                        equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word2, *target_word.get(2).unwrap()));

                        if num_slice > 3 && !avx_iszero(equal){
                            let word3 = _mm256_lddqu_si256(slice_ptr.get(3).unwrap().add(i as usize) as *const __m256i);
                            // current equal and with previous equal
                            equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word3, *target_word.get(3).unwrap()));
                        }
                    }
                }
                let eq_mask = _mm256_movemask_epi8(equal);
                bv.push(mem::transmute::<i32, u32>(eq_mask));
                i += BYTE_WORD;
            }
        }
        let rb1 = BitVec::from_vec(&mut bv,len as usize);
        let card = rb1.cardinality();
        let mut check_ratio = card as f64/len as f64;

        remain = remain - 8*num_slice;
        println!("remain: {}", remain);

        if remain>0 && card!=0{
            bitpack.skip_n_byte((num_slice * len) as usize);
            dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
            bitpack.finish_read_byte();
            // let start = Instant::now();
            let mut iterator = BVIter::new(&rb1);
            // check the decimal part
            let mut it = iterator.next();
            let mut dec_cur = 0;

            let mut dec_pre = 0;
            let mut dec = 0;
            let mut delta = 0;
            if it!=None{
                dec_cur = it.unwrap();
                if dec_cur!=0{
                    bitpack.skip(dec_cur * remain as usize);
                }
                dec = bitpack.read_bits(remain as usize).unwrap();
                if dec==dec_byte{
                    res.set(dec_cur,true);
                }
                it = iterator.next();
                dec_pre = dec_cur;
            }
            while it!=None{
                dec_cur = it.unwrap();
                delta = dec_cur-dec_pre;
                if delta != 1 {
                    bitpack.skip((delta-1) * remain as usize);
                }
                dec = bitpack.read_bits(remain as usize).unwrap();
                if dec==dec_byte{
                    res.set(dec_cur,true);
                }
                it = iterator.next();
                dec_pre=dec_cur;
            }
            // println!("read the remain {} bits of dec",remain);
            remain = 0;
        }

        println!("Number of qualified int items:{}", res.cardinality());
    }

    pub(crate) fn buff_simd512_range_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain = dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = RoaringBitmap::new();
        let mut res = RoaringBitmap::new();
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        println!("fixed target:{}", fixed_target);
        let mut byte_count = 0;
        let mut cur_tar = 0u8;
        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                if cur as u64>fixed_target{
                    res.insert(i);
                }
            }
            remain = 0;
        }else {
            remain-=8;
            byte_count+=1;
            let chunk = bitpack.read_n_byte(len as usize).unwrap();
            cur_tar = (fixed_target >> remain) as u8;
            unsafe { rb1 = range_simd_myroaring(chunk, &mut res, cur_tar); }
        }


        // println!("Number of qualified items in bitmap:{}", rb1.cardinality());

        while (remain>0){
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;
                let mut cur_rb = RoaringBitmap::new();
                if rb1.len()!=0{
                    // let start = Instant::now();
                    let mut iterator = rb1.iter();
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;
                    let mut dec_pre:u32 = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    // shift right to get corresponding byte
                    cur_tar = (fixed_target >> remain as u64) as u8;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip_n_byte((dec_cur) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // println!("{} first match: {}", dec_cur, dec);
                        if dec>cur_tar{
                            res.insert(dec_cur);
                        }
                        else if dec == cur_tar{
                            cur_rb.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip_n_byte((delta-1) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // if dec_cur<10{
                        //     println!("{} first match: {}", dec_cur, dec);
                        // }
                        if dec>cur_tar{
                            res.insert(dec_cur);
                        }
                        else if dec == cur_tar{
                            cur_rb.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    if len - dec_pre>1 {
                        bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                    }
                }
                else{
                    bitpack.skip_n_byte((len) as usize);
                    break;
                }
                rb1 = cur_rb;
                // println!("read the {}th byte of dec",byte_count);
                // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
            }
            // else we have to read by bits
            else {
                cur_tar =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                bitpack.finish_read_byte();
                if rb1.len()!=0{
                    // let start = Instant::now();
                    let mut iterator = rb1.iter();
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre:u32 = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(((dec_cur) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>cur_tar{
                            res.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip(((delta-1) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>cur_tar{
                            res.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    println!("read the remain {} bits of dec",remain);
                    remain = 0;
                }
                else{
                    break;
                }
            }
        }

        println!("Number of qualified items:{}", res.len());
    }

    pub(crate) fn buff_simd512_equal_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        // println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        // println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        // println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = RoaringBitmap::new();
        let mut res = RoaringBitmap::new();
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int {
            // println!("Number of qualified items for equal:{}", 0);
            return;
        }
        let fixed_target = (fixed_part-base_int ) as u64;
        let mut dec_byte = fixed_target as u8;
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        // let start = Instant::now();

        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                if cur as u64==fixed_target{
                    res.insert(i);
                }
            }
            remain = 0;
        }else {
            remain-=8;
            byte_count+=1;
            let chunk = bitpack.read_n_byte(len as usize).unwrap();
            // let start = Instant::now();
            dec_byte = (fixed_target >> remain) as u8;
            unsafe { rb1 = equal_simd_myroaring(chunk, dec_byte); }
            // let duration = start.elapsed();
            // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
            // let mut i =0;
            // for &c in chunk {
            //     if c == dec_byte {
            //         rb1.insert(i);
            //     };
            //     i+=1;
            // }

        }
        // rb1.run_optimize();
        // let duration = start.elapsed();
        // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
        // println!("Number of qualified items for equal:{}", rb1.len());

        while (remain>0){
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;
                let mut cur_rb = RoaringBitmap::new();
                if rb1.len()!=0{
                    // let start = Instant::now();
                    let mut iterator = rb1.iter();
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;
                    let mut dec_pre:u32 = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    // shift right to get corresponding byte
                    dec_byte = (fixed_target >> remain as u64) as u8;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip_n_byte((dec_cur) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // println!("{} first match: {}", dec_cur, dec);
                        if dec == dec_byte{
                            cur_rb.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip_n_byte((delta-1) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // if dec_cur<10{
                        //     println!("{} first match: {}", dec_cur, dec);
                        // }
                        if dec == dec_byte{
                            cur_rb.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    if len - dec_pre>1 {
                        bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                    }
                }
                else{
                    bitpack.skip_n_byte((len) as usize);
                    break;
                }
                rb1 = cur_rb;
                // println!("read the {}th byte of dec",byte_count);
                // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
            }
            // else we have to read by bits
            else {
                dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                bitpack.finish_read_byte();
                if rb1.len()!=0{
                    // let start = Instant::now();
                    let mut iterator = rb1.iter();
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre:u32 = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(((dec_cur) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec==dec_byte{
                            res.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip(((delta-1) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec==dec_byte{
                            res.insert(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    // println!("read the remain {} bits of dec",remain);
                    remain = 0;
                }
                else{
                    break;
                }
            }
        }
        println!("Number of qualified int items:{}", res.len());
    }

    pub(crate) fn buff_range_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain = dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        let mut byte_count = 0;
        let mut cur_tar = 0u8;
        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                if cur as u64>fixed_target{
                    res.set(i,true);
                }
            }
            remain = 0;
        }else {
            remain-=8;
            byte_count+=1;
            let chunk = bitpack.read_n_byte(len as usize).unwrap();
            cur_tar = (fixed_target >> remain) as u8;
            let mut i =0;
            for &c in chunk {
                if c >cur_tar{
                    res.set(i, true);
                }
                else if c == cur_tar {
                    rb1.set(i, true);
                };
                i+=1;
            }
        }


        // println!("Number of qualified items in bitmap:{}", rb1.cardinality());

        while (remain>0){
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;
                let mut cur_rb = BitVec::from_elem(len, false);;
                if rb1.cardinality()!=0{
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;
                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    // shift right to get corresponding byte
                    cur_tar = (fixed_target >> remain as u64) as u8;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip_n_byte((dec_cur) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // println!("{} first match: {}", dec_cur, dec);
                        if dec>cur_tar{
                            res.set(dec_cur, true);
                        }
                        else if dec == cur_tar{
                            cur_rb.set(dec_cur, true);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip_n_byte((delta-1) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // if dec_cur<10{
                        //     println!("{} first match: {}", dec_cur, dec);
                        // }
                        if dec>cur_tar{
                            res.set(dec_cur, true);
                        }
                        else if dec == cur_tar{
                            cur_rb.set(dec_cur, true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    if len - dec_pre>1 {
                        bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                    }
                }
                else{
                    bitpack.skip_n_byte((len) as usize);
                    break;
                }
                rb1 = cur_rb;
                // println!("read the {}th byte of dec",byte_count);
                // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
            }
            // else we have to read by bits
            else {
                cur_tar =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                bitpack.finish_read_byte();
                if rb1.cardinality()!=0{
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(dec_cur * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>cur_tar{
                            res.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip((delta-1) * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>cur_tar{
                            res.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    println!("read the remain {} bits of dec",remain);
                    remain = 0;
                }
                else{
                    break;
                }
            }
        }

        println!("Number of qualified items:{}", res.cardinality());
    }


    pub(crate) fn buff_equal_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap() as usize;
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = BitVec::from_elem(len, false);
        let mut res = BitVec::from_elem(len, false);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int {
            println!("Number of qualified items for equal:{}", 0);
            return;
        }
        let fixed_target = (fixed_part-base_int ) as u64;
        let mut dec_byte = fixed_target as u8;
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        let start = Instant::now();
        let mut count = 0;

        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap();
                if cur as u64==fixed_target{
                    res.set(i,true);
                }
            }
            remain = 0;
        }else {
            remain-=8;
            byte_count+=1;
            let chunk = bitpack.read_n_byte(len as usize).unwrap();
            dec_byte = (fixed_target >> remain) as u8;
            let mut i =0;
            for &c in chunk {
                if c == dec_byte {
                    rb1.set(i, true);
                    // count += 1;
                };
                i+=1;
            }
        }
        // rb1.run_optimize();
        let duration = start.elapsed();
        println!("Time elapsed in first byte is: {:?}", duration);
        println!("Number of qualified items for equal:{}", rb1.cardinality());

        while (remain>0){
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;
                let mut cur_rb = BitVec::from_elem(len, false);
                if rb1.cardinality()!=0{
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;
                    let mut dec_pre = 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    // shift right to get corresponding byte
                    dec_byte = (fixed_target >> remain as u64) as u8;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip_n_byte((dec_cur) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // println!("{} first match: {}", dec_cur, dec);
                        if dec == dec_byte{
                            cur_rb.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip_n_byte((delta-1) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // if dec_cur<10{
                        //     println!("{} first match: {}", dec_cur, dec);
                        // }
                        if dec == dec_byte{
                            cur_rb.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    if len - dec_pre>1 {
                        bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                    }
                }
                else{
                    bitpack.skip_n_byte((len) as usize);
                    break;
                }
                rb1 = cur_rb;
                // println!("read the {}th byte of dec",byte_count);
                // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
            }
            // else we have to read by bits
            else {
                dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
                bitpack.finish_read_byte();
                if rb1.cardinality()!=0{
                    // let start = Instant::now();
                    let mut iterator = BVIter::new(&rb1);
                    // check the decimal part
                    let mut it = iterator.next();
                    let mut dec_cur = 0;

                    let mut dec_pre= 0;
                    let mut dec = 0;
                    let mut delta = 0;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip(dec_cur * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec==dec_byte{
                            res.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip((delta-1) * remain as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec==dec_byte{
                            res.set(dec_cur,true);
                        }
                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    println!("read the remain {} bits of dec",remain);
                    remain = 0;
                }
                else{
                    break;
                }
            }
        }
        println!("Number of qualified int items:{}", res.cardinality());
    }

    // todo: buff max in given range, fix the logics.
    pub(crate) fn buff_max_range(&self, bytes: Vec<u8>,s:u32, e:u32, window:u32) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let lower = bitpack.read(32).unwrap();
        let higher = bitpack.read(32).unwrap();
        let ubase_int= (lower as u64)|((higher as u64)<<32);
        let base_int = unsafe { mem::transmute::<u64, i64>(ubase_int) };
        // println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        // println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        // println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        let mut max = u64::min_value();
        let mut max_vec = Vec::new();

        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        let mut byte_max = u8::min_value();
        let mut cur_s= s;
        // let start = Instant::now();

        if remain<8{
            bitpack.skip(((s) * remain) as usize);
            for i in s..e {
                if i==cur_s+window{
                    cur_s = i;
                    max_vec.push(max);
                    max = u64::min_value();
                }
                cur = bitpack.read_bits(remain as usize).unwrap() as u64;
                if cur >max{
                    max = cur ;
                    res.remove_range(cur_s as u64 .. i as u64);
                    res.add(i);
                }
                else if cur == max {
                    res.add(i);
                }
            }
            if e==cur_s+window{
                cur_s = e;
                max_vec.push(max);
                max = u64::min_value();
            }
            remain = 0;
        }else {
            remain-=8;
            byte_count+=1;
            bitpack.skip_n_byte(s as usize);
            let chunk = bitpack.read_n_byte((e-s) as usize).unwrap();
            let mut i =s;
            for &c in chunk{
                if i==cur_s+window{
                    cur_s = i;
                    max_vec.push((byte_max as u64) << (remain as u64));
                    byte_max = u8::min_value();
                }
                if c > byte_max {
                    byte_max=c;
                    rb1.remove_range(cur_s as u64 .. i as u64);
                    rb1.add(i);
                }
                else if c == byte_max {
                    rb1.add(i);
                }
                i+=1;
            }
            if i==cur_s+window{
                cur_s = i;
                max_vec.push((byte_max as u64) << (remain as u64));
                byte_max = u8::min_value();
            }
            bitpack.skip_n_byte((len-e) as usize);
        }
        // rb1.run_optimize();
        // let duration = start.elapsed();
        // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
        // println!("Number of qualified items for max:{}", rb1.cardinality());

        while (remain>0){
            byte_max = u8::min_value();
            let mut cur_s= s;
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;

                let mut dec_cur = 0;
                let mut dec_pre:u32 = 0;
                let mut dec = 0;
                let mut delta = 0;
                let mut cur_rb = Bitmap::create();

                {
                    let mut iterator = rb1.iter();
                    // let start = Instant::now();
                    // check the decimal part
                    let mut it = iterator.next();
                    // shift right to get corresponding byte
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip_n_byte((dec_cur) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // println!("{} first match: {}", dec_cur, dec);
                        if dec > byte_max{
                            byte_max = dec;
                            cur_rb.add(dec_cur);

                        }else if dec == byte_max {
                            cur_rb.add(dec_cur);
                        }
                        it = iterator.next();
                        dec_pre = dec_cur;
                    }
                    while it!=None{
                        dec_cur = it.unwrap();

                        if dec_cur >= cur_s+window{
                            let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                            *max_elem = *max_elem | ((byte_max as u64)<< remain);
                            byte_max = u8::min_value();
                            cur_s= cur_s+window;
                        }

                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip_n_byte((delta-1) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // if dec_cur<10{
                        //     println!("{} first match: {}", dec_cur, dec);
                        // }
                        if dec > byte_max{
                            byte_max = dec;
                            cur_rb.remove_range(cur_s as u64 .. dec_cur as u64);
                            cur_rb.add(dec_cur);
                        }else if dec == byte_max {
                            cur_rb.add(dec_cur);
                        }

                        it = iterator.next();
                        dec_pre=dec_cur;
                    }
                    assert_eq!((cur_s-s)/window+1, (e-s)/window);
                    /// set max for last window
                    let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                    *max_elem = *max_elem | ((byte_max as u64)<< remain);
                    byte_max = u8::min_value();
                    cur_s= cur_s+window;


                    if len - dec_pre>1 {
                        bitpack.skip_n_byte(((len - dec_pre - 1)) as usize);
                    }
                }
                rb1 = cur_rb;
                // println!("read the {}th byte of dec",byte_count);
                // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
            }
            // else we have to read by bits
            else {
                byte_max = u8::min_value();
                bitpack.finish_read_byte();
                // let start = Instant::now();
                let mut iterator = rb1.iter();
                // check the decimal part
                let mut it = iterator.next();
                let mut dec_cur = 0;

                let mut dec_pre:u32 = 0;
                let mut dec = 0;
                let mut delta = 0;
                if it!=None{
                    dec_cur = it.unwrap();
                    if dec_cur!=0{
                        bitpack.skip(((dec_cur) * remain) as usize);
                    }
                    dec = bitpack.read_bits(remain as usize).unwrap();
                    if dec>byte_max{
                        byte_max = dec;
                        res.add(dec_cur);
                    }
                    else if dec == byte_max {
                        res.add(dec_cur);
                    }

                    it = iterator.next();
                    dec_pre = dec_cur;
                }
                while it!=None{
                    dec_cur = it.unwrap();
                    if dec_cur >= cur_s+window{
                        let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                        *max_elem = *max_elem | (byte_max as u64);
                        byte_max = u8::min_value();
                        cur_s= cur_s+window;
                    }
                    delta = dec_cur-dec_pre;
                    if delta != 1 {
                        bitpack.skip(((delta-1) * remain) as usize);
                    }
                    dec = bitpack.read_bits(remain as usize).unwrap();
                    if dec>byte_max{
                        byte_max = dec;
                        res.remove_range(cur_s as u64 .. dec_cur as u64);
                        res.add(dec_cur);
                    }
                    else if dec==byte_max{
                        res.add(dec_cur);
                    }
                    it = iterator.next();
                    dec_pre=dec_cur;
                }
                assert_eq!((cur_s-s)/window+1, (e-s)/window);
                /// set max for last window
                let mut max_elem = max_vec.get_mut(((cur_s-s)/window) as usize).unwrap();
                *max_elem = *max_elem | (byte_max as u64);
                byte_max = u8::min_value();
                cur_s= cur_s+window;
                // println!("read the remain {} bits of dec",remain);
                remain = 0;
            }
        }
        let max_vec_f64 : Vec<f64> = max_vec.iter().map(|&x| (x as i64 +base_int) as f64 / 2.0f64.powi(dlen as i32)).collect();
        println!("Number of qualified max items:{}", res.cardinality());
        println!("Max value:{:?}", max_vec_f64);
    }

    // pub(crate) fn simd_range_filter(&self, bytes: Vec<u8>,pred:f64) {
    //     let prec = (self.scale as f32).log10() as i32;
    //     let prec_delta = get_precision_bound(prec);
    //     let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
    //     let mut bound = PrecisionBound::new(prec_delta);
    //     let ubase_int = bitpack.read(32).unwrap();
    //     let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
    //     println!("base integer: {}",base_int);
    //     let len = bitpack.read(32).unwrap();
    //     println!("total vector size:{}",len);
    //     let ilen = bitpack.read(32).unwrap();
    //     println!("bit packing length:{}",ilen);
    //     let dlen = bitpack.read(32).unwrap();
    //     bound.set_length(ilen as u64, dlen as u64);
    //     // check integer part and update bitmap;
    //     let mut rb1 = Bitmap::create();
    //     let mut res = Bitmap::create();
    //     let target = pred;
    //     let (int_part, dec_part) = bound.fetch_components(target);
    //     let int_target = (int_part-base_int as i64) as u32;
    //     let dec_target = dec_part as u32;
    //     println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
    //
    //     let mut int_vec:Vec<u8> = Vec::new();
    //
    //     let start = Instant::now();
    //     for i in 0..len {
    //         int_vec.push(bitpack.read(ilen as usize).unwrap() as u8);
    //     }
    //     let lane = 16;
    //     assert!(int_vec.len() % lane == 0);
    //     let mut pre_vec = u8x16::splat(int_target as u8);
    //     for i in (0..int_vec.len()).step_by(lane) {
    //         let cur_word = u8x16::from_slice_unaligned(&int_vec[i..]);
    //         let m = cur_word.gt(pre_vec);
    //         for j in 0..lane{
    //             if m.extract(j){
    //                 res.add((i + j) as u32);
    //             }
    //         }
    //         let m = cur_word.eq(pre_vec);
    //         for j in 0..lane{
    //             if m.extract(j){
    //                 rb1.add((i + j) as u32);
    //             }
    //         }
    //     }
    //
    //
    //     // rb1.run_optimize();
    //     // res.run_optimize();
    //
    //     let duration = start.elapsed();
    //     println!("Time elapsed in splitBD simd filtering int part is: {:?}", duration);
    //     println!("Number of qualified int items:{}", res.cardinality());
    //
    //     let start = Instant::now();
    //     let mut iterator = rb1.iter();
    //     // check the decimal part
    //     let mut it = iterator.next();
    //     let mut dec_cur = 0;
    //     let mut dec_pre:u32 = 0;
    //     let mut dec = 0;
    //     let mut delta = 0;
    //     if it!=None{
    //         dec_cur = it.unwrap();
    //         if dec_cur!=0{
    //             bitpack.skip(((dec_cur) * dlen) as usize);
    //         }
    //         dec = bitpack.read(dlen as usize).unwrap();
    //         if dec>dec_target{
    //             res.add(dec_cur);
    //         }
    //         // println!("index qualified {}, decimal:{}",dec_cur,dec);
    //         it = iterator.next();
    //         dec_pre = dec_cur;
    //     }
    //     while it!=None{
    //         dec_cur = it.unwrap();
    //         //println!("index qualified {}",dec_cur);
    //         delta = dec_cur-dec_pre;
    //         if delta != 1 {
    //             bitpack.skip(((delta-1) * dlen) as usize);
    //         }
    //         dec = bitpack.read(dlen as usize).unwrap();
    //         // if dec_cur<10{
    //         //     println!("index qualified {}, decimal:{}",dec_cur,dec);
    //         // }
    //         if dec>dec_target{
    //             res.add(dec_cur);
    //         }
    //         it = iterator.next();
    //         dec_pre=dec_cur;
    //     }
    //     let duration = start.elapsed();
    //     println!("Time elapsed in splitBD simd filtering fraction part is: {:?}", duration);
    //     println!("Number of qualified items:{}", res.cardinality());
    // }
}