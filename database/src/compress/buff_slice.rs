use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{SCALE, CompressionMethod};
use crate::segment::Segment;
use std::time::{SystemTime, Instant};
use crate::methods::prec_double::{get_precision_bound, PrecisionBound};
use crate::simd::vectorize_query::{range_simd_myroaring, equal_simd_myroaring};
use std::mem;
use log::info;
use serde::{Deserialize, Serialize};
use tsz::stream::BufferedWriter;
use itertools::Itertools;
use crate::compress::PRECISION_MAP;
use std::arch::x86_64::{__m256i, _mm256_set1_epi8, _mm256_lddqu_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8, _mm256_cmpgt_epi8, _mm256_and_si256, _mm256_or_si256, _mm256_testz_si256, _mm256_set1_epi64x, _mm256_setr_epi64x, _mm256_set1_epi32, _mm256_shuffle_epi8};
use std::ptr::eq;
use my_bit_vec::BitVec;
use std::slice::Iter;
use parquet::basic::Type::BYTE_ARRAY;
use num::Float;
use crate::methods::bit_packing::BitPack;

pub const BYTE_WORD:u32 = 32u32;
pub const REVERSE_i64:i64 = -9205322385119247871i64;
pub const SF0:i64 = 0b0000000000000000000000000000000000000000000000000000000000000000;
pub const SF1:i64 = 0b0000000100000001000000010000000100000001000000010000000100000001;
pub const SF2:i64 = 0b0000001000000010000000100000001000000010000000100000001000000010;
pub const SF3:i64 = 0b0000001100000011000000110000001100000011000000110000001100000011;
// pub const SHUFFLE_MASK: __m256i = unsafe { _mm256_setr_epi64x(SF0,SF1,SF2,SF3) };
// pub const INVERSE_MASK: __m256i = unsafe { _mm256_set1_epi64x(REVERSE_i64) };

pub fn run_buff_slice_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = BuffSliceCompress::new(10,10,scl);
    let start1 = Instant::now();
    let mut compressed= comp.buff_slice_encode(&mut seg);

    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in buff slice compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.buff_slice_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in buff slice decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.buff_slice_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in buff slice range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.buff_slice_equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in buff slice equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    // comp.buff_slice_range_filter_nosimd(comp_sum,pred);
    comp.buff_slice_buffsum(comp_sum);
    let duration5 = start5.elapsed();
    // println!("Time elapsed in buff slice range no simd function() is: {:?}", duration5);
    println!("Time elapsed in buff slice sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    // comp.buff_slice_equal_filter_nosimd(comp_max,pred);
    comp.buff_slice_buffmax(comp_max);
    let duration6 = start6.elapsed();
    // println!("Time elapsed in buff slice eqaul no simd function() is: {:?}", duration6);
    println!("Time elapsed in buff slice max function() is: {:?}", duration6);

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


pub fn run_buff_slice_scalar_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = BuffSliceCompress::new(10,10,scl);
    let start1 = Instant::now();
    let mut compressed= comp.buff_slice_encode(&mut seg);

    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_sum = compressed.clone();
    let comp_max = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in buff slice compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.buff_slice_decode_bs(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in buff slice decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.buff_slice_range_filter_nosimd(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in buff slice range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.buff_slice_equal_filter_nosimd(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in buff slice equal filter function() is: {:?}", duration4);

    let start5 = Instant::now();
    comp.buff_slice_sum(comp_sum);
    let duration5 = start5.elapsed();
    // println!("Time elapsed in buff slice range no simd function() is: {:?}", duration5);
    println!("Time elapsed in buff slice sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    comp.buff_slice_max(comp_max);
    let duration6 = start6.elapsed();
    // println!("Time elapsed in buff slice eqaul no simd function() is: {:?}", duration6);
    println!("Time elapsed in buff slice max function() is: {:?}", duration6);

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

#[inline]
pub fn flip(x:u8)->u8{
    let offset = 1u8<< 7;
    x^offset
}

#[inline]
pub fn ceil(x:u32,y:u32)->u32{
   (x-1)/y+1
}

#[inline]
pub fn floor(x:u32,y:u32)->u32{
    x/y
}

#[inline]
pub fn avx_iszero(a: __m256i)-> bool{
    unsafe { return _mm256_testz_si256(a, a) == 1 }
}




/// by Chunwei
/// set simd i8 word with u8 input
pub fn set_pred_word(pred:u8) -> __m256i{
    let predicate = unsafe { mem::transmute::<u8, i8>(pred) };
    println!("current predicate:{}",pred);
    let pred_word = unsafe { _mm256_set1_epi8(predicate) };
    pred_word
}

/// by Chunwei
pub fn inverse_movemask(input:u32,shuffle_mask:__m256i,inverse_mask: __m256i) -> __m256i{
    let val = unsafe { mem::transmute::<u32, i32>(input) };
    // println!("current predicate:{}",val);
    let input_simd = unsafe { _mm256_set1_epi32(val) };
    let input_shuflle= unsafe{ _mm256_shuffle_epi8(input_simd, shuffle_mask) };
    let simd_and = unsafe { _mm256_and_si256(input_shuflle, inverse_mask) };
    let greater = unsafe { _mm256_cmpeq_epi8(inverse_mask, simd_and) };
    greater
}






#[derive(Clone)]
pub struct BuffSliceCompress {
    chunksize: usize,
    batchsize: usize,
    pub(crate) scale: usize
}

impl BuffSliceCompress {
    pub fn new(chunksize: usize, batchsize: usize, scale: usize) -> Self {
        BuffSliceCompress { chunksize, batchsize, scale}
    }

    fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        let mut bd_vec = Vec::new();
        let mut dec_vec = Vec::new();

        let mut t:u32 =0;
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        let mut bound = PrecisionBound::new(prec_delta);
        let start = Instant::now();
        for val in seg.get_data(){
            let v = bound.precision_bound((*val).into());
            bd_vec.push(v);
            t+=1;
            bound.cal_length(v);
//            println!("{}=>{}",(*val).into(),v);
//            let preu =  unsafe { mem::transmute::<f64, u64>((*val).into()) };
//            let bdu =  unsafe { mem::transmute::<f64, u64>(v) };
//            println!("{:#066b} => {:#066b}", preu, bdu);
        }
        let duration = start.elapsed();
        println!("Time elapsed in bound double function() is: {:?}", duration);
        let start1 = Instant::now();
        let (int_len,dec_len) = bound.get_length();
        // let (int_len,dec_len) = (4u64,19u64);
        let ilen = int_len as usize;
        let dlen = dec_len as usize;
        println!("int_len:{},dec_len:{}",int_len,dec_len);
        //let cap = (int_len+dec_len)* t as u64 /8;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);
        let mut i= 0;
        for bd in bd_vec{
            let (int_part, dec_part) = bound.fetch_components(bd);
            // if i<10{
            //     println!("cur: {}",bd);
            //     println!("{}th, int: {}, decimal: {} in form:{}",i,int_part,dec_part as f64*1.0f64/(2i64.pow(dec_len as u32)) as f64, dec_part);
            // }
            // i += 1;
            bitpack_vec.write(int_part as u32, ilen).unwrap();
            dec_vec.push(dec_part);
        }
        let duration1 = start1.elapsed();
        println!("Time elapsed in dividing double function() is: {:?}", duration1);

        for d in dec_vec {
            bitpack_vec.write(d as u32, dlen).unwrap();
        }
        let vec = bitpack_vec.into_vec();

        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
    }

    // byteslice padding right for the trailing bits
    pub fn buff_slice_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
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
        let dec_len = *(PRECISION_MAP.get(&prec).unwrap()) as u64;
        bound.set_length(0,dec_len);
        let mut min = i64::max_value();
        let mut max = i64::min_value();
        let start1= Instant::now();
        for bd in seg.get_data(){
            let fixed = bound.fetch_fixed_aligned((*bd).into());
            if fixed<min {
                min = fixed;
            }
            else if fixed>max {
                max = fixed;
            }
            fixed_vec.push(fixed);
        }
        let duration1 = start1.elapsed();
        println!("Time elapsed in bit extract double function() is: {:?}", duration1);

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
            // add right padding for last slice.
            let padding = 8-remain;
            for i in fixed_vec{
                bitpack_vec.write_byte(flip((((i-base_fixed64) as u64 )<<padding) as u8)).unwrap();
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
                // add right padding for last slice.
                let padding = 8-remain;
                for d in fixed_u64 {
                    bitpack_vec.write_byte(flip((d<<padding) as u8)).unwrap();
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

    pub fn buff_slice_decode(&self, bytes: Vec<u8>) -> Vec<f64>{
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

        if remain<8{
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            let padding = 8-remain;
            for &x in chunk {
                expected_datapoints.push((base_int + (flip(x)>>padding) as i64 ) as f64 / dec_scl);
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
                println!("read remaining {} bits of dec",remain);
                println!("length for fixed:{}", fixed_vec.len());
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                let padding = 8-remain;
                for (&cur_fixed, &x) in fixed_vec.iter().zip(chunk.iter()){
                    f_cur = (base_int + ((cur_fixed)|(flip(x) as u64 >> padding)) as i64) as f64 / dec_scl;
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

    /// decode in buff way and max
    pub fn buff_slice_buffmax(&self, bytes: Vec<u8>){
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

        let mut index = 0;
        let mut fixed_vec:Vec<u64> = Vec::new();

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk;
        let mut f_cur = 0f64;
        let mut res = BitVec::from_elem(len as usize, false);
        let mut max = f64::min_value();


        if remain<8{
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            let padding = 8-remain;
            index = 0;
            for &x in chunk {
                f_cur=(base_int + (flip(x)>>padding) as i64 ) as f64 / dec_scl;
                if f_cur>max{
                    max = f_cur;
                    res.clear();
                    res.set(index,true);
                }
                else if f_cur==max{
                    res.set(index,true);
                }
                index+=1;
            }
            remain=0
        }
        else {
            bytec+=1;
            remain -= 8;
            chunk = bitpack.read_n_byte(len as usize).unwrap();

            if remain == 0 {
                index = 0;
                for &x in chunk {
                    f_cur = (base_int + flip(x) as i64) as f64 / dec_scl;

                    if f_cur>max{
                        max = f_cur;
                        res.clear();
                        res.set(index,true);
                    }
                    else if f_cur==max{
                        res.set(index,true);
                    }
                    index+=1
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

                    index = 0;
                    for (cur_fixed,cur_chunk) in fixed_vec.iter().zip(chunk.iter()){
                        f_cur =  (base_int + ((*cur_fixed)|(flip(*cur_chunk) as u64)) as i64 ) as f64 / dec_scl;

                        if f_cur>max{
                            max = f_cur;
                            res.clear();
                            res.set(index,true);
                        }
                        else if f_cur==max{
                            res.set(index,true);
                        }
                        index+=1;
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
                println!("read remaining {} bits of dec",remain);
                println!("length for fixed:{}", fixed_vec.len());
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                let padding = 8-remain;
                index = 0;
                for (&cur_fixed, &x) in fixed_vec.iter().zip(chunk.iter()){
                    f_cur = (base_int + ((cur_fixed)|(flip(x) as u64 >> padding)) as i64) as f64 / dec_scl;
                    if f_cur>max{
                        max = f_cur;
                        res.clear();
                        res.set(index,true);
                    }
                    else if f_cur==max{
                        res.set(index,true);
                    }
                    index+=1;
                }
            }
        }
        // for i in 0..10{
        //     println!("{}th item:{}",i,expected_datapoints.get(i).unwrap())
        // }

        println!("Number of qualified max items:{}", res.cardinality());
        println!("Max value:{}", max);
    }

    /// decode in buff way and sum
    pub fn buff_slice_buffsum(&self, bytes: Vec<u8>){
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

        let mut fixed_vec:Vec<u64> = Vec::new();

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk;
        let mut f_cur = 0f64;
        let mut sum = 0.0;


        if remain<8{
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            let padding = 8-remain;
            sum = 0.0;
            for &x in chunk {
                f_cur=(base_int + (flip(x)>>padding) as i64 ) as f64 / dec_scl;
                sum+=f_cur;
            }
            remain=0
        }
        else {
            bytec+=1;
            remain -= 8;
            chunk = bitpack.read_n_byte(len as usize).unwrap();

            if remain == 0 {
                sum = 0.0;
                for &x in chunk {
                    f_cur = (base_int + flip(x) as i64) as f64 / dec_scl;
                    sum+=f_cur;
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

                    sum = 0.0;
                    for (cur_fixed,cur_chunk) in fixed_vec.iter().zip(chunk.iter()){
                        f_cur =  (base_int + ((*cur_fixed)|(flip(*cur_chunk) as u64)) as i64 ) as f64 / dec_scl;
                        sum+=f_cur;
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
                println!("read remaining {} bits of dec",remain);
                println!("length for fixed:{}", fixed_vec.len());
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                let padding = 8-remain;
                sum = 0.0;
                for (&cur_fixed, &x) in fixed_vec.iter().zip(chunk.iter()){
                    f_cur = (base_int + ((cur_fixed)|(flip(x) as u64 >> padding)) as i64) as f64 / dec_scl;
                    sum+=f_cur;
                }
            }
        }
        println!("sum value:{}",  sum);

    }

    /// load all data, deocde and max
    pub fn buff_slice_max(&self, bytes: Vec<u8>){
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


        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk0:&[u8];
        let mut chunk1:&[u8];
        let mut chunk2:&[u8];
        let mut chunk3:&[u8];
        let mut f_cur = 0f64;
        let num = ceil(remain, 8);
        println!("Number of chunks:{}", num);

        let padding = num*8-ilen-dlen;
        let mut res = BitVec::from_elem(len as usize, false);
        let mut max = f64::min_value();


        match num {
            1=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((flip(*chunk0.get(index).unwrap()) as u64)>>padding) as i64 ) as f64 / dec_scl;
                    if f_cur>max{
                        max = f_cur;
                        res.clear();
                        res.set(index,true);
                    }
                    else if f_cur==max{
                        res.set(index,true);
                    }

                }
            }
            2=>{
                chunk0 = bitpack.read_n_byte_unmut(0, len as usize).unwrap().clone();
                chunk1 = bitpack.read_n_byte_unmut(len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(8-padding)) ) |
                        ((flip(*chunk1.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    if f_cur>max{
                        max = f_cur;
                        res.clear();
                        res.set(index,true);
                    }
                    else if f_cur==max{
                        res.set(index,true);
                    }
                }
            }
            3=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur=(base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(8-padding) )as u64) |
                        ((flip(*chunk2.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    if f_cur>max{
                        max = f_cur;
                        res.clear();
                        res.set(index,true);
                    }
                    else if f_cur==max{
                        res.set(index,true);
                    }
                }
            }
            4=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                chunk3 = bitpack.read_n_byte_unmut(3*len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                   f_cur = (base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(24-padding) )as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk2.get(index).unwrap()) as u64)<<(8-padding)) as u64) |
                        ((flip(*chunk3.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    if f_cur>max{
                        max = f_cur;
                        res.clear();
                        res.set(index,true);
                    }
                    else if f_cur==max{
                        res.set(index,true);
                    }
                }
            }
            _ => {panic!("bit length greater than 32 is not supported yet.")}
        }
        println!("Number of qualified max items:{}", res.cardinality());
        println!("Max value:{}", max);
    }


    pub fn buff_slice_max_range(&self, bytes: Vec<u8>,s:u32, e:u32, window:u32) {
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


        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk0:&[u8];
        let mut chunk1:&[u8];
        let mut chunk2:&[u8];
        let mut chunk3:&[u8];
        let mut f_cur = 0f64;
        let num = ceil(remain, 8);
        println!("Number of chunks:{}", num);

        let padding = num*8-ilen-dlen;
        let mut res = BitVec::from_elem(len as usize, false);
        let mut max = f64::min_value();
        let mut max_vec = Vec::new();
        let mut cur_s= s as usize;


        match num {
            1=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((flip(*chunk0.get(index).unwrap()) as u64)>>padding) as i64 ) as f64 / dec_scl;
                    if index< s as usize {
                        continue;
                    }else if index>= e as usize {
                        break;
                    }
                    if index==cur_s+window as usize{
                        max_vec.push(max);
                        // println!("{}",max);
                        max =f64::min_value();
                        cur_s=index;
                    }

                    if f_cur > max{
                        max =  f_cur;
                        res.remove_range(cur_s  , index );
                        res.set(index,true);
                    }
                    else if f_cur == max {
                        res.set(index,true);
                    }

                }
            }
            2=>{
                chunk0 = bitpack.read_n_byte_unmut(0, len as usize).unwrap().clone();
                chunk1 = bitpack.read_n_byte_unmut(len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(8-padding)) ) |
                        ((flip(*chunk1.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    if index< s as usize {
                        continue;
                    }else if index>= e as usize {
                        break;
                    }
                    if index==cur_s+window as usize{
                        max_vec.push(max);
                        // println!("{}",max);
                        max =f64::min_value();
                        cur_s=index;
                    }

                    if f_cur > max{
                        max =  f_cur;
                        res.remove_range(cur_s  , index );
                        res.set(index,true);
                    }
                    else if f_cur == max {
                        res.set(index,true);
                    }

                }
            }
            3=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur=(base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(8-padding) )as u64) |
                        ((flip(*chunk2.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    if index< s as usize {
                        continue;
                    }else if index>= e as usize {
                        break;
                    }
                    if index==cur_s+window as usize{
                        max_vec.push(max);
                        // println!("{}",max);
                        max =f64::min_value();
                        cur_s=index;
                    }

                    if f_cur > max{
                        max =  f_cur;
                        res.remove_range(cur_s  , index );
                        res.set(index,true);
                    }
                    else if f_cur == max {
                        res.set(index,true);
                    }

                }
            }
            4=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                chunk3 = bitpack.read_n_byte_unmut(3*len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(24-padding) )as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk2.get(index).unwrap()) as u64)<<(8-padding)) as u64) |
                        ((flip(*chunk3.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    if index< s as usize {
                        continue;
                    }else if index>= e as usize {
                        break;
                    }
                    if index==cur_s+window as usize{
                        max_vec.push(max);
                        // println!("{}",max);
                        max =f64::min_value();
                        cur_s=index;
                    }

                    if f_cur > max{
                        max =  f_cur;
                        res.remove_range(cur_s  , index );
                        res.set(index,true);
                    }
                    else if f_cur == max {
                        res.set(index,true);
                    }

                }
            }
            _ => {panic!("bit length greater than 32 is not supported yet.")}
        }
        assert_eq!((cur_s as u32-s)/window+1, (e-s)/window);
        /// set max for last window
        max_vec.push(max);

        // println!("Number of qualified max items:{}", res.cardinality());
        println!("Max value:{}", max_vec.len());
    }

    /// load all data, deocde and sum
    pub fn buff_slice_sum(&self, bytes: Vec<u8>){
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


        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen+ilen;
        let mut bytec = 0;
        let mut chunk0:&[u8];
        let mut chunk1:&[u8];
        let mut chunk2:&[u8];
        let mut chunk3:&[u8];
        let mut f_cur = 0f64;
        let num = ceil(remain, 8);
        println!("Number of chunks:{}", num);

        let padding = num*8-ilen-dlen;
        let mut sum = 0.0;


        match num {
            1=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((flip(*chunk0.get(index).unwrap()) as u64)>>padding) as i64 ) as f64 / dec_scl;
                    sum+=f_cur;

                }
            }
            2=>{
                chunk0 = bitpack.read_n_byte_unmut(0, len as usize).unwrap().clone();
                chunk1 = bitpack.read_n_byte_unmut(len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(8-padding)) ) |
                        ((flip(*chunk1.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    sum+=f_cur;
                }
            }
            3=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur=(base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(8-padding) )as u64) |
                        ((flip(*chunk2.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    sum+=f_cur;
                }
            }
            4=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                chunk3 = bitpack.read_n_byte_unmut(3*len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                    f_cur = (base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(24-padding) )as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk2.get(index).unwrap()) as u64)<<(8-padding)) as u64) |
                        ((flip(*chunk3.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl;
                    sum+=f_cur;
                }
            }
            _ => {panic!("bit length greater than 32 is not supported yet.")}
        }
        println!("sum value:{}",  sum);
    }

    pub fn buff_slice_decode_bs(&self, bytes: Vec<u8>) -> Vec<f64>{
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
        let mut chunk0:&[u8];
        let mut chunk1:&[u8];
        let mut chunk2:&[u8];
        let mut chunk3:&[u8];
        let mut f_cur = 0f64;
        let num = ceil(remain, 8);
        println!("Number of chunks:{}", num);

        let padding = num*8-ilen-dlen;
        match num {
            1=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                for index in 0..len as usize{
                    expected_datapoints.push((base_int + ((flip(*chunk0.get(index).unwrap()) as u64)>>padding) as i64 ) as f64 / dec_scl);
                }
            }
            2=>{
                chunk0 = bitpack.read_n_byte_unmut(0, len as usize).unwrap().clone();
                chunk1 = bitpack.read_n_byte_unmut(len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                    expected_datapoints.push((base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(8-padding)) ) |
                        ((flip(*chunk1.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl);
                }
            }
            3=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                for index in 0..len as usize{
                    expected_datapoints.push((base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(8-padding) )as u64) |
                        ((flip(*chunk2.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl);
                }
            }
            4=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                chunk3 = bitpack.read_n_byte_unmut(3*len as usize, len as usize).unwrap();
                for index in 0..len as usize{
                    expected_datapoints.push((base_int + ((((flip(*chunk0.get(index).unwrap()) as u64)<<(24-padding) )as u64)|
                        (((flip(*chunk1.get(index).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk2.get(index).unwrap()) as u64)<<(8-padding)) as u64) |
                        ((flip(*chunk3.get(index).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl);

                }
            }
            _ => {panic!("bit length greater than 32 is not supported yet.")}
        }

        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }


    pub fn buff_slice_decode_condition(&self, bytes: Vec<u8>,cond:Iter<usize>) -> Vec<f64>{
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
        let mut chunk0:&[u8];
        let mut chunk1:&[u8];
        let mut chunk2:&[u8];
        let mut chunk3:&[u8];
        let mut f_cur = 0f64;
        let num = ceil(remain, 8);
        println!("Number of chunks:{}", num);

        let padding = num*8-ilen-dlen;
        match num {
            1=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                for &elem in cond{
                    expected_datapoints.push((base_int + ((flip(*chunk0.get(elem).unwrap()) as u64)>>padding) as i64 ) as f64 / dec_scl);
                }
            }
            2=>{
                chunk0 = bitpack.read_n_byte_unmut(0, len as usize).unwrap().clone();
                chunk1 = bitpack.read_n_byte_unmut(len as usize, len as usize).unwrap();
                for &elem in cond{
                    expected_datapoints.push((base_int + ((((flip(*chunk0.get(elem).unwrap()) as u64)<<(8-padding)) ) |
                        ((flip(*chunk1.get(elem).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl);
                }
            }
            3=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                for &elem in cond{
                    expected_datapoints.push((base_int + ((((flip(*chunk0.get(elem).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk1.get(elem).unwrap()) as u64)<<(8-padding) )as u64) |
                        ((flip(*chunk2.get(elem).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl);
                }
            }
            4=>{
                chunk0 = bitpack.read_n_byte_unmut(0,len as usize).unwrap();
                chunk1 = bitpack.read_n_byte_unmut(len as usize,len as usize).unwrap();
                chunk2 = bitpack.read_n_byte_unmut(2*len as usize,len as usize).unwrap();
                chunk3 = bitpack.read_n_byte_unmut(3*len as usize, len as usize).unwrap();
                for &elem in cond{
                    expected_datapoints.push((base_int + ((((flip(*chunk0.get(elem).unwrap()) as u64)<<(24-padding) )as u64)|
                        (((flip(*chunk1.get(elem).unwrap()) as u64)<<(16-padding)) as u64)|
                        (((flip(*chunk2.get(elem).unwrap()) as u64)<<(8-padding)) as u64) |
                        ((flip(*chunk3.get(elem).unwrap()) as u64)>>padding) as u64) as i64 ) as f64 / dec_scl);

                }
            }
            _ => {panic!("bit length greater than 32 is not supported yet.")}
        }

        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }


    pub(crate) fn buff_slice_range_filter(&self, bytes: Vec<u8>, pred:f64) {
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
        let num_slice = ceil(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        bound.set_length(ilen as u64, dlen as u64);
        let mut res = BitVec::from_elem(len as usize, false);
        let mut resv = Vec::new();
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);

        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        println!("fixed target:{}", fixed_target);
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
                pred_u8= flip((fixed_target << pad) as u8);
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }

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
                let gt_mask = _mm256_movemask_epi8(greater);
                resv.push(mem::transmute::<i32, u32>(gt_mask));
                i += BYTE_WORD;
            }
        }
        res.set_storage(&resv);
        println!("Number of qualified items:{}", res.cardinality());
    }


    pub(crate) fn buff_slice_range_filter_condition(&self, bytes: Vec<u8>, pred:f64, cond:Iter<usize>) -> BitVec<u32> {
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
        let num_slice = ceil(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        let mut condition = BitVec::from_elem(len as usize, false);
        for &elem in cond{
            condition.set(elem,true);
        }


        bound.set_length(ilen as u64, dlen as u64);
        let mut res = BitVec::from_elem(len as usize, false);
        let mut resv = Vec::new();
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);

        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return res;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        println!("fixed target:{}", fixed_target);
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
                pred_u8= flip((fixed_target << pad) as u8);
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }

        let mut i = 0;
        let mut bv_elem = 0;
        let mut gt_mask = 0;
        let shuffle_mask: __m256i = unsafe { _mm256_setr_epi64x(SF0,SF1,SF2,SF3) };
        let inverse_mask: __m256i = unsafe { _mm256_set1_epi64x(REVERSE_i64) };

        unsafe {
            while i <= len - BYTE_WORD {
                bv_elem = condition.get_entry((i / BYTE_WORD) as usize);
                if (bv_elem!=0){
                    let mut equal =inverse_movemask(bv_elem,shuffle_mask, inverse_mask);

                    let mut word = _mm256_lddqu_si256(slice_ptr.get(0).unwrap().add(i as usize) as *const __m256i);
                    let mut greater = _mm256_and_si256(equal, _mm256_cmpgt_epi8(word, *target_word.get(0).unwrap()));
                    equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word, *target_word.get(0).unwrap()));

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
                    gt_mask = _mm256_movemask_epi8(greater);
                    resv.push(mem::transmute::<i32, u32>(gt_mask));
                }
                else {
                    resv.push(0);
                }
                i += BYTE_WORD;
            }
        }
        res.set_storage(&resv);
        println!("Number of qualified items:{}", res.cardinality());
        return res;
    }


    pub(crate) fn buff_slice_range_smaller_filter_condition(&self, bytes: Vec<u8>, pred:f64, cond:Iter<usize>) -> BitVec<u32> {
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
        let num_slice = ceil(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        let mut condition = BitVec::from_elem(len as usize, false);
        for &elem in cond{
            condition.set(elem,true);
        }


        bound.set_length(ilen as u64, dlen as u64);
        let mut res = BitVec::from_elem(len as usize, false);
        let mut resv = Vec::new();
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);

        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return res;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        println!("fixed target:{}", fixed_target);
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
                pred_u8= flip((fixed_target << pad) as u8);
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }

        let mut i = 0;
        let mut bv_elem = 0;
        let mut gt_mask = 0;
        let shuffle_mask: __m256i = unsafe { _mm256_setr_epi64x(SF0,SF1,SF2,SF3) };
        let inverse_mask: __m256i = unsafe { _mm256_set1_epi64x(REVERSE_i64) };

        unsafe {
            while i <= len - BYTE_WORD {
                bv_elem = condition.get_entry((i / BYTE_WORD) as usize);
                if (bv_elem!=0){
                    let mut equal =inverse_movemask(bv_elem,shuffle_mask, inverse_mask);

                    let mut word = _mm256_lddqu_si256(slice_ptr.get(0).unwrap().add(i as usize) as *const __m256i);
                    let mut greater = _mm256_and_si256(equal, _mm256_cmpgt_epi8( *target_word.get(0).unwrap(),word));
                    equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word, *target_word.get(0).unwrap()));

                    if num_slice > 1 && !avx_iszero(equal){
                        let word1 = _mm256_lddqu_si256(slice_ptr.get(1).unwrap().add(i as usize) as *const __m256i);
                        // previous equal and current greater, then append (or) to previous greater
                        greater =_mm256_or_si256( greater,_mm256_and_si256(equal,_mm256_cmpgt_epi8( *target_word.get(1).unwrap(),word1)));
                        // current equal and with previous equal
                        equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word1, *target_word.get(1).unwrap()));

                        if num_slice > 2 && !avx_iszero(equal){
                            let word2 = _mm256_lddqu_si256(slice_ptr.get(2).unwrap().add(i as usize) as *const __m256i);
                            // previous equal and current greater, then append (or) to previous greater
                            greater =_mm256_or_si256( greater,_mm256_and_si256(equal,_mm256_cmpgt_epi8(*target_word.get(2).unwrap(),word2)));
                            // current equal and with previous equal
                            equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word2, *target_word.get(2).unwrap()));

                            if num_slice > 3 && !avx_iszero(equal){
                                let word3 = _mm256_lddqu_si256(slice_ptr.get(3).unwrap().add(i as usize) as *const __m256i);
                                // previous equal and current greater, then append (or) to previous greater
                                greater =_mm256_or_si256( greater,_mm256_and_si256(equal,_mm256_cmpgt_epi8( *target_word.get(3).unwrap(),word3)));
                                // current equal and with previous equal
                                equal = _mm256_and_si256(equal,_mm256_cmpeq_epi8(word3, *target_word.get(3).unwrap()));
                            }
                        }
                    }
                    gt_mask = _mm256_movemask_epi8(greater);
                    resv.push(mem::transmute::<i32, u32>(gt_mask));
                }
                else {
                    resv.push(0);
                }
                i += BYTE_WORD;
            }
        }
        res.set_storage(&resv);
        println!("Number of qualified items:{}", res.cardinality());
        return res;
    }


    pub(crate) fn buff_slice_equal_filter(&self, bytes: Vec<u8>, pred:f64) {
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
        let remain =dlen+ilen;
        let num_slice = ceil(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut res = BitVec::from_elem(len as usize, false);
        let mut resv  = Vec::new();
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;

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
                pred_u8= flip((fixed_target << pad) as u8);
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }


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
                resv.push(mem::transmute::<i32, u32>(eq_mask));
                i += BYTE_WORD;
            }
        }
        res.set_storage(&resv);
        println!("Number of qualified items:{}", res.cardinality());
    }



    pub(crate) fn buff_slice_range_filter_nosimd(&self, bytes: Vec<u8>, pred:f64) {
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
        let num_slice = ceil(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        bound.set_length(ilen as u64, dlen as u64);
        let mut res = BitVec::from_elem(len, false);
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);

        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;
        println!("fixed target:{}", fixed_target);
        for cnt in 0..num_slice{
            let chunk = bitpack.read_n_byte_unmut(cnt as usize * len, len).unwrap();
            slice_ptr.push(chunk.as_ptr());
            data.push(chunk);
            let mut pred_u8= 0;
            if (remain-8*cnt>=8){
                pred_u8 =(fixed_target >> (remain - 8 * (1 + cnt))) as u8;
            }
            else {
                let pad = 8*(1+cnt)-remain;
                pred_u8= (fixed_target << pad) as u8;
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }


        let mut i = 0;
        unsafe {
            while i < len {
                let mut tbd = true;
                let mut word = flip(*slice_ptr.get(0).unwrap().add(i));
                if word>*target_byte.get(0).unwrap(){
                    tbd = false;
                    res.set(i,true);
                    i += 1;
                    continue
                }
                else if word<*target_byte.get(0).unwrap(){
                    tbd = false;
                    i += 1;
                    continue
                }

                if num_slice > 1 && tbd{
                    let mut word1= flip(*slice_ptr.get(1).unwrap().add(i));
                    if word1>*target_byte.get(1).unwrap(){
                        tbd = false;
                        res.set(i,true);
                        i += 1;
                        continue
                    }
                    else if word1<*target_byte.get(1).unwrap(){
                        tbd = false;
                        i += 1;
                        continue
                    }

                    if num_slice > 2 && tbd{
                        let mut word2 = flip(*slice_ptr.get(2).unwrap().add(i));
                        if word2>*target_byte.get(2).unwrap(){
                            tbd = false;
                            res.set(i,true);
                            i += 1;
                            continue
                        }
                        else if word2<*target_byte.get(2).unwrap(){
                            tbd = false;
                            i += 1;
                            continue
                        }

                        if num_slice > 3 && tbd{
                            let mut word3 = flip(*slice_ptr.get(3).unwrap().add(i));
                            if word3>*target_byte.get(3).unwrap(){
                                tbd = false;
                                res.set(i,true);
                                i += 1;
                                continue
                            }
                            else if word3<*target_byte.get(3).unwrap(){
                                tbd = false;
                                i += 1;
                                continue
                            }
                        }
                    }
                }
                i += 1;
            }
        }
        println!("Number of qualified items:{}", res.cardinality());
    }


    pub(crate) fn buff_slice_equal_filter_nosimd(&self, bytes: Vec<u8>, pred:f64) {
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
        let remain =dlen+ilen;
        let num_slice = ceil(remain, 8);
        let mut data = Vec::new();
        let mut target_byte = Vec::new();
        let mut target_word = Vec::new();
        let mut slice_ptr = Vec::new();

        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut res = BitVec::from_elem(len as usize, false);
        let mut resv  = Vec::new();
        let target = pred;
        let fixed_part = bound.fetch_fixed_aligned(target);
        if fixed_part<base_int{
            println!("Number of qualified items:{}", len);
            return;
        }
        let fixed_target = (fixed_part-base_int) as u64;

        for cnt in 0..num_slice{
            let chunk = bitpack.read_n_byte_unmut(cnt as usize*len, len).unwrap();
            slice_ptr.push(chunk.as_ptr());
            data.push(chunk);
            let mut pred_u8= 0;
            if (remain-8*cnt>=8){
                pred_u8 =flip((fixed_target >> (remain - 8 * (1 + cnt))) as u8);
            }
            else {
                let pad = 8*(1+cnt)-remain;
                pred_u8= flip((fixed_target << pad) as u8);
            }
            target_byte.push(pred_u8);
            target_word.push(set_pred_word(pred_u8));

        }


        let mut i = 0;
        unsafe {
            while i < len {
                let mut tbd = true;
                let mut word = *slice_ptr.get(0).unwrap().add(i);
                if word!=*target_byte.get(0).unwrap(){
                    tbd = false;
                    i += 1;
                    continue
                }
                if num_slice > 1 && tbd{
                    let mut word1= *slice_ptr.get(1).unwrap().add(i);
                    if word1!=*target_byte.get(1).unwrap(){
                        tbd = false;
                        i += 1;
                        continue
                    }

                    if num_slice > 2 && tbd{
                        let mut word2 = *slice_ptr.get(2).unwrap().add(i);
                        if word2!=*target_byte.get(2).unwrap(){
                            tbd = false;
                            i += 1;
                            continue
                        }

                        if num_slice > 3 && tbd{
                            let mut word3 = *slice_ptr.get(3).unwrap().add(i);
                            if word3!=*target_byte.get(3).unwrap(){
                                tbd = false;
                                i += 1;
                                continue
                            }
                        }
                    }
                }
                res.set(i,true);
                i += 1;
            }
        }
        res.set_storage(&resv);
        println!("Number of qualified items:{}", res.cardinality());
    }
}

impl<'a, T> CompressionMethod<T> for BuffSliceCompress
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
            self.buff_slice_encode(seg);
        }
        let duration = start.elapsed();
        info!("Time elapsed in buff-slice function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[test]
fn test_ceil(){
    let c = ceil(7,3);
    assert_eq!(c, 3);

    let c = ceil(7,8);
    assert_eq!(c,1);


    let avx0 = set_pred_word(0);
    assert!(avx_iszero(avx0));

    let avx1 = set_pred_word(1);
    assert!(!avx_iszero(avx1));

    let test = 4096u32;
    let shuffle_mask: __m256i = unsafe { _mm256_setr_epi64x(SF0,SF1,SF2,SF3) };
    let inverse_mask: __m256i = unsafe { _mm256_set1_epi64x(REVERSE_i64) };

    let expand = inverse_movemask(test,shuffle_mask, inverse_mask);
    println!("{:?}",expand);
    assert_eq!(unsafe { mem::transmute::<u32, i32>(test) }, unsafe { _mm256_movemask_epi8(expand) })

}