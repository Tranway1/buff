use crate::segment::Segment;
use std::time::{Instant, SystemTime};
use serde::{Serialize, Deserialize};
use tsz::stream::BufferedWriter;
use crate::methods::prec_double::{get_precision_bound, PrecisionBound, FIRST_ONE};
use crate::methods::bit_packing::{BitPack, BYTE_BITS};
use std::mem;
use crate::methods::compress::{CompressionMethod, SCALE, TEST_FILE};
use croaring::Bitmap;
use tsz::StdEncoder;
use rustfft::num_traits::real::Real;
use log::info;
use std::collections::HashMap;
use crate::client::construct_file_iterator_skip_newline;
use itertools::Itertools;
use nalgebra::max;
use std::iter::FromIterator;
use crate::outlier::MAJOR;
use std::fs::File;
use std::io::{Write};
use crate::compress::PRECISION_MAP;
use std::slice::Iter;
use my_bit_vec::BitVec;

pub const SAMPLE:usize = 2000usize;
pub const OUTLIER_R:f32 = 0.1f32;
pub const MAJOR_R:f32 = 0.6f32; //todo: default 0.6 for buff

#[derive(Clone)]
pub struct SplitBDDoubleCompress {
    chunksize: usize,
    batchsize: usize,
    pub(crate) scale: usize
}

impl SplitBDDoubleCompress {
    pub fn new(chunksize: usize, batchsize: usize, scale: usize) -> Self {
        SplitBDDoubleCompress { chunksize, batchsize, scale}
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

    pub fn offset_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut int_vec = Vec::new();
        let mut dec_vec = Vec::new();

        let mut t:u32 = seg.get_data().len() as u32;
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        println!("precision {}, precision delta:{}", prec, prec_delta);

        let mut bound = PrecisionBound::new(prec_delta);
        // let start1 = Instant::now();
        let dec_len = *(PRECISION_MAP.get(&prec).unwrap()) as u64;
        bound.set_length(0,dec_len);
        let mut min = i64::max_value();
        let mut max = i64::min_value();

        for bd in seg.get_data(){
            let (int_part, dec_part) = bound.fetch_components((*bd).into());
            if int_part<min {
                min = int_part;
            }
            if int_part>max {
                max = int_part;
            }
            int_vec.push(int_part);
            dec_vec.push(dec_part as u32);
        }
        let delta = max-min;
        let base_int = min as i32;
        println!("base integer: {}, max:{}",base_int,max);
        let ubase_int = unsafe { mem::transmute::<i32, u32>(base_int) };
        let base_int64:i64 = base_int as i64;
        let mut single_int = false;
        let mut cal_int_length = 0.0;
        if delta == 0 {
            single_int = true;
        }else {
            cal_int_length = (delta as f64).log2().ceil();
        }

        bound.set_length(cal_int_length as u64, dec_len);
        let ilen = cal_int_length as usize;
        let dlen = dec_len as usize;
        println!("int_len:{},dec_len:{}",cal_int_length as u64,dec_len);
        //let cap = (int_len+dec_len)* t as u64 /8;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(ubase_int,32);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);

        // let duration1 = start1.elapsed();
        // println!("Time elapsed in dividing double function() is: {:?}", duration1);

        // let start1 = Instant::now();

        for i in int_vec{
            bitpack_vec.write((i-base_int64) as u32, ilen).unwrap();
        }

        for d in dec_vec {
            // j += 1;
            bitpack_vec.write(d, dlen).unwrap();
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


    // encoder for V3
    pub fn single_pass_byte_fixed_encode<'a,T>(&self, seg: &mut Segment<T>, mi:i64, ma:i64) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut t:u32 = seg.get_data().len() as u32;
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        println!("precision {}, precision delta:{}", prec, prec_delta);

        let mut bound = PrecisionBound::new(prec_delta);
        // let start1 = Instant::now();
        let dec_len = *(PRECISION_MAP.get(&prec).unwrap()) as u64;
        let mut min = mi;
        let mut max = ma;

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

        let start1 = Instant::now();
        let mut remain = fixed_len;
        let mut bytec = 0;

        if remain<8{
            for bd in seg.get_data(){
                let fixed = bound.fetch_fixed_aligned((*bd).into());
                bitpack_vec.write_bits((fixed-base_fixed64) as u32, remain).unwrap();
            }
            remain = 0;
        }
        else {
            bytec+=1;
            remain -= 8;
            let mut fixed_u64 = Vec::new();
            let mut cur_u64 = 0u64;
            for bd in seg.get_data(){
                let fixed = bound.fetch_fixed_aligned((*bd).into());
                cur_u64 = (fixed-base_fixed64) as u64;
                bitpack_vec.write_byte((cur_u64>>remain) as u8);
                fixed_u64.push(cur_u64);
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

        let duration1 = start1.elapsed();
        println!("Time elapsed in writing double function() is: {:?}", duration1);

        // println!("total number of dec is: {}", j);
        let vec = bitpack_vec.into_vec();


        let origin = t * mem::size_of::<T>() as u32;
        info!("original size:{}", origin);
        info!("compressed size:{}", vec.len());
        let ratio = vec.len() as f64 /origin as f64;
        print!("{}",ratio);
        vec
        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // encoder for buff
    pub fn byte_fixed_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
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
            else if fixed>max {
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


    // profiling frequency of items for each sub-column.
    pub fn byte_level_dist<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut fixed_vec = Vec::new();

        let mut frequency: [u32; 256] = [0; 256];
        let mut index = (0..256).collect_vec();
        let mut bits_u8 = 0;

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
            println!("write the {}th column with {} bits==============",bytec, remain);
            frequency = [0; 256];
            index = (0..256).collect_vec();
            for i in fixed_vec{
                bitpack_vec.write_bits((i-base_fixed64) as u32, remain).unwrap();
                bits_u8 = ((((i-base_fixed64) as u64) << (64 - remain))>> (64-remain)) as usize;
                frequency[bits_u8] += 1;
            }
            let permutation = permutation::sort(&frequency[..]);
            let ordered_index = permutation.apply_slice(&index[..]);
            let ordered_freq = permutation.apply_slice(&frequency[..]);
            for (a,b) in ordered_index.iter().zip(ordered_freq.iter()){
                println!("{}->{}",a,*b as f32/ t as f32)
            }
            remain = 0;
        }
        else {
            bytec+=1;
            remain -= 8;
            println!("write the {}th byte==============",bytec);
            let mut fixed_u64 = Vec::new();
            let mut cur_u64 = 0u64;
            frequency = [0; 256];
            index = (0..256).collect_vec();
            if remain>0{
                // let mut k = 0;
                fixed_u64 = fixed_vec.iter().map(|x|{
                    cur_u64 = (*x-base_fixed64) as u64;
                    bitpack_vec.write_byte((cur_u64>>remain) as u8);
                    bits_u8 = (cur_u64 >> remain) as usize;
                    frequency[bits_u8] += 1;
                    cur_u64
                }).collect_vec();
            }
            else {
                fixed_u64 = fixed_vec.iter().map(|x|{
                    cur_u64 = (*x-base_fixed64) as u64;
                    bitpack_vec.write_byte((cur_u64) as u8);
                    bits_u8 = (cur_u64 as u8) as usize;
                    frequency[bits_u8] += 1;
                    cur_u64
                }).collect_vec();
            }
            let permutation = permutation::sort(&frequency[..]);
            let ordered_index = permutation.apply_slice(&index[..]);
            let ordered_freq = permutation.apply_slice(&frequency[..]);
            for (a,b) in ordered_index.iter().zip(ordered_freq.iter()){
                println!("{}->{}",a,*b as f32/ t as f32)
            }


            while (remain>=8){
                bytec+=1;
                remain -= 8;
                println!("write the {}th byte==============",bytec);
                frequency = [0; 256];
                index = (0..256).collect_vec();frequency = [0; 256];
                index = (0..256).collect_vec();
                if remain>0{
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte((*d >>remain) as u8).unwrap();
                        bits_u8 = ((*d >>remain) as u8) as usize;
                        frequency[bits_u8] += 1;
                    }
                }
                else {
                    for d in &fixed_u64 {
                        bitpack_vec.write_byte(*d as u8).unwrap();
                        bits_u8 = (*d as u8) as usize;
                        frequency[bits_u8] += 1;
                    }
                }

                let permutation = permutation::sort(&frequency[..]);
                let ordered_index = permutation.apply_slice(&index[..]);
                let ordered_freq = permutation.apply_slice(&frequency[..]);
                for (a,b) in ordered_index.iter().zip(ordered_freq.iter()){
                    println!("{}->{}",a,*b as f32/ t as f32)
                }

            }
            if (remain>0){
                bitpack_vec.finish_write_byte();
                println!("write remaining {} bits==============",remain);
                frequency = [0; 256];
                for d in fixed_u64 {
                    bitpack_vec.write_bits(d as u32, remain as usize).unwrap();
                    bits_u8 = ((d << (64-remain))>> (64-remain)) as usize;
                    frequency[bits_u8] += 1;
                }

                let permutation = permutation::sort(&frequency[..]);
                let ordered_index = permutation.apply_slice(&index[..]);
                let ordered_freq = permutation.apply_slice(&frequency[..]);
                for (a,b) in ordered_index.iter().zip(ordered_freq.iter()){
                    println!("{}->{}",a,*b as f32/ t as f32)
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

    // encoder for RAPG
    pub fn byte_residue_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let mut fixed_vec = Vec::new();

        let mut t:u32 = seg.get_data().len() as u32;
        let prec = (self.scale as f32).log10() as i32;
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

        // if more than two bytes for integer section, use category byte compression
        if (ilen>15){
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
            println!("outlier ratio: {}",ol_conter);
            for i in 0..adjusted{
                // println!("{}: {}",i, bits_counter[i]);
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

            let mut outlier = Vec::new();
            if cols_outlier>0{
                // write number of outlier cols
                println!("sub column for outlier: {}",cols_outlier);
                bitpack_vec.write(cols_outlier, 8);
                let mut cur_u64 = 0u64;
                let mut cur_u8 = 0u8;
                let mut fixed_u64 = Vec::new();
                // write the first outlier column
                remain -= 8;
                bytec += 1;
                let mut ind = 0;
                let mut bm_outlier = Bitmap::create();
                fixed_u64 = fixed_vec.iter().map(|x|{
                    cur_u64 = (*x-base_fixed64) as u64;
                    cur_u8 = (cur_u64>>remain) as u8;
                    if cur_u8 != 0{
                        outlier.push(cur_u8);
                        bm_outlier.add(ind);
                    }
                    ind += 1;
                    cur_u64
                }).collect_vec();

                let mut n_outlier = bm_outlier.cardinality() as u32;
                bm_outlier.run_optimize();
                let mut bm_vec = bm_outlier.serialize();
                let mut bm_size = bm_vec.len() as u32;
                println!("bitmap size in {} byte: {}", bytec, bm_size);
                println!("number of outliers in {} byte: {}", bytec, n_outlier);
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
                    ind = 0;
                    remain -= 8;
                    bytec += 1;
                    for d in &fixed_u64 {
                        cur_u8 = (*d >>remain) as u8;
                        if cur_u8 != 0{
                            outlier.push(cur_u8);
                            bm_outlier.add(ind);
                        }
                        ind += 1;
                    }

                    n_outlier = bm_outlier.cardinality() as u32;
                    bm_outlier.run_optimize();
                    bm_vec = bm_outlier.serialize();
                    bm_size = bm_vec.len() as u32;
                    println!("bitmap size in {} byte: {}", bytec, bm_size);
                    println!("number of outliers in {} byte: {}", bytec, n_outlier);
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

        }
        else{
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

    // encoder for RAPG-outlier with single u8 column (micro benchmark for buff outlier)
    pub fn byte_residue_encode_outlieru8<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
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
        major = MAJOR as u8;

        if frequency>-1{
            println!("sparse coding with major item: {} with frequency: {}", major, frequency);
            freq_vec.push(major);
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
            let mut bm_outlier = Bitmap::create();
            fixed_u64 = fixed_vec.iter().map(|x|{
                cur_u64 = (*x-base_fixed64) as u64;
                cur_u8 = (cur_u64>>remain) as u8;
                if cur_u8 != f_item{
                    outlier.push(cur_u8);
                    bm_outlier.add(ind);
                }
                ind += 1;
                cur_u64
            }).collect_vec();

            let mut n_outlier = bm_outlier.cardinality() as u32;
            bm_outlier.run_optimize();
            let mut bm_vec = bm_outlier.serialize();
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
                        bm_outlier.add(ind);
                    }
                    ind += 1;
                }

                n_outlier = bm_outlier.cardinality() as u32;
                bm_outlier.run_optimize();
                bm_vec = bm_outlier.serialize();
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


    pub fn byte_fixed_decode(&self, bytes: Vec<u8>) -> Vec<f64>{
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
        let mut f_cur = 0f64;

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
                    f_cur = (base_int + ((cur_fixed)|(bitpack.read_bits( remain as usize).unwrap() as u64)) as i64) as f64 / dec_scl;
                    // todo: this is for value reconstruction
                    // format!("{:.2}", f_cur);
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


    pub fn byte_residue_decode(&self, bytes: Vec<u8>) -> Vec<f64>{
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
        if ilen>15 {
            let outlier_cols = bitpack.read(8).unwrap();
            println!("number of outlier cols:{}",outlier_cols);

            // if there are outlier cols, then read outlier cols first
            if outlier_cols>0{
                fixed_vec = vec![0u64; len as usize];
                for round in 0..outlier_cols{
                    bytec+=1;
                    remain -= 8;
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}",bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                    bitpack.finish_read_byte();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                    let mut outlier_iter = outliers.iter();
                    let mut outlier_value = outlier_iter.next().unwrap();

                    let mut bm_iterator = bitmap_outlier.iter();
                    println!("bitmap cardinaliry: {}",bitmap_outlier.cardinality());
                    let mut ind = bm_iterator.next().unwrap();

                    let mut i = 0;
                    let mut counter = 0;
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

    pub fn byte_residue_decode_with_precision(&self, bytes: Vec<u8>,precision:u32) -> Vec<f64>{
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

        if precision == 0{
            remain = ilen;
            let mut byte_needed = ilen / 8;
            if ilen % 8 >0{
                byte_needed += 1;
                remain = byte_needed * 8;
            }
        }
        else {
            let bits_needed = *(PRECISION_MAP.get(&(precision as i32)).unwrap()) as u32;
            assert!(dlen >= bits_needed);
            remain = ilen+bits_needed;
            let fixed_byte = (dlen+ilen) / 8;
            let mut byte_needed = (ilen+bits_needed) / 8;
            let extra_bits = (ilen+bits_needed) % 8;
            if byte_needed < fixed_byte {
                if extra_bits > 0 {
                    byte_needed += 1;
                    remain = byte_needed * 8;
                }
            }
            println!("adjusted dec bits to decode:{}", remain);
        }

        // check if there are potential outiler cols
        if ilen>15 {
            let outlier_cols = bitpack.read(8).unwrap();
            println!("number of outlier cols:{}",outlier_cols);

            // if there are outlier cols, then read outlier cols first
            if outlier_cols>0{
                fixed_vec = vec![0u64; len as usize];
                for round in 0..outlier_cols{
                    bytec+=1;
                    remain -= 8;
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}",bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                    bitpack.finish_read_byte();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                    let mut outlier_iter = outliers.iter();
                    let mut outlier_value = outlier_iter.next().unwrap();

                    let mut bm_iterator = bitmap_outlier.iter();
                    println!("bitmap cardinaliry: {}",bitmap_outlier.cardinality());
                    let mut ind = bm_iterator.next().unwrap();

                    let mut i = 0;
                    let mut counter = 0;
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


    // decode compression with majority sparse coding
    pub fn byte_residue_decode_majority(&self, bytes: Vec<u8>) -> Vec<f64>{
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
            let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
            bitpack.finish_read_byte();

            let n_outlier = bitpack.read(32).unwrap() as usize;
            let outliers = bitpack.read_n_byte(n_outlier).unwrap();
            let mut outlier_iter = outliers.iter();
            let mut outlier_value = outlier_iter.next().unwrap();

            let mut bm_iterator = bitmap_outlier.iter();
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
                let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();
                let mut outlier_value = outlier_iter.next().unwrap();

                let mut bm_iterator = bitmap_outlier.iter();
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

    pub fn byte_fixed_sum(&self, bytes: Vec<u8>) -> f64{
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
        sum+= sum_base as f64;
        println!("sum is: {:?}",sum);
        sum
    }


    pub fn byte_residue_sum(&self, bytes: Vec<u8>) -> f64{
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

        if ilen>15{

            let outlier_cols = bitpack.read(8).unwrap();
            println!("number of outlier cols:{}",outlier_cols);

            // if there are outlier cols, then read outlier cols first
            if outlier_cols>0 {
                for round in 0..outlier_cols {
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
        }
        else{
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

    pub fn byte_residue_sum_with_precision(&self, bytes: Vec<u8>,precision:u32) -> f64{
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

        if precision == 0{
            remain = ilen;
            let mut byte_needed = ilen / 8;
            if ilen % 8 >0{
                byte_needed += 1;
                remain = byte_needed * 8;
            }
        }
        else {
            let bits_needed = *(PRECISION_MAP.get(&(precision as i32)).unwrap()) as u32;
            assert!(dlen >= bits_needed);
            remain = ilen+bits_needed;
            let fixed_byte = (dlen+ilen) / 8;
            let mut byte_needed = (ilen+bits_needed) / 8;
            let extra_bits = (ilen+bits_needed) % 8;
            if byte_needed < fixed_byte {
                if extra_bits > 0 {
                    byte_needed += 1;
                    remain = byte_needed * 8;
                }
            }
            println!("adjusted dec bits to decode:{}", remain);
        }

        if ilen>15{

            let outlier_cols = bitpack.read(8).unwrap();
            println!("number of outlier cols:{}",outlier_cols);

            // if there are outlier cols, then read outlier cols first
            if outlier_cols>0 {
                for round in 0..outlier_cols {
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
        }
        else{
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


    pub fn byte_residue_sum_majority(&self, bytes: Vec<u8>) -> f64{
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


    pub(crate) fn byte_fixed_range_filter(&self, bytes: Vec<u8>, pred:f64) {
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
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
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
                    res.add(i);
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
                    res.add(i);
                }
                else if c == cur_tar {
                    rb1.add(i);
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
                let mut cur_rb = Bitmap::create();
                if rb1.cardinality()!=0{
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
                            res.add(dec_cur);
                        }
                        else if dec == cur_tar{
                            cur_rb.add(dec_cur);
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
                            res.add(dec_cur);
                        }
                        else if dec == cur_tar{
                            cur_rb.add(dec_cur);
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
                            res.add(dec_cur);
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
                            res.add(dec_cur);
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


    pub(crate) fn byte_residue_range_filter_majority(&self, bytes: Vec<u8>, pred:f64) {
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
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
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
                let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                res.or_inplace(&bitmap_outlier);
                bitmap_outlier.flip_inplace(0..(len as u64));
                rb1.or_inplace(&bitmap_outlier);
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                bitpack.skip_n_byte(n_outlier).unwrap();

                bitpack.finish_read_byte();
            }
            else{
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                let mut flip_bitmap = bitmap_outlier.flip(0..(len as u64));
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();

                let mut bm_iterator = bitmap_outlier.iter();
                println!("bitmap cardinaliry: {} and hit outliers", bitmap_outlier.cardinality());
                let mut ind = 0;

                for &val in outlier_iter{
                    ind = bm_iterator.next().unwrap();
                    if val==cur_tar{
                        rb1.add(ind);
                    }
                    else if val>cur_tar{
                        res.add(ind);
                    }
                }
                // check if major is also qualified
                if cur_tar<first_major{
                    res.or_inplace(&flip_bitmap);
                }else if cur_tar==first_major {
                    rb1.or_inplace(&flip_bitmap)
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
                    let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                    res.or_inplace(&rb1.and(&bitmap_outlier));
                    bitmap_outlier.flip_inplace(0..(len as u64));
                    rb1.and_inplace(&bitmap_outlier);
                    bitpack.finish_read_byte();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    bitpack.skip_n_byte(n_outlier).unwrap();

                    bitpack.finish_read_byte();
                }
                else {
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}", bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                    let mut flip_bitmap = bitmap_outlier.flip(0..(len as u64));
                    bitpack.finish_read_byte();
                    let mut cur_rb = Bitmap::create();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                    let mut outlier_iter = outliers.iter();

                    let mut bm_iterator = bitmap_outlier.iter();
                    println!("bitmap cardinaliry: {}", bitmap_outlier.cardinality());
                    let mut ind = 0;

                    for &val in outlier_iter{
                        ind = bm_iterator.next().unwrap();
                        if val==cur_tar{
                            cur_rb.add(ind);
                        }
                        else if val>cur_tar{
                            res.add(ind);
                        }
                    }
                    // check if major is also qualified
                    if cur_tar<cur_major{
                        flip_bitmap.and_inplace(&rb1);
                        res.or_inplace(&flip_bitmap);
                    }else if cur_tar==cur_major {
                        cur_rb.or_inplace(&flip_bitmap);
                    }

                    rb1.and_inplace(&cur_rb);


                    bitpack.finish_read_byte();
                }
                // println!("read the {}th byte of outlier, get {} records need futher check",byte_count,rb1.cardinality());
            }

            while (remain>0){
                // if we can read by byte
                if remain>=8{
                    remain-=8;
                    byte_count+=1;
                    let mut cur_rb = Bitmap::create();
                    if rb1.cardinality()!=0{
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
                                res.add(dec_cur);
                            }
                            else if dec == cur_tar{
                                cur_rb.add(dec_cur);
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
                                res.add(dec_cur);
                            }
                            else if dec == cur_tar{
                                cur_rb.add(dec_cur);
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
                                res.add(dec_cur);
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
                                res.add(dec_cur);
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
                for i in 0..len {
                    cur = bitpack.read_bits(remain as usize).unwrap();
                    if cur as u64>fixed_target{
                        res.add(i);
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
                        res.add(i);
                    }
                    else if c == cur_tar {
                        rb1.add(i);
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
                    let mut cur_rb = Bitmap::create();
                    if rb1.cardinality()!=0{
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
                                res.add(dec_cur);
                            }
                            else if dec == cur_tar{
                                cur_rb.add(dec_cur);
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
                                res.add(dec_cur);
                            }
                            else if dec == cur_tar{
                                cur_rb.add(dec_cur);
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
                                res.add(dec_cur);
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
                                res.add(dec_cur);
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

    pub(crate) fn byte_fixed_equal_filter(&self, bytes: Vec<u8>, pred:f64) {
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
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
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
                    res.add(i);
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
                    rb1.add(i);
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
                let mut cur_rb = Bitmap::create();
                if rb1.cardinality()!=0{
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
                            cur_rb.add(dec_cur);
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
                            cur_rb.add(dec_cur);
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
                            res.add(dec_cur);
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
                            res.add(dec_cur);
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

    pub(crate) fn byte_residue_equal_filter_majority(&self, bytes: Vec<u8>, pred:f64) {
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
        let mut remain =dlen+ilen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
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
                let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                bitmap_outlier.flip_inplace(0..(len as u64));
                rb1.or_inplace(&bitmap_outlier);
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                bitpack.skip_n_byte(n_outlier).unwrap();

                bitpack.finish_read_byte();
            }
            else {
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();

                let mut bm_iterator = bitmap_outlier.iter();
                println!("bitmap cardinaliry: {} and hit outliers", bitmap_outlier.cardinality());
                let mut ind = 0;

                for &val in outlier_iter{
                    ind = bm_iterator.next().unwrap();
                    if val==dec_byte{
                        rb1.add(ind);
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
                    let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                    bitmap_outlier.flip_inplace(0..(len as u64));
                    rb1.and_inplace(&bitmap_outlier);
                    bitpack.finish_read_byte();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    bitpack.skip_n_byte(n_outlier).unwrap();

                    bitpack.finish_read_byte();
                }
                else {
                    let bm_len = bitpack.read(32).unwrap() as usize;
                    println!("bitmap length: {}", bm_len);
                    let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                    let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                    bitpack.finish_read_byte();
                    let mut cur_rb = Bitmap::create();

                    let n_outlier = bitpack.read(32).unwrap() as usize;
                    let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                    let mut outlier_iter = outliers.iter();

                    let mut bm_iterator = bitmap_outlier.iter();
                    println!("bitmap cardinaliry: {}", bitmap_outlier.cardinality());
                    let mut ind = 0;

                    for &val in outlier_iter{
                        ind = bm_iterator.next().unwrap();
                        if val==dec_byte{
                            cur_rb.add(ind);
                        }
                    }
                    rb1.and_inplace(&cur_rb);

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
                    let mut cur_rb = Bitmap::create();
                    if rb1.cardinality()!=0{
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
                                cur_rb.add(dec_cur);
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
                                cur_rb.add(dec_cur);
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
                                res.add(dec_cur);
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
                                res.add(dec_cur);
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
                for i in 0..len {
                    cur = bitpack.read_bits(remain as usize).unwrap();
                    if cur as u64==fixed_target{
                        res.add(i);
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
                        rb1.add(i);
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
                    let mut cur_rb = Bitmap::create();
                    if rb1.cardinality()!=0{
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
                                cur_rb.add(dec_cur);
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
                                cur_rb.add(dec_cur);
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
                                res.add(dec_cur);
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
                                res.add(dec_cur);
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

    pub(crate) fn byte_fixed_max(&self, bytes: Vec<u8>) {
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

        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);
        let mut byte_count = 0;
        let mut byte_max = u8::min_value();
        // let start = Instant::now();

        if remain<8{
            for i in 0..len {
                cur = bitpack.read_bits(remain as usize).unwrap() as u64;
                if cur >max{
                    max = cur ;
                    res.clear();
                    res.add(i);
                }
                else if cur == max {
                    res.add(i);
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
                    rb1.add(i);
                }
                else if c == byte_max {
                    rb1.add(i);
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
                            cur_rb.add(dec_cur);
                        }else if dec == byte_max {
                            cur_rb.add(dec_cur);
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
                    delta = dec_cur-dec_pre;
                    if delta != 1 {
                        bitpack.skip(((delta-1) * remain) as usize);
                    }
                    dec = bitpack.read_bits(remain as usize).unwrap();
                    if dec>byte_max{
                        byte_max = dec;
                        res.clear();
                        res.add(dec_cur);
                    }
                    else if dec==byte_max{
                        res.add(dec_cur);
                    }
                    it = iterator.next();
                    dec_pre=dec_cur;
                }
                // println!("read the remain {} bits of dec",remain);
                remain = 0;
                max = max | (byte_max as u64);
            }
        }
        let max_f = (max as i64 +base_int) as f64 / 2.0f64.powi(dlen as i32);
        println!("Number of qualified max items:{}", res.cardinality());
        println!("Max value:{}", max_f);
    }

    pub(crate) fn byte_residue_max_majority(&self, bytes: Vec<u8>) {
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
            let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
            rb1 = bitmap_outlier.flip(0..(len as u64));
            bitpack.finish_read_byte();

            let n_outlier = bitpack.read(32).unwrap() as usize;
            let outliers = bitpack.read_n_byte(n_outlier).unwrap();
            let mut outlier_iter = outliers.iter();

            let mut bm_iterator = bitmap_outlier.iter();
            println!("bitmap cardinaliry: {} and hit outliers", bitmap_outlier.cardinality());
            let mut ind = 0;

            // iterate all outlier cols
            for &val in outlier_iter{
                ind = bm_iterator.next().unwrap();
                if val>byte_max{
                    byte_max = val;
                    rb1.clear();
                    rb1.add(ind);
                }
                else if val==byte_max{
                    rb1.add(ind);
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

                let mut cur_rb = Bitmap::create();
                let bm_len = bitpack.read(32).unwrap() as usize;
                println!("bitmap length: {}", bm_len);
                let bm_mat = bitpack.read_n_byte(bm_len).unwrap();
                let mut bitmap_outlier = Bitmap::deserialize(bm_mat);
                cur_rb = rb1.and(&bitmap_outlier.flip(0..(len as u64)));
                let mut bm_iterator = bitmap_outlier.iter();
                bitpack.finish_read_byte();

                let n_outlier = bitpack.read(32).unwrap() as usize;
                let outliers = bitpack.read_n_byte(n_outlier).unwrap();
                let mut outlier_iter = outliers.iter();


                let max_bitmap = rb1.and(&bitmap_outlier);
                if max_bitmap.cardinality()==0{
                    rb1 = cur_rb;
                    bitpack.finish_read_byte();
                    continue;
                }else{
                    let mut ind = 0;
                    let mut max_iterator = max_bitmap.iter();
                    let mut max_it = max_iterator.next();
                    let mut max_cur = max_it.unwrap();
                    for &val in outlier_iter{
                        ind = bm_iterator.next().unwrap();
                        if ind==max_cur{
                            if val>byte_max{
                                byte_max = val;
                                cur_rb.clear();
                                cur_rb.add(ind);
                            }
                            else if val==byte_max{
                                cur_rb.add(ind);
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
                                cur_rb.add(dec_cur);
                            }else if dec == byte_max {
                                cur_rb.add(dec_cur);
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
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip(((delta-1) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.clear();
                            res.add(dec_cur);
                        }
                        else if dec==byte_max{
                            res.add(dec_cur);
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
                        res.add(i);
                    }
                    else if cur == max {
                        res.add(i);
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
                        rb1.add(i);
                    }
                    else if c == byte_max {
                        rb1.add(i);
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
                                cur_rb.add(dec_cur);
                            }else if dec == byte_max {
                                cur_rb.add(dec_cur);
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
                        delta = dec_cur-dec_pre;
                        if delta != 1 {
                            bitpack.skip(((delta-1) * remain) as usize);
                        }
                        dec = bitpack.read_bits(remain as usize).unwrap();
                        if dec>byte_max{
                            byte_max = dec;
                            res.clear();
                            res.add(dec_cur);
                        }
                        else if dec==byte_max{
                            res.add(dec_cur);
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


}

impl<'a, T> CompressionMethod<T> for SplitBDDoubleCompress
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
            self.offset_encode(seg);
        }
        let duration = start.elapsed();
        info!("Time elapsed in splitBD function() is: {:?}", duration);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[test]
fn test_sub_column_profiler() {
    let file_iter = construct_file_iterator_skip_newline::<f64>("/home/cc/float_comp/signal/time_series_120rpm-c2-current.csv", 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,100000);
    let start1 = Instant::now();
    let mut compressed = comp.byte_level_dist(&mut seg);
    let duration1 = start1.elapsed();
}

#[test]
fn test_sort_by_array(){
    let names = vec!["Bob", "Steve", "Jane"];
    let salary = vec![10, 5, 15];
    let mut freq  = [0; 256];
    let mut index = (0..256).collect_vec();
    freq[0]+=255;
    println!("first item:{} ", freq[0]);
    let permutation = permutation::sort(&freq[..]);
    let ordered_index = permutation.apply_slice(&index[..]);
    let ordered_freq = permutation.apply_slice(&freq[..]);
    println!("first item:{} ", ordered_freq[255]);
}

#[test]
fn test_vec_array() {
    let length = 1000000;

    let start = Instant::now();
    let mut bitpack0 = BitPack::<Vec<u8>>::with_capacity(8);
    let mut vec:Vec<usize> = Vec::new();
    for i in 0..length{
        vec.push(i);
    }

    for j in vec.iter(){
        bitpack0.write_byte((*j >> 8) as u8) ;
    }
    for j in vec.iter(){
        bitpack0.write_byte(*j as u8) ;
    }
    let bytes = bitpack0.as_slice();
    println!("{}", bytes.len());

    let duration = start.elapsed();
    println!("Time elapsed in sum 100 million usize vec is: {:?}", duration);



    let start = Instant::now();
    let mut bitpack1 = BitPack::<Vec<u8>>::with_capacity(8);
    let mut bitpack2 = BitPack::<Vec<u8>>::with_capacity(8);
    let mut out_vec = Vec::new();
    out_vec.push(&mut bitpack1);
    out_vec.push(&mut bitpack2);
    let mut shared = out_vec.as_mut_slice();


    let mut vec:Vec<usize> = Vec::new();
    for i in 0..length{
        vec.push(i);
    }

    let mut cur = 0usize;
    for j in vec.iter(){
        cur = *j;
        let mut rsh = 0;
        for bp in shared.iter_mut(){
            cur = cur>>rsh;
            (*bp).write_byte(cur as u8);
            rsh+=8;
        }
    }

    let mut res:Vec<u8> = Vec::new();
    res.append(&mut bitpack1.into_vec());
    res.append(&mut bitpack2.into_vec());

    println!("{}", res.len());

    let duration = start.elapsed();
    println!("Time elapsed in sum 100 million usize vec is: {:?}", duration);


    let start = Instant::now();
    let mut vec:Vec<usize> = Vec::new();
    for i in 0..length{
        vec.push(i);
    }

    let mut sum = 0;
    for j in vec.iter(){
        sum+=*j
    }
    println!("{}", sum);
    let duration = start.elapsed();
    println!("Time elapsed in sum 100 million usize iter is: {:?}", duration);

    let start = Instant::now();
    let mut vec:Vec<usize> = Vec::new();
    for i in 0..length{
        vec.push(i);
    }

    let mut sum = 0;
    for j in vec{
        sum+=j
    }
    println!("{}", sum);
    let duration = start.elapsed();
    println!("Time elapsed in sum 100 million usize vec is: {:?}", duration);

    let start = Instant::now();
    let mut vec:Vec<usize> = Vec::new();
    for i in 0..length{
        vec.push(i);
    }

    let sum = vec.iter().fold(0,|acc,&x| acc+x);
    println!("{}", sum);
    let duration = start.elapsed();
    println!("Time elapsed in sum 100 million usize vec is: {:?}", duration);
}
