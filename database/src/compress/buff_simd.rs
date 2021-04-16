use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::SCALE;
use crate::segment::Segment;
use std::time::{SystemTime, Instant};
use crate::compress::split_double::SplitBDDoubleCompress;
use crate::methods::prec_double::{get_precision_bound, PrecisionBound};
use crate::methods::bit_packing::{BitPack, BYTE_BITS};
use crate::simd::vectorize_query::{range_simd_myroaring, equal_simd_myroaring};
use std::mem;
use log::info;
use croaring::Bitmap;
use myroaring::RoaringBitmap;
use serde::{Serialize, Deserialize};
use crate::compress::buff_slice::{flip, floor, set_pred_word, BYTE_WORD, avx_iszero};
use crate::compress::PRECISION_MAP;
use itertools::Itertools;
use std::ops::{BitAnd, BitOr};
use std::arch::x86_64::{_mm256_lddqu_si256, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_movemask_epi8, __m256i};

pub const SIMD_THRESHOLD:f64 = 0.018;

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
    comp.buff_simd_equal_filter_with_slice(comp_sum,pred);
    let duration5 = start5.elapsed();
    println!("Time elapsed in buff simd sum function() is: {:?}", duration5);

    let start6 = Instant::now();
    // comp.byte_fixed_max(comp_max);
    let duration6 = start6.elapsed();
    println!("Time elapsed in buff simd max function() is: {:?}", duration6);


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
        let mut check_ratio =0.0f64;

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
            rb1.optimize();
            res.optimize();
            check_ratio = rb1.len() as f64/len as f64;
            println!("to check ratio: {}", check_ratio);
        }


        // println!("Number of qualified items in bitmap:{}", rb1.cardinality());

        while (remain>0){
            // if we can read by byte
            if remain>=8{
                remain-=8;
                byte_count+=1;
                if check_ratio>SIMD_THRESHOLD{
                    let mut temp_res = RoaringBitmap::new();
                    let chunk = bitpack.read_n_byte(len as usize).unwrap();
                    cur_tar = (fixed_target >> remain) as u8;
                    let mut eq_bm;
                    unsafe { eq_bm = range_simd_myroaring(chunk, &mut temp_res, cur_tar); }

                    res = res.bitor(temp_res.bitand(&rb1));

                    rb1 = rb1.bitand(eq_bm);

                    rb1.optimize();
                    res.optimize();
                    check_ratio = rb1.len() as f64/len as f64;
                    println!("to check ratio: {}",check_ratio );
                }
                else{
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
                            dec = flip(bitpack.read_byte().unwrap());
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
                            dec = flip(bitpack.read_byte().unwrap());
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
                }

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
        let mut check_ratio =0.0f64;

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
            let start = Instant::now();
            let chunk = bitpack.read_n_byte(len as usize).unwrap();
            dec_byte = (fixed_target >> remain) as u8;
            unsafe { rb1 = equal_simd_myroaring(chunk, dec_byte); }
            rb1.optimize();
            check_ratio =rb1.len() as f64/len as f64;
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
                    unsafe { rb1 = rb1.bitand(equal_simd_myroaring(chunk, dec_byte)); }
                    rb1.optimize();
                    check_ratio =rb1.len() as f64/len as f64;
                    println!("to check ratio: {}", check_ratio);
                }else {
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
                            dec = flip(bitpack.read_byte().unwrap());
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
                            dec = flip(bitpack.read_byte().unwrap());
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
                }

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
        let mut rb1 = RoaringBitmap::new();
        let mut res = RoaringBitmap::new();
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
                rb1.insert_direct_u32(i, mem::transmute::<i32, u32>(eq_mask));
                i += BYTE_WORD;
            }
        }
        rb1.optimize();
        let mut check_ratio = rb1.len() as f64/len as f64;

        remain = remain - 8*num_slice;
        println!("remain: {}", remain);

        if remain>0 && rb1.len()!=0{
            bitpack.skip_n_byte((num_slice * len) as usize);
            dec_byte =(((fixed_target as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
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

        println!("Number of qualified int items:{}", res.len());
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