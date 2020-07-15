use crate::segment::Segment;
use std::time::{Instant, SystemTime};
use serde::{Serialize, Deserialize};
use tsz::stream::BufferedWriter;
use crate::methods::prec_double::{get_precision_bound, PrecisionBound, FIRST_ONE};
use crate::methods::bit_packing::{BitPack, BYTE_BITS};
use std::mem;
use crate::methods::compress::{CompressionMethod, SCALE};
use croaring::Bitmap;
use tsz::StdEncoder;
use rustfft::num_traits::real::Real;
use log::info;
use std::collections::HashMap;
use packed_simd::{u8x8,u8x16,u8x32};
use crate::client::construct_file_iterator_skip_newline;


lazy_static! {
    static ref PRECISION_MAP: HashMap<i32, i32> =[(1, 5),
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

#[derive(Clone)]
pub struct SplitBDDoubleCompress {
    chunksize: usize,
    batchsize: usize,
    scale: usize
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

    pub fn byte_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
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
            dec_vec.push(dec_part as u64);
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
        let mut remain = dec_len;
        let mut bytec = 0;

        if (remain>=8){
            bytec+=1;
            remain -= 8;
            if remain>0{
                // let mut k = 0;
                for d in &dec_vec {
                    // if k<10{
                    //     println!("write {}th value {} with first byte {}",k, *d, (*d >> remain))
                    // }
                    // k+=1;
                    bitpack_vec.write_byte((*d >> remain) as u8).unwrap();
                }
            }
            else {
                for d in &dec_vec {
                    bitpack_vec.write_byte(*d as u8).unwrap();
                }
            }

            println!("write the {}th byte of dec",bytec);
        }
        while (remain>=8){
            bytec+=1;
            remain -= 8;
            if remain>0{
                for d in &dec_vec {
                    bitpack_vec.write_byte((*d >>remain) as u8).unwrap();
                }
            }
            else {
                for d in &dec_vec {
                    bitpack_vec.write_byte(*d as u8).unwrap();
                }
            }


            println!("write the {}th byte of dec",bytec);
        }
        if (remain>0){
            bitpack_vec.finish_write_byte();
            for d in dec_vec {
                bitpack_vec.write_bits(d as u32, remain as usize).unwrap();
            }
            println!("write remaining {} bits of dec",remain);
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

    fn fast_encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        // 1482892260 is the Unix timestamp of the start of the stream
        let mut encoder = StdEncoder::new(0, w);

        let mut dec_vec = Vec::new();

        let mut t:u32 =0;
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bound = PrecisionBound::new(prec_delta);
        let start = Instant::now();
        t = seg.get_data().len() as u32;
        let duration = start.elapsed();
        println!("Time elapsed in bound double function() is: {:?}", duration);
        let start1 = Instant::now();
        // let (int_len,dec_len) = bound.get_length();
        let (int_len,dec_len) = (4u64,18u64);
        bound.set_length(int_len,dec_len);
        let ilen = int_len as usize;
        let dlen = dec_len as usize;
        println!("int_len:{},dec_len:{}",int_len,dec_len);
        //let cap = (int_len+dec_len)* t as u64 /8;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        bitpack_vec.write(t, 32);
        bitpack_vec.write(ilen as u32, 32);
        bitpack_vec.write(dlen as u32, 32);
        let mut i =0;
        for bd in seg.get_data(){
            let (int_part, dec_part) = bound.fetch_components((*bd).into());
            // if i<10{
            //     println!("cur: {}",(*bd).into());
            //     println!("{}th, int: {}, decimal: {} in form:{}",i,int_part,dec_part as f64*1.0f64/(2i64.pow(dec_len as u32)) as f64, dec_part);
            // }
            // i+=1;
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

    pub fn byte_decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
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
        let mut int_vec:Vec<i32> = Vec::new();
        let mut dec_vec = Vec::new();
        let mut cur_intf = 0f64;

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            int_vec.push(cur as i32 + base_int);
        }

        let mut dec = 0;
        let dec_scl:f64 = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        let mut remain = dlen;
        let mut bytec = 0;
        let mut chunk;

        if (remain>=8){
            bytec+=1;
            remain -= 8;
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            if remain == 0 {
                for (int_comp,dec_comp) in int_vec.iter().zip(chunk.iter()){
                    cur_intf = *int_comp as f64;
                    expected_datapoints.push(cur_intf + (((*dec_comp)) as f64) / dec_scl);
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
                    dec_vec.push(((*x) as u32)<<remain)
                }
            }
            println!("read the {}th byte of dec",bytec);
        }
        while (remain>=8){
            bytec+=1;
            remain -= 8;
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            if remain == 0 {
                // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                for (int_comp,dec_comp,cur_chunk) in izip!(&int_vec,&dec_vec,chunk){
                    cur_intf = *int_comp as f64;
                    // if j<10{
                    //     println!("{}th item {}, decimal:{}",j, cur_f,*dec_comp);
                    // }
                    // j += 1;
                    expected_datapoints.push( cur_intf + (((*dec_comp)|((*cur_chunk) as u32)) as f64) / dec_scl);
                }
            }
            else{
                let mut it = chunk.into_iter();
                dec_vec=dec_vec.into_iter().map(|x| x|((*(it.next().unwrap()) as u32)<<remain)).collect();
            }

            println!("read the {}th byte of dec",bytec);
        }
        if (remain>0){
            // let mut j =0;
            bitpack.finish_read_byte();
            println!("read remaining {} bits of dec",remain);
            println!("length for int:{} and length for dec: {}", int_vec.len(),dec_vec.len());
            for (int_comp,dec_comp) in int_vec.iter().zip(dec_vec.iter()){
                cur_intf = *int_comp as f64;
                // let cur_f = cur_intf + (((*dec_comp)|(bitpack.read_bits( remain as usize).unwrap() as u32)) as f64) / dec_scl;
                // if j<20{
                //     println!("{}th item {}, decimal:{}",j, cur_f,*dec_comp);
                // }
                // j += 1;
                expected_datapoints.push( cur_intf + (((*dec_comp)|(bitpack.read_bits( remain as usize).unwrap() as u32)) as f64) / dec_scl);
            }
        }

        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }


    pub fn sum_with_precision(&self, bytes: Vec<u8>, precision:u32) -> f64{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;

        let mut remain = dlen;
        let mut processed = 0;
        let mut sum_int:i64 = (len as i32 * base_int) as i64;
        let mut sum_dec:u64 = 0;
        let mut sum = 0.0f64;

        if precision == 0{
            for i in 0..len {
                cur = bitpack.read(ilen as usize).unwrap();
                sum_int+= cur as i64;
            }
            sum =sum_int as f64;
            println!("sum of integer part:{}", sum);
            return sum
        }
        else {
            let bits_needed = *(PRECISION_MAP.get(&(precision as i32)).unwrap()) as u32;
            assert!(dlen>=bits_needed);
            remain = bits_needed;
            let dec_byte = dlen/8;
            let mut byte_needed = bits_needed/8;
            let extra_bits = bits_needed%8;
            if byte_needed<dec_byte{
                if extra_bits>0{
                    byte_needed += 1;
                    remain = byte_needed*8;
                }
            }
            println!("adjusted dec bits to decode:{}",remain);

            for i in 0..len {
                cur = bitpack.read(ilen as usize).unwrap();
                sum_int+= cur as i64;
            }
            sum =sum_int as f64;
            println!("sum of integer part:{}", sum);

            let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
            // println!("Scale for decimal:{}", dec_scl);

            let mut bytec = 0;
            let mut chunk;

            if (remain>=8){
                bytec+=1;
                remain -= 8;
                processed += 8;
                dec_scl = 2.0f64.powi(processed);
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                for dec_comp in chunk.iter(){
                    sum_dec += (*dec_comp) as u64;
                }
                sum = sum+(sum_dec as f64)/dec_scl;
                sum_dec=0;
                println!("sum the {}th byte of dec",bytec);
                println!("now sum :{}", sum);
                if remain == 0 {
                    return sum
                }
            }
            while (remain>=8){
                bytec+=1;
                remain -= 8;
                processed += 8;
                dec_scl = 2.0f64.powi(processed);
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                for dec_comp in chunk.iter(){
                    sum_dec += (*dec_comp) as u64;
                }
                sum = sum+(sum_dec as f64)/dec_scl;
                sum_dec=0;
                println!("sum the {}th byte of dec",bytec);
                println!("now sum :{}", sum);
                if remain == 0 {
                    return sum
                }
            }
            if (remain>0){
                // let mut j =0;
                bitpack.finish_read_byte();
                println!("sum remaining {} bits of dec",remain);
                processed += remain as i32;
                dec_scl = 2.0f64.powi(processed);
                for i in 0..len {
                    sum_dec += (bitpack.read_bits( remain as usize).unwrap() as u64);
                }
                sum = sum+(sum_dec as f64)/dec_scl;
            }
            println!("final sum :{}", sum);
            sum
        }
    }

    pub fn byte_sum(&self, bytes: Vec<u8>) -> f64{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;

        let mut remain = dlen;
        let mut processed = 0;
        let mut sum_int:i64 = (len as i32 * base_int) as i64;
        let mut sum_dec:u64 = 0;
        let mut sum = 0.0f64;

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            sum_int+= cur as i64;
        }
        sum =sum_int as f64;
        println!("sum of integer part:{}", sum);

        let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
        // println!("Scale for decimal:{}", dec_scl);

        let mut bytec = 0;
        let mut chunk;

        if (remain>=8){
            bytec+=1;
            remain -= 8;
            processed += 8;
            dec_scl = 2.0f64.powi(processed);
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            for dec_comp in chunk.iter(){
                sum_dec += (*dec_comp) as u64;
            }
            sum = sum+(sum_dec as f64)/dec_scl;
            sum_dec=0;
            println!("sum the {}th byte of dec",bytec);
            println!("now sum :{}", sum);
            if remain == 0 {
                return sum
            }
        }
        while (remain>=8){
            bytec+=1;
            remain -= 8;
            processed += 8;
            dec_scl = 2.0f64.powi(processed);
            chunk = bitpack.read_n_byte(len as usize).unwrap();
            for dec_comp in chunk.iter(){
                sum_dec += (*dec_comp) as u64;
            }
            sum = sum+(sum_dec as f64)/dec_scl;
            sum_dec=0;
            println!("sum the {}th byte of dec",bytec);
            println!("now sum :{}", sum);
            if remain == 0 {
                return sum
            }
        }
        if (remain>0){
            // let mut j =0;
            bitpack.finish_read_byte();
            println!("sum remaining {} bits of dec",remain);
            processed += remain as i32;
            dec_scl = 2.0f64.powi(processed);
            for i in 0..len {
                sum_dec += (bitpack.read_bits( remain as usize).unwrap() as u64);
            }
            sum = sum+(sum_dec as f64)/dec_scl;
        }
        println!("sum is: {:?}",sum);
        sum
    }

    pub fn decode_with_precision(&self, bytes: Vec<u8>, precision:u32) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
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
        let mut int_vec:Vec<i32> = Vec::new();
        let mut dec_vec = Vec::new();
        let mut cur_intf = 0f64;
        let mut remain = dlen;
        let mut processed = 0;

        if precision == 0{
            for i in 0..len {
                cur = bitpack.read(ilen as usize).unwrap();
                expected_datapoints.push(cur as f64);
            }
            println!("Number of precision scan items:{}", expected_datapoints.len());
            expected_datapoints
        }
        else {
            let bits_needed = *(PRECISION_MAP.get(&(precision as i32)).unwrap()) as u32;
            assert!(dlen>=bits_needed);
            remain = bits_needed;
            let dec_byte = dlen/8;
            let mut byte_needed = bits_needed/8;
            let extra_bits = bits_needed%8;
            if byte_needed<dec_byte{
                if extra_bits>0{
                    byte_needed += 1;
                    remain = byte_needed*8;
                }
            }
            println!("adjusted dec bits to decode:{}",remain);

            for i in 0..len {
                cur = bitpack.read(ilen as usize).unwrap();
                int_vec.push(cur as i32 + base_int);
            }

            let mut dec = 0;
            let mut dec_scl:f64 = 2.0f64.powi(dlen as i32);
            println!("Scale for decimal:{}", dec_scl);

            let mut bytec = 0;
            let mut chunk;

            if (remain>=8){
                bytec+=1;
                remain -= 8;
                processed += 8;
                dec_scl = 2.0f64.powi(processed);
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                if remain == 0 {
                    for (int_comp,dec_comp) in int_vec.iter().zip(chunk.iter()){
                        cur_intf = *int_comp as f64;
                        //todo: this is problemetic.
                        expected_datapoints.push(cur_intf + (((*dec_comp)) as f64) / dec_scl);
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
                        dec_vec.push(((*x) as u32)<<remain)
                    }
                }
                println!("read the {}th byte of dec",bytec);
            }
            while (remain>=8){
                bytec+=1;
                remain -= 8;
                processed += 8;
                dec_scl = 2.0f64.powi(processed);
                chunk = bitpack.read_n_byte(len as usize).unwrap();
                if remain == 0 {
                    // dec_vec=dec_vec.into_iter().map(|x| x|(bitpack.read_byte().unwrap() as u32)).collect();
                    for (int_comp,dec_comp,cur_chunk) in izip!(&int_vec,&dec_vec,chunk){
                        cur_intf = *int_comp as f64;
                        // if j<10{
                        //     println!("{}th item {}, decimal:{}",j, cur_f,*dec_comp);
                        // }
                        // j += 1;
                        expected_datapoints.push( cur_intf + (((*dec_comp)|((*cur_chunk) as u32)) as f64) / dec_scl);
                    }
                }
                else{
                    let mut it = chunk.into_iter();
                    dec_vec=dec_vec.into_iter().map(|x| x|((*(it.next().unwrap()) as u32)<<remain)).collect();
                }

                println!("read the {}th byte of dec",bytec);
            }
            if (remain>0){
                // let mut j =0;
                bitpack.finish_read_byte();
                println!("read remaining {} bits of dec",remain);
                println!("length for int:{} and length for dec: {}", int_vec.len(),dec_vec.len());
                processed += remain as i32;
                dec_scl = 2.0f64.powi(processed);
                for (int_comp,dec_comp) in int_vec.iter().zip(dec_vec.iter()){
                    cur_intf = *int_comp as f64;
                    // if j<10{
                    //     println!("{}th item {}, decimal:{}",j, cur_f,*dec_comp);
                    // }
                    // j += 1;
                    expected_datapoints.push( cur_intf + (((*dec_comp)|(bitpack.read_bits( remain as usize).unwrap() as u32)) as f64) / dec_scl);
                }
            }
            println!("Number of precision scan items:{}", expected_datapoints.len());
            expected_datapoints
        }
    }

    pub fn decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
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
        let mut int_vec:Vec<i32> = Vec::new();

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            int_vec.push(cur as i32 + base_int);
        }

        let mut dec = 0;
        let dec_scl = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);
        let mut j = 0;
        let mut cur_intf = 0f64;
        for int_comp in int_vec{
            cur_intf = int_comp as f64;
            dec = bitpack.read(dlen as usize).unwrap();
            // if j<10{
            //     println!("{}th item {}, decimal:{}",j, cur_intf + (dec as f64) / dec_scl,dec);
            // }
            // j += 1;
            expected_datapoints.push(cur_intf + (dec as f64) / dec_scl);
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    pub fn sum(&self, bytes: Vec<u8>) -> f64{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;

        let mut int_sum:i64 =  len as i64 * base_int as i64;
        let mut dec_sum = 0;

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            int_sum = int_sum + cur as i64;
        }
        println!("sum of integer part:{}", int_sum);

        let mut dec = 0;
        let dec_scl = 2.0f64.powi(dlen as i32);
        println!("Scale for decimal:{}", dec_scl);

        for j in 0..len{
            dec = bitpack.read(dlen as usize).unwrap();
            dec_sum+=dec as u64;
        }
        let sum = int_sum as f64 + (dec_sum as f64) / dec_scl;
        println!("int sum: {}, f64 sum: {}", dec_sum, dec_sum as f64);
        println!("sum of column:{}", sum);
        sum
    }

    pub(crate) fn bit_decode(&self, bytes: Vec<u8>) -> Vec<f64>{
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
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
        let mut int_vec:Vec<i32> = Vec::new();

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            int_vec.push(cur as i32 + base_int);
        }

        let mut dec = 0;
        // let dec_scl:f64 = 2.0f64.powi(dlen as i32);
        // println!("Scale for decimal:{}", dec_scl);
        let mut j = 0;
        let mut cur_u64 = 0u64;
        let mut sign = 0;
        let mut sign_u64 = 0u64;
        let mut cur_f = 0f64;
        let mut int_abs = 0;
        let mut leading_zero = 0;
        let mut exp = 0;
        let mut exp_bin = 0u64;
        let dec_lf = (32 - dlen) as u64;
        for int_comp in int_vec{
            dec = bitpack.read(dlen as usize).unwrap() as u64;
            dec = dec << dec_lf;
            sign_u64 =unsafe { mem::transmute::<i64, u64>(int_comp as i64) };
            sign_u64 = sign_u64 & FIRST_ONE;
            int_abs = int_comp.abs() as u64;
            cur_u64 = int_abs << 32;
            cur_u64 = cur_u64 | dec;
            leading_zero = cur_u64.leading_zeros();
            exp = 31i32-leading_zero as i32;
            exp_bin = (exp + 1023i32) as u64;
            exp_bin = exp_bin << 52;
            exp_bin = exp_bin | sign_u64;
            cur_u64 = cur_u64 << (leading_zero+1);
            cur_u64 = cur_u64 >> 12;
            cur_u64 = cur_u64 | exp_bin;
            cur_f = unsafe { mem::transmute::<u64, f64>(cur_u64) };
            // if j<10{
            //     println!("{}th item {}, with int part:{}, exp:{}",j, cur_f,int_abs,exp_bin);
            // }
            // j += 1;
            expected_datapoints.push(cur_f );
            cur_u64 = 0u64;
        }
        println!("Number of scan items:{}", expected_datapoints.len());
        expected_datapoints
    }

    pub(crate) fn range_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        let target = pred;
        let (int_part, dec_part) = bound.fetch_components(target);
        if int_part<base_int as i64{
            println!("Number of qualified items:{}", len);
            return;
        }
        let int_target = (int_part-base_int as i64) as u32;
        let dec_target = dec_part as u32;
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);

        // let start = Instant::now();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            // if i<10{
            //     println!("{}th value: {}",i,cur);
            // }
            if cur>int_target{
                res.add(i);
            }
            else if cur == int_target {
                rb1.add(i);
            };
        }
        // rb1.run_optimize();
        // res.run_optimize();
        // let duration = start.elapsed();
        // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
        // println!("Number of qualified int items:{}", res.cardinality());
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
                    bitpack.skip(((dec_cur) * dlen) as usize);
                }
                dec = bitpack.read(dlen as usize).unwrap();
                if dec>dec_target{
                    res.add(dec_cur);
                }
                // println!("index qualified {}, decimal:{}",dec_cur,dec);
                it = iterator.next();
                dec_pre = dec_cur;
            }
            while it!=None{
                dec_cur = it.unwrap();
                //println!("index qualified {}",dec_cur);
                delta = dec_cur-dec_pre;
                if delta != 1 {
                    bitpack.skip(((delta-1) * dlen) as usize);
                }
                dec = bitpack.read(dlen as usize).unwrap();
                // if dec_cur<10{
                //     println!("index qualified {}, decimal:{}",dec_cur,dec);
                // }
                if dec>dec_target{
                    res.add(dec_cur);
                }
                it = iterator.next();
                dec_pre=dec_cur;
            }
            // let duration = start.elapsed();
            // println!("Time elapsed in splitBD filtering fraction part is: {:?}", duration);
        }
        println!("Number of qualified items:{}", res.cardinality());
    }

    pub(crate) fn equal_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        let target = pred;
        let (int_part, dec_part) = bound.fetch_components(target);
        if int_part<base_int as i64{
            println!("Number of qualified items for equal:{}", 0);
            return;
        }
        let int_target = (int_part-base_int as i64) as u32;
        let dec_target = dec_part as u32;
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);

        // let start = Instant::now();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            if cur == int_target {
                rb1.add(i);
            };
        }
        // rb1.run_optimize();
        // let duration = start.elapsed();
        // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
        println!("Number of qualified int items:{}", res.cardinality());
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
                    bitpack.skip(((dec_cur) * dlen) as usize);
                }
                dec = bitpack.read(dlen as usize).unwrap();
                if dec==dec_target{
                    res.add(dec_cur);
                }
                // println!("index qualified {}, decimal:{}",dec_cur,dec);
                it = iterator.next();
                dec_pre = dec_cur;
            }
            while it!=None{
                dec_cur = it.unwrap();
                //println!("index qualified {}",dec_cur);
                delta = dec_cur-dec_pre;
                if delta != 1 {
                    bitpack.skip(((delta-1) * dlen) as usize);
                }
                dec = bitpack.read(dlen as usize).unwrap();
                // if dec_cur<10{
                //     println!("index qualified {}, decimal:{}",dec_cur,dec);
                // }
                if dec==dec_target{
                    res.add(dec_cur);
                }
                it = iterator.next();
                dec_pre=dec_cur;
            }
            // let duration = start.elapsed();
            // println!("Time elapsed in splitBD filtering fraction part is: {:?}", duration);
        }
        println!("Number of qualified items for equal:{}", rb1.cardinality());
    }

    pub(crate) fn byte_range_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain = dlen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        let target = pred;
        let (int_part, dec_part) = bound.fetch_components(target);
        if int_part<base_int as i64{
            println!("Number of qualified items:{}", len);
            return;
        }
        let int_target = (int_part-base_int as i64) as u32;
        let mut dec_byte = dec_part as u8;

        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            if cur>int_target{
                res.add(i);
            }
            else if cur == int_target {
                rb1.add(i);
            };
        }
        // println!("Number of qualified items in bitmap:{}", rb1.cardinality());
        let mut byte_count = 0;
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
                    dec_byte = (dec_part >> remain as u64) as u8;
                    if it!=None{
                        dec_cur = it.unwrap();
                        if dec_cur!=0{
                            bitpack.skip_n_byte((dec_cur) as usize);
                        }
                        dec = bitpack.read_byte().unwrap();
                        // println!("{} first match: {}", dec_cur, dec);
                        if dec>dec_byte{
                            res.add(dec_cur);
                        }
                        else if dec == dec_byte{
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
                        if dec>dec_byte{
                            res.add(dec_cur);
                        }
                        else if dec == dec_byte{
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
                dec_byte =(((dec_part as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
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
                        if dec>dec_byte{
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
                        if dec>dec_byte{
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

    pub(crate) fn byte_equal_filter(&self, bytes: Vec<u8>, pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);

        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        let mut remain =dlen;
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut cur;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        let target = pred;
        let (int_part, dec_part) = bound.fetch_components(target);
        if int_part<base_int as i64{
            println!("Number of qualified items for equal:{}", 0);
            return;
        }
        let int_target = (int_part-base_int as i64) as u32;
        let mut dec_byte = dec_part as u8;
        // println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);

        // let start = Instant::now();
        for i in 0..len {
            cur = bitpack.read(ilen as usize).unwrap();
            if cur == int_target {
                rb1.add(i);
            };
        }
        // rb1.run_optimize();
        // let duration = start.elapsed();
        // println!("Time elapsed in splitBD filtering int part is: {:?}", duration);
        println!("Number of qualified items for equal:{}", rb1.cardinality());
        let mut byte_count = 0;
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
                    dec_byte = (dec_part >> remain as u64) as u8;
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
                dec_byte =(((dec_part as u8)<< ((BYTE_BITS - remain as usize) as u8)) >> ((BYTE_BITS - remain as usize) as u8));
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

    pub(crate) fn simd_range_filter(&self, bytes: Vec<u8>,pred:f64) {
        let prec = (self.scale as f32).log10() as i32;
        let prec_delta = get_precision_bound(prec);
        let mut bitpack = BitPack::<&[u8]>::new(bytes.as_slice());
        let mut bound = PrecisionBound::new(prec_delta);
        let ubase_int = bitpack.read(32).unwrap();
        let base_int = unsafe { mem::transmute::<u32, i32>(ubase_int) };
        println!("base integer: {}",base_int);
        let len = bitpack.read(32).unwrap();
        println!("total vector size:{}",len);
        let ilen = bitpack.read(32).unwrap();
        println!("bit packing length:{}",ilen);
        let dlen = bitpack.read(32).unwrap();
        bound.set_length(ilen as u64, dlen as u64);
        // check integer part and update bitmap;
        let mut rb1 = Bitmap::create();
        let mut res = Bitmap::create();
        let target = pred;
        let (int_part, dec_part) = bound.fetch_components(target);
        let int_target = (int_part-base_int as i64) as u32;
        let dec_target = dec_part as u32;
        println!("target value with integer part:{}, decimal part:{}",int_target,dec_target);

        let mut int_vec:Vec<u8> = Vec::new();

        let start = Instant::now();
        for i in 0..len {
            int_vec.push(bitpack.read(ilen as usize).unwrap() as u8);
        }
        let lane = 16;
        assert!(int_vec.len() % lane == 0);
        let mut pre_vec = u8x16::splat(int_target as u8);
        for i in (0..int_vec.len()).step_by(lane) {
            let cur_word = u8x16::from_slice_unaligned(&int_vec[i..]);
            let m = cur_word.gt(pre_vec);
            for j in 0..lane{
                if m.extract(j){
                    res.add((i + j) as u32);
                }
            }
            let m = cur_word.eq(pre_vec);
            for j in 0..lane{
                if m.extract(j){
                    rb1.add((i + j) as u32);
                }
            }
        }


        // rb1.run_optimize();
        // res.run_optimize();

        let duration = start.elapsed();
        println!("Time elapsed in splitBD simd filtering int part is: {:?}", duration);
        println!("Number of qualified int items:{}", res.cardinality());

        let start = Instant::now();
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
                bitpack.skip(((dec_cur) * dlen) as usize);
            }
            dec = bitpack.read(dlen as usize).unwrap();
            if dec>dec_target{
                res.add(dec_cur);
            }
            // println!("index qualified {}, decimal:{}",dec_cur,dec);
            it = iterator.next();
            dec_pre = dec_cur;
        }
        while it!=None{
            dec_cur = it.unwrap();
            //println!("index qualified {}",dec_cur);
            delta = dec_cur-dec_pre;
            if delta != 1 {
                bitpack.skip(((delta-1) * dlen) as usize);
            }
            dec = bitpack.read(dlen as usize).unwrap();
            // if dec_cur<10{
            //     println!("index qualified {}, decimal:{}",dec_cur,dec);
            // }
            if dec>dec_target{
                res.add(dec_cur);
            }
            it = iterator.next();
            dec_pre=dec_cur;
        }
        let duration = start.elapsed();
        println!("Time elapsed in splitBD simd filtering fraction part is: {:?}", duration);
        println!("Number of qualified items:{}", res.cardinality());
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

// #[test]
// fn splitbd_decode_with_precision() {
//
// }