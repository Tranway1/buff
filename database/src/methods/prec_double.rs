use log::{warn,info};
use std::mem;
use crate::methods::bit_packing::BitPack;
use crate::client::construct_file_iterator_skip_newline;
use std::time::{SystemTime, Instant};
use crate::segment::Segment;
use std::fs::File;
use std::io::{LineWriter, Write};
use croaring::Bitmap;
use rust_decimal::prelude::*;
use histogram::Histogram;

/// END_MARKER is a special bit sequence used to indicate the end of the stream
pub const EXP_MASK: u64 = 0b0111111111110000000000000000000000000000000000000000000000000000;
pub const FIRST_ONE: u64 = 0b1000000000000000000000000000000000000000000000000000000000000000;
pub const NEG_ONE: u64 = 0b1111111111111111111111111111111111111111111111111111111111111111;

pub struct PrecisionBound {
    position: u64,
    precision: f64,
    precision_exp : i32,
    int_length: u64,
    decimal_length: u64,
}


impl PrecisionBound {
    pub fn new(precision:f64) -> Self {
        let mut e = PrecisionBound { position: 0, precision: precision, precision_exp : 0, int_length: 0, decimal_length: 0 };
        let xu = unsafe { mem::transmute::<f64, u64>(precision) };
        e.precision_exp = ((xu & EXP_MASK) >> 52) as i32 - 1023 as i32;
        e
    }


    pub fn precision_bound(&mut self, orig: f64)-> f64{
        let a = 0u64;
        let mut mask = !a;
        let mut ret = 0f64;
        let mut pre = orig;
        let mut cur = 0f64;
        let origu = unsafe { mem::transmute::<f64, u64>(orig) };
        let mut curu = 0u64;
        curu = origu & (mask<<self.position) | (1u64<<self.position);
        cur = unsafe { mem::transmute::<u64,f64>(curu) };
        pre = cur;
        let mut bounded = self.is_bounded(orig,cur);
        if bounded{
            // find first bit where is not bounded
            while bounded{
                if self.position==52{
                    return pre
                }
                self.position+=1;
                curu = origu & (mask<<self.position) | (1u64<<self.position);
                cur = unsafe { mem::transmute::<u64,f64>(curu) };
                if !self.is_bounded(orig,cur){
                    bounded = false;
                    break;
                }
                pre = cur;
            }
        }else {
            // find the first bit where is bounded
            while !bounded{
                if self.position==0 {
                    break;
                }
                self.position-=1;
                curu = origu & (mask<<self.position) | (1u64<<self.position);
                cur = unsafe { mem::transmute::<u64,f64>(curu) };
                if self.is_bounded(orig,cur){
                    bounded = true;
                    pre = cur;
                    break;
                }
            }
        }
        pre
    }

//    iter over all bounded double and set the length for each one
    pub fn cal_length(&mut self, x:f64){
        let xu = unsafe { mem::transmute::<f64, u64>(x) };
        let trailing_zeros = xu.trailing_zeros();
        let exp = ((xu & EXP_MASK) >> 52) as i32 - 1023 as i32;
       // println!("trailing_zeros:{}",trailing_zeros);
       // println!("exp:{}",exp);
        let mut dec_length = 0;
        if 52<=trailing_zeros {
            if exp<0{
                dec_length = (-exp) as u64;
                if exp<self.precision_exp{
                    dec_length =0;
                }
            }

        }
        else if (52-trailing_zeros as i32)>exp{
            dec_length = ((52-trailing_zeros) as i32-exp) as u64;
        }

        if exp>=0{
            if (exp+1) as u64 >self.int_length {
                self.int_length = (exp+1) as u64;
            }
        }
        if dec_length >self.decimal_length{
            self.decimal_length = dec_length as u64;
            // let xu =  unsafe { mem::transmute::<f64, u64>(x)};
            // println!("{} with dec_length:{}, bounded => {:#066b}",x, dec_length, xu);
        }
   // println!("int len :{}, dec len:{}",self.int_length,self.decimal_length );
    }

    pub fn get_length(& self) -> (u64,u64){
        (self.int_length, self.decimal_length)
    }

    #[inline]
    pub fn set_length(&mut self, ilen:u64, dlen:u64){
        self.decimal_length = dlen;
        self.int_length = ilen;
    }

    // this is for dataset with same power of 2, power>1
    #[inline]
    pub fn fast_fetch_components_large(& self, bd:f64,exp:i32) -> (i64,u64){
        let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
        // let sign = bdu&FIRST_ONE;
        let mut int_part = 0u64;
        let mut dec_part = 0u64;
        let dec_move = 64u64-self.decimal_length;
        // if exp>=0{
        dec_part = bdu << (12 + exp) as u64;
        int_part = ((bdu << 11)| FIRST_ONE )>> (63-exp) as u64;
        // dec_part = bdu << 18 as u64;
        // int_part = ((bdu << 11)| FIRST_ONE )>> 57 as u64;
        // }else if exp<self.precision_exp{
        //     dec_part=0u64;
        // }else{
        //     dec_part = (((bdu << (12)) >>1) | FIRST_ONE) >> ((-exp - 1) as u64);
        // }
        // int_part = int_part|sign;
        // let signed_int = unsafe { mem::transmute::<u64, i64>(int_part) };
        //let signed_int = bd.trunc() as i64;
        (int_part as i64,dec_part >> dec_move)
    }

    #[inline]
    pub fn fetch_components(& self, bd:f64) -> (i64,u64){
        let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
        let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023 as i32;
        let sign = bdu&FIRST_ONE;
        let mut int_part = 0u64;
        let mut dec_part = 0u64;
        // todo: check whether we can remove those if branch
        if exp>=0{
            dec_part = bdu << (12 + exp) as u64;
            int_part = ((bdu << 11)| FIRST_ONE )>> (63-exp) as u64;
            if sign!=0{
                int_part = !int_part;
                // this is an approximation for simplification.
                // dec_part = !dec_part;
                // more accurate representation
                dec_part = !dec_part+1;
            }
        }else if exp<self.precision_exp{
            dec_part=0u64;
            if sign!=0{
                int_part = NEG_ONE;
                dec_part = !dec_part;
            }
        }else{
            dec_part = ((bdu << (11)) | FIRST_ONE) >> ((-exp - 1) as u64);
            if sign!=0{
                int_part = NEG_ONE;
                dec_part = !dec_part;
            }
        }

        let signed_int = unsafe { mem::transmute::<u64, i64>(int_part) };
        //let signed_int = bd.trunc() as i64;
        (signed_int, dec_part >> 64u64-self.decimal_length)
    }


    ///byte aligned version of spilt double
    #[inline]
    pub fn fetch_fixed_aligned(&self, bd:f64) -> i64{
        let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
        let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023 as i32;
        let sign = bdu&FIRST_ONE;
        let mut fixed = 0u64;
        if exp<self.precision_exp{
            fixed = 0u64;
        }else{
            fixed = ((bdu << (11)) | FIRST_ONE) >> (63-exp-self.decimal_length as i32) as u64;
            if sign!=0{
                fixed = !(fixed-1);
            }
        }
        let signed_int = unsafe { mem::transmute::<u64, i64>(fixed) };
        signed_int
    }


    pub fn finer(&self, input:f64) -> Vec<u8>{
        print!("finer results:");
        let mut org = input.abs();
        let mut cur = 0.5f64;
        let mut pre = 0f64;
        let mut mid = 0f64;
        let mut low =0.0;
        let mut high = 1.0;
        // check if input between 0 and 1;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(1);
        if org == 0.0{
            bitpack_vec.write(0,1);
            print!("0");
            let vec = bitpack_vec.into_vec();
            return vec;
        }else if org ==0.5 {
            bitpack_vec.write(1,1);
            print!("1");
            let vec = bitpack_vec.into_vec();
            return vec;
        }
        else if (org<1.0 && org>0.0){
            mid = (low+high)/2.0;
            while(!self.is_bounded(org,mid)){
                if org<mid{
                    bitpack_vec.write(0,1);
                    print!("0");
                    high = mid;
                }
                else {
                    bitpack_vec.write(1,1);
                    print!("1");
                    low = mid;
                }
                mid = (low+high)/2.0;
            }

        }
        if mid == org{
            bitpack_vec.write(0,1);
            print!("0");
        }else {
            bitpack_vec.write(1,1);
            print!("1");
        }
        let vec = bitpack_vec.into_vec();
        println!("{:?} after conversion: {:?}", vec, vec);
        vec
    }

    #[inline]
    pub fn is_bounded(&self, a:f64, b:f64)-> bool{
        let delta =  a-b;
        if delta.abs()<self.precision{
            return true
        }
        return false
    }
}


pub fn to_u32(slice: &[u8]) -> u32 {
    slice.iter().rev().fold(0, |acc, &b| acc*2 + b as u32)
}

pub fn get_precision_bound(precision: i32) -> f64{
    let mut str = String::from("0.");
    for pos in 0..precision{
        str.push('0');
    }
    str.push_str("49");
    let error = str.parse().unwrap();
    error
}




#[test]
fn test_precision_bounded_decimal() {
    let mut bound = PrecisionBound::new(0.000005);
    let vec = bound.finer(0.2324563);
    let vec = bound.finer(0.25);
    let vec = bound.finer(0.125);
    let vec = bound.finer(0.1239);
}

#[test]
fn test_exp_double() {
    let mut pre = 1.51f64;
    let mut preu = unsafe { mem::transmute::<f64, u64>(pre) };
    let exp = ((preu & EXP_MASK) >> 52) as i32 - 1023 as i32;
    println!("exponent is {}", exp.trailing_zeros());

    let mut bound = PrecisionBound::new(0.00005);
    let bdpre = bound.precision_bound(pre);
    let mut bdpreu = unsafe { mem::transmute::<f64, u64>(bdpre) };
    println!("preu : {:#066b}", bdpreu);
    bound.cal_length(bdpre);
    bound.cal_length(pre);
    let (int_len, dec_len) = bound.get_length();
    println!("int_len:{}, dec_len:{}",int_len, dec_len);
    let (int_part,dec_part) = bound.fetch_components(bdpre);
    println!("int_len:{:#066b}, dec_len:{:#066b}",int_part, dec_part);



}

#[test]
fn test_precision_bounded_double() {
    let mut pre = 0.1f64;
    let mut cur = 0.2f64;
    let mut preu = unsafe { mem::transmute::<f64, u64>(pre) };
    let curu = unsafe { mem::transmute::<f64, u64>(cur) };
    println!("{:#066b}^{:#066b}={:#066b}", preu, curu, preu^curu);


    let a = 0u64;
    let b = 1u64;
    let mut mask = !a;
    let mut after = 1.0;
    let mut res = 0u64;
    for i in 0..53 {
        res = preu & (mask<<i)|(1u64<<i);
        println!("{:#066b} after conversion: {:#066b}", mask<<i, res);
        after = unsafe { mem::transmute::<u64, f64>(res) };
        println!("{} float number got: {}, formatted : {:.1}",i,after,after);
    }
    let mut bound = PrecisionBound::new(0.05);
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = 0.2f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = 33.3f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = -33.3f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = 0.4f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = 0.5f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = 0.6f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);

    pre = 36.7f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = -36.7f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = 242.8f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
    pre = 2.9f64;
    preu =  unsafe { mem::transmute::<f64, u64>(pre) };
    let bd = bound.precision_bound(pre);
    let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
    println!("{:#066b} => {:#066b}", preu, bdu);
    println!("float number got: {}, formatted : {:.1}",bd,bd);
}


#[test]
fn run_benchmark_operations() {

    let file_iter = construct_file_iterator_skip_newline::<f64>("../UCRArchive2018/Kernel/randomwalkdatasample1k-40k", 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap().collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let mut cur = 0f64;
    let mut scl = 100000f64;
    let mut sum  = 0.0;
    let start = Instant::now();
    // let minValue = *(seg.get_data().iter().min().unwrap());
    // let maxValue = *(seg.get_data().iter().max().unwrap());
    for x in 0..10 {
        for item in seg.get_data(){
            cur = (*item)* scl;
            sum += cur;
        }
    }
    let duration = start.elapsed();
    // println!("min:{}, max:{}", minValue,maxValue);
    println!("Time elapsed in multiply 400 million f64 is: {:?}", duration);
    println!("multiply and sum: {}", sum);

    sum = 0.0;

    let start0 = Instant::now();
    for x in 0..10 {
        for item in seg.get_data(){
            cur = (*item)/ scl;
            sum += cur;
        }
    }
    let duration0 = start0.elapsed();
    println!("Time elapsed in divede 400 million f64 is: {:?}", duration0);
    println!("division and sum: {}", sum);
    sum = 0.0;

    let start1 = Instant::now();
    for x in 0..10 {
        for item in seg.get_data(){
            cur += (*item)- scl;
            sum += cur;
        }
    }
    let duration1 = start1.elapsed();
    println!("Time elapsed in sub 400 million f64 is: {:?}", duration1);
    println!("subtraction and sum: {}", sum);
    sum = 0.0;

    let start2 = Instant::now();
    for x in 0..10 {
        for item in seg.get_data(){
            cur = (*item)+ scl;
            sum += cur;
        }
    }
    let duration2 = start2.elapsed();
    println!("Time elapsed in add 400 million f64 is: {:?}", duration2);
    println!("addition and sum: {}", sum);
    sum = 0.0;

    let start3 = Instant::now();
    let mut curu = 0u64;
    for x in 0..10 {
        for item in seg.get_data(){
            curu = unsafe { mem::transmute::<f64, u64>(*item) };
            curu = curu>>1;
            // curu = curu|EXP_MASK;
            curu = curu|1024u64;
            curu = curu<<1;
            sum += unsafe { mem::transmute::<u64, f64>(curu) };
        }
    }
    let duration3 = start3.elapsed();
    println!("Time elapsed in converse and right shift 400 million f64 is: {:?}", duration3);
    println!("bitopts and sum: {}", sum);
}

#[test]
fn test_getlength4decimal() {
    let int_part = 2f64;
    let mut cur = 0f64;
    let err = 0.00000000005f64;
    println!("err:{}", err);
    let file = File::create("test_configs/precision.csv").unwrap();
    let mut file = LineWriter::new(file);

    for precision in 1..16{
        let mut str = String::from("0.");
        for pos in 0..precision{
            str.push('0');
        }
        str.push_str("49");
        let error = str.parse().unwrap();
        let div = 10i64.pow(precision);
        //let error = (0.5f64/div as f64).abs();
        let mut bound = PrecisionBound::new(error);
        for num in 1i64..div{
            let mut str_cur = String::from("0.");
            str_cur.push_str(format!("{:0width$}", num, width = precision as usize).as_ref());
            // cur = int_part+num as f64/div as f64;
            cur = str_cur.parse().unwrap();
            let v = bound.precision_bound(cur+int_part);
            // println!("{}th for precision:{}, cur:{}->v:{}", num,precision, cur, v);
            bound.cal_length(v);
        }
        let (int_len,dec_len) = bound.get_length();
        println!("length for integer part:{}, decimal part:{} to save {} position of decimal with error;{}", int_len,dec_len,precision,error);
        file.write_all(format!("{},{}\n", precision, dec_len).as_ref());
    }
    file.flush();
}



#[test]
fn test_bitmap_serialize() {
    use croaring::Bitmap;

    let mut bitmap: Bitmap = (100..800).collect();
    println!("bitmap size without compression: {}",bitmap.get_serialized_size_in_bytes());

    assert_eq!(bitmap.cardinality(), 700);
    assert!(bitmap.run_optimize());
    println!("bitmap size with compression: {}",bitmap.get_serialized_size_in_bytes());
    let ser = bitmap.serialize();
    let mut bitmap_new = Bitmap::deserialize(&ser);
    assert_eq!(bitmap_new.cardinality(), 700);
    println!("deserialize bitmap size with compression: {}",bitmap_new.get_serialized_size_in_bytes());
    bitmap_new.remove_run_compression();
    println!("deserialize bitmap size with decompression: {}",bitmap_new.get_serialized_size_in_bytes());
    bitmap_new.flip_inplace(1..801);
    assert_eq!(bitmap_new.cardinality(), 100);
    println!("bitmap size without compression: {}",bitmap_new.get_serialized_size_in_bytes());
    assert!(bitmap_new.run_optimize());
    println!("bitmap size with compression: {}",bitmap_new.get_serialized_size_in_bytes());



}

#[test]
fn test_bitmap() {
    let mut rb1 = Bitmap::create();
    rb1.add(0);
    rb1.add(2);
    rb1.add(3);
    rb1.add(4);
    rb1.add(5);
    rb1.add(100);
    rb1.add(1000);
    rb1.run_optimize();

    let mut rb2 = Bitmap::create();
    rb2.add(0);
    rb2.add(4);
    rb2.add(1000);
    rb2.run_optimize();

    let mut rb3 = Bitmap::create();

    assert_eq!(rb1.cardinality(), 7);
    assert!(rb1.contains(3));

    rb1.and_inplace(&rb2);
    rb3.add(5);
    rb3.or_inplace(&rb1);

    let mut rb4 = Bitmap::fast_or(&[&rb1, &rb2, &rb3]);

    rb1.and_inplace(&rb2);
    println!("{:?}", rb1);

    rb3.add(5);
    rb3.or_inplace(&rb1);

    println!("{:?}", rb1);

    rb3.add(5);
    rb3.or_inplace(&rb1);

    println!("{:?}", rb3.to_vec());
    println!("{:?}", rb3);
    println!("{:?}", rb4);

    rb4 = Bitmap::fast_or(&[&rb1, &rb2, &rb3]);

    println!("{:?}", rb4);

    let base = -1i32;
    let div  = base as i64;
    println!("div: {}",div)
}

#[test]
fn test_decimal() {
// Using an integer followed by the decimal points
    let scaled = Decimal::new(200000002, 2);
    let example = Decimal::from_parts(1, 1, 0, false, 0);

    println!("{}",scaled.to_string());
    println!("example: {}",example.to_string());
}

#[test]
fn test_varible_length_hist() {
    let file_iter = construct_file_iterator_skip_newline::<f64>("../UCRArchive2018/Kernel/randomwalkdatasample1k-40k", 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap().collect();
    let clone_vec = file_vec.clone();
    let mut histogram = Histogram::new();
    let mut dec_hist = Histogram::new();

    let mut bound = PrecisionBound::new(0.0000005);
    let start = Instant::now();
    for val in file_vec{

        let bd = bound.precision_bound(val);
        bound.cal_length(bd);
        histogram.increment(bound.get_length().1);

    }
    let duration = start.elapsed();
    println!("Time elapsed in cal_length is: {:?}", duration);
    // print percentiles from the histogram
    println!("Percentiles: p50: {} ns p90: {} ns p99: {} ns p999: {}",
             histogram.percentile(50.0).unwrap(),
             histogram.percentile(90.0).unwrap(),
             histogram.percentile(99.0).unwrap(),
             histogram.percentile(99.9).unwrap(),
    );

    println!("Latency (ns): Min: {} Avg: {} Max: {} StdDev: {}",
             histogram.minimum().unwrap(),
             histogram.mean().unwrap(),
             histogram.maximum().unwrap(),
             histogram.stddev().unwrap(),
    );

    let start = Instant::now();
    for val in clone_vec{
        let cur_str = val.to_string();
        let string:  Vec<&str> = cur_str.split('.').collect();
        if string.len()>1{
            dec_hist.increment(string.get(1).unwrap().len() as u64);
        }
        else { dec_hist.increment(0); }
    }
    let duration = start.elapsed();
    println!("Time elapsed in checking bits is: {:?}", duration);

    println!("Percentiles: p10: {} ns p15: {} ns p50: {} ns p90: {}",
             dec_hist.percentile(10.0).unwrap(),
             dec_hist.percentile(15.0).unwrap(),
             dec_hist.percentile(50.0).unwrap(),
             dec_hist.percentile(90.0).unwrap(),
    );

    println!("Latency (ns): Min: {} Avg: {} Max: {} StdDev: {}",
             dec_hist.minimum().unwrap(),
             dec_hist.mean().unwrap(),
             dec_hist.maximum().unwrap(),
             dec_hist.stddev().unwrap(),
    );


}