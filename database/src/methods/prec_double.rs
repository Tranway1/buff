use log::{warn,info};
use std::mem;
use crate::methods::bit_packing::BitPack;
use crate::client::construct_file_iterator_skip_newline;
use std::time::{SystemTime, Instant};
use crate::segment::Segment;

/// END_MARKER is a special bit sequence used to indicate the end of the stream
pub const EXP_MASK: u64 = 0b0111111111110000000000000000000000000000000000000000000000000000;
pub const FIRST_ONE: u64 = 0b1000000000000000000000000000000000000000000000000000000000000000;


pub struct PrecisionBound {
    position: u64,
    precision: f64,
    precision_exp : i32,
    int_length: u64,
    decimal_length: u64
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
                    info!("full precision.");
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
//        println!("trailing_zeros:{}",trailing_zeros);
//        println!("exp:{}",exp);
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
//    println!("int len :{}, dec len:{}",self.int_length,self.decimal_length );
    }

    pub fn get_length(& self) -> (u64,u64){
        (self.int_length, self.decimal_length)
    }

    pub fn fetch_components(& self, bd:f64) -> (u64,u64){
        let bdu = unsafe { mem::transmute::<f64, u64>(bd) };
        let exp = ((bdu & EXP_MASK) >> 52) as i32 - 1023 as i32;
        let mut int_part = 0u64;
        let mut dec_part = 0u64;
        if exp>=0{
            dec_part = bdu << (12 + exp) as u64;
            int_part = (((bdu << 12) >> 1)| FIRST_ONE )>> (11+52-exp) as u64
        }else if exp<self.precision_exp{
            dec_part=0u64;
        }else{

            dec_part = (((bdu << (12)) >>1) | FIRST_ONE) >> ((-exp - 1) as u64);
        }
        (int_part,dec_part >> 64u64-self.decimal_length)
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
    let start = Instant::now();
    for x in 0..10 {
        for item in seg.get_data(){
            cur = (*item)* scl;
        }
    }
    let duration = start.elapsed();
    println!("Time elapsed in multiply 400 million f64 is: {:?}", duration);

    let start0 = Instant::now();
    for x in 0..10 {
        for item in seg.get_data(){
            cur = (*item)/ scl;
            cur = (*item)*cur
        }
    }
    let duration0 = start0.elapsed();
    println!("Time elapsed in divede 400 million f64 is: {:?}", duration0);

    let start1 = Instant::now();
    for x in 0..10 {
        for item in seg.get_data(){
            cur += (*item)- scl;
        }
    }
    let duration1 = start1.elapsed();
    println!("Time elapsed in sub 400 million f64 is: {:?}", duration1);

    let start2 = Instant::now();
    for x in 0..10 {
        for item in seg.get_data(){
            cur = (*item)+ scl;
        }
    }
    let duration2 = start2.elapsed();
    println!("Time elapsed in add 400 million f64 is: {:?}", duration2);

    let start3 = Instant::now();
    let mut curu = 0u64;
    for x in 0..10 {
        for item in seg.get_data(){
            curu = unsafe { mem::transmute::<f64, u64>(*item) };
            curu>>32;
            curu&EXP_MASK;
            curu|EXP_MASK;
        }
    }
    let duration3 = start3.elapsed();
    println!("Time elapsed in converse and right shift 400 million f64 is: {:?}", duration3);
}

#[test]
fn test_getlength4decimal() {
    let int_part = 2f64;
    let mut cur = 0f64;
    let err = 0.00000000005f64;
    println!("err:{}", err);
    for precision in 1..10{
        let mut str = String::from("0.");
        for pos in 0..precision{
            str.push('0');
        }
        str.push_str("49");
        let error = str.parse().unwrap();
        let div = 10i32.pow(precision);
        //let error = (0.5f64/div as f64).abs();
        let mut bound = PrecisionBound::new(error);
        for num in 1..div{
            let mut str_cur = String::from("0.");
            str_cur.push_str(format!("{:0width$}", num, width = precision as usize).as_ref());
            // cur = int_part+num as f64/div as f64;
            cur = str_cur.parse().unwrap();
            let v = bound.precision_bound(cur);
            //println!("{}th for precision:{}, cur:{}->v:{}", num,precision, cur, v);
            bound.cal_length(v);
        }
        let (int_len,dec_len) = bound.get_length();
        println!("length for integer part:{}, decimal part:{} to save {} position of decimal with error;{}", int_len,dec_len,precision,error);
    }
}