use log::{warn,info};
use std::mem;
use crate::methods::bit_packing::BitPack;

/// END_MARKER is a special bit sequence used to indicate the end of the stream
pub const EXP_MASK: u64 = 0b0111111111110000000000000000000000000000000000000000000000000000;
pub const FIRST_ONE: u64 = 0b1000000000000000000000000000000000000000000000000000000000000000;


pub struct PrecisionBound {
    position: u64,
    precision: f64,
    int_length: u64,
    decimal_length: u64
}


impl PrecisionBound {
    pub fn new(precision:f64) -> Self {
        PrecisionBound { position: 0, precision: precision, int_length: 0, decimal_length: 0 }
    }

    pub fn precision_bound(&mut self, orig: f64)-> f64{
        let a = 0u64;
        let mut mask = !a;
        let mut ret = 0f64;
        let mut pre = orig;
        let mut cur = 0f64;
        let origu = unsafe { mem::transmute::<f64, u64>(orig) };
        let mut curu = 0u64;
        curu = origu & (mask<<self.position);
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
                curu = origu & (mask<<self.position);
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
                curu = origu & (mask<<self.position);
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
            }

        }
        else if (52-trailing_zeros as i32)>exp{
            dec_length = ((52-trailing_zeros) as i32-exp) as u64;
        }

        if exp>=0{
            if (exp+1) as u64 >self.int_length {
                self.int_length = (exp+1) as u64;
            }
            if dec_length >self.decimal_length{
                self.decimal_length = dec_length as u64
            }
        }else{
            if (dec_length+1) >self.decimal_length{
                self.decimal_length = (dec_length+1) as u64
            }
        }
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
        }else{
            dec_part = (((bdu << (12) as u64) >>1) | FIRST_ONE) >> (-exp-1) as u64;
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
        res = preu & (mask<<i);
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