use log::{warn,info};
use std::mem;
use crate::methods::bit_packing::BitPack;

pub struct PrecisionBound {
    position: u64,
    precision: f64
}


impl PrecisionBound {
    pub fn new(precision:f64) -> Self {
        PrecisionBound { position: 0, precision: precision}
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



#[test]
fn test_precision_bounded_decimal() {
    let mut bound = PrecisionBound::new(0.000005);
    let vec = bound.finer(0.2324563);
    let vec = bound.finer(0.25);
    let vec = bound.finer(0.125);
    let vec = bound.finer(0.1239);
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