use std::collections::HashMap;
use crate::methods::bit_packing::{num_bits, BitPack, delta_num_bits, differential};
use std::borrow::Borrow;
use std::mem;
use log::{info, trace, warn};
use std::hash::Hash;
use num::Num;

//#[derive(Hash, Eq, PartialEq, Debug)]
pub struct FCMCompressor<'a,T> {
    input: Vec<T>,
    len: usize,
    array_list_map: HashMap<&'a Vec<T>, T>,
    count: usize,
    level: usize,
    delta: bool,
}
//
//impl<'a,T> FCMCompressor<'a,T>
//    where T:Eq+Hash+Clone+Copy+Num{
//
//    pub fn new(input: Vec<T>, level: usize, def: bool) -> FCMCompressor<'a,T> {
//        FCMCompressor{
//            input: input,
//            len: input.len(),
//            array_list_map: HashMap::new(),
//            count: 0,
//            level: level,
//            delta: def
//        }
//
//    }
//
//    fn  get_prev_list(&self) -> Option<Vec<T>> {
//        let start = self.count - self.level;
//        if start >= 0 {
//            let ret: Vec<T> = self.input[start..self.count].to_vec();
//            return Some(ret);
//        }
//        None
//    }
//
//
//    fn predict(&self) -> Option<T>  {
//        if self.count >= self.level {
//            Some(*self.array_list_map.get(&serialize(self.get_prev_list().unwrap()).unwrap()).unwrap())
//        } else {
//            None
//        }
//
//    }
//
//    fn  update(&mut self) {
//        let prev = self.get_prev_list();
//        let cur = self.input[self.count].borrow();
//        match prev {
//            Some(list) => {
//                self.array_list_map.insert(list,*cur);
//            },
//            None => (),
//        }
//    }
//
//    pub fn differential_compress(&mut self) -> Vec<u8>  {
//        let mut mydata = self.input.as_slice();
//        let mut num_bits= mem::size_of::<T>();
//        if self.delta{
//            let delta_vec = differential(mydata);
//            mydata = delta_vec.as_slice();
//        }
//        info!("Number of bits: {}", num_bits);
//        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
//        for (pos, &e) in mydata.iter().enumerate() {
//            self.count = pos;
//            let predict = self.predict();
//            match predict {
//                Some(val) => {
//                    bitpack_vec.write(1,1).unwrap();
//                },
//                None =>{
//                    bitpack_vec.write(0,0).unwrap();
//                    bitpack_vec.write(self.input[pos], num_bits as usize).unwrap();
//                },
//            }
//            self.update();
//
//        }
//        let vec = bitpack_vec.into_vec();
//        info!("Length of compressed data: {}", vec.len());
//        let ratio= vec.len() as f32 / (mydata.len() as f32*4.0);
//        print!("{}",ratio);
//        vec
//    }
//
//    pub fn delta_compress(&mut self) -> Vec<u8>  {
//        let mut mydata = self.input.as_slice();
//        let mut num_bits = 0;
//        if self.delta{
//            let (bits,delta_vec) = delta_num_bits(mydata);
//            num_bits = bits;
//            mydata = delta_vec.as_slice();
//        }
//        else {
//            num_bits = num_bits(mydata);
//        }
//        info!("Number of bits: {}", num_bits);
//        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
//        for (pos, &e) in mydata.iter().enumerate() {
//            self.count = pos;
//            let predict = self.predict();
//            match predict {
//                Some(val) => {
//                    bitpack_vec.write(1,1).unwrap();
//                },
//                None =>{
//                    bitpack_vec.write(0,0).unwrap();
//                    bitpack_vec.write(self.input[pos], num_bits as usize).unwrap();
//                },
//            }
//            self.update();
//
//        }
//        let vec = bitpack_vec.into_vec();
//        info!("Length of compressed data: {}", vec.len());
//        let ratio= vec.len() as f32 / (mydata.len() as f32*4.0);
//        print!("{}",ratio);
//        vec
//    }
//
//}
//
//pub fn serialize<T>(input:Vec<T>) -> Result<Vec<u8>,()>{
//    match bincode::serialize(&input) {
//        Ok(key) => Ok(key),
//        Err(_)  => Err(())
//    }
//}
//

#[test]
fn test_hashmap() {
    let data = vec![53,3,567,8932];
    let data10 = vec![53,3,567,8932];
    let mut map = HashMap::new();
    map.insert(&data,9);
    map.insert(&data10,19);
    //println!("Time elapsed in compress function() is: {:?}", duration);
    let decompress = map.get(&data).unwrap();
    println!("expected datapoints: {:?}", decompress);

}

