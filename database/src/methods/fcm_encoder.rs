use std::collections::HashMap;
use crate::methods::bit_packing::{num_bits, BitPack, delta_num_bits, differential, diff_num_bits};
use std::borrow::Borrow;
use std::mem;
use log::{info, trace, warn};
use std::hash::Hash;
use num::Num;
use serde::Serialize;
use serde::export::PhantomData;

//#[derive(Hash, Eq, PartialEq, Debug)]
pub struct FCMCompressor<'a,T> {
    orgin: Vec<i32>,
    input: Vec<u32>,
    len: usize,
    array_list_map: HashMap<Vec<u32>, u32>,
    count: usize,
    level: usize,
    delta: bool,
    phantom: PhantomData<&'a T>
}

impl<'a,T> FCMCompressor<'a,T>
    where T:Eq+Hash+Clone+Copy+Num+Into<i32>{

    pub fn new(input: Vec<T>, level: usize, def: bool) -> FCMCompressor<'a,T> {
        FCMCompressor{
            orgin: input.iter().map(|&x|x.into()).collect(),
            len: input.len(),
            input: vec![],
            array_list_map: HashMap::new(),
            count: 0,
            level: level,
            delta: def,
            phantom: PhantomData
        }

    }

    fn  get_prev_list(&self) -> Option<Vec<u32>> {
        let start = self.count as i32 - self.level as i32;
        if start > 0 {
            let ret: Vec<u32> = self.input[start as usize..self.count].to_vec();
            return Some(ret);
        }
        None
    }


    fn predict(&self) -> Option<u32>  {
        if self.count >= self.level {
            let vec_u32 = self.get_prev_list();
            match vec_u32{
                Some(pre_vec) => {
                    let val = self.array_list_map.get(&pre_vec);
                    match val {
                        Some(&pred) => Some(pred),
                        None => None
                    }
                }
                None => None
            }
//            Some(*self.array_list_map.get(&serialize(self.get_prev_list().unwrap()).unwrap()).unwrap())
        } else {
            None
        }

    }

    fn update(&mut self) {
        let prev = self.get_prev_list();
        let cur = self.input[self.count].borrow();
        match prev {
            Some(list) => {
                self.array_list_map.insert(list, *cur);
            },
            None => (),
        }
    }

    pub fn differential_compress(&mut self) -> Vec<u8>  {
        let mut cp_data = self.input.clone();
        let mut mydata = cp_data.as_slice();
        let mut delta_data:Vec<u32> ;
        let vec_len = mydata.len();
        let mut num_bits= mem::size_of::<T>();
        if self.delta{
            delta_data= differential(mydata);
            mydata = delta_data.as_slice();
        }
        info!("Number of bits: {}", num_bits);
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        for (pos, &e) in mydata.iter().enumerate() {
            self.count = pos;
            let predict = self.predict();
            match predict {
                Some(val) => {
                    if val==e{
                        bitpack_vec.write(1,1).unwrap();
                    }
                    else {
                        bitpack_vec.write(e, num_bits as usize).unwrap();
                    }
                },
                None =>{
                    bitpack_vec.write(e, num_bits as usize).unwrap();
                },
            }
            self.update();

        }
        let vec = bitpack_vec.into_vec();
        info!("Length of compressed data: {}", vec.len());
        let ratio= vec.len() as f32 / (vec_len as f32* mem::size_of::<T>() as f32);
        print!("{}",ratio);
        vec
    }

    pub fn delta_compress(&mut self) -> Vec<u8>  {
        let mut mydata = &self.orgin.clone();
        let mut num_bit = 0;
        let mut bits = 0;
        let mut delta_vec: Vec<u32> = vec![];
        let mut hit = 0;
        if self.delta{
            let (b, vec) = diff_num_bits(mydata);
            num_bit = b;
            self.input = vec;
        }
        else {
            let (b, vec) = delta_num_bits(mydata);
            num_bit = b;
            self.input = vec.clone();
        }
        info!("Number of bits: {}", num_bit);
        num_bit +=1;
        let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
        let offset_data = self.input.clone();
        for (pos, &e) in offset_data.iter().enumerate() {
            self.count = pos;
            let predict = self.predict();
            match predict {
                Some(val) => {
                    if val==e{
                        bitpack_vec.write(1,1).unwrap();
                        hit += 1;
                    }
                    else {
                        bitpack_vec.write(e, num_bit as usize).unwrap();
                    }
                },
                None =>{
                    //bitpack_vec.write(0,0).unwrap();
                    bitpack_vec.write(self.input[pos], num_bit as usize).unwrap();
                },
            }
            self.update();

        }
        let vec = bitpack_vec.into_vec();
        info!("number of hit: {}", hit);
        info!("Length of compressed data: {}", vec.len());
        info!("number of entries in hashmap:{}",self.array_list_map.len());
        let ratio= vec.len() as f32 / (mydata.len() as f32* mem::size_of::<T>() as f32);
        print!("{}",ratio);
        vec
    }

}

pub fn serialize<T>(input:Vec<T>) -> Result<Vec<u8>,()>
 where T: Serialize{
    match bincode::serialize(&input) {
        Ok(key) => Ok(key),
        Err(_)  => Err(())
    }
}


#[test]
fn test_hashmap() {
    let data = vec![53u32,3u32,567u32,8932u32];
    let data10 = vec![53u32,3u32,567u32,8932u32];
    let mut map = HashMap::new();
    let cur = &data;
    //map.insert(data,9);
    map.insert(data10,19);
    //println!("Time elapsed in compress function() is: {:?}", duration);
    let decompress = map.get(&data).unwrap();
    println!("expected datapoints: {:?}", decompress);

}

