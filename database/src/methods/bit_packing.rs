use bitpacking::{BitPacker4x};
use log::{info, trace, warn};
use futures::future::err;
use num::Num;
use std::mem;
use crate::client::construct_file_iterator_skip_newline;
use std::fmt::Debug;
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, Instant};
use crate::segment::Segment;
use std::any::Any;

pub const MAX_BITS: usize = 32;
pub const BYTE_BITS: usize = 8;


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitPack<B> {
    buff: B,
    cursor: usize,
    bits: usize
}


impl<B> BitPack<B> {
    #[inline]
    pub fn new(buff: B) -> Self {
        BitPack { buff: buff, cursor: 0, bits: 0 }
    }

    #[inline]
    pub fn sum_bits(&self) -> usize {
        self.cursor * BYTE_BITS + self.bits
    }

    #[inline]
    pub fn with_cursor(&mut self, cursor: usize) -> &mut Self {
        self.cursor = cursor;
        self
    }

    #[inline]
    pub fn with_bits(&mut self, bits: usize) -> &mut Self {
        self.bits = bits;
        self
    }
}

impl<B: AsRef<[u8]>> BitPack<B> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.buff.as_ref()
    }
}

impl<'a> BitPack<&'a mut [u8]> {
    /// ```
    /// use bitpack::BitPack;
    /// use time_series_start::methods::bit_packing::BitPack;
    ///
    /// let mut buff = [0; 2];
    ///
    /// {
    ///     let mut bitpack = BitPack::<&mut [u8]>::new(&mut buff);
    ///     bitpack.write(10, 4).unwrap();
    ///     bitpack.write(1021, 10).unwrap();
    ///     bitpack.write(3, 2).unwrap();
    /// }
    ///
    /// assert_eq!(buff, [218, 255]);
    /// ```

    pub fn write(&mut self, mut value: u32, mut bits: usize) -> Result<(), usize> {
        if bits > MAX_BITS || self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        }
        if bits < MAX_BITS {
            value &= ((1 << bits) - 1) as u32;
        }

        loop {
            let bits_left = BYTE_BITS - self.bits;

            if bits <= bits_left {
                self.buff[self.cursor] |= (value as u8) << self.bits as u8;
                self.bits += bits;

                if self.bits >= BYTE_BITS {
                    self.cursor += 1;
                    self.bits = 0;
                }

                break
            }

            let bb = value & ((1 << bits_left) - 1) as u32;
            self.buff[self.cursor] |= (bb as u8) << self.bits as u8;
            self.cursor += 1;
            self.bits = 0;
            value >>= bits_left as u32;
            bits -= bits_left;
        }
        Ok(())
    }

    /***
    read bits less then BYTE_BITS
     */
    pub fn write_bits(&mut self, mut value: u32, mut bits: usize) -> Result<(), usize> {
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        }
        value &= ((1 << bits) - 1) as u32;

        loop {
            let bits_left = BYTE_BITS - self.bits;

            if bits <= bits_left {
                self.buff[self.cursor] |= (value as u8) << self.bits as u8;
                self.bits += bits;

                if self.bits >= BYTE_BITS {
                    self.cursor += 1;
                    self.bits = 0;
                }

                break
            }

            let bb = value & ((1 << bits_left) - 1) as u32;
            self.buff[self.cursor] |= (bb as u8) << self.bits as u8;
            self.cursor += 1;
            self.bits = 0;
            value >>= bits_left as u32;
            bits -= bits_left;
        }
        Ok(())
    }
}


impl<'a> BitPack<&'a [u8]> {
    /// ```
    /// use bitpack::BitPack;
    /// use time_series_start::methods::bit_packing::BitPack;
    ///
    /// let mut buff = [218, 255];
    ///
    /// let mut bitpack = BitPack::<&[u8]>::new(&buff);
    /// assert_eq!(bitpack.read(4).unwrap(), 10);
    /// assert_eq!(bitpack.read(10).unwrap(), 1021);
    /// assert_eq!(bitpack.read(2).unwrap(), 3);
    /// ```
    pub fn read(&mut self, mut bits: usize) -> Result<u32, usize> {
        if bits > MAX_BITS || self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        };

        let mut bits_left = 0;
        let mut output = 0;
        loop {
            let byte_left = BYTE_BITS - self.bits;

            if bits <= byte_left {
                let mut bb = self.buff[self.cursor] as u32;
                bb >>= self.bits as u32;
                bb &= ((1 << bits) - 1) as u32;
                output |= bb << bits_left;
                self.bits += bits;
                break
            }

            let mut bb = self.buff[self.cursor] as u32;
            bb >>= self.bits as u32;
            bb &= ((1 << byte_left) - 1) as u32;
            output |= bb << bits_left;
            self.bits += byte_left;
            bits_left += byte_left as u32;
            bits -= byte_left;

            if self.bits >= BYTE_BITS {
                self.cursor += 1;
                self.bits -= BYTE_BITS;
            }
        }
        Ok(output)
    }

    /***
    read bits less than BYTE_BITS
     */

    #[inline]
    pub fn read_bits(&mut self, mut bits: usize) -> Result<u8, usize> {
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            println!("buff length: {}, cursor: {}, bits: {}", self.buff.len(), self.cursor,self.bits);
            return Err(bits);
        };

        let mut bits_left = 0;
        let mut output = 0;
        loop {
            let byte_left = BYTE_BITS - self.bits;

            if bits <= byte_left {
                let mut bb = self.buff[self.cursor] as u32;
                bb >>= self.bits as u32;
                bb &= ((1 << bits) - 1) as u32;
                output |= bb << bits_left;
                self.bits += bits;
                break
            }

            let mut bb = self.buff[self.cursor] as u32;
            bb >>= self.bits as u32;
            bb &= ((1 << byte_left) - 1) as u32;
            output |= bb << bits_left;
            self.bits += byte_left;
            bits_left += byte_left as u32;
            bits -= byte_left;

            if self.bits >= BYTE_BITS {
                self.cursor += 1;
                self.bits -= BYTE_BITS;
            }
        }
        Ok(output as u8)
    }

    #[inline]
    pub fn finish_read_byte(&mut self){
        self.cursor += 1;
        self.bits = 0;
        // println!("cursor now at {}" , self.cursor)
    }

    #[inline]
    pub fn read_byte(&mut self) -> Result<u8, usize> {
        self.cursor += 1;
        let output = self.buff[self.cursor] as u8;
        Ok(output)
    }

    #[inline]
    pub fn read_n_byte(&mut self,n:usize) -> Result<&[u8], usize> {
        self.cursor += 1;
        let end = self.cursor+n;
        let output = &self.buff[self.cursor..end];
        self.cursor += n-1;
        Ok(output)
    }

    #[inline]
    pub fn read_n_byte_unmut(&self,start:usize, n:usize) -> Result<&[u8], usize> {
        let s = start+self.cursor + 1;
        let end =s+n;
        let output = &self.buff[s..end];
        Ok(output)
    }

    #[inline]
    pub fn skip_n_byte(&mut self, mut n: usize) -> Result<(), usize> {
        self.cursor += n;
        // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
        Ok(())
    }
    #[inline]
    pub fn skip(&mut self, mut bits: usize) -> Result<(), usize> {
        if self.buff.len() * BYTE_BITS < self.sum_bits() + bits {
            return Err(bits);
        };
        // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
        // println!("try to skip {} bits",bits);
        let bytes = bits/BYTE_BITS;
        let left = bits%BYTE_BITS;

        let cur_bits = (self.bits +left);
        self.cursor = self.cursor + bytes + cur_bits/BYTE_BITS;
        self.bits = cur_bits%BYTE_BITS;

        // println!("current cursor{}, current bits:{}",self.cursor,self.bits);
        Ok(())
    }
}

impl Default for BitPack<Vec<u8>> {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl BitPack<Vec<u8>> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::new(Vec::with_capacity(capacity))
    }

    /// ```
    /// use bitpack::BitPack;
    /// use time_series_start::methods::bit_packing::BitPack;
    ///
    /// let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(2);
    /// bitpack_vec.write(10, 4).unwrap();
    /// bitpack_vec.write(1021, 10).unwrap();
    /// bitpack_vec.write(3, 2).unwrap();
    ///
    /// assert_eq!(bitpack_vec.as_slice(), [218, 255]);
    /// #
    /// # let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
    /// # assert_eq!(bitpack.read(4).unwrap(), 10);
    /// # assert_eq!(bitpack.read(10).unwrap(), 1021);
    /// # assert_eq!(bitpack.read(2).unwrap(), 3);
    /// ```
    #[inline]
    pub fn write(&mut self, value: u32, bits: usize) -> Result<(), usize> {
        let len = self.buff.len();

        if let Some(bits) = (self.sum_bits() + bits).checked_sub(len * BYTE_BITS) {
            self.buff.resize(len + (bits + BYTE_BITS - 1) / BYTE_BITS, 0x0);
        }

        let mut bitpack = BitPack {
            buff: self.buff.as_mut_slice(),
            cursor: self.cursor,
            bits: self.bits
        };

        bitpack.write(value, bits)?;

        self.bits = bitpack.bits;
        self.cursor = bitpack.cursor;

        Ok(())
    }

    /***
    read bits less then BYTE_BITS
     */
    #[inline]
    pub fn write_bits(&mut self, value: u32, bits: usize) -> Result<(), usize> {
        let len = self.buff.len();

        if let Some(bits) = (self.sum_bits() + bits).checked_sub(len * BYTE_BITS) {
            self.buff.resize(len + (bits + BYTE_BITS - 1) / BYTE_BITS, 0x0);
        }

        let mut bitpack = BitPack {
            buff: self.buff.as_mut_slice(),
            cursor: self.cursor,
            bits: self.bits
        };

        bitpack.write_bits(value, bits)?;

        self.bits = bitpack.bits;
        self.cursor = bitpack.cursor;

        Ok(())
    }



    #[inline]
    pub fn write_bytes(&mut self, value: &mut Vec<u8>) -> Result<(), usize> {
        self.buff.append(value);
        Ok(())
    }

    #[inline]
    pub fn write_byte(&mut self, value: u8) -> Result<(), usize> {
        self.buff.push(value);
        Ok(())
    }

    #[inline]
    pub fn finish_write_byte(&mut self) {
        let len = self.buff.len();
        self.buff.resize(len + 1, 0x0);
        self.bits = 0;
        self.cursor = len;
        // println!("cursor now at {}" , self.cursor)
    }

    #[inline]
    pub fn into_vec(self) -> Vec<u8> {
        // println!("buff length: {}, cursor: {}, bits: {}", self.buff.len(), self.cursor,self.bits);
        self.buff
    }
}

pub(crate) fn num_bits(mydata: &[u32]) -> u8{
    let mut xor:u32 = 0;
    for &b in mydata {
        xor = xor | b;
    }
    let lead = xor.leading_zeros();
    let bits:u8 = (32 -lead) as u8;
    bits
}

pub fn delta_num_bits(mydata: &[i32]) -> (i32, u8,Vec<u32>){
    // info!("10th vec: {},{},{},{}", mydata[0],mydata[1],mydata[2],mydata[3]);
    let mut vec = Vec::new();
    let mut xor:u32 = 0;
    let mut delta = 0u32;
    let mut min = 0i32;
    let minValue = mydata.iter().min();
    info!("min value:{}",minValue.unwrap());
    match minValue {
        Some(&val) => min = val,
        None => panic!("empty"),
    }

    for &b in mydata {
        delta = (b - min) as u32;
        vec.push(delta);
        xor = xor | delta;
    }
    let lead = xor.leading_zeros();
    let bits:u8 = (32 -lead) as u8;
    info!("10th vec: {},{},{},{}", vec[0],vec[1],vec[2],vec[3]);
    (min, bits, vec)
}

/*
zigzag encoding: Maps negative values to positive values while going back and
  forth (0 = 0, -1 = 1, 1 = 2, -2 = 3, 2 = 4, -3 = 5, 3 = 6 ...)
 */
#[inline]
pub fn zigzag(origin: i32) -> u32{
    let zzu = (origin << 1) ^ (origin >> 31);
    let orgu= unsafe { mem::transmute::<i32, u32>(zzu) };
    orgu
}

#[inline]
pub fn unzigzag(origin: u32) -> i32{
    let zzu = (origin >> 1) as i32 ^ -((origin & 1) as i32);
    zzu
}

// delta calculation for sprintz
pub fn zigzag_delta_num_bits(mydata: &[i32]) -> (i32, u8,Vec<u32>){
    info!("10th vec: {},{},{},{}", mydata[0],mydata[1],mydata[2],mydata[3]);
    let mut vec = Vec::new();
    let mut xor:u32 = 0;
    let mut delta = 0i32;
    let mut zz = 0u32;
    let base = mydata[0];
    let mut pre = mydata[0];

    for &b in mydata {
        delta = b - pre;
        zz = zigzag(delta);
        vec.push(zz);
        xor = xor | zz;
        pre = b;
    }
    let lead = xor.leading_zeros();
    let bits:u8 = (32 -lead) as u8;
    info!("10th vec: {},{},{},{}", vec[0],vec[1],vec[2],vec[3]);
    (base, bits, vec)
}

pub(crate) fn delta_64num_bits(mydata: &[i64]) -> (u8,Vec<u64>){
    info!("10th vec: {},{},{},{}", mydata[0],mydata[1],mydata[2],mydata[3]);
    let mut vec = Vec::new();
    let mut xor:u64 = 0;
    let mut delta = 0u64;
    let mut min = 0i64;
    let minValue = mydata.iter().min();
    info!("min value:{}",minValue.unwrap());
    match minValue {
        Some(&val) => min = val,
        None => panic!("empty"),
    }

    for &b in mydata {
        delta = (b - min) as u64;
        vec.push(delta);
        xor = xor | delta;
    }
    let lead = xor.leading_zeros();
    let bits:u8 = (64 -lead) as u8;
    info!("10th vec: {},{},{},{}", vec[0],vec[1],vec[2],vec[3]);
    (bits,vec)
}

pub(crate) fn split_num_bits(mydata: &[i32], scl: usize) -> (u8,Vec<u32>,u8,Vec<u32>){
    info!("10th vec: {},{},{},{}", mydata[0],mydata[1],mydata[2],mydata[3]);
    let mut intpart = Vec::new();
    let mut decpart = Vec::new();
    let mut intxor:u32 = 0;
    let mut decxor:u32 = 0;
    let mut delta = 0u32;
    let mut min = 0i32;
    let minValue = mydata.iter().min();
    info!("min value:{}",minValue.unwrap());
    match minValue {
        Some(&val) => min = val,
        None => panic!("empty"),
    }

    for &b in mydata {
        delta = (b - min) as u32;
        let div = delta / scl as u32;
        let rem = delta % scl as u32;
        intpart.push(div);
        decpart.push(rem);
        intxor = intxor | div;
        decxor = decxor | rem;
    }
    let intlead = intxor.leading_zeros();
    let declead = decxor.leading_zeros();
    let intbits:u8 = (32 -intlead) as u8;
    let decbits:u8 = (32 -declead) as u8;
    info!("10th vec: {},{},{},{}", intpart[0],intpart[1],intpart[2],intpart[3]);
    (intbits,intpart,decbits,decpart)
}

pub(crate) fn split_64num_bits(mydata: &[i64], scl: usize) -> (i32, u8,Vec<u64>,u8,Vec<u64>){
    info!("10th vec: {},{},{},{}", mydata[0],mydata[1],mydata[2],mydata[3]);
    let mut intpart = Vec::new();
    let mut decpart = Vec::new();
    let mut intxor:u64 = 0;
    let mut decxor:u64 = 0;
    let mut delta = 0u64;
    let mut min = 0i64;
    let minValue = mydata.iter().min();
    let maxValue = mydata.iter().max();
    println!("min:{}, max:{}",minValue.unwrap(),maxValue.unwrap());
    info!("min value:{}",minValue.unwrap());
    match minValue {
        Some(&val) => min = val,
        None => panic!("empty"),
    }
    let mut min_int = min/scl as i64;
    if min < 0{
        min_int -= 1;
    }
    println!("min int: {}", min_int);
    min = min_int * scl as i64;


    for &b in mydata {
        delta = (b - min) as u64;
        let div = delta / scl as u64;
        let rem = delta % scl as u64;
        intpart.push(div);
        decpart.push(rem);
        intxor = intxor | div;
        decxor = decxor | rem;
    }
    let intlead = intxor.leading_zeros();
    let declead = decxor.leading_zeros();
    let intbits:u8 = (64 -intlead) as u8;
    let decbits:u8 = (64 -declead) as u8;
    info!("10th vec: {},{},{},{}", intpart[0],intpart[1],intpart[2],intpart[3]);
    (min_int as i32,intbits,intpart,decbits,decpart)
}

pub(crate) fn diff_num_bits(mydata: &[i32]) -> (u8,Vec<u32>){
    info!("10th vec: {},{},{},{}", mydata[0],mydata[1],mydata[2],mydata[3]);
    let mut vec:Vec<i32> = Vec::new();
    let mut pre = 0;
    let mut diff = 0;
    let mut i= 0;

    for &b in mydata {
        //println!("b {} - pre {}",b,pre);
        diff = b - pre ;
        vec.push(diff);
        pre = b;
        i+=1;
    }
    info!("10th vec: {},{},{},{}", vec[0],vec[1],vec[2],vec[3]);
    let (base, bits,vec1) = delta_num_bits(&vec);
    info!("10th vec: {},{},{},{}", vec1[0],vec1[1],vec1[2],vec1[3]);
    (bits,vec1)
}

pub(crate) fn differential<T: Clone+Copy+Num>(mydata: &[T]) -> Vec<T>{
    let mut vec = Vec::new();
    let mut xor:u32 = 0;
    let mut delta = T::zero();
    let mut pre = T::zero();
    let mut i = 0;

    for &b in mydata {
        if i==0 {
            delta = b;
        }
        else {
            delta = b-pre;
        }
        vec.push(delta);
        pre = b;
        i += 1;
    }
    vec
}

pub(crate) fn delta_offset<T: Clone+Copy+Num+PartialOrd>(mydata: &[T]) -> Vec<T>{
    let mut vec = Vec::new();
    let mut xor:u32 = 0;
    let mut delta = T::zero();
    let mut min = T::zero();
    for &b in mydata {
        if b<min {
            min = b;
        }
    }
    // avoid using 2
    // min = min - T::one() - T::one();
    for &b in mydata {
        delta = (b - min);
        vec.push(delta);
    }
    vec
}

pub(crate) fn BP_encoder(mydata: &[i32]) -> Vec<u8>{
    let (base, num_bits, delta_vec) = delta_num_bits(mydata);
    println!("base int:{}",base);
    info!("Number of bits: {}", num_bits);
    info!("10th vec: {},{},{},{}", delta_vec[0],delta_vec[1],delta_vec[2],delta_vec[3]);
    let ubase_int = unsafe { mem::transmute::<i32, u32>(base) };
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    bitpack_vec.write(ubase_int,32);
    bitpack_vec.write(delta_vec.len() as u32, 32);
    bitpack_vec.write(num_bits as u32, 8);
    for &b in delta_vec.as_slice() {
        bitpack_vec.write(b, num_bits as usize).unwrap();
    }
    let vec = bitpack_vec.into_vec();
    info!("Length of compressed data: {}", vec.len());
    let ratio= vec.len() as f32 / (mydata.len() as f32*4.0);
    print!("{}",ratio);
    vec
}

pub(crate) fn bp_double_encoder<'a, T>(mydata: &[T], scl:usize) -> Vec<u8>
    where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
    let ldata: Vec<i32> = mydata.into_iter().map(|x| ((*x).into()* scl as f64).ceil() as i32).collect::<Vec<i32>>();
    let (base, num_bits, delta_vec) = delta_num_bits(ldata.as_ref());
    println!("base int:{}",base);
    println!("Number of bits: {}", num_bits);
    // info!("10th vec: {},{},{},{}", delta_vec[0],delta_vec[1],delta_vec[2],delta_vec[3]);
    let ubase_int = unsafe { mem::transmute::<i32, u32>(base) };
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    bitpack_vec.write(ubase_int,32);
    bitpack_vec.write(delta_vec.len() as u32, 32);
    bitpack_vec.write(num_bits as u32, 8);
    let mut i =0 ;
    for &b in delta_vec.as_slice() {
        // if i<10{
        //     println!("{}th value: {}",i,b);
        // }
        // i+=1;
        bitpack_vec.write(b, num_bits as usize).unwrap();

    }
    let vec = bitpack_vec.into_vec();
    info!("Length of compressed data: {}", vec.len());
    let ratio= vec.len() as f32 / (mydata.len() as f32*mem::size_of::<T>() as f32);
    print!("{}",ratio);
    vec
}

pub(crate) fn sprintz_double_encoder<'a, T>(mydata: &[T], scl:usize) -> Vec<u8>
    where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
    let ldata: Vec<i32> = mydata.into_iter().map(|x| ((*x).into()* scl as f64).ceil() as i32).collect::<Vec<i32>>();
    let (base, num_bits, delta_vec) = zigzag_delta_num_bits(ldata.as_ref());
    println!("base int:{}",base);
    info!("Number of bits: {}", num_bits);
    info!("10th vec: {},{},{},{}", delta_vec[0],delta_vec[1],delta_vec[2],delta_vec[3]);
    let ubase_int = unsafe { mem::transmute::<i32, u32>(base) };
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    bitpack_vec.write(ubase_int,32);
    bitpack_vec.write(delta_vec.len() as u32, 32);
    bitpack_vec.write(num_bits as u32, 8);
    let mut i =0 ;
    let start = Instant::now();
    for &b in delta_vec.as_slice() {
        // if i<10{
        //     println!("{}th value: {}",i,b);
        // }
        // i+=1;
        bitpack_vec.write(b, num_bits as usize).unwrap();

    }
    let vec = bitpack_vec.into_vec();
    let duration2 = start.elapsed();
    println!("Time elapsed in writing double function() is: {:?}", duration2);

    info!("Length of compressed data: {}", vec.len());
    let ratio= vec.len() as f32 / (mydata.len() as f32*mem::size_of::<T>() as f32);
    print!("{}",ratio);
    vec
}



pub(crate) fn split_double_encoder<'a, T>(mydata: &[T], scl:usize) -> Vec<u8>
    where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{
    let ldata: Vec<i64> = mydata.into_iter().map(|x| ((*x).into()* scl as f64) as i64).collect::<Vec<i64>>();
    let (base_int, int_bits, int_vec,dec_bits,dec_vec) = split_64num_bits(ldata.as_slice(),scl);
    println!("base int:{}",base_int);
    println!("Number of int bits: {}; number of decimal bits: {}", int_bits, dec_bits);
    println!("10th decimal vec: {},{},{},{}", dec_vec[0],dec_vec[1],dec_vec[2],dec_vec[3]);
    println!("10th integer vec: {},{},{},{}", int_vec[0],int_vec[1],int_vec[2],int_vec[3]);
    let ubase_int = unsafe { mem::transmute::<i32, u32>(base_int) };
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    bitpack_vec.write(ubase_int,32);
    bitpack_vec.write(ldata.len() as u32, 32);
    bitpack_vec.write(int_bits as u32, 8);
    bitpack_vec.write(dec_bits as u32, 8);
    let mut i =0 ;
    for &b in int_vec.as_slice() {
        // if i<10{
        //     println!("{}th value: {}",i,b);
        // }
        // i+=1;
        bitpack_vec.write(b as u32, int_bits as usize).unwrap();
    }
    for &d in dec_vec.as_slice() {
        bitpack_vec.write(d as u32,dec_bits as usize).unwrap();
    }
    let vec = bitpack_vec.into_vec();
    info!("compressed size: {}", vec.len());
    let ratio= vec.len() as f32 / (mydata.len() as f32*mem::size_of::<T>() as f32);
    print!("{}",ratio);
    vec
}

pub(crate) fn split_int_encoder(mydata: &[i32], scl:usize) -> Vec<u8>{
    let (int_bits, int_vec,dec_bits,dec_vec) = split_num_bits(mydata,scl);
    //info!("Number of int bits: {}; number of decimal bits: {}", int_bits, dec_bits);
    //info!("10th decimal vec: {},{},{},{}", dec_vec[0],dec_vec[1],dec_vec[2],dec_vec[3]);
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    for &b in int_vec.as_slice() {
        bitpack_vec.write(b, int_bits as usize).unwrap();
    }
    for &d in dec_vec.as_slice() {
        bitpack_vec.write(d, dec_bits as usize).unwrap();
    }
    let vec = bitpack_vec.into_vec();
    info!("compressed size: {}", vec.len());
    let ratio= vec.len() as f32 / (mydata.len() as f32*8.0);
    print!("{}",ratio);
    vec
}

pub(crate) fn deltaBP_encoder(mydata: &[i32]) -> Vec<u8>{
    let (num_bits, delta_vec) = diff_num_bits(mydata);
    info!("Number of bits: {}", num_bits);
    info!("10th vec: {},{},{},{}", delta_vec[0],delta_vec[1],delta_vec[2],delta_vec[3]);
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    for &b in delta_vec.as_slice() {
        bitpack_vec.write(b, num_bits as usize).unwrap();
    }
    let vec = bitpack_vec.into_vec();
    info!("Length of compressed data: {}", vec.len());
    let ratio= vec.len() as f32 / (mydata.len() as f32*4.0);
    print!("{}",ratio);
    vec

}


#[test]
fn test_smallbit() {
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(1);
    bitpack_vec.write(1, 1).unwrap();
    bitpack_vec.write(0, 1).unwrap();
    bitpack_vec.write(0, 1).unwrap();
    bitpack_vec.write(1, 1).unwrap();

    let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
    assert_eq!(bitpack.read(1).unwrap(), 1);
    assert_eq!(bitpack.read(1).unwrap(), 0);
    assert_eq!(bitpack.read(1).unwrap(), 0);
    assert_eq!(bitpack.read(1).unwrap(), 1);
}

#[test]
fn test_bigbit() {
    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    bitpack_vec.write(255, 8).unwrap();
    bitpack_vec.write(65535, 16).unwrap();
    bitpack_vec.write(65535, 16).unwrap();
    bitpack_vec.write(255, 8).unwrap();
    bitpack_vec.write(65535, 16).unwrap();

    let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
    assert_eq!(bitpack.read(8).unwrap(), 255);
    assert_eq!(bitpack.read(16).unwrap(), 65535);
    assert_eq!(bitpack.read(16).unwrap(), 65535);
    assert_eq!(bitpack.read(8).unwrap(), 255);
    assert_eq!(bitpack.read(16).unwrap(), 65535);
}

#[test]
fn test_moresmallbit() {
    let input = [
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0,
        1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1,
        0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    ];

    let mut bitpack_vec = BitPack::<Vec<u8>>::with_capacity(8);
    for &b in &input[..] {
        bitpack_vec.write(b, 1).unwrap();
    }

    let mut bitpack = BitPack::<&[u8]>::new(bitpack_vec.as_slice());
    for &b in &input[..] {
        assert_eq!(bitpack.read(1).unwrap(), b);
    }
}

#[test]
fn test_xor_f64() {
    let input= [0f64,1.0,2.0,3.0,4.0,100.0,20000.0,0.0001,0.001,0.01,0.1,1.0,1.1,1.2,2.1,1.3,1.31,1.2999999,1.3000000,1.3099999,1.3100001,1.31101,1.31102, 1.31902,1.31111,-1.311,10000.3,10000.31,10000.2999999,10000.3000000,10000.3099999,10000.3100001];
    let mut pre = 0u64;
    let mut cur = 0u64;
    let a = 0u64;
    let b = 1u64;
    let x_or = a ^ b;
    let x = 3.1415f32;
    let xu = unsafe { mem::transmute::<f32, u32>(x) };
    println!("{:#066b}",xu);
    let y = 3.1415f64;
    let yu = unsafe { mem::transmute::<f64, u64>(y) };
    println!("{:#066b}",yu);
    println!("{:#066b}^{:#066b}={:#066b}", a, b, x_or);
    for &ele in &input{
        cur = unsafe { mem::transmute::<f64, u64>(ele) };
        let xor = cur ^ pre;
        println!("{:#066b}  XOR", xor);
        println!("{:#066b}  {}", cur,ele);
        pre = cur;

    }
}


#[test]
fn run_gorilla_example_buff() {
    let file_vec =  [0.66f64,1.41,1.41,1.50,2.72,3.14];

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.to_vec(),None,None);
    let comp = GorillaCompress::new(10,10);
    let start = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration = start.elapsed();
    println!("Time elapsed in {:?} gorilla compress function() is: {:?}",comp.type_id(), duration);

    let start1 = Instant::now();
    let decomp = comp.decode(compressed);
    let duration1 = start1.elapsed();
    println!("decompressed vec:{:?}", decomp);
    println!("Time elapsed in {:?} decompress function() is: {:?}",comp.type_id(), duration1);
}

#[test]
fn test_zigzag(){
    let org =  0i32;
    println!("zigzag:{}",unzigzag(1));
    assert_eq!(unzigzag(zigzag(org)),org);
    assert_eq!(unzigzag(zigzag(1)),1);
    assert_eq!(unzigzag(zigzag(-1)),-1);
    assert_eq!(unzigzag(zigzag(100)),100);
    assert_eq!(unzigzag(zigzag(-100)),-100);
}

#[test]
fn test_to_bytes(){
    let value: u32 = 0x1FFFF;
    let bytes = value.to_be_bytes();
    println!("{:?}" , bytes);
}

#[test]
fn test_xor_on_file() {
    let file_iter = construct_file_iterator_skip_newline::<f64>("../UCRArchive2018/Kernel/randomwalkdatasample1k-10k", 1, ',');
    let ve: Vec<f64> = file_iter.unwrap().collect();
    let input = &ve[..20];
    let mut pre = 0u64;
    let mut cur = 0u64;
    let a = 0u64;
    let b = 1u64;
    let x_or = a ^ b;
    println!("{:#066b}^{:#066b}={:#066b}", a, b, x_or);
    for &ele in input{
        cur = unsafe { mem::transmute::<f64, u64>(ele) };
        let xor = cur ^ pre;
        println!("{:#066b}  XOR", xor);
        println!("{:#066b}  {}", cur,ele);
        pre = cur;

    }
}