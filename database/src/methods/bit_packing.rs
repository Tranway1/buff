use bitpacking::{BitPacker, BitPacker4x};
use log::{info, trace, warn};
use futures::future::err;
use num::Num;
use std::mem;
use crate::client::construct_file_iterator_skip_newline;

pub const MAX_BITS: usize = 32;
const BYTE_BITS: usize = 8;


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

    pub fn into_vec(self) -> Vec<u8> {
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

pub(crate) fn delta_num_bits(mydata: &[i32]) -> (u8,Vec<u32>){
    info!("10th vec: {},{},{},{}", mydata[0],mydata[1],mydata[2],mydata[3]);
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
    (bits,vec)
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
    let (bits,vec1) = delta_num_bits(&vec);
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

pub(crate) fn BP_encoder(mydata: &[i32]) -> Vec<u8>{
    let (num_bits, delta_vec) = delta_num_bits(mydata);
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
    let input= [0f64,1.0,2.0,3.0,4.0,100.0,20000.0,0.0001,0.001,0.01,0.1,1.0,1.1,1.2,1.3,1.31,1.2999999,1.3000000,1.3099999,1.3100001,1.31101,1.31102, 1.31902,1.31111,-1.311,10000.3,10000.31,10000.2999999,10000.3000000,10000.3099999,10000.3100001];
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