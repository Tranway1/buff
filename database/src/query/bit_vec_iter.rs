use my_bit_vec::{BitVec, BitBlock};
use std::ops::Range;
use parity_snappy::{compress, decompress};

///
/// by Chunwei
/// this is iterator for position of true values
pub struct BVIter<'a> {
    storage: &'a [u32],
    len: usize,
    cached: u32,
    pointer: usize
}

impl <'a> BVIter<'a>{
    #[inline]
    pub fn new( bv: &'a BitVec<u32>) -> Self {
        let storage: &'a [u32]= bv.storage();
        let len:usize = storage.len();
        let pointer:usize = 0;
        let cached: u32 = storage[pointer];
        BVIter{
            storage,
            len,
            cached,
            pointer
        }
    }
}

impl <'a> Iterator for BVIter<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        while self.cached == 0{
            if self.pointer == self.len-1{
                return None
            }
            self.pointer += 1;
            self.cached = self.storage[self.pointer];
        }
        let t= self.cached & !(self.cached-1);
        self.cached ^= t;
        let res = (self.pointer << 5) + (t-1).count_ones() as usize;
        Some(res)
    }

}

#[inline]
fn get_right_most1(mut cached: u32, )->u32{
    let t= cached & !(cached-1);
    cached ^= t;
    (t-1).count_ones()
}

///
/// apply snappy compression on
#[inline]
pub fn bit_vec_compress(input: &[u8]) -> Vec<u8> {
    compress(input)
}


///
/// apply snappy decompression on
#[inline]
pub fn bit_vec_decompress(input: &[u8]) -> Vec<u8> {
    decompress(input).unwrap()
}

#[test]
fn test_right_most1() {
    assert_eq!(get_right_most1(4),2);
    assert_eq!(get_right_most1(1),0)
}

#[test]
fn test_BVIter() {
    let bv = BitVec::from_bytes(&[0b10100000, 0b00010010]);
    let mut iter = BVIter::new(&bv);
    let mut t = iter.next();
    let mut cur = 0;
    while t!=None{
        cur = t.unwrap();
        println!("true records: {}", cur);
        t=iter.next();
    }

}
