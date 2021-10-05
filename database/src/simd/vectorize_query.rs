// use packed_simd::i32x4;
// use packed_simd::u8x32;
use std::time::Instant;
use core::arch::x86_64::*;
use core::cmp;
use core::mem::size_of;
use croaring::Bitmap;
use crate::methods::bit_packing::BitPack;
use rand::{Rng, thread_rng};
use std::mem;
use self::myroaring::RoaringBitmap;
use crate::compress::buff_slice::flip;
use my_bit_vec::{BitBlock, BitVec};
use std::borrow::BorrowMut;
use crate::query::bit_vec_iter::BVIter;
use std::mem::transmute;

extern crate myroaring;

const VECTOR_SIZE: usize = size_of::<__m256i>();
const VEC_LEN: usize =  100000000;
// const PRED: u8 =  127;
const PRED: u8 =  200;

pub fn get_random_byte_vec(n: usize) -> Vec<u8> {
    let random_bytes: Vec<u8> = (0..n).map(|_| rand::random::<u8>()).collect();
    random_bytes
}


unsafe fn sum_simd(x: &[u8]) {
    let haystack = x;
    let start_ptr = haystack.as_ptr();
    let end_ptr = haystack[haystack.len()..].as_ptr();
    let mut ptr = start_ptr;
    let mut sum = _mm256_set1_epi8(0);
    // println!("simd sum: {:?}", sum);
    let ep = end_ptr.sub(VECTOR_SIZE);
    while ptr <= ep {
        let word1 = _mm256_lddqu_si256(ptr as *const __m256i);
        // println!("current word: {:?}", word1);
        sum = _mm256_add_epi8(sum, word1);
        // println!("simd sum: {:?}", sum);
        ptr = ptr.add(VECTOR_SIZE);
    }
    println!("simd sum: {:?}", sum);
}


unsafe fn filtering_simd(x: &[u8]) {
    let haystack = x;
    let start_ptr = haystack.as_ptr();
    let end_ptr = haystack[haystack.len()..].as_ptr();
    let mut ptr = start_ptr;
    // translate into i8 logics
    let offset:u8 = 1u8<<7;
    let predicate = mem::transmute::<u8, i8>(offset^PRED);
    let mut pred_word = _mm256_set1_epi8(predicate);
    let ep = end_ptr.sub(VECTOR_SIZE);
    // let mut gt = Vec::<i32>::new();
    let mut qualified = 0;
    while ptr <= ep {
        let word1 = _mm256_lddqu_si256(  ptr as *const __m256i);
        let gter = _mm256_cmpgt_epi8(word1, pred_word);
        let mask = _mm256_movemask_epi8(gter);
        // println!("{:?}{:?}{:b}", word1, pred_word, mask);
        // gt.push(mask);
        qualified += _popcnt32(mask);
        ptr = ptr.add(VECTOR_SIZE);
    }
    println!("\tsimd greater than filter: {}", qualified);
}

unsafe fn filtering_simd_myroaring(x: &[u8]) {
    let haystack = x;
    let start_ptr = haystack.as_ptr();
    let end_ptr = haystack[haystack.len()..].as_ptr();
    let mut ptr = start_ptr;
    // translate into i8 logics
    let offset:u8 = 1u8<<7;
    let predicate = mem::transmute::<u8, i8>(offset^PRED);
    let mut pred_word = _mm256_set1_epi8(predicate);
    let ep = end_ptr.sub(VECTOR_SIZE);
    // let mut gt = Vec::<i32>::new();
    let mut res = RoaringBitmap::new();
    let mut qualified = 0;
    let mut count = 0;
    while ptr <= ep {
        let word1 = _mm256_lddqu_si256(  ptr as *const __m256i);
        let gter = _mm256_cmpgt_epi8(word1, pred_word);
        let mask = _mm256_movemask_epi8(gter);
        res.insert_direct_u32(count,mem::transmute::<i32, u32>(mask));
        // println!("{:?}{:?}{:b}", word1, pred_word, mask);
        // gt.push(mask);
        // qualified += _popcnt32(mask);
        count += 32;
        ptr = ptr.add(VECTOR_SIZE);
    }
    println!("\tsimd greater than bitmap: {}", res.len());
}


pub unsafe fn range_simd_myroaring(x: &[u8], res: &mut RoaringBitmap,pred:u8)  -> RoaringBitmap{
    let haystack = x;
    let start_ptr = haystack.as_ptr();
    let end_ptr = haystack[haystack.len()..].as_ptr();
    let mut ptr = start_ptr;

    let predicate = mem::transmute::<u8, i8>(flip(pred));
    let mut pred_word = _mm256_set1_epi8(predicate);
    let ep = end_ptr.sub(VECTOR_SIZE);
    let mut middle = RoaringBitmap::new();
    // let mut gt = Vec::<i32>::new();
    let mut qualified = 0;
    let mut count = 0;
    while ptr <= ep {
        let word1 = _mm256_lddqu_si256(  ptr as *const __m256i);
        let gter = _mm256_cmpgt_epi8 (word1, pred_word);
        let gt_mask = _mm256_movemask_epi8(gter);
        res.insert_direct_u32(count,mem::transmute::<i32, u32>(gt_mask));

        let equal = _mm256_cmpeq_epi8(word1, pred_word);
        let eq_mask = _mm256_movemask_epi8(equal);
        middle.insert_direct_u32(count,mem::transmute::<i32, u32>(eq_mask));

        // println!("{:?}{:?}{:b}", word1, pred_word, mask);
        // gt.push(mask);
        // qualified += _popcnt32(mask);
        count += 32;
        ptr = ptr.add(VECTOR_SIZE);
    }
    println!("\tsimd greater than bitmap: {}", middle.len());
    middle
}

pub unsafe fn equal_simd_myroaring(x: &[u8], pred:u8)  -> RoaringBitmap{
    let haystack = x;
    let start_ptr = haystack.as_ptr();
    let end_ptr = haystack[haystack.len()..].as_ptr();
    let mut ptr = start_ptr;

    let predicate = mem::transmute::<u8, i8>(flip(pred));
    println!("current predicate:{}",pred);
    let mut pred_word = _mm256_set1_epi8(predicate);
    let ep = end_ptr.sub(VECTOR_SIZE);
    let mut middle = RoaringBitmap::new();
    // let mut gt = Vec::<i32>::new();
    let mut qualified = 0;
    let mut count = 0;
    while ptr <= ep {
        let word1 = _mm256_lddqu_si256(  ptr as *const __m256i);

        let equal = _mm256_cmpeq_epi8(word1, pred_word);
        let eq_mask = _mm256_movemask_epi8(equal);
        // middle.insert_direct_u32(count,mem::transmute::<i32, u32>(eq_mask));

        // println!("{:?}{:?}{:b}", word1, pred_word, mask);
        // gt.push(mask);
        qualified += _popcnt32(eq_mask);
        count += 32;
        ptr = ptr.add(VECTOR_SIZE);
    }
    println!("\tsimd equal bitmap: {}", qualified);
    middle
}

/// this it range query operator with my bit-vec
pub unsafe fn range_simd_mybitvec(x: &[u8], mut res: &mut BitVec, pred:u8) -> BitVec{
    let haystack = x;
    let len = haystack.len();
    let start_ptr = haystack.as_ptr();
    let end_ptr = haystack[len..].as_ptr();
    let mut ptr = start_ptr;

    let predicate = mem::transmute::<u8, i8>(flip(pred));
    let mut pred_word = _mm256_set1_epi8(predicate);
    let ep = end_ptr.sub(VECTOR_SIZE);
    let mut resv = Vec::new();
    let mut midv =  Vec::new();
    while ptr <= ep {
        let word1 = _mm256_lddqu_si256(  ptr as *const __m256i);
        let gter = _mm256_cmpgt_epi8 (word1, pred_word);
        let gt_mask = _mm256_movemask_epi8(gter);
        resv.push(mem::transmute::<i32, u32>(gt_mask));

        let equal = _mm256_cmpeq_epi8(word1, pred_word);
        let eq_mask = _mm256_movemask_epi8(equal);
        midv.push(mem::transmute::<i32, u32>(eq_mask));

        ptr = ptr.add(VECTOR_SIZE);
    }
    res.set_storage(&resv);
    // println!("\tsimd greater than filtering: {}", res.cardinality());
    BitVec::from_vec(&mut midv,len)
}

/// this it equality query operator with my bit-vec
pub unsafe fn equal_simd_mybitvec(x: &[u8], pred:u8)  -> BitVec{
    let haystack = x;
    let len = haystack.len();
    let start_ptr = haystack.as_ptr();
    let end_ptr = haystack[len..].as_ptr();
    let mut ptr = start_ptr;

    let predicate = mem::transmute::<u8, i8>(flip(pred));
    // println!("current predicate:{}",pred);
    let mut pred_word = _mm256_set1_epi8(predicate);
    let ep = end_ptr.sub(VECTOR_SIZE);
    let mut bitvec = Vec::new();
    while ptr <= ep {
        let word1 = _mm256_lddqu_si256(  ptr as *const __m256i);

        let equal = _mm256_cmpeq_epi8(word1, pred_word);
        let eq_mask = _mm256_movemask_epi8(equal);
        bitvec.push(mem::transmute::<i32, u32>(eq_mask));

        ptr = ptr.add(VECTOR_SIZE);
    }
    let bitv = BitVec::from_vec(&mut bitvec,len);
    println!("\tsimd equal bitmap: {}", bitv.cardinality());
    bitv
}

fn filtering_range_progressive(x: &[u8], bit_vec: &BitVec) {
    let len = x.len();
    let mut res = BitVec::from_elem(len, false);
    let mut rb = BitVec::from_elem(len, false);
    let mut iterator = BVIter::new(bit_vec);
    let mut it = iterator.next();
    let mut bitpack = BitPack::<&[u8]>::new(x);
    let mut dec_cur = 0;
    let mut dec_pre = 0;
    let mut delta = 0;
    let mut dec = 0;
    let pred = PRED;
    if it!=None{
        dec_cur = it.unwrap();
        if dec_cur!=0{
            bitpack.skip_n_byte((dec_cur) as usize);
        }
        dec = bitpack.read_byte().unwrap();
        // println!("{} first match: {}", dec_cur, dec);
        if dec > pred{
            res.set(0,true);
        }
        else if dec == pred{
            rb.set(0,true);
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
        if dec > pred{
            res.set(dec_cur, true);
        }
        else if dec == pred{
            rb.set(dec_cur,true);
        }
        it = iterator.next();
        dec_pre=dec_cur;
    }

    println!("\tprogressive greater than filtering: {}, equal: {}", res.cardinality(),rb.cardinality());
}

fn filtering_equal_progressive( x: &[u8], bit_vec: &BitVec) {
    let len = x.len();
    let mut res = BitVec::from_elem(len, false);
    let mut rb = BitVec::from_elem(len, false);
    let mut iterator = BVIter::new(bit_vec);
    let mut it = iterator.next();
    let mut bitpack = BitPack::<&[u8]>::new(x);
    let mut dec_cur = 0;
    let mut dec_pre = 0;
    let mut delta = 0;
    let mut dec = 0;
    let pred = PRED;
    if it!=None{
        dec_cur = it.unwrap();
        if dec_cur!=0{
            bitpack.skip_n_byte((dec_cur) as usize);
        }
        dec = bitpack.read_byte().unwrap();
        // println!("{} first match: {}", dec_cur, dec);
        if dec == pred{
            rb.set(0,true);
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
        if dec == pred{
            rb.set(dec_cur,true);
        }
        it = iterator.next();
        dec_pre=dec_cur;
    }

    println!("\tprogressive equal: {}",rb.cardinality());
}


#[test]
fn test_single_vecu8() {
    let in_vec = get_random_byte_vec(VEC_LEN);

    let start = Instant::now();
    let mut sum = 0u64;
    for &val in in_vec.iter(){
        sum += val as u64;
    }
    println!("single thread sum: {}", sum);
    let duration = start.elapsed();
    println!("Time elapsed in single thread sum function() is: {:?}", duration);
}

#[test]
fn test_simd_vecu8() {
    let in_vec = get_random_byte_vec(VEC_LEN);

    let start = Instant::now();
    // let mut sum = reduce(in_vec.as_ref());
    // println!("single thread sum: {}", sum);
    let duration = start.elapsed();
    println!("Time elapsed in single thread sum function() is: {:?}", duration);
}

#[test]
fn test_simd_avx_vecu8() {
    let in_vec = get_random_byte_vec(VEC_LEN);

    let start = Instant::now();
    unsafe { sum_simd(in_vec.as_slice()); }
    let duration = start.elapsed();
    println!("Time elapsed in simd sum function() is: {:?}", duration);
}

#[test]
fn test_progressive_filtering_vecu8() {
    // let mut ratios = vec![0.00001, 0.0001,0.001, 0.01,0.02, 0.04, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.1];
    let mut ratios = vec![0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.1];
    let repeat = 8;
    for ratio in ratios{
        let mut prf = 0;
        let mut srf = 0;
        let mut sef = 0;
        let mut pef = 0;
        unsafe {
            for i in 0..repeat {
                let in_vec = get_random_byte_vec(VEC_LEN);
                let mut cur_rb = BitVec::from_elem(VEC_LEN, false);
                let checks = (VEC_LEN as f64 * ratio) as usize;
                let mut indices: Vec<u32> = (0..(VEC_LEN - 1) as u32).collect();
                thread_rng().shuffle(&mut indices);
                let mut count = 0;
                for ind in indices {
                    count += 1;
                    // println!("get:{}", ind);
                    cur_rb.set(ind as usize, true);
                    if count > checks {
                        break;
                    }
                }
                println!("\tbitmap cadinality: {}", cur_rb.cardinality() as f64 / VEC_LEN as f64);

                let start = Instant::now();
                unsafe { filtering_range_progressive(in_vec.as_slice(), &cur_rb); }
                let duration = start.elapsed();
                prf += duration.as_micros();
                println!("\tTime elapsed in progressive filtering greater than function() is: {:?}", duration);

                let start1 = Instant::now();
                unsafe { filtering_equal_progressive(in_vec.as_slice(), &cur_rb); }
                let duration1 = start1.elapsed();
                pef += duration1.as_micros();
                println!("\tTime elapsed in progressive filtering equal function() is: {:?}", duration1);

                let start2 = Instant::now();
                let mut res = BitVec::from_elem(VEC_LEN, false);
                range_simd_mybitvec(in_vec.as_slice(), &mut res, PRED);
                res.and(&cur_rb);
                println!("\tsimd range bitmap: {}", res.cardinality());
                let duration2 = start2.elapsed();
                srf += duration2.as_micros();
                println!("\tTime elapsed in simd greater than function() is: {:?}", duration2);

                let start3 = Instant::now();
                let mut eqbitv= equal_simd_mybitvec(in_vec.as_slice(), PRED);
                eqbitv.and(&cur_rb);
                println!("\tsimd equal bitmap: {}", eqbitv.cardinality());
                let duration3 = start3.elapsed();
                sef += duration3.as_micros();
                println!("\tTime elapsed in simd equal function() is: {:?}", duration3);


                println!("\t------");
            }
        }
        println!("AVG time elapsed in ratio, simd, progressive filtering is: {}, {:?}, {:?},{:?},{:?}", ratio, prf as f64/1000.0/repeat as f64,  pef as f64/1000.0/repeat as f64,  srf as f64/1000.0/repeat as f64,  sef as f64/1000.0/repeat as f64);
    }

}

#[test]
fn test_insert_bitword() {
    let mut b = RoaringBitmap::new();
    let inserted = b.insert_direct_u32(0,128);
    let inserted = b.insert_direct_u32(64,127);
    let inserted = b.insert_direct_u32(64000,127);
    assert!(inserted);

    // for i in 0..7 {
    assert!(b.contains(7));
    // }
    for i in 64..70 {
        assert!(b.contains(i));
    }
    assert_eq!(b.len(),15);
    let mut iter = b.iter();
    let mut cur =  iter.next();
    while (cur!=None){
        println!("{}", cur.unwrap());
        cur = iter.next();
    }
}

#[test]
fn test_bitmap_insert(){
    let mut indices: Vec<u32> = (0..(VEC_LEN-1) as u32).collect();
    thread_rng().shuffle(&mut indices);
    let round = 5;
    let size = VEC_LEN/10;

    // test croaring bitmap
    let start =Instant::now();
    for i in 0..round{
        let mut croaring = Bitmap::create();
        for &ind in &indices[0..(size)]{
            croaring.add(ind);
        }
    }
    let duration = start.elapsed();
    println!("croaring bitmap insert performance for {} inserts: {}ms", size, duration.as_millis() as f64 /round as f64);

    // test roaring-rs bitmap
    let start =Instant::now();
    for i in 0..round{
        let mut roaring_rs = RoaringBitmap::new();
        for  &ind in &indices[0..(size)]{
            roaring_rs.insert(ind);
        }
    }
    let duration = start.elapsed();
    println!("roaring-rs bitmap insert performance for {} inserts: {}ms", size, duration.as_millis() as f64 /round as f64);


    // test bit_vec bitmap
    let start1 =Instant::now();
    for i in 0..round{
        // let mut bitv = BitVec::with_capacity(VEC_LEN);
        let mut bitv = BitVec::from_elem(VEC_LEN, false);
        // let mut bitv= BitVec::new();
        for  &ind in &indices[0..size]{
            bitv.set(ind as usize, true);
        }
        // println!("{}th",i);
    }
    let duration1 = start1.elapsed();
    println!("bit_vec bitmap insert performance for {} inserts: {}ms", size, duration1.as_millis() as f64 /round as f64);


    // test croaring bitmap iter
    let mut croaring = Bitmap::create();
    for &ind in &indices[0..(size)]{
        croaring.add(ind);
    }

    let start =Instant::now();
    croaring.run_optimize();
    for i in 0..round{
        let mut sum = 0;
        let mut iter = croaring.iter();
        let mut t = iter.next();
        let mut cur = 0;
        while t!=None{
            cur = t.unwrap();
            sum += cur as usize;
            t=iter.next();
        }
        println!("croaring sum: {}", sum);

    }
    let duration = start.elapsed();
    println!("croaring bitmap insert performance for {} inserts: {}ms", size, duration.as_millis() as f64 /round as f64);

    // test roaring-rs bitmap iter
    let mut roaring_rs = RoaringBitmap::new();
    for  &ind in &indices[0..(size)]{
        roaring_rs.insert(ind);
    }
    let start =Instant::now();
    roaring_rs.optimize();
    for i in 0..round{
        let mut sum = 0 as usize;
        let mut iter = roaring_rs.iter();
        let mut t = iter.next();
        let mut cur = 0;
        while t!=None{
            cur = t.unwrap();
            sum += cur as usize;
            t=iter.next();
        }
        println!("roaring-rs sum: {}", sum);
    }
    let duration = start.elapsed();
    println!("roaring-rs bitmap iter performance for {} inserts: {}ms", size, duration.as_millis() as f64 /round as f64);


    // test bit_vec bitmap iter
    // let mut bitv = BitVec::with_capacity(VEC_LEN);

    let mut bitv = BitVec::from_elem(VEC_LEN, false);

    // let mut bitv= BitVec::new();
    for  &ind in &indices[0..size]{
        bitv.set(ind as usize, true);
    }
    let start1 =Instant::now();

    for i in 0..round{
        let mut sum = 0;
        let mut iter = BVIter::new(&bitv);
        let mut t = iter.next();
        let mut cur = 0;
        while t!=None{
            cur = t.unwrap();
            sum += cur;
            t=iter.next();
        }
        println!("bit_vec sum: {}", sum);
    }
    let duration1 = start1.elapsed();
    println!("bit_vec bitmap iter performance for {} inserts: {}ms", size, duration1.as_millis() as f64 /round as f64);


}


#[test]
fn test_bit_vec_ser_deser(){
    let mut vec = vec![32u32;10];
    let bin = bincode::serialize(&vec).unwrap();
    let decoded: Vec<u32> = bincode::deserialize(&bin[..]).unwrap();
    assert_eq!(vec,decoded);

    let mut bitv = BitVec::from_vec(&mut vec,330);
    let encoded  = bitv.to_binary();
    let nbitv = BitVec::from_binary(&encoded);
    assert_eq!(nbitv.len(),330);
    assert_eq!(nbitv.storage().to_vec(),vec);
    assert_eq!(bitv.cardinality(),10);
}