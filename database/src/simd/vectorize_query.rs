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

extern crate myroaring;

const VECTOR_SIZE: usize = size_of::<__m256i>();
const VEC_LEN: usize =  100000000;
const PRED: u8 =  127;


pub fn get_random_byte_vec(n: usize) -> Vec<u8> {
    let random_bytes: Vec<u8> = (0..n).map(|_| rand::random::<u8>()).collect();
    random_bytes
}


// fn reduce(x: &[u8]) -> u8 {
//     assert!(x.len() % 32 == 0);
//     let mut sum = u8x32::splat(0); // [0, 0, 0, 0]
//     for i in (0..x.len()).step_by(32) {
//         sum += u8x32::from_slice_unaligned(&x[i..]);
//     }
//     sum.wrapping_sum()
// }

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

fn filtering_progressive(x: &[u8], bit_vec:Bitmap) {
    let mut res = Bitmap::create();
    let mut iterator = bit_vec.iter();
    let mut it = iterator.next();
    let mut bitpack = BitPack::<&[u8]>::new(x);
    let mut dec_cur = 0;
    let mut dec_pre:u32 = 0;
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
            res.add(0);
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
            res.add(dec_cur);
        }

        it = iterator.next();
        dec_pre=dec_cur;
    }

    println!("\tprogressive greater than filtering: {}", res.cardinality());
}


// #[test]
// fn test_simd() {
//     let a = i32x4::new(1, 1, 3, 3);
//     let b = i32x4::new(2, 2, 0, 0);
//
// // ge: >= (Greater Eequal; see also lt, le, gt, eq, ne).
//     let m = a.ge(i32x4::splat(2));
//
//     if m.any() {
//         // all / any / none allow coherent control flow
//         let d = m.select(a, b);
//         assert_eq!(d, i32x4::new(2, 2, 3, 3));
//     }
// }



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
    let mut ratios = vec![0.00001, 0.0001,0.0005,0.001, 0.005,0.01,0.012,0.014,0.016,0.018, 0.02, 0.04, 0.06, 0.08, 0.1];
    let repeat = 6;
    for ratio in ratios{
        let mut pf = 0;
        let mut sf = 0;
        for i in 0..repeat{
            let in_vec = get_random_byte_vec(VEC_LEN);
            let mut cur_rb = Bitmap::create();
            let checks = (VEC_LEN as f64 * ratio) as usize;
            let mut indices: Vec<u32> = (0..(VEC_LEN-1) as u32).collect();
            thread_rng().shuffle(&mut indices);
            let mut count = 0;
            for ind in indices{
                count += 1;
                // println!("get:{}", ind);
                cur_rb.add(ind);
                if count>checks{
                    break;
                }
            }
            println!("\tbitmap cadinality: {}", cur_rb.cardinality() as f64/VEC_LEN as f64);

            let start = Instant::now();
            unsafe { filtering_progressive(in_vec.as_slice(), cur_rb); }
            let duration = start.elapsed();
            pf += duration.as_micros();
            println!("\tTime elapsed in progressive filtering greater than function() is: {:?}", duration);

            let start1 = Instant::now();
            unsafe { filtering_simd(in_vec.as_slice()); }
            let duration1 = start1.elapsed();
            sf += duration1.as_micros();
            println!("\tTime elapsed in simd greater than function() is: {:?}", duration1);

            let start2 = Instant::now();
            unsafe { filtering_simd_myroaring(in_vec.as_slice()); }
            let duration2 = start2.elapsed();
            // sf += duration2.as_micros();
            println!("\tTime elapsed in simd greater than function() is: {:?}", duration2);



            println!("\t------");
        }
        println!("AVG time elapsed in ratio, simd, progressive filtering is: {}, {:?}, {:?}", ratio, pf as f64/1000.0/repeat as f64,  sf as f64/1000.0/repeat as f64);
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