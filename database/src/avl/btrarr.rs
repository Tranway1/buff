use std::collections::HashMap;
use crate::stats::{Stats, merge_adjacent};
use crate::client::construct_file_iterator_skip_newline;
use std::time::{SystemTime, Instant};
use num::Float;
use crate::compress::split_double::SplitBDDoubleCompress;
use crate::segment::Segment;
use crate::compress::btr_array::BtrArrayIndex;
use rand::Rng;

pub const K_DEGREE: usize = 8;

fn foo(){
    let mut results:HashMap<i32, Vec<f64>> = HashMap::new();
    for k in 1..10 {
        results.insert(k, Vec::new());
    }
    let mut fnum = 1.0;
    for v in 1..10 {
        results.get_mut(&v).unwrap().push(fnum);
        fnum+=1.0;
    }

    for v in 1..10 {
        println!("{}th vector: {:?}",v, results.get(&v).unwrap());
    }
}


pub fn run_btr_array_index(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = BtrArrayIndex::new(10,10,scl);
    let start1 = Instant::now();

    let (compressed, ind) = comp.encode(&mut seg);

    let duration1 = start1.elapsed();
    let comp_cp = ind.clone();
    let comp_eq = ind.clone();
    let comp_sum = ind.clone();
    let comp_max = ind.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd byte compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    // comp.byte_fixed_decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd byte decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    // comp.byte_fixed_range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd byte range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.max_no_partial(file_vec.as_ref(), comp_sum);
    let duration4 = start4.elapsed();
    println!("Time elapsed in btr max with no partial is: {:?}", duration4);

    let start5 = Instant::now();
    comp.max_one_partial(file_vec.as_ref(), comp_max);
    let duration5 = start5.elapsed();
    println!("Time elapsed in btr max with one partial is: {:?}", duration5);

    let start6 = Instant::now();
    comp.max_two_partial(file_vec.as_ref(), comp_eq);
    let duration6 = start6.elapsed();
    println!("Time elapsed in btr max with two partial  is: {:?}", duration6);


    println!("Performance:{},{},{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration5.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration6.as_nanos() as f64 / 1024.0/1024.0


    )
}




pub fn update_stats(map: &mut HashMap<i32, Vec<Stats<f64>>>, stats: Stats<f64>, stats_level: i32){
    if map.contains_key(&stats_level){
        let cur = map.get_mut(&stats_level).unwrap();
        cur.push(stats);

        // new nodes from current layer form a new node for the upper layer
        if cur.len()%K_DEGREE == 0{
            // fold K nodes
            let mut base = cur.len()-K_DEGREE;
            let mut res = cur[base].clone();
            base+=1;
            while base<cur.len(){
                res = merge_adjacent(&res,cur.get(base).unwrap());
                base+=1;
            }

            // update the upper layer
            let mut layer = stats_level+1;
            update_stats(map,res,layer);
        }

    }
    else {
        let mut new_vec = Vec::new();
        new_vec.push(stats);
        map.insert(stats_level, new_vec);
    }


}

#[test]
fn test_given_min_max() {
    // foo();
    let mut rng = rand::thread_rng();
    let mut results:HashMap<i32, Vec<Stats<f64>>> = HashMap::new();
    for x in 0..200{
        update_stats(&mut results, Stats::new(x,x+1, rng.gen::<f64>(),rng.gen::<f64>(),rng.gen::<usize>(),rng.gen::<f64>(),rng.gen::<f64>()),1);
    }

    for v in results.keys() {
        println!("{}th level vector with length {} : {:?}",v, results.get(v).unwrap().len(),results.get(v).unwrap());
    }

}