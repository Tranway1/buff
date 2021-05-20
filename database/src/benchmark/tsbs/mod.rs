use crate::benchmark::{BENCH_DATA, read_a_file, get_comp_file, get_csv_file};
use crate::client::construct_file_iterator_skip_newline;
use std::collections::{HashSet, HashMap};
use std::iter::FromIterator;
use crate::compress::split_double::SplitBDDoubleCompress;
use std::time::Instant;
use crate::compress::buff_slice::BuffSliceCompress;
use crate::compress::sprintz::SprintzDoubleCompress;
use crate::compress::gorilla::{GorillaCompress, GorillaBDCompress};
use crate::methods::compress::{SnappyCompress, GZipCompress};

pub fn tsbs_bench(compression: &str, query: &str){
    if query=="project"{
        let longtitude = get_comp_file("r_longitude.csv",compression);
        let latitude = get_comp_file("r_latitude.csv",compression);
        let r_tag = get_csv_file("r_tags_id.csv");
        let t_id = get_csv_file("t_id_south.csv");
        let scl = 100000;
        let start = Instant::now();

        let mut i = r_tag.len()-1;
        let mut cands = Vec::new();
        let mut id_set:HashSet<usize> = HashSet::from_iter(t_id.iter().cloned());
        let mut id_loc = Vec::new();
        // get tag id
        for &e in r_tag.iter().rev() {
            if id_set.contains(&e){
                cands.insert(0,i);
                id_set.remove(&e);
                id_loc.insert(0,e);
                if id_set.is_empty(){
                    break;
                }
            }
            i-=1;
        }
        let mut iter = cands.iter();
        let mut lati = Vec::new();
        let mut longti = Vec::new();

        println!("integer join runtime: {:?}",start.elapsed());

        match compression{
            "buff" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.buff_decode_condition(latitude,iter.clone());
                longti = comp.buff_decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in buff project function() is: {:?}", duration6);
            },
            "buff-major" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.buff_major_decode_condition(latitude,iter.clone());
                longti = comp.buff_major_decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in buff-major project function() is: {:?}", duration6);

            },
            "gorilla" => {
                let comp = GorillaCompress::new(10,10);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in gorilla project function() is: {:?}", duration6);
            },
            "gorillabd" => {
                let comp = GorillaBDCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in gorillabd project function() is: {:?}", duration6);
            },

            "snappy" => {
                let comp = SnappyCompress::new(10,10);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in snappy project function() is: {:?}", duration6);
            },

            "gzip" => {
                let comp = GZipCompress::new(10,10);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in gzip project function() is: {:?}", duration6);

            },

            "fixed" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.fixed_decode_condition(latitude,iter.clone());
                longti = comp.fixed_decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in fixed project function() is: {:?}", duration6);

            },
            "sprintz" => {
                let comp = SprintzDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();
                println!("Time elapsed in sprintz project function() is: {:?}", duration6);

            },
            "buff-slice" => {
                let comp = BuffSliceCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.buff_slice_decode_condition(latitude,iter.clone());
                longti = comp.buff_slice_decode_condition(longtitude,iter);
                let duration6 = start6.elapsed();

                println!("Time elapsed in buff-slice project function() is: {:?}", duration6);

            },
            _ => {panic!("Compression not supported yet.")}
        }

        let duration = start.elapsed();
        println!("extracted latitude values: {:?}",lati);
        println!("extracted latitude values: {:?}",longti);
        println!("Time elapsed in tsbs last-loc function() is: {:?}", duration);

    }
    else if query=="mat" { }

}

#[test]
fn test_file_read(){
    let file_iter = construct_file_iterator_skip_newline::<usize>("script/data/t_id_south.csv", 0, ',');
    let file_vec: Vec<usize> = file_iter.unwrap().collect();

    for &e in file_vec.iter().rev() {
        println!("Element at position {}", e);
    }
}