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
use my_bit_vec::BitVec;
use crate::query::bit_vec_iter::BVIter;

pub fn tsbs_bench(compression: &str, query: &str){
    let mut other = 0.0;
    let mut fl_time = 0.0;
    let mut total = 0.0;
    let mut duration6 ;
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
                duration6 = start6.elapsed();

                println!("Time elapsed in buff project function() is: {:?}", duration6);
            },
            "buff-major" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.buff_major_decode_condition(latitude,iter.clone());
                longti = comp.buff_major_decode_condition(longtitude,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff-major project function() is: {:?}", duration6);

            },
            "gorilla" => {
                let comp = GorillaCompress::new(10,10);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in gorilla project function() is: {:?}", duration6);
            },
            "gorillabd" => {
                let comp = GorillaBDCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in gorillabd project function() is: {:?}", duration6);
            },

            "snappy" => {
                let comp = SnappyCompress::new(10,10);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in snappy project function() is: {:?}", duration6);
            },

            "gzip" => {
                let comp = GZipCompress::new(10,10);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in gzip project function() is: {:?}", duration6);

            },

            "fixed" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.fixed_decode_condition(latitude,iter.clone());
                longti = comp.fixed_decode_condition(longtitude,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in fixed project function() is: {:?}", duration6);

            },
            "sprintz" => {
                let comp = SprintzDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.decode_condition(latitude,iter.clone());
                longti = comp.decode_condition(longtitude,iter);
                duration6 = start6.elapsed();
                println!("Time elapsed in sprintz project function() is: {:?}", duration6);

            },
            "buff-slice" => {
                let comp = BuffSliceCompress::new(10,10,scl);
                let start6 = Instant::now();
                lati = comp.buff_slice_decode_condition(latitude,iter.clone());
                longti = comp.buff_slice_decode_condition(longtitude,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff-slice project function() is: {:?}", duration6);

            },
            _ => {panic!("Compression not supported yet.")}
        }

        let duration = start.elapsed();
        fl_time= duration6.as_micros() as f64/1000.0664;
        total = duration.as_micros() as f64/1000.0f64;
        other = total-fl_time;
        println!("extracted latitude values: {:?}",lati);
        println!("extracted latitude values: {:?}",longti);
        println!("Time elapsed in tsbs last-loc function() is: {:?}", duration);

    }
    else if query=="single" {
        let fuel = get_comp_file("d_fuel_state.csv",compression);
        let r_tag = get_csv_file("r_tags_id.csv");
        let t_id = get_csv_file("t_id_south.csv");
        let scl = 10;
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
        let mut f = Vec::new();
        println!("integer join runtime: {:?}",start.elapsed());

        match compression{
            "buff" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.buff_decode_condition(fuel,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff single function() is: {:?}", duration6);
            },
            "buff-major" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.buff_major_decode_condition(fuel,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff-major single function() is: {:?}", duration6);

            },
            "gorilla" => {
                let comp = GorillaCompress::new(10,10);
                let start6 = Instant::now();
                f = comp.decode_condition(fuel,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in gorilla single function() is: {:?}", duration6);
            },
            "gorillabd" => {
                let comp = GorillaBDCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.decode_condition(fuel,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in gorillabd single function() is: {:?}", duration6);
            },

            "snappy" => {
                let comp = SnappyCompress::new(10,10);
                let start6 = Instant::now();
                f = comp.decode_condition(fuel,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in snappy single function() is: {:?}", duration6);
            },

            "gzip" => {
                let comp = GZipCompress::new(10,10);
                let start6 = Instant::now();
                f = comp.decode_condition(fuel,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in gzip single function() is: {:?}", duration6);

            },

            "fixed" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.fixed_decode_condition(fuel,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in fixed single function() is: {:?}", duration6);

            },
            "sprintz" => {
                let comp = SprintzDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.decode_condition(fuel,iter);
                duration6 = start6.elapsed();
                println!("Time elapsed in sprintz single function() is: {:?}", duration6);

            },
            "buff-slice" => {
                let comp = BuffSliceCompress::new(10,10,scl);
                let start6 = Instant::now();
                comp.buff_slice_range_smaller_filter_condition(fuel,0.1,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff-slice single function() is: {:?}", duration6);

            },
            _ => {panic!("Compression not supported yet.")}
        }
        let mut res = Vec::new();

        if f.len()>0{
            for (&id, &f_level) in id_loc.iter().zip(f.iter()){
                if f_level<0.1{
                    res.push(id);
                }
            }
        }
        let duration = start.elapsed();
        fl_time= duration6.as_micros() as f64/1000.0664;
        total = duration.as_micros() as f64/1000.0f64;
        other = total-fl_time;
        println!("extracted low_fuel values: {:?}",res);
        println!("Time elapsed in tsbs low-fuel function() is: {:?}", duration);
    }
    else if query=="range" {
        let fuel = get_comp_file("dt_cur_load.csv",compression);
        let r_tag = get_csv_file("d_tags_id.csv");
        let t_id = get_csv_file("t_id_west.csv");
        let scl = 10000;
        let pred = 0.9;
        let mut f = BitVec::new();
        let len  = r_tag.len();
        let start = Instant::now();

        let mut i = len-1;
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
        println!("integer join runtime: {:?}",start.elapsed());

        match compression{
            "buff" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.buff_range_filter_condition(fuel,pred,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff range function() is: {:?}", duration6);
            },
            "buff-major" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.buff_range_filter_majority_condition(fuel,pred,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff-major range function() is: {:?}", duration6);

            },
            "gorilla" => {
                let comp = GorillaCompress::new(10,10);
                let start6 = Instant::now();
                f = comp.range_filter_condition(fuel,pred,iter,len);
                duration6 = start6.elapsed();

                println!("Time elapsed in gorilla range function() is: {:?}", duration6);
            },
            "gorillabd" => {
                let comp = GorillaBDCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.range_filter_condition(fuel,pred, iter, len);
                duration6 = start6.elapsed();

                println!("Time elapsed in gorillabd range function() is: {:?}", duration6);
            },

            "snappy" => {
                let comp = SnappyCompress::new(10,10);
                let start6 = Instant::now();
                f = comp.range_filter_condition(fuel,pred, iter, len);
                duration6 = start6.elapsed();

                println!("Time elapsed in snappy range function() is: {:?}", duration6);
            },

            "gzip" => {
                let comp = GZipCompress::new(10,10);
                let start6 = Instant::now();
                f = comp.range_filter_condition(fuel,pred, iter, len);
                duration6 = start6.elapsed();

                println!("Time elapsed in gzip range function() is: {:?}", duration6);

            },

            "fixed" => {
                let comp = SplitBDDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.fixed_range_filter_condition(fuel,pred,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in fixed range function() is: {:?}", duration6);

            },
            "sprintz" => {
                let comp = SprintzDoubleCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.range_filter_condition(fuel,pred,iter);
                duration6 = start6.elapsed();
                println!("Time elapsed in sprintz range function() is: {:?}", duration6);

            },
            "buff-slice" => {
                let comp = BuffSliceCompress::new(10,10,scl);
                let start6 = Instant::now();
                f = comp.buff_slice_range_filter_condition(fuel,pred,iter);
                duration6 = start6.elapsed();

                println!("Time elapsed in buff-slice range function() is: {:?}", duration6);

            },
            _ => {panic!("Compression not supported yet.")}
        }

        let duration = start.elapsed();
        fl_time= duration6.as_micros() as f64/1000.0664;
        total = duration.as_micros() as f64/1000.0f64;
        other = total-fl_time;
        println!("extracted cur_load values: {}",f.cardinality());
        println!("Time elapsed in tsbs cur_load function() is: {:?}", duration);

    }


    println!("summary: {},{},{},{},{}", query, compression, fl_time, other, total);


}

#[test]
fn test_file_read(){
    let file_iter = construct_file_iterator_skip_newline::<usize>("script/data/t_id_south.csv", 0, ',');
    let file_vec: Vec<usize> = file_iter.unwrap().collect();

    for &e in file_vec.iter().rev() {
        println!("Element at position {}", e);
    }
}