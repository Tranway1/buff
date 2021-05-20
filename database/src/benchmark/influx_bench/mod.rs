use crate::benchmark::{BENCH_DATA, read_a_file, get_comp_file};
use std::time::Instant;
use crate::compress::sprintz::SprintzDoubleCompress;
use crate::compress::split_double::SplitBDDoubleCompress;
use crate::methods::compress::{GZipCompress, SnappyCompress};
use crate::compress::gorilla::{GorillaCompress, GorillaBDCompress};


pub fn influx_bench(compression: &str, query: &str){
    let comp_max = get_comp_file("acr_temperature.csv",compression);
    let scl = 10000;
    let window = 240*7;
    let start = 500*window;
    let end = 600*window;


    match compression{

        "buff" => {
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.byte_fixed_max(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in buff max function() is: {:?}", duration6);

            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.buff_max_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in buff max_groupby function() is: {:?}", duration6);

            }
        },
        "buff-major" => {
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.buff_max_majority(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in buff-major max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.buff_max_majority_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in buff-major max_groupby function() is: {:?}", duration6);

            }
        },
        "gorilla" => {
            let comp = GorillaCompress::new(10,10);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in gorilla max function() is: {:?}", duration6);

            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in gorilla max_groupby function() is: {:?}", duration6);

            }
        },
        "gorillabd" => {
            let comp = GorillaBDCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in gorillabd max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in gorillabd max_groupby function() is: {:?}", duration6);

            }
        },

        "snappy" => {
            let comp = SnappyCompress::new(10,10);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in snappy max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in snappy max_groupby function() is: {:?}", duration6);

            }
        },

        "gzip" => {
            let comp = GZipCompress::new(10,10);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in gzip max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in gzip max_groupby function() is: {:?}", duration6);

            }
        },

        "fixed" => {
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.fixed_max(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in fixed bp max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.fixed_max_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in fixed bp max_groupby function() is: {:?}", duration6);

            }
        },
        "sprintz" => {
            let comp = SprintzDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                let duration6 = start6.elapsed();
                println!("Time elapsed in sprintz max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                let duration6 = start6.elapsed();
                println!("Time elapsed in sprintz max_groupby function() is: {:?}", duration6);

            }
        },
        "buff-slice" => {
            println!("Time elapsed in buff-slice max not supported");
            if query=="max"{

            }else if query=="max_groupby"{

            }
        },

        _ => {panic!("Compression not supported yet.")}
    }
}