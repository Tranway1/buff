use crate::benchmark::{BENCH_DATA, read_a_file, get_comp_file};
use std::time::Instant;
use crate::compress::sprintz::SprintzDoubleCompress;
use crate::compress::split_double::SplitBDDoubleCompress;
use crate::methods::compress::{GZipCompress, SnappyCompress};
use crate::compress::gorilla::{GorillaCompress, GorillaBDCompress};
use crate::compress::buff_slice::BuffSliceCompress;
use crate::compress::scaled_slice::ScaledSliceCompress;


pub fn influx_bench(compression: &str, query: &str){
    let comp_max = get_comp_file("acr_temperature.csv",compression);
    let scl = 10000;
    let window = 240*7;
    let start = 500*window;
    let end = 600*window;
    let starttime = Instant::now();
    let mut other = 0.0;
    let mut fl_time = 0.0;
    let mut total = 0.0;
    let mut duration6=starttime.elapsed() ;

    match compression{

        "buff" => {
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.byte_fixed_max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in buff max function() is: {:?}", duration6);

            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.buff_max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in buff max_groupby function() is: {:?}", duration6);

            }
        },
        "buff-major" => {
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.buff_max_majority(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in buff-major max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.buff_max_majority_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in buff-major max_groupby function() is: {:?}", duration6);

            }
        },
        "gorilla" => {
            let comp = GorillaCompress::new(10,10);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in gorilla max function() is: {:?}", duration6);

            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in gorilla max_groupby function() is: {:?}", duration6);

            }
        },
        "gorillabd" => {
            let comp = GorillaBDCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in gorillabd max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in gorillabd max_groupby function() is: {:?}", duration6);

            }
        },

        "snappy" => {
            let comp = SnappyCompress::new(10,10);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in snappy max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in snappy max_groupby function() is: {:?}", duration6);

            }
        },

        "gzip" => {
            let comp = GZipCompress::new(10,10);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in gzip max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in gzip max_groupby function() is: {:?}", duration6);

            }
        },

        "fixed" => {
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.fixed_max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in fixed bp max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.fixed_max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in fixed bp max_groupby function() is: {:?}", duration6);

            }
        },
        "sprintz" => {
            let comp = SprintzDoubleCompress::new(10,10,scl);
            if query=="max"{
                let start6 = Instant::now();
                comp.max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in sprintz max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                let start6 = Instant::now();
                comp.max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in sprintz max_groupby function() is: {:?}", duration6);

            }
        },
        "buff-slice" => {
           let comp = BuffSliceCompress::new(10,10,scl);

            if query=="max"{
                let start6 = Instant::now();
                comp.buff_slice_buffmax(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in buff-slice max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                // println!("Time elapsed in buff-slice max groupby not supported");
                let start6 = Instant::now();
                comp.buff_slice_max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in buff-slice max groupby function() is: {:?}", duration6);
            }
        },

        "scaled-slice" => {
            let comp = ScaledSliceCompress::new(10,10,scl);

            if query=="max"{
                let start6 = Instant::now();
                comp.scaled_slice_max(comp_max);
                duration6 = start6.elapsed();
                println!("Time elapsed in scaled-slice max function() is: {:?}", duration6);
            }else if query=="max_groupby"{
                // println!("Time elapsed in buff-slice max groupby not supported");
                let start6 = Instant::now();
                comp.scaled_slice_max_range(comp_max,start,end,window);
                duration6 = start6.elapsed();
                println!("Time elapsed in scaled-slice max groupby function() is: {:?}", duration6);
            }
        },

        _ => {panic!("Compression not supported yet.")}
    }
    let duration = starttime.elapsed();
    fl_time= duration6.as_micros() as f64/1000.0664;
    total = duration.as_micros() as f64/1000.0f64;
    other = total-fl_time;
    println!("summary: {},{},{},{},{}", query, compression, fl_time, other, total);
}