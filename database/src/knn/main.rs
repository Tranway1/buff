use tsz::stream::BufferedReader;
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};
use time_series_start::knn::{slurp_file, classify};
use std::env;
use log::{ info};

fn main() {

    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let train_set = &args[1];
    let test_set = &args[2];
    let precision = args[3].parse::<i32>().unwrap();

    let training_set = slurp_file(&Path::new(train_set),precision);
    let validation_sample = slurp_file(&Path::new(test_set),precision);

    let num_correct = validation_sample.iter()
        .filter(|x| {
            // println!("first {},{},{}",x.label,x.pixels[0],x.pixels[1]);
            classify(training_set.as_slice(), x.pixels.as_slice()) == x.label
        })
        .count();

    println!("{},{},Percentage correct,{}",train_set,precision,
             num_correct as f64 / validation_sample.len() as f64 );
}