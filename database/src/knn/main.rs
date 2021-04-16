use tsz::stream::BufferedReader;
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};
use time_series_start::knn::{paa_file, classify, paa_buff_file, fft_file, grail_file, get_gamma};
use std::env;
use log::{ info};
use std::borrow::Borrow;

fn main() {

    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let train_set = &args[1];
    let test_set = &args[2];
    let precision = args[3].parse::<f64>().unwrap();
    let window = 1;
    let train_name = train_set.split('/').last().unwrap();
    let root: &str = "../ucr_dict/";
    let gm=get_gamma(&Path::new("../database/script/data/gamma_ucr_new.csv"));
    if !(gm.contains_key(train_name)){
        return;
    }
    let gamma= *gm.get(train_name).unwrap() as usize;
    // println!("dataset: {} with gamma: {}",train_name,gamma);

    let dict_file = format!("{}{}", root, train_name);
    // println!("dict file: {} ",dict_file);


    // let window = args[4].parse::<i32>().unwrap();

    // let training_set = paa_buff_file(&Path::new(train_set),window,precision);
    // let validation_sample = paa_buff_file(&Path::new(test_set),window,precision);

    // let training_set = fft_file(&Path::new(train_set),precision);
    // let validation_sample = fft_file(&Path::new(test_set),precision);

    let training_set = grail_file(&Path::new(train_set),&Path::new(&dict_file), gamma,usize::max_value());
    let validation_sample = grail_file(&Path::new(test_set),&Path::new(&dict_file),gamma,usize::max_value());

    let num_correct = validation_sample.iter()
        .filter(|x| {
            // println!("first {},{},{}",x.label,x.pixels[0],x.pixels[1]);
            classify(training_set.as_slice(), x.pixels.as_slice()) == x.label
        })
        .count();

    println!("{},{},{},Percentage correct,{}",train_set,precision,window,
             num_correct as f64 / validation_sample.len() as f64 );
}