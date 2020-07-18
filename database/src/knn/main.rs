use tsz::stream::BufferedReader;
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};
use time_series_start::knn::{slurp_file, classify};

fn main() {
    let training_set = slurp_file(&Path::new("../UCRArchive2018/CBF/CBF_TRAIN"));
    let validation_sample = slurp_file(&Path::new("../UCRArchive2018/CBF/CBF_TRAIN"));

    let num_correct = validation_sample.iter()
        .filter(|x| {
            classify(training_set.as_slice(), x.pixels.as_slice()) == x.label
        })
        .count();

    println!("Percentage correct: {}%",
             num_correct as f64 / validation_sample.len() as f64 * 100.0);
}