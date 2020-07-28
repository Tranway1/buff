use tsz::stream::BufferedReader;
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};
use rust_decimal::prelude::FromStr;
use crate::methods::prec_double::{get_precision_bound, PrecisionBound};

pub struct LabelPixel {
    pub label: isize,
    pub pixels: Vec<f64>
}


pub fn slurp_file(file: &Path, prec:i32) -> Vec<LabelPixel> {
    let prec_delta = get_precision_bound(prec);
    let mut bound = PrecisionBound::new(prec_delta);
    BufReader::new(File::open(file).unwrap())
        .lines()
        .skip(1)
        .map(|line| {
            let line = line.unwrap();
            let mut iter = line.trim()
                .split(',')
                .map(|x| f64::from_str(x).unwrap());

            LabelPixel {
                label: iter.next().unwrap() as isize,
                pixels: {if prec<0 {
                    iter.collect()
                }else{
                    iter.map(|x|{
                        let bd = bound.precision_bound(x);
                        // println!("before: {}, after: {}",x,bd);
                        bd
                    }).collect()
                }
                    }
            }
        })
        .collect()
}

fn distance_sqr(x: &[f64], y: &[f64]) -> f64 {
    // run through the two vectors, summing up the squares of the differences
    x.iter()
        .zip(y.iter())
        .fold(0f64, |s, (&a, &b)| s + (a - b) * (a - b))
}

pub fn classify(training: &[LabelPixel], pixels: &[f64]) -> isize {
    training
        .iter()
        // find element of `training` with the smallest distance_sqr to `pixel`
        .min_by(|x, y| distance_sqr(x.pixels.as_slice(), pixels).partial_cmp(&(distance_sqr(y.pixels.as_slice(), pixels))).unwrap()).unwrap()
        .label
}