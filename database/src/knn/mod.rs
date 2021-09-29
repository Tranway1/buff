use tsz::stream::BufferedReader;
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, BufRead};
use rust_decimal::prelude::FromStr;
use crate::methods::prec_double::{get_precision_bound, PrecisionBound};
use itertools::Itertools;
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use ndarray::Array2;
use std::collections::HashMap;
use core::mem;
use crate::methods::compress::TEST_FILE;


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

pub fn get_gamma(gammas: &Path) -> HashMap<String,isize>{
    let mut mymap = HashMap::new();
    // mymap.insert("foo".to_string(), "bar".to_string());
    // println!("{}", mymap.find_equiv(&"foo"));
    // println!("{}", mymap.find_equiv(&"not there"));
    let lines = BufReader::new(File::open(gammas).unwrap()).lines().skip(0)
        .map(|x| x.unwrap())
        .collect::<Vec<String>>();

    for line in lines{
        // println!("{}",line);
        let mut iter = line.trim().split(',');
        let fname =  iter.next().unwrap().split('/').last().unwrap();
        // println!("{}",fname);
        let gamma_arr:Vec<isize> = iter.map(|x|isize::from_str(x).unwrap()).collect();
        let gamma = gamma_arr.get(7).unwrap();
        mymap.insert(fname.to_string(), *gamma);
    }

    // BufReader::new(File::open(gammas).unwrap())
    //     .lines()
    //     .skip(0)
    //     .map(|line| {
    //         let line = line.unwrap();
    //
    //     });

    mymap
}

pub fn fft_ifft_ratio(data: &[f64], ratio: f64) -> Vec<f64>{
    let size = data.len();
    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(size);

    let mut input: Vec<Complex<f64>> = data.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();

    let mut output: Vec<Complex<f64>> = vec![Complex::zero(); size];
    fft.process(&mut input, &mut output);
    let mut comp:Vec<f64> = output.iter().map(|x|x.re).collect();
    let mut icomp:Vec<f64> = output.into_iter().map(|x|x.im).collect();

    // println!("compressed real part {:?}",comp );
    // println!("compressed image part {:?}",icomp );


    let mut iplanner = FFTplanner::new(true);
    let isize = comp.len();
    let ifft = iplanner.plan_fft(isize);

    let mut ioutput: Vec<Complex<f64>> = vec![Complex::zero(); isize];

    let mut ccomp = Vec::new();

    for (re,im) in comp.iter().zip(icomp.iter()){
        ccomp.push(Complex::new(*re,*im));
    }

    let k= (ratio * size as f64) as usize;
    // println!("k:{}, length: {}",k,size);
    let mut m = 0;
    for (i, a) in ccomp.iter_mut().enumerate(){
        //&& i<=le-k
        if i>=k && i<=size-k {
            m+=1;
            *a = Complex::zero();
        }
    }
    // println!("number trimmed: {}",m);
    // let mut icomp: Vec<Complex<f32>> = comp.iter()
    // 	.map(|x| Complex::new(*x,Zero::zero()))
    // 	.collect();

    ifft.process(&mut ccomp, &mut ioutput);

    let ires:Vec<f64>  = ioutput.iter().map(|c| c.re).collect();
    let fres =ires.iter().map(|&x|{x/isize as f64 }).collect::<Vec<_>>();
    // let mut sse = 0.0;
    // for (&a, &b) in data.iter().zip(fres.as_slice().iter()) {
    //     sse = sse + ( (a - b) / a );
        // println!("{},{},cur err: {}",a, b, (a - b) / a);
    // }

    // println!("err rate: {}",sse);

    // println!("decompressed {:?}",fres );
    fres
}

pub fn fft_file(file: &Path, ratio:f64) -> Vec<LabelPixel> {

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
                pixels: {
                    let ts = iter.collect();
                    if ratio>=1.0 {
                        ts
                    }else{
                        fft_ifft_ratio(&ts,ratio)
                }
                }
            }
        })
        .collect()
}

pub fn paa_file(file: &Path, wsize:i32) -> Vec<LabelPixel> {
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
                pixels: {
                    let vec = iter.collect();
                    if wsize<=1 {
                        vec
                    }else{
                        let vsize = vec.len();
                        let mut paa_data= vec.chunks(wsize as usize)
                            .map(|x| {
                                x.iter().fold(0f64, |sum, &i| sum + i
                                ) / (wsize as f64)
                            });
                        let mut unpaa = Vec::new();

                        let mut val:f64 = 0.0;
                        for i in 0..vsize as i32 {
                            if i%wsize == 0{
                                val=paa_data.next().unwrap();
                            }
                            unpaa.push(val);
                        }
                        unpaa
                    }
                }
            }
        })
        .collect()
}


pub fn paa_buff_file(file: &Path, wsize:i32, prec:i32) -> Vec<LabelPixel> {
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
                pixels: {
                    let vec = iter.collect();
                    if wsize<=1 {
                        if prec<0 {
                            vec
                        }else{
                            vec.into_iter().map(|x|{
                                let bd = bound.precision_bound(x);
                                // println!("before: {}, after: {}",x,bd);
                                bd
                            }).collect()
                        }
                    }else{
                        let vsize = vec.len();
                        let mut paa_data= vec.chunks(wsize as usize)
                            .map(|x| {
                                x.iter().fold(0f64, |sum, &i| sum + i
                                ) / (wsize as f64)
                            });
                        let mut unpaa = Vec::new();

                        let mut val:f64 = 0.0;
                        for i in 0..vsize as i32 {
                            if i%wsize == 0{
                                val=paa_data.next().unwrap();
                            }
                            unpaa.push(val);
                        }
                        if prec<0 {
                            unpaa
                        }else{
                            unpaa.into_iter().map(|x|{
                                let bd = bound.precision_bound(x);
                                // println!("before: {}, after: {}",x,bd);
                                bd
                            }).collect()
                        }
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

#[test]
fn test_paa_on_file() {
    paa_file(&Path::new("../UCRArchive2018/Kernel/randomwalkdatasample1k-40k"), 1);
}


#[test]
fn test_get_gammas() {
    let gm=get_gamma(&Path::new("/Users/chunwei/research/TimeSeriesDB/database/script/data/gamma_ucr_new.csv"));
    println!("map size: {}", gm.len());
    let key = "ACSF1_TRAIN";
    println!("{}", gm.get(key).unwrap());
}