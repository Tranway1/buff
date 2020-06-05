use nalgebra::{MatrixMN, Dim, Scalar, DMatrix, DVector, DMatrixSlice};
use core::fmt;
use std::cmp::max;
use ndarray;
use ndarray::{ArrayD, ArrayViewMut1, ArrayView1, Array1};
use ndarray::Array2;
use rustfft::{FFTplanner,FFTnum};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{Mul, Neg, Div, Add};
use num::{Num, Float};
use std::borrow::BorrowMut;

pub struct LCCE<T> {
    x: Array2<T>,
    dictionary: Array2<T>,
    nn: usize,
    compression: usize
}

impl<'a,T: FFTnum+ PartialOrd+ Clone + Float + Serialize + Deserialize<'a>> LCCE<T> {
    pub fn new(x: Array2<T>, dictionary: Array2<T>, nn: usize, compression: usize) -> Self {
        LCCE {
            x: x,
            dictionary: dictionary,
            nn: nn,
            compression:compression
        }
    }

    pub fn run(&mut self){
        let (nrows_x, ncols_x) = (self.x.rows(),self.x.cols());
        let (nrows_dic, ncols_dic) = (self.dictionary.rows(),self.dictionary.cols());
        //let mut array = Array2::zeros((4, 3));
        //array[[1, 1]] = 7;
        let mut values: Array2<T> =  Array2::zeros((nrows_x, self.nn));
        let mut nn_pos: Array2<T> = Array2::zeros((nrows_x, self.nn));
        let mut d: Array2<T> = Array2::zeros((nrows_x, nrows_dic));

        for i in 0..nrows_x {
            for j in 0..nrows_dic {
                let cc_seq =  nccc_compressed(self.x.row_mut(i), self.dictionary.row_mut(j), self.compression);
                let value = T::one()-*cc_seq.to_vec().iter().fold(None, |min, x| match min {
                    None => Some(x),
                    Some(y) => Some(if x > y { x } else { y }),
                }).unwrap();
                d[[i,j]] = value;
            }
        }
        for i in 0..self.nn {
            for j in 0..nrows_dic {

            }
        }
    }
}


pub fn nccc_compressed<'a,T: FFTnum + PartialOrd + Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayViewMut1<T>, dic_row: ArrayViewMut1<T>, comp: usize) ->Array1<T>{
    let len = max(xrow.len(),dic_row.len());
    let fftlen = len; // fft length calculation
    let size = len;
    let compression = len * comp /100;
    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(size);

    let mut xinput: Vec<Complex<T>> = xrow.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();

    let mut xoutput: Vec<Complex<T>> = vec![Complex::zero(); size];

    let mut dic_input: Vec<Complex<T>> = dic_row.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();

    let mut dic_output: Vec<Complex<T>> = vec![Complex::zero(); size];

    fft.process(&mut xinput, &mut xoutput);
    fft.process(&mut dic_input, &mut dic_output);

    leading_fft(&mut xoutput,compression);
    leading_fft(&mut dic_output,compression);

    let mut mul: Vec<Complex<T>> = xoutput.iter().zip(dic_output).map(|(x,y)| x.mul(y.conj())).collect();
//        let d:Vec<T> = mul.iter().map(|x| x.re).collect();


    /* ifft*/
    let mut iplanner = FFTplanner::new(true);
    let ifft = iplanner.plan_fft(size);

    let mut ioutput: Vec<Complex<T>> = vec![Complex::zero(); size];

    ifft.process(&mut mul, &mut ioutput);

    let mut reioutput:Vec<T> = ioutput.iter().map(|c| c.re).collect();
    let mut cc_sequence  = Array1::from_vec(reioutput);

    let norm = l2_norm(xrow.view()) * l2_norm(dic_row.view());
    let res=cc_sequence.mapv_into(|e|e/norm);
    res
}

//fn l2_norm<T> (x: ArrayViewMut1<T>) -> T {
//    x.dot(&x)
//}
//
//fn normalize<T:Div> (mut x: Array1<T>) -> Array1<T> {
//    let norm = l2_norm(x.view_mut());
//    x.mapv_inplace(|e| (e / norm));
//    x
//}
//
fn leading_fft<T:Neg+Num+Clone> (x: &mut Vec<Complex<T>>, k: usize) {
    let m = x.len()/2 + 1;
    for (i, a) in x.iter_mut().enumerate(){
        if i>m-k/2 && i<m+k/2{
            *a = Complex::zero();
        }
    }
}

//
//fn l1_norm<T:Add> (x: ArrayView1<T>) -> T {
//    x.fold(0., |acc, elem| acc + elem.abs())
//}

fn l2_norm<T:Clone+FFTnum+Num+Float> (x: ArrayView1<T>) -> T {
    x.dot(&x) .sqrt()
}

fn normalize<T: Div+ Clone+FFTnum+Num+Float> (mut x: Array1<T>) -> Array1<T> {
    let norm = l2_norm(x.view());
    x.mapv_inplace(|e| e/norm);
    x
}



#[test]
fn test_crate(){

    use rand::prelude::*;

    if rand::random() { // generates a boolean
        // Try printing a random unicode code point (probably a bad idea)!
        println!("char: {}", rand::random::<char>());
    }

    let mut rng = rand::thread_rng();
    let y: f64 = rng.gen(); // generates a float between 0 and 1

    let mut dic_nums = Vec::<f32>::with_capacity(800);

    for _ in 0..800 {
        // 1 (inclusive) to 21 (exclusive)
        dic_nums.push(rng.gen_range(1.0, 10.0));
    }
    print!("{:?}, ", dic_nums);

    let mut nums = Vec::<f32>::with_capacity(4000);

    for _ in 0..4000 {
        // 1 (inclusive) to 21 (exclusive)
        nums.push(rng.gen_range(1.0, 10.0));
    }

    let x = Array2::<f32>::from_shape_vec((100,40), nums).unwrap();
    let dic = Array2::<f32>::from_shape_vec((20,40), dic_nums).unwrap();
    let mut lcce = LCCE::new(x,dic,20,15);
    lcce.run();
    let x = array![1., 2., 3., 4., 5.];
    println!("||x||_2 = {}", l2_norm(x.view()));
    //println!("||x||_1 = {}", l1_norm(x.view()));
    println!("Normalizing x yields {:?}", normalize(x));
}