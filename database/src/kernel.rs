use core::fmt;
use std::cmp::max;
use std::mem;
use ndarray;
use ndarray::{ArrayD, ArrayViewMut1, ArrayView1, Array1, LinalgScalar};
use ndarray::Array2;
use rustfft::{FFTplanner,FFTnum};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::ops::{Mul, Neg, Div, Add};
use num::{Num, Float,FromPrimitive};
use std::borrow::BorrowMut;
use ndarray_linalg::*;
use ndarray_linalg::eigh;
use std::time::Instant;
use crate::segment::Segment;
use crate::methods::compress::CompressionMethod;

#[derive(Clone )]
pub struct Kernel<T> {
    dictionary: Array2<T>,
    ffted_dictionary: Vec<Vec<Complex<T>>> ,
    gamma: usize,
    coeffs: usize,
    eigen_vec: Array2<T>,
    in_eigen_val: Array2<T>,
    batchsize: usize
}

impl<'a,T: FFTnum + PartialOrd + std::fmt::Debug + Clone + Float  +Scalar + Lapack+ Serialize> Kernel<T> {
    pub fn new( dictionary: Array2<T>, gamma: usize, coeffs: usize, batchsize: usize) -> Self {
        Kernel {
            dictionary: dictionary,
            ffted_dictionary: Vec::new(),
            gamma: gamma,
            coeffs: coeffs,
            eigen_vec:  Array2::zeros((1, 1)),
            in_eigen_val: Array2::zeros((1, 1)),
            batchsize: batchsize
        }
    }

    // this is optimized version
    pub fn dict_pre_process(&mut self){
        self.ffted_dictionary = fft_preprocess(&self.dictionary,self.coeffs);

        let (nrows_dic, ncols_dic) = (self.dictionary.rows(),self.dictionary.cols());
        let mut w: Array2<T> = Array2::zeros((nrows_dic, nrows_dic));
        let mut dist_comp:usize= 0;
        for i in 0..nrows_dic {
            for j in 0..nrows_dic {
                w[[i,j]] = opt_sinkcompressed(self.dictionary.row(i), self.dictionary.row(j), self.gamma, self.coeffs, self.ffted_dictionary[i].as_ref(), self.ffted_dictionary[j].as_ref());
                dist_comp = dist_comp+1;
            }
        }
        let (val, vecs) = w.clone().eigh(UPLO::Upper).unwrap();
        println!("eigenvalues = \n{:?}", val);
        //println!("V = \n{:?}", vecs);
        let size_eig = val.len();
        let mut one_2d = Array2::ones((size_eig,1));
        let mut val_2d = Array2::from_shape_vec((size_eig,1), val.to_vec()).unwrap();
        let mut inv_val_2d= one_2d/val_2d;
        let sqrt_val = inv_val_2d.map(|x|Scalar::from_real(x.sqrt()));
        let mut dia = Array2::eye(size_eig);
        let mut va = dia.dot(&sqrt_val);
        self.in_eigen_val = va;
        self.eigen_vec = vecs;
        println!("eigen vec shape:{}*{}", self.eigen_vec.rows(),self.eigen_vec.cols())
    }

    // optimized version
    pub fn run(&self, x: Array2<T>){
        let mut x_fft = fft_preprocess(&x,self.coeffs);

        let start = Instant::now();
        let (nrows_x, ncols_x) = (x.rows(),x.cols());
        let (nrows_dic, ncols_dic) = (self.dictionary.rows(),self.dictionary.cols());
        let mut e: Array2<T> = Array2::zeros((nrows_x, nrows_dic));
        let mut dist_comp:usize= 0;

        for i in 0..nrows_x {
            //println!("{}",i);
            for j in 0..nrows_dic {
                e[[i,j]] = opt_sinkcompressed(x.row(i), self.dictionary.row(j), self.gamma, self.coeffs, x_fft[i].as_ref(), self.ffted_dictionary[j].as_ref());
                dist_comp = dist_comp + 1;
            }
        }

        let mut z_exact = e.dot(&self.eigen_vec);
        z_exact = z_exact.dot(&self.in_eigen_val);

        let duration = start.elapsed();
        println!("Time elapsed in run kernel function() is: {:?}", duration);
    }


    pub fn rbfdict_pre_process(&mut self){
        let (nrows_dic, ncols_dic) = (self.dictionary.rows(),self.dictionary.cols());
        let mut w: Array2<T> = Array2::zeros((nrows_dic, nrows_dic));
        let mut dist_comp:usize= 0;
        for i in 0..nrows_dic {
            for j in 0..nrows_dic {
                w[[i,j]] = rbfkernel(self.dictionary.row(i), self.dictionary.row(j), 0.4);
                dist_comp = dist_comp+1;
            }
        }
        let (val, vecs) = w.clone().eigh(UPLO::Upper).unwrap();
        println!("eigenvalues = \n{:?}", val);
        //println!("V = \n{:?}", vecs);
        let size_eig = val.len();
        let mut one_2d = Array2::ones((size_eig,1));
        let mut val_2d = Array2::from_shape_vec((size_eig,1), val.to_vec()).unwrap();
        let mut inv_val_2d= one_2d/val_2d;
        let sqrt_val = inv_val_2d.map(|x|Scalar::from_real(x.sqrt()));
        let mut dia = Array2::eye(size_eig);
        let mut va = dia.dot(&sqrt_val);
        self.in_eigen_val = va;
        self.eigen_vec = vecs;
        println!("eigen vec shape:{}*{}", self.eigen_vec.rows(),self.eigen_vec.cols())
    }

    // original version
    pub fn rbfrun(&self, mut x: Array2<T>){
        let mut x_fft = fft_preprocess(&x,self.coeffs);

        let start = Instant::now();
        let (nrows_x, ncols_x) = (x.rows(),x.cols());
        let (nrows_dic, ncols_dic) = (self.dictionary.rows(),self.dictionary.cols());
        let mut e: Array2<T> = Array2::zeros((nrows_x, nrows_dic));
        let mut dist_comp:usize= 0;

        for i in 0..nrows_x {
            //println!("{}",i);
            for j in 0..nrows_dic {
                e[[i,j]] = rbfkernel(x.row(i), self.dictionary.row(j), 0.4);
                dist_comp = dist_comp + 1;
            }
        }

        let mut z_exact = e.dot(&self.eigen_vec);
        z_exact = z_exact.dot(&self.in_eigen_val);

        let duration = start.elapsed();
        println!("Time elapsed in rbfrun kernel function() is: {:?}", duration);
    }

}


impl<T> CompressionMethod<T> for Kernel<T>
    where T:FFTnum + PartialOrd + std::fmt::Debug + Clone + Float + Scalar + Lapack+ Serialize{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'a>(&self, segs: &mut Vec<Segment<T>>) {
        let mut batch_vec: Vec<T> = Vec::new();
        // println!("segments size: {}", segs.len());
        for seg in segs {
            batch_vec.extend(seg.get_data().clone());

        }
        let belesize = batch_vec.len();
        // println!("vec for matrix length: {}", belesize);
        let mut x = Array2::from_shape_vec((self.batchsize,belesize/self.batchsize),mem::replace(&mut batch_vec, Vec::with_capacity(belesize))).unwrap();
        // println!("matrix shape: {} * {}", x.rows(), x.cols());
        self.rbfrun(x);
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

pub fn opt_sinkcompressed<'a,T: FFTnum + PartialOrd +std::fmt::Debug+ Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayView1<T>, dic_row: ArrayView1<T>, gamma: usize, k:usize, x_fft: &Vec<Complex<T>>, dict_fft: &Vec<Complex<T>>) ->T {
    /*Shift INvariant Kernel*/
    let sim = opt_sum_exp_nccc_compressed(xrow, dic_row, gamma, k, x_fft, dict_fft)/((opt_sum_exp_nccc_compressed(xrow, xrow, gamma, k, x_fft, x_fft)*(opt_sum_exp_nccc_compressed(dic_row, dic_row, gamma, k, dict_fft, dict_fft))).sqrt());

    sim
}

pub fn opt_sum_exp_nccc_compressed<'a,T: FFTnum + PartialOrd+std::fmt::Debug + Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayView1<T>, dic_row: ArrayView1<T>, gamma: usize, k:usize, x_fft: &Vec<Complex<T>>, dict_fft: &Vec<Complex<T>>,) ->T {
    let mut sim = opt_nccc_compressed(xrow, dic_row, k, x_fft, dict_fft) ;
    let exp_sim = sim.mapv(|e| (e * FromPrimitive::from_usize(gamma).unwrap()).exp());
    let sum = exp_sim.sum();
    //println!("sum:{:?}", sum);
    sum
}

pub fn opt_nccc_compressed<'a,T: FFTnum + PartialOrd +std::fmt::Debug + Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayView1<T>, dic_row: ArrayView1<T>, comp: usize, x_fft: &Vec<Complex<T>>, dict_fft: &Vec<Complex<T>>) ->Array1<T>{
    let len = max(xrow.len(),dic_row.len());
    let fftlen = ((2*len-1) as f64).log2().ceil(); // fft length calculation
    let mut size = (fftlen.exp2()) as usize;

    let mut mul: Vec<Complex<T>> = x_fft.iter().zip(dict_fft).map(|(x,y)| x.mul(y.conj())).collect();

    /* ifft*/
    let mut iplanner = FFTplanner::new(true);
    let ifft = iplanner.plan_fft(size);
    //println!("size:{:?}",size);
    let mut ioutput: Vec<Complex<T>> = vec![Complex::zero(); size];
    //let mut cpxoutput: Vec<Complex<T>> = vec![Complex::zero(); size];

    ifft.process(&mut mul, &mut ioutput);
    //ifft.process(&mut cpx,&mut cpxoutput);
    //println!("cpxput:{:?}",cpxoutput);
    //println!("ioutput:{:?}",ioutput);
    let mut reioutput:Vec<T> = ioutput.iter().map(|c| c.re).collect();
    //println!("reioutput:{:?}",reioutput);

    //
    let to_del = size-2*len+1;
    for i in 0..to_del{
        reioutput.remove(len);
    }

    let mut cc_sequence  = Array1::from_vec(reioutput);
    //println!("cc_sequence:{:?}",cc_sequence);

    let norm = l2_norm(xrow.view()) * l2_norm(dic_row.view()) * FromPrimitive::from_usize(size).unwrap();
    //println!("norm:{:?}",norm);

    let res=cc_sequence.mapv_into(|e|e/norm);
    //println!("normlized:{:?}",res);
    res
}


pub fn fft_preprocess<'a,T: FFTnum + PartialOrd +std::fmt::Debug + Clone + Float + Serialize + Deserialize<'a>>(x: &Array2<T>,  comp: usize) -> Vec<Vec<Complex<T>>> {
    let nrows = x.rows();
    let mut ffted_x: Vec<Vec<Complex<T>>> = Vec::with_capacity(nrows);
    for i in 0..nrows {
        let row = x.row(i);
        ffted_x.push(fft_with_leading(row,comp))
    }
    ffted_x
}

pub fn fft_with_leading<'a,T: FFTnum + PartialOrd +std::fmt::Debug + Clone + Float + Serialize + Deserialize<'a>>( xrow: ArrayView1<T>, comp: usize)->Vec<Complex<T>>{
    let len = xrow.len();
    let fftlen = ((2*len-1) as f64).log2().ceil(); // fft length calculation
    let mut size = (fftlen.exp2()) as usize;

    let mut xinput: Vec<Complex<T>> = xrow.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();
    xinput.resize(size,Complex::zero());
    //println!("xinput:{:?}",xinput);
    let mut xoutput: Vec<Complex<T>> = vec![Complex::zero(); size];

    {
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(size);
        fft.process(&mut xinput, &mut xoutput);
    }
    //println!("xoutput:{:?}",xoutput);
    leading_fft(&mut xoutput,comp);
    return xoutput

}

pub fn sinkcompressed<'a,T: FFTnum + PartialOrd +std::fmt::Debug+ Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayView1<T>, dic_row: ArrayView1<T>, gamma: usize, k:usize) ->T {
    /*Shift INvariant Kernel*/
    let sim = sum_exp_nccc_compressed(xrow, dic_row, gamma, k)/((sum_exp_nccc_compressed(xrow, xrow, gamma, k)*(sum_exp_nccc_compressed(dic_row, dic_row, gamma, k))).sqrt());

    sim
}

pub fn rbfkernel<'a,T: FFTnum + PartialOrd +std::fmt::Debug+ Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayView1<T>, dic_row: ArrayView1<T>, sigma: f32) ->T {
    /*Radial basis function kernel*/
    let diff = &xrow - &dic_row;
    //println!("diff:{:?}",diff);
    let l2norm = l2_norm(diff.view());
    let sigma_t = FromPrimitive::from_f32(sigma).unwrap();
    let sim = T::one()/((l2norm/FromPrimitive::from_usize(2).unwrap()* sigma_t * sigma_t).exp());
    sim
}

pub fn sum_exp_nccc_compressed<'a,T: FFTnum + PartialOrd+std::fmt::Debug + Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayView1<T>, dic_row: ArrayView1<T>, gamma: usize, k:usize) ->T {
    let mut sim = nccc_compressed(xrow, dic_row, k) ;
    let exp_sim = sim.mapv(|e| (e * FromPrimitive::from_usize(gamma).unwrap()).exp());
    let sum = exp_sim.sum();
    //println!("sum:{:?}", sum);
    sum
}

pub fn nccc_compressed<'a,T: FFTnum + PartialOrd +std::fmt::Debug + Clone + Float + Serialize + Deserialize<'a>>(xrow: ArrayView1<T>, dic_row: ArrayView1<T>, comp: usize) ->Array1<T>{
    let len = max(xrow.len(),dic_row.len());
    let fftlen = ((2*len-1) as f64).log2().ceil(); // fft length calculation
    let mut size = (fftlen.exp2()) as usize;
    //let compression = len * comp /100;

    //println!("xrow:{:?}",xrow);
    let mut xinput: Vec<Complex<T>> = xrow.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();
    xinput.resize(size,Complex::zero());
    //println!("xinput:{:?}",xinput);
    let mut xoutput: Vec<Complex<T>> = vec![Complex::zero(); size];

    let mut dic_input: Vec<Complex<T>> = dic_row.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();
    dic_input.resize(size,Complex::zero());

    let mut dic_output: Vec<Complex<T>> = vec![Complex::zero(); size];
    {
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(size);
        fft.process(&mut xinput, &mut xoutput);
        fft.process(&mut dic_input, &mut dic_output);
    }
    let mut cpx = xoutput.clone();
    //println!("xoutput:{:?}",xoutput);
    leading_fft(&mut xoutput,comp);
    leading_fft(&mut dic_output,comp);
    //println!("xoutput:{:?}",xoutput);
    let mut mul: Vec<Complex<T>> = xoutput.iter().zip(dic_output).map(|(x,y)| x.mul(y.conj())).collect();
//        let d:Vec<T> = mul.iter().map(|x| x.re).collect();
    //println!("mul:{:?}",mul);

    /* ifft*/
    let mut iplanner = FFTplanner::new(true);
    let ifft = iplanner.plan_fft(size);
    //println!("size:{:?}",size);
    let mut ioutput: Vec<Complex<T>> = vec![Complex::zero(); size];
    //let mut cpxoutput: Vec<Complex<T>> = vec![Complex::zero(); size];

    ifft.process(&mut mul, &mut ioutput);
    //ifft.process(&mut cpx,&mut cpxoutput);
    //println!("cpxput:{:?}",cpxoutput);
    //println!("ioutput:{:?}",ioutput);
    let mut reioutput:Vec<T> = ioutput.iter().map(|c| c.re).collect();
    //println!("reioutput:{:?}",reioutput);

    //
    let to_del = size-2*len+1;
    for i in 0..to_del{
        reioutput.remove(len);
    }

    let mut cc_sequence  = Array1::from_vec(reioutput);
    //println!("cc_sequence:{:?}",cc_sequence);

    let norm = l2_norm(xrow.view()) * l2_norm(dic_row.view()) * FromPrimitive::from_usize(size).unwrap();
    //println!("norm:{:?}",norm);

    let res=cc_sequence.mapv_into(|e|e/norm);
    //println!("normlized:{:?}",res);
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
    let le = x.len();
    for (i, a) in x.iter_mut().enumerate(){
        if i>=k && i<=le-k {
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
fn test_kernel(){

    use rand::prelude::*;

    if rand::random() { // generates a boolean
        // Try printing a random unicode code point (probably a bad idea)!
        println!("char: {}", rand::random::<char>());
    }

    let mut rng = rand::thread_rng();
    let y: f64 = rng.gen(); // generates a float between 0 and 1

    let mut dic_nums = Vec::<f32>::with_capacity(400);

    for _ in 0..400 {
        // 1 (inclusive) to 21 (exclusive)
        dic_nums.push(rng.gen_range(-1.0, 1.0));
    }
    //print!("{:?}, ", dic_nums);
//
    let mut nums = Vec::<f32>::with_capacity(2000);
//
    for _ in 0..2000 {

        nums.push(rng.gen_range(-1.0, 1.0));
    }
//
    //let x = Array2::<f32>::from_shape_vec((100,20), nums).unwrap();
    let mut x = array![
[41.978,44.135,43.157,42.803,42.802,42.672,42.826,43.359,42.875,43.156],
[42.944,42.538,41.731,42.069,41.371,39.89,39.728,41.223,40.713,42.503],
[43.226,43.818,43.375,43.751,42.286,44.058,44.739,46.354,48.355,49.314],
[50.89,52.083,52.909,50.614,49.477,49.57,49.078,49.581,50.214,49.834],
[49.049,48.494,49.298,49.282,48.585,49.332,47.191,48.429,48.453,48.331],
[48.346,47.564,47.394,46.009,45.966,44.624,46.227,47.14,47.309,47.779],
[48.379,49.402,47.357,47.128,47.291,47.665,47.862,48.106,49.097,47.727],
[46.644,45.982,46.267,45.742,44.671,42.223,41.234,43.402,44.148,46.537],
[46.397,45.721,44.332,44.742,44.921,46.214,45.983,44.234,45.16,44.096],
[44.973,44.394,45.277,44.881,44.735,44.47,45.751,44.211,42.38,41.479],
[42.049,43.956,43.553,42.185,42.303,41.601,40.385,40.776,40.31,39.87],
[37.498,38.657,36.979,37.317,38.332,38.675,40.225,39.954,40.74,40.659],
[40.58,42.118,43.352,45.932,45.415,45.349,45.5,46.362,45.593,45.898],
[45.325,43.888,44.32,41.39,40.869,42.786,42.5,41.648,42.069,42.004],
[43.541,42.777,42.282,42.132,42.46,40.613,40.73,41.842,43.133,43.],
[42.671,42.384,43.039,42.72,43.223,43.475,42.043,42.718,44.165,44.566],
[45.162,45.355,46.235,44.233,43.076,44.076,43.423,42.647,41.937,42.718],
[41.925,40.493,40.339,40.621,41.24,42.408,43.807,41.917,42.199,42.57],
[42.284,43.286,42.847,43.94,44.045,44.685,45.798,45.074,47.823,46.541],
[45.417,46.165,47.619,47.22,47.668,49.084,48.501,48.529,48.064,47.916],
[48.101,47.641,47.594,48.604,49.299,48.716,47.419,47.02,47.339,45.836],
[45.67,44.682,44.534,44.351,44.858,46.702,46.113,45.736,46.109,44.942],
[43.958,44.439,43.94,44.158,44.935,45.587,45.929,46.616,43.972,41.99],
[41.677,43.875,44.464,45.582,44.592,46.577,45.527,45.34,43.8,45.685],
[45.357,44.675,43.888,44.169,43.581,43.503,44.611,44.238,44.009,44.946],
[45.968,47.128,47.966,48.848,50.708,50.707,51.183,52.564,51.299,49.794],
[49.994,49.669,48.211,48.198,47.835,48.197,49.547,47.155,45.739,45.942],
[45.736,45.627,45.965,46.284,46.536,47.231,46.588,46.563,47.561,47.429],
[48.654,46.779,46.65,47.71,49.253,49.804,51.038,51.817,50.865,52.022],
[51.744,52.958,53.988,53.673,54.707,53.272,53.037,52.243,51.883,53.35]
];
    //let dic = Array2::<f32>::from_shape_vec((20,20), dic_nums).unwrap();
    let mut dic = array![
[41.978,44.135,43.157,42.803,42.802,42.672,42.826,43.359,42.875,43.156],
[42.944,42.538,41.731,42.069,41.371,39.89,39.728,41.223,40.713,42.503],
[43.226,43.818,43.375,43.751,42.286,44.058,44.739,46.354,48.355,49.314],
[50.89,52.083,52.909,50.614,49.477,49.57,49.078,49.581,50.214,49.834],
[49.049,48.494,49.298,49.282,48.585,49.332,47.191,48.429,48.453,48.331],
[48.346,47.564,47.394,46.009,45.966,44.624,46.227,47.14,47.309,47.779],
[48.379,49.402,47.357,47.128,47.291,47.665,47.862,48.106,49.097,47.727],
[46.644,45.982,46.267,45.742,44.671,42.223,41.234,43.402,44.148,46.537],
[46.397,45.721,44.332,44.742,44.921,46.214,45.983,44.234,45.16,44.096],
[44.973,44.394,45.277,44.881,44.735,44.47,45.751,44.211,42.38,41.479]
];
    let (e, vecs) = dic.clone().eigh(UPLO::Upper).unwrap();
//    println!("eigenvalues = \n{:?}", e);
//    println!("V = \n{:?}", vecs);
    let av = dic.dot(&vecs);


    let mut lcce = Kernel::new(dic,3,4,20);
    lcce.run(x);

    //println!("AV = \n{:?}", av);


//    let mut kernel = Kernel::new(x,dic,20,15);
//    //kernel.run();
//    let x = array![1., 2., 3., 4., 5.];
//    println!("||x||_2 = {}", l2_norm(x.view()));
//    //println!("||x||_1 = {}", l1_norm(x.view()));
//    println!("Normalizing x yields {:?}", normalize(x));
}



#[test]
fn test_fft() {
    let mut size = 16;
    let mut xrow = Array1::from_vec(vec![41.978,44.135,43.157,42.803,42.802,42.672,42.826,43.359,42.875,43.156]);
    let mut dic_row = Array1::from_vec(vec![41.978,44.135,43.157,42.803,42.802,42.672,42.826,43.359,42.875,43.156]);
    let mut xinput: Vec<Complex<f32>>  = xrow.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();
    xinput.resize(size,Complex::zero());
    println!("xinput:{:?}",xinput);
    let mut xoutput= vec![Complex::zero(); size];
    let mut dic_input:Vec<Complex<f32>> = dic_row.iter()
        .map(|x| Complex::new(*x,Zero::zero()))
        .collect();
    dic_input.resize(size,Complex::zero());

    let mut planner = FFTplanner::new(false);
    let fft = planner.plan_fft(size);
    fft.process(&mut xinput, &mut xoutput);
    println!("xoutput:{:?}",xoutput);


    /* ifft*/
    let mut iplanner = FFTplanner::new(true);
    let ifft = iplanner.plan_fft(size);
    println!("size:{:?}",size);
    let mut ioutput: Vec<Complex<f32>> = vec![Complex::zero(); size];
    let mut cpxoutput: Vec<Complex<f32>> = vec![Complex::zero(); size];

    ifft.process(&mut xoutput,&mut cpxoutput);
    println!("cpxput:{:?}",cpxoutput);


}
