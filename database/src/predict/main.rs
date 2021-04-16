use std::env;
use log::{error, info, warn};
use log4rs;
use time_series_start::predict::cr_predictor::{est_gorilla_cr, est_sprintz_cr, est_dict_cr, est_gzip_cr, est_buff_cr, est_snappy_cr};
use time_series_start::client::construct_file_iterator_skip_newline;

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let input_file = &args[1];
    let compression = &args[2];
    let int_scale = args[3].parse::<usize>().unwrap();
    let sample_r = args[4].parse::<f64>().unwrap();

    let file_iter = construct_file_iterator_skip_newline::<f64>(input_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap().collect();
    let len = file_vec.len();
    let sample = (len as f64 * sample_r) as usize;
    let mut p_ratio = match compression.as_str(){
        "gorilla" => {
            est_gorilla_cr(file_vec,sample,int_scale)
        },
        "buff" => {
            est_buff_cr(file_vec,sample,int_scale)
        },

        "snappy" => {
            est_snappy_cr(file_vec,sample,int_scale)
        },

        "gzip" => {
            est_gzip_cr(file_vec,sample,int_scale)
        },

        "sprintz" => {
            est_sprintz_cr(file_vec,sample,int_scale)
        },

        "dict" => {
            est_dict_cr(file_vec,sample,int_scale)
        },


        _ => {panic!("Compression not supported yet.")}
    };

    println!("ARGS: {}, {}, {}, {}, {}",input_file, compression, int_scale,sample_r,p_ratio);
}