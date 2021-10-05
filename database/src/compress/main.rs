use std::env;
use log::{error, info, warn};
use log4rs;
use buff::compress::buff_simd::{run_buff_simd_encoding_decoding, run_buff_encoding_decoding_mybitvec, run_buff_majority_encoding_decoding};
use buff::compress::buff_slice::{run_buff_slice_encoding_decoding, run_buff_slice_scalar_encoding_decoding};

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let input_file = &args[1];
    let compression = &args[2];
    let int_scale = args[3].parse::<usize>().unwrap();
    let pred = args[4].parse::<f64>().unwrap();

    println!("ARGS: {}, {}, {}, {}, ",input_file, compression, int_scale,pred);
    match compression.as_str(){
        "buff" => {
            run_buff_encoding_decoding_mybitvec(input_file,int_scale,pred);
        },
        "buff-simd" => {
            run_buff_simd_encoding_decoding(input_file,int_scale,pred);
        },
        "buff-slice" => {
            run_buff_slice_encoding_decoding(input_file,int_scale,pred);
        },
        "buff-slice-scalar" => {
            run_buff_slice_scalar_encoding_decoding(input_file,int_scale,pred);
        },
        "buff-major" => {
            run_buff_majority_encoding_decoding(input_file,int_scale,pred);
        },
        _ => {panic!("Compression not supported yet.")}
    }

}