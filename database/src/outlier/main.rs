use rand::distributions::Distribution;
use std::env;
use buff::outlier::{gen_u8_with_outlier, outlier_byteall_encoding_decoding, outlier_byte_majority_encoding_decoding};
use log::{info};

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();


    let args: Vec<String> = env::args().collect();
    info!("input args{:?}", args);
    let size = args[1].parse::<u64>().unwrap();
    let o_ratio = args[2].parse::<f64>().unwrap();
    let sparse = &args[3];

    // gen_u8_with_outlier(0.01,100000);
    // let size =100000000;
    // let o_ratio = 0.9;
    let vec_u8 = gen_u8_with_outlier(o_ratio,size).unwrap();

    match sparse.as_str(){
        "byte" => {outlier_byteall_encoding_decoding(vec_u8, size, o_ratio, 0,69.0);},
        "sparse" => {outlier_byte_majority_encoding_decoding(vec_u8, size, o_ratio, 0,69.0);},

        _ => {panic!("Compression not supported yet -- outlier micro-benchmark.")}

    }
}
