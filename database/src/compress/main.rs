use std::env;
use time_series_start::methods::compress::{test_grilla_compress_on_file, test_grilla_compress_on_int_file, test_zlib_compress_on_file, test_zlib_compress_on_int_file, test_BP_compress_on_int, test_paa_compress_on_file, test_paa_compress_on_int_file, test_fourier_compress_on_file, test_snappy_compress_on_file, test_snappy_compress_on_int_file, test_deflate_compress_on_file, test_deflate_compress_on_int_file, test_gzip_compress_on_file, test_gzip_compress_on_int_file, test_FCM_compress_on_int, test_deltaBP_compress_on_int, test_DFCM_compress_on_int, test_offsetgrilla_compress_on_file, test_offsetgrilla_compress_on_int_file, test_split_compress_on_int, test_splitbd_compress_on_file, test_grillabd_compress_on_file, test_split_compress_on_file};
use time_series_start::compress::{run_bpsplit_encoding_decoding, run_gorilla_encoding_decoding, run_gorillabd_encoding_decoding, run_splitbd_encoding_decoding, run_snappy_encoding_decoding, run_gzip_encoding_decoding, run_bp_double_encoding_decoding, run_sprintz_double_encoding_decoding};
use log::{error, info, warn};
use log4rs;

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let input_file = &args[1];
    let compression = &args[2];
    let int_scale = args[3].parse::<usize>().unwrap();
    let pred = args[4].parse::<f64>().unwrap();

    print!("ARGS: {}, {}, {}, {}, ",input_file, compression, int_scale,pred);
    match compression.as_str(){
        "bpsplit" => {
            run_bpsplit_encoding_decoding(input_file,int_scale,pred);
        },
        "gorilla" => {
            run_gorilla_encoding_decoding(input_file,int_scale,pred);
        },
        "gorillabd" => {
            run_gorillabd_encoding_decoding(input_file,int_scale,pred);
        },
        "splitbd" => {
            run_splitbd_encoding_decoding(input_file,int_scale,pred);
        },
        "snappy" => {
            run_snappy_encoding_decoding(input_file,int_scale,pred);
        },

        "gzip" => {
            run_gzip_encoding_decoding(input_file,int_scale,pred);
        },
        "bp" => {
            run_bp_double_encoding_decoding(input_file,int_scale,pred);
        },
        "sprintz" => {
            run_sprintz_double_encoding_decoding(input_file,int_scale,pred);
        },

        _ => {panic!("Compression not supported yet.")}
    }

}