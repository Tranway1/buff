use std::env;
use time_series_start::compress::{run_bpsplit_encoding_decoding, run_gorilla_encoding_decoding, run_gorillabd_encoding_decoding, run_snappy_encoding_decoding, run_gzip_encoding_decoding, run_bp_double_encoding_decoding, run_sprintz_double_encoding_decoding, run_parquet_write_filter, run_splitbd_byte_encoding_decoding, run_splitdouble_byte_encoding_decoding, run_splitdouble_encoding_decoding, run_splitdouble_byte_residue_encoding_decoding, run_splitdouble_byte_residue_majority_encoding_decoding};
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

    println!("ARGS: {}, {}, {}, {}, ",input_file, compression, int_scale,pred);
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
        "splitdouble" => {
            run_splitdouble_encoding_decoding(input_file,int_scale,pred);
        },
        "bytedec" => {
            run_splitbd_byte_encoding_decoding(input_file,int_scale,pred);
        },
        "byteall" => {
            run_splitdouble_byte_encoding_decoding(input_file,int_scale,pred);
        },
        "RAPG" => {
            run_splitdouble_byte_residue_encoding_decoding(input_file,int_scale,pred);
        },
        "RAPG-major" => {
            run_splitdouble_byte_residue_majority_encoding_decoding(input_file,int_scale,pred);
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

        "dict" => {
            run_parquet_write_filter(input_file, int_scale, pred, "dict");
        },
        "plain" => {run_parquet_write_filter(input_file, int_scale, pred, "plain");},
        "pqgzip" => {run_parquet_write_filter(input_file, int_scale, pred, "pqgzip");},
        "pqsnappy" => {run_parquet_write_filter(input_file, int_scale, pred, "pqsnappy");},

        _ => {panic!("Compression not supported yet.")}
    }

}