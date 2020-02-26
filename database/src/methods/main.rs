use std::env;
use time_series_start::methods::compress::{test_grilla_compress_on_file, test_grilla_compress_on_int_file, test_zlib_compress_on_file, test_zlib_compress_on_int_file, test_BP_compress_on_int, test_paa_compress_on_file, test_paa_compress_on_int_file, test_fourier_compress_on_file, test_snappy_compress_on_file, test_snappy_compress_on_int_file, test_deflate_compress_on_file, test_deflate_compress_on_int_file, test_gzip_compress_on_file, test_gzip_compress_on_int_file, test_FCM_compress_on_int, test_deltaBP_compress_on_int};
use log::{error, info, warn};
use log4rs;

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let input_file = &args[1];
    let data_type = &args[2];
    let compression = &args[3];
    print!("{}, {}, {}, ",input_file, data_type, compression);
    match compression.as_str(){
        "grilla" => {
            match data_type.as_str() {
                "f32" => test_grilla_compress_on_file::<f32>(input_file),
                "f64" => test_grilla_compress_on_file::<f64>(input_file),
                "u32" => test_grilla_compress_on_int_file(input_file),
                _ => panic!("Data type not supported yet for grilla."),
            }
        },
        "zlib" => {
            match data_type.as_str() {
                "f32" => test_zlib_compress_on_file::<f32>(input_file),
                "f64" => test_zlib_compress_on_file::<f64>(input_file),
                "u32" => test_zlib_compress_on_int_file(input_file),
                _ => panic!("Data type not supported yet for zlib."),
            }
        },
        "paa" => {
            match data_type.as_str() {
                "f32" => test_paa_compress_on_file::<f32>(input_file),
                "f64" => test_paa_compress_on_file::<f64>(input_file),
                "u32" => test_paa_compress_on_int_file(input_file),
                _ => panic!("Data type not supported yet for paa."),
            }
        },
        "fourier" => {
            match data_type.as_str() {
                "f32" => test_fourier_compress_on_file::<f32>(input_file),
                "f64" => test_fourier_compress_on_file::<f64>(input_file),
//                "u32" => test_fourier_compress_on_int_file(input_file),
                _ => panic!("Data type not supported yet for fourier."),
            }
        },
        "snappy" => {
            match data_type.as_str() {
                "f32" => test_snappy_compress_on_file::<f32>(input_file),
                "f64" => test_snappy_compress_on_file::<f64>(input_file),
                "u32" => test_snappy_compress_on_int_file(input_file),
                _ => panic!("Data type not supported yet for snappy."),
            }
        },
        "deflate" => {
            match data_type.as_str() {
                "f32" => test_deflate_compress_on_file::<f32>(input_file),
                "f64" => test_deflate_compress_on_file::<f64>(input_file),
                "u32" => test_deflate_compress_on_int_file(input_file),
                _ => panic!("Data type not supported yet for deflate."),
            }
        },
        "gzip" => {
            match data_type.as_str() {
                "f32" => test_gzip_compress_on_file::<f32>(input_file),
                "f64" => test_gzip_compress_on_file::<f64>(input_file),
                "u32" => test_gzip_compress_on_int_file(input_file),
                _ => panic!("Data type not supported yet for gzip."),
            }
        },
        "bp" => {
            match data_type.as_str() {
                "u32" => test_BP_compress_on_int(input_file),
                _ => panic!("Data type not supported yet for BP."),
            }
        },
        "deltabp" => {
            match data_type.as_str() {
                "i32" => test_deltaBP_compress_on_int(input_file),
                _ => panic!("Data type not supported yet for BP."),
            }
        },
        "dfcm" => {
            match data_type.as_str() {
                "u32" => test_FCM_compress_on_int(input_file),
                _ => panic!("Data type not supported yet for BP."),
            }
        },
        _ => {panic!("Compression not supported yet.")}
    }

}