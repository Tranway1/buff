use std::env;
use time_series_start::methods::compress::{test_grilla_compress_on_file, test_grilla_compress_on_int_file, test_zlib_compress_on_file, test_zlib_compress_on_int_file, test_BP_compress_on_int, test_paa_compress_on_file, test_paa_compress_on_int_file, test_fourier_compress_on_file, test_snappy_compress_on_file, test_snappy_compress_on_int_file, test_deflate_compress_on_file, test_deflate_compress_on_int_file, test_gzip_compress_on_file, test_gzip_compress_on_int_file, test_FCM_compress_on_int, test_deltaBP_compress_on_int, test_DFCM_compress_on_int, test_offsetgrilla_compress_on_file, test_offsetgrilla_compress_on_int_file, test_split_compress_on_int, test_splitbd_compress_on_file, test_grillabd_compress_on_file, test_split_compress_on_file};
use log::{error, info, warn};
use log4rs;

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let input_file = &args[1];
    let data_type = &args[2];
    let compression = &args[3];
    let mut int_scale = 0;
    if data_type.eq("u32")||data_type.eq("i32") {
        int_scale = args[4].parse::<i32>().unwrap();
    }
    print!("{}, {}, {}-{}, ",input_file, data_type, compression,int_scale);
    match compression.as_str(){
        "ofsgorilla" => {
            match data_type.as_str() {
                "f32" => test_offsetgrilla_compress_on_file::<f32>(input_file),
                "f64" => test_offsetgrilla_compress_on_file::<f64>(input_file),
                "i32" => test_offsetgrilla_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for offset+gorilla."),
            }
        },
        "gorilla" => {
            match data_type.as_str() {
                "f32" => test_grilla_compress_on_file::<f32>(input_file),
                "f64" => test_grilla_compress_on_file::<f64>(input_file),
                "i32" => test_grilla_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for gorilla."),
            }
        },
        "gorillabd" => {
            match data_type.as_str() {
//                "f32" => test_grilla_compress_on_file::<f32>(input_file),
                "f64" => test_grillabd_compress_on_file::<f64>(input_file),
//                "i32" => test_grilla_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for gorilla."),
            }
        },
        "splitbd" => {
            match data_type.as_str() {
                "f32" => test_splitbd_compress_on_file::<f32>(input_file),
                "f64" => test_splitbd_compress_on_file::<f64>(input_file),
//                "i32" => test_grilla_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for gorilla."),
            }
        },
        "zlib" => {
            match data_type.as_str() {
                "f32" => test_zlib_compress_on_file::<f32>(input_file),
                "f64" => test_zlib_compress_on_file::<f64>(input_file),
                "i32" => test_zlib_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for zlib."),
            }
        },
        "paa" => {
            match data_type.as_str() {
                "f32" => test_paa_compress_on_file::<f32>(input_file),
                "f64" => test_paa_compress_on_file::<f64>(input_file),
                "i32" => test_paa_compress_on_int_file(input_file,int_scale),
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
                "i32" => test_snappy_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for snappy."),
            }
        },
        "deflate" => {
            match data_type.as_str() {
                "f32" => test_deflate_compress_on_file::<f32>(input_file),
                "f64" => test_deflate_compress_on_file::<f64>(input_file),
                "i32" => test_deflate_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for deflate."),
            }
        },
        "gzip" => {
            match data_type.as_str() {
                "f32" => test_gzip_compress_on_file::<f32>(input_file),
                "f64" => test_gzip_compress_on_file::<f64>(input_file),
                "i32" => test_gzip_compress_on_int_file(input_file,int_scale),
                _ => panic!("Data type not supported yet for gzip."),
            }
        },
        "bp" => {
            match data_type.as_str() {
                "i32" => test_BP_compress_on_int(input_file,int_scale),
                _ => panic!("Data type not supported yet for BP."),
            }
        },
        "split" => {
            match data_type.as_str() {
                "i32" => test_split_compress_on_int(input_file,int_scale),
                "f32" => test_split_compress_on_file::<f32>(input_file,int_scale),
                "f64" => test_split_compress_on_file::<f64>(input_file,int_scale),
                _ => panic!("Data type not supported yet for BP."),
            }
        },
        "deltabp" => {
            match data_type.as_str() {
                "i32" => test_deltaBP_compress_on_int(input_file,int_scale),
                _ => panic!("Data type not supported yet for DELTA-BP."),
            }
        },
        "dfcm" => {
            match data_type.as_str() {
                "i32" => test_DFCM_compress_on_int(input_file,int_scale),
                _ => panic!("Data type not supported yet for DFCM."),
            }
        },
        "fcm" => {
            match data_type.as_str() {
                "i32" => test_FCM_compress_on_int(input_file,int_scale),
                _ => panic!("Data type not supported yet for FCM."),
            }
        },
        _ => {panic!("Compression not supported yet.")}
    }

}