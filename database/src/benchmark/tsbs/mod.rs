use crate::benchmark::{BENCH_DATA, read_a_file};

pub fn tsbs_bench(compression: &str, query: &str){
    let longtitude = get_file("r_longitude.csv",compression);
    let latitude = get_file("r_latitude.csv",compression);

    match compression{
        "buff" => {

        },
        "buff-major" => {

        },
        "gorilla" => {

        },
        "gorillabd" => {

        },

        "snappy" => {

        },

        "gzip" => {

        },

        "fixed" => {

        },
        "sprintz" => {

        },
        "buff-slice" => {

        },
        _ => {panic!("Compression not supported yet.")}
    }
}

pub fn get_file(file:&str, compression: &str) ->Vec<u8>
{
    let mut path = BENCH_DATA.to_owned();
    path.push_str(file);
    path.push_str(".");
    path.push_str(compression);
    let binary = read_a_file(&path).unwrap();
    binary
}