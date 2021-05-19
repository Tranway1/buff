use time_series_start::benchmark::tsbs::tsbs_bench;
use time_series_start::benchmark::influx_bench::influx_bench;
use std::env;
use log::info;
use time_series_start::benchmark::BENCH_DATA;

fn main() {
    log4rs::init_file("config/log4rs.yaml", Default::default()).unwrap();

    let args: Vec<String> = env::args().collect();
    info!("input args{:?}",args);
    let bench = &args[1];
    let compression = &args[2];
    // let int_scale = args[3].parse::<usize>().unwrap();
    let query = &args[3];
    let pred = args[4].parse::<f64>().unwrap();

    println!("ARGS: {}, {}, {}",bench, compression,pred);
    match bench.as_str(){
        "tsbs" => {
            tsbs_bench(compression, query);
        },
        "influx" => {
            influx_bench(compression, query);
        },
        _ => {panic!("benchmark not supported yet.")}
    }

}