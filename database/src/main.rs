use time_series_start::{run_test, run_single_test};
use std::env;

fn main() {
	let args: Vec<String> = env::args().collect();

	let config_file = &args[1];
	let data_type = &args[2];
	let comp = &args[3];
	let num_comp = args[4].parse::<i32>().ok().expect("I wasn't given an integer!");


	match data_type.as_str() {
		"f32" => run_single_test::<f32>(config_file,comp, num_comp),
		"f64" => run_single_test::<f64>(config_file,comp, num_comp),
//		"u32" => run_single_test::<u32>(config_file),
		_ => panic!("Data type not supported yet"),
	}
    
}
