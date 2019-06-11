use time_series_start::run_test;
use std::env;

fn main() {
	let args: Vec<String> = env::args().collect();

	let config_file = &args[1];
	let data_type = &args[2];

	match data_type.as_str() {
		"f32" => run_test::<f32>(config_file),
		"f64" => run_test::<f64>(config_file),
		_ => panic!("Data type not supported yet"),
	}
    
}
