use std::env;

fn main() {
	let args: Vec<String> = env::args().collect();

	let config_file = &args[1];
	let data_type = &args[2];
	let comp = &args[3];
	let num_comp = args[4].parse::<i32>().ok().expect("I wasn't given an integer!");


	match data_type.as_str() {
		_ => panic!("Data type not supported yet"),
	}
    
}
