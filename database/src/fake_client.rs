extern crate futures;
use std::vec::IntoIter;
use futures::stream::IterOk;
use futures::stream;
use std::io::{BufReader,BufRead};
use std::fs::File;

/* client trait stats, produce stream, gnerate values, generation vs source file, montior with instants */

pub fn construct_stream(file: &str) -> Result<IterOk<IntoIter<f32>,()>,()> {
	let mut rv = Vec::new();

	let f = match File::open(file) {
		Ok(f) => f,
		Err(_) => return Err(()),
	};
	let reader = BufReader::new(f);
	for line in reader.lines() {
		match line {
			Ok(line) => {
				for item in line.split(',').skip(1) {
					match item.parse::<f32>() {
						Ok(i) => rv.push(i),
						Err(_) => return Err(()),
					}
				}
			} 
			Err(_) => return Err(()),
		}
	}

	Ok(stream::iter_ok::<_,()>(rv))
}