#[macro_use]
extern crate serde_derive;
extern crate bincode;
#[macro_use]
extern crate futures;
extern crate toml_loader;

use serde::Serialize;
use std::fmt::Debug;
use serde::de::DeserializeOwned;
use crate::buffer_pool::{SegmentBuffer,VDBufferPool};
use std::path::Path;
use toml_loader::Loader;

mod buffer_pool;
mod dictionary;
mod file_handler;
mod segment;
mod methods;
mod future_signal;
mod client;

use client::{construct_file_client,Amount,RunPeriod,Frequency};


pub fn run_test<T,B>(config_file: &str) 
	where T: Copy + Send + Serialize + DeserializeOwned + Debug,
		  B: SegmentBuffer<T>,
{
	let config = match Loader::from_file(Path::new(config_file)) {
		Ok(config) => config,
		Err(e) => panic!("{:?}", e),
	};

	/* Construct the file manager to be used */
	let fm = match config.lookup("file_handler") {
		Some(config) => {
			let fm_type = config.lookup("file_manager").expect("A file manager must be provided");
			match fm_type.as_str().expect("A file manager must be provided as a string") {
				"Rocks" => {
					let params = fm_type.lookup("params").expect("A RocksDB file manager requires parameters");
					let path = params.lookup("path").expect("RocksDB requires a path be provided").as_str().expect("Rocks file path must be provided as string");
					let mut db_opts = rocksdb::Options::default();
					db_opts.create_if_missing(true);
					match rocksdb::DB::open(&db_opts, path) {
						Ok(x) => Some(x),
						Err(e) => panic!("Failed to create database: {:?}", e),
					}
				}
				_ => panic!("An invalid file manager type was provided"),
			}
		}
		None => None,
	};


	let _buf = match fm {
		Some(fm) => {
			match config.lookup("buffer") {
					Some(config) => {
						let buf_type = config.lookup("buffer").expect("A buffer type provided");
						match buf_type.as_str().expect("Buffer type must be provided as string") {
							"Clock" => Some(VDBufferPool::<f32,rocksdb::DB>::new(50,fm)),
							_ => panic!("An invalid buffer type was provided"),
						}
					}
					None => None,
			}
		}
		None => panic!("No implementation for buffer w/o file manager"),
	};
	
	/* Construct the clients */
	let _clients: Vec<f32> = Vec::new();
	for client_table in config.lookup("clients")
							  .expect("At least one client must be provided")
							  .as_table()
							  .expect("The clients must be provided as a TOML table")
							  .values()
	{
		let client_type = client_table.lookup("type").expect("The client type must be provided");
		let data_type = config.lookup("data_type").expect("The type of the data used by the database must be declared").as_str().expect("The type of data used must be the database must be provided");
		match client_type.as_str().expect("The client type must be provided as a string") {
			"file" => {
				let _path = client_type.lookup("path")
									  .expect("The file client must be provided a path argument to be constructed")
									  .as_str()
									  .expect("The file path for the client must be provided as a string");
				let _amount = match client_type.lookup("amount") {
					Some(value) => Amount::Limited (value.as_integer().expect("The client amount argument must be specified as an integer") as u64),
					None => Amount::Unlimited,
				};
				match data_type {
					"f32" => unimplemented!(), //construct_file_client::<f32>(path)
					_ => panic!("Provided data type is not currently supported by file client"),
				}
			},
			"gen" => unimplemented!(),
			"external" => unimplemented!(),
			_ => panic!("Provided client type is not supported"),
		}
	}
}