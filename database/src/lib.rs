#[macro_use]
extern crate serde_derive;
extern crate bincode;
#[macro_use]
extern crate futures;
extern crate toml_loader;

use std::time::SystemTime;
use crate::client::construct_file_client;
use crate::segment::Segment;
use rocksdb::DBVector;
use crate::file_handler::FileManager;
use futures::future::join_all;
use std::str::FromStr;
use serde::Serialize;
use std::fmt::Debug;
use serde::de::DeserializeOwned;
use crate::buffer_pool::{SegmentBuffer,ClockBuffer,NoFmClockBuffer};
use crate::future_signal::{BufferedSignal};
use std::path::Path;
use toml_loader::{Loader};
use std::time::{Duration,Instant};
use tokio::timer::Interval;
use tokio::prelude::*;
use tokio::runtime::{Builder,Runtime};
use std::sync::{Arc,Mutex};

mod buffer_pool;
mod dictionary;
mod file_handler;
mod segment;
mod methods;
mod future_signal;
mod client;

use client::{construct_file_client_skip_newline,Amount,RunPeriod,Frequency};

const DEFAULT_BUF_SIZE: usize = 150;
const DEFAULT_DELIM: char = '\n';

pub fn run_test<T: 'static>(config_file: &str) 
	where T: Copy + Send + Sync + Serialize + DeserializeOwned + Debug + FromStr,
{

	let config = match Loader::from_file(Path::new(config_file)) {
		Ok(config) => config,
		Err(e) => panic!("{:?}", e),
	};

	/* Get segment size */
	let seg_size = config
					.lookup("segment_size")
					.expect("A segment size must be provided")
					.as_integer()
					.expect("The segment size argument must be provided as an integer") as usize;


	/* Construct the file manager to be used */
	let fm = match config.lookup("file_handler") {
		Some (config) => {
			let fm_type = config.lookup("file_manager").expect("A file manager must be provided");
			match fm_type.as_str().expect("A file manager must be provided as a string") {
				"Rocks" => {
					let params = config.lookup("params").expect("A RocksDB file manager requires parameters");
					let path = params.lookup("path").expect("RocksDB requires a path be provided").as_str().expect("Rocks file path must be provided as string");
					let mut db_opts = rocksdb::Options::default();
					db_opts.create_if_missing(true);
					match rocksdb::DB::open(&db_opts, path) {
						Ok(x) =>  {
							let boxed_x = Box::new(x);
							Some(boxed_x)
						}
						Err(e) => panic!("Failed to create RocksFM object: {:?}", e),
					}
				}
				x => panic!("File manager type, {:?}, not supported yet", x),
			}
		}
		None => None,
	};


	/* Construct the buffer to be used */
	let buffer_size = match config.lookup("buffer") {
		Some(value) => value.lookup("buffer_size").map_or(DEFAULT_BUF_SIZE, |v| v.as_integer().expect("The buffer size should be provided as an integer") as usize),
		None => DEFAULT_BUF_SIZE,
	};

	let buf_option: Option<Box<Arc<Mutex<(SegmentBuffer<T> + Send + Sync)>>>> = match fm {
		Some(fm) => {
			match config.lookup("buffer") {
					Some(config) => {
						let buf_type = config.lookup("type").expect("A buffer type must be provided");
						match buf_type.as_str().expect("Buffer type must be provided as string") {
							"Clock" => Some(Box::new(Arc::new(Mutex::new(ClockBuffer::<T,rocksdb::DB>::new(buffer_size,*fm))))),
							x => panic!("The buffer type, {:?}, is not currently supported to run with a file manager", x),
						}
					}
					None => None,
			}
		}
		None => {
			match config.lookup("buffer") {
				Some(config) => {
					let buf_type = config.lookup("type").expect("A buffer type must be provided");
					match buf_type.as_str().expect("Buffer type must be provided as a string") {
						"NoFmClock" => Some(Box::new(Arc::new(Mutex::new(NoFmClockBuffer::<T>::new(buffer_size))))),
						x => panic!("The buffer type, {:?}, is not currently supported to run without a file manager", x),
					}
				}
				None => None,
			}
		}
	};
	
	/* Construct the clients */
	let mut signals: Vec<Box<(Future<Item=Option<SystemTime>,Error=()> + Send + Sync)>> = Vec::new();
	let mut signal_id = 0;

	for client_config in config.lookup("clients")
							  .expect("At least one client must be provided")
							  .as_table()
							  .expect("The clients must be provided as a TOML table")
							  .values()
	{
		let client_type = client_config.lookup("type").expect("The client type must be provided");
		
		let amount = match client_config.lookup("amount") {
			Some(value) => Amount::Limited (value.as_integer().expect("The client amount argument must be specified as an integer") as u64),
			None => Amount::Unlimited,
		};

		let run_period = match client_config.lookup("run_period") {
			Some(table) => {
				let secs = match table.lookup("sec") {
					Some(sec_value) => sec_value.as_integer().expect("The sec argument in run period must be provided as an integer") as u64,
					None => 0, 
				};
				let nano_secs = match table.lookup("nano_sec") {
					Some(nano_sec_value) => nano_sec_value.as_integer().expect("The nano_sec argument in run period must be provided as an integer") as u32,
					None => 0, 
				};

				if secs == 0 && nano_secs == 0 {
					panic!("The run period was provided a value of 0 for both secs and nano_secs. This is not allowed as the signal will start and immediately exit"); 
				}

				RunPeriod::Finite(Duration::new(secs,nano_secs))
			}
			None => RunPeriod::Indefinite,
		};

		let frequency = match client_config.lookup("interval") {
			Some(table) => {
				let secs = match table.lookup("sec") {
					Some(sec_value) => sec_value.as_integer().expect("The sec argument in run period must be provided as an integer") as u64,
					None => 0, 
				};
				let nano_secs = match table.lookup("nano_sec") {
					Some(nano_sec_value) => nano_sec_value.as_integer().expect("The nano_sec argument in run period must be provided as an integer") as u32,
					None => 0, 
				};

				if secs == 0 && nano_secs == 0 {
					panic!("The interval period was provided with a value of 0 for both secs and nano_secs. This is not allowed as the signal will have no delay"); 
				}

				let interval = Duration::new(secs,nano_secs);

				let start_secs = match table.lookup("start_sec") {
					Some(sec_value) => sec_value.as_integer().expect("The start sec argument in run period must be provided as an integer") as u64,
					None => 0, 
				};
				let start_nano_secs = match table.lookup("start_nano_sec") {
					Some(nano_sec_value) => nano_sec_value.as_integer().expect("The start nano_sec argument in run period must be provided as an integer") as u32,
					None => 0, 
				};

				let start = Instant::now() + Duration::new(start_secs,start_nano_secs);
				Frequency::Delayed(Interval::new(start,interval))
			}
			None => Frequency::Immediate,
		};

		match client_type.as_str().expect("The client type must be provided as a string") {
			"file" => {
				let params = client_config.lookup("params").expect("The file client must provide a params table");
				let reader_type =  params.lookup("reader_type")
										 .expect("A file client must provide a reader types in the params table")
										 .as_str()
										 .expect("The reader type must be provided as a string");
				
				let path = params
							.lookup("path")
							.expect("The file client parameters must provide a file path argument")
							.as_str()
							.expect("The file path for the client must be provided as a string");

				let delim = match params.lookup("delim") {
					Some(value) => value.as_str()
										.expect("The file delimiter must be privded as a string")
										.chars().next().expect("The provided delimiter must have some value"),
					None => DEFAULT_DELIM,
				};

				let client: Box<(Stream<Item=T,Error=()> + Sync + Send)> = match reader_type {
					"NewlineAndSkip" => {

						let skip_val = match params.lookup("skip") {
							Some(skip_val) => skip_val.as_integer().expect("The skip value must be provided as an integer") as usize,
							None => 0,
						};
					;
						Box::new(construct_file_client_skip_newline::<T>(path, skip_val, delim, amount, run_period, frequency).expect("Client could not be properly produced"))
					}
					"DeserializeDelim" => Box::new(construct_file_client::<T>(path, delim as u8, amount, run_period, frequency).expect("Client could not be properly produced")),
					x => panic!("The specified file reader, {:?}, is not supported yet", x),
				};

				match &buf_option {
					Some(buf) => signals.push(Box::new(BufferedSignal::new(signal_id, client, seg_size, *buf.clone(), |i,j| i >= j, |_| (), false))),
					None => panic!("Buffer and File manager provided not supported yet"),
				}
			}
			x => panic!("The provided type, {:?}, is not currently supported", x),
		}
		signal_id += 1;
	}

	/* Construct the runtime */
	let mut rt = match config.lookup("runtime") {
		None => Builder::new()
					.after_start(|| println!("Threads have been constructed"))
					.build()
					.expect("Failed to produce a default runtime"),

		Some(value) => {
			let core_threads = value.lookup("core_threads")
									.expect("Core threads field required by custom runtime")
									.as_integer()
									.expect("Core threads should be provided as an integer") as usize;

			let blocking_threads = value.lookup("blocking_threads")
										.expect("Blocking threads field required by custom runtime")
										.as_integer()
										.expect("Blocking threads should be provided as an integer") as usize;

			Builder::new()
					.core_threads(core_threads)
					.blocking_threads(blocking_threads)
					.after_start(|| println!("Threads have been constructed"))
					.build()
					.expect("Failed to produce the custom runtime")
		}
	};

	for res in rt.block_on(join_all(signals)).expect("The signals failed to join") {
		match res {
			None => println!("Failed to produce time stamp"),
			Some(x) => println!("Return value: {:?}", x),
		}
	}

}
