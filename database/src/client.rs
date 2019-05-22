use futures::stream::iter_ok;
use tokio::timer::Interval;
use std::time::Instant;
use std::io::{BufReader,BufRead};
use std::str::FromStr;
use std::fs::File;

use std::time::Duration;
use rand::distributions::*;
use rand::prelude::*;

use std::sync::RwLock;
use crate::buffer_pool::VDBufferPool;
use std::sync::Arc;
use tokio::prelude::*;

use crate::future_signal::Signal;


pub enum Amount {
	Limited (u64),
	Unlimited,
}

pub enum RunPeriod {
	Finite (Duration),
	Indefinite,
}

pub enum Frequency {
	Immediate,
	Delayed(Interval),
}


pub struct Client<T,U> 
	where T: Stream<Item=U,Error=()>
{
	producer: T,
	amount: Amount,
	run_period: RunPeriod,
	frequency: Frequency,
	start: Instant,
	produced: Option<u64>,
}

	
pub fn client_from_stream<T,U>(producer: T, amount: Amount, run_period: RunPeriod,
			   interval_args: Option<(Instant,Duration)>)
			    -> impl Stream<Item=U,Error=()>
	where T: Stream<Item=U,Error=()>,
{
	let frequency = match interval_args {
		Some((start,dur)) => Frequency::Delayed(Interval::new(start,dur)),
		None => Frequency::Immediate,
	};

	let produced = match amount {
		Amount::Limited(_) => Some(0),
		Amount::Unlimited  => None,
	};

	Client { 
		producer: producer,
		amount: amount,
		run_period: run_period,
		frequency: frequency,
		start: Instant::now(),
		produced: produced,
	}
}

pub fn client_from_iter<T,U>(producer: T, amount: Amount, run_period: RunPeriod,
				   interval_args: Option<(Instant,Duration)>)
				    -> impl Stream<Item=U,Error=()>
	where T: Iterator<Item=U>
{
	let frequency = match interval_args {
		Some((start,dur)) => Frequency::Delayed(Interval::new(start,dur)),
		None => Frequency::Immediate,
	};

	let produced = match amount {
		Amount::Limited(_) => Some(0),
		Amount::Unlimited  => None,
	};

	Client { 
		producer:iter_ok(producer),
		amount: amount,
		run_period: run_period,
		frequency: frequency,
		start: Instant::now(),
		produced: produced,
	}
}


impl<T,U> Stream for Client<T,U> 
	where T: Stream<Item=U,Error=()>
{
	type Item = U;
	type Error = ();

	fn poll(&mut self) -> Poll<Option<U>,()> {
		
		/* Terminate stream if hit time-limit */
		if let RunPeriod::Finite(dur) = self.run_period {
			let now = Instant::now();
			let time = now.duration_since(self.start); 
			if time >= dur { return Ok(Async::Ready(None)) }
		}

		/* Terminate stream if hit max production */
		if let Amount::Limited(max_items) = self.amount {
			if let Some(items) = self.produced {
				if items >= max_items { return Ok(Async::Ready(None)) }
			}
		}

		/* Either poll to determine if enough time has passed or
		 * immediately get the value depending on Frequency Mode
		 * Must call poll on the stream within the client
		 */
		match &mut self.frequency {
			Frequency::Immediate => {
				let poll_val = try_ready!(self.producer.poll());
				Ok(Async::Ready(poll_val))
			}
			Frequency::Delayed(interval) => {
				match interval.poll() {
					Ok(Async::NotReady) => Ok(Async::NotReady),
					Err(e) => { 
						println!("{:?}", e); 
						Err(())
					}
					_ =>  {
						let poll_val = try_ready!(self.producer.poll());
						Ok(Async::Ready(poll_val))
					}
				}
				
			}
		}

	}
}

/* Must use type annotation on function to declare what to 
 * parse CSV entries as 
 */
fn construct_file_iterator<T>(file: &str, skip_val: usize, delim: char) -> Result<impl Iterator<Item=T>,()> 
	where T: FromStr
{
	let f = match File::open(file) {
		Ok(f) => f,
		Err(_) => return Err(()),
	};

	Ok(BufReader::new(f)
		.lines()
		.filter_map(Result::ok)
		.flat_map(|line: String| {
			line.split(delim)
				.skip(skip_val)
				.filter_map(|item: &str| item.parse::<T>().ok())
				.collect::<Vec<T>>()
				.into_iter()
		})
	)
}

/* An example of how to combine an iterator and IterClient constructor */
pub fn construct_file_client<T>(file: &str, skip_val: usize, delim: char, amount: Amount, run_period: RunPeriod,
				   		 interval_args: Option<(Instant,Duration)>)
				    	 -> Result<impl Stream<Item=T,Error=()>,()>
	where T: FromStr,
{
	let producer = construct_file_iterator::<T>(file, skip_val, delim)?;
	Ok(client_from_iter(producer, amount, run_period, interval_args))
}

#[test]
fn construct_client() {
	let client = construct_file_client::<f32>("../UCRArchive2018/Ham/Ham_TEST", 1, ',', Amount::Unlimited, RunPeriod::Indefinite, None).unwrap();
	let buffer: Arc<RwLock<VDBufferPool<f32>>>  = Arc::new(RwLock::new(VDBufferPool::new()));
	let sig1 = Signal::new(1, client, 400, buffer.clone(),|i,j| i >= j);
}