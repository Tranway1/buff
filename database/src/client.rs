use std::sync::mpsc::channel;
use std::thread;
use std::marker::PhantomData;
use serde::de::DeserializeOwned;
use std::sync::Mutex;
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
use std::sync::Arc;
use tokio::prelude::*;

use std::borrow::Borrow;

#[derive(PartialEq)]
pub enum Amount {
	Limited (u64),
	Unlimited,
}

#[derive(PartialEq)]
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
			   frequency: Frequency)
			    -> impl Stream<Item=U,Error=()>
	where T: Stream<Item=U,Error=()>,
{
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
				   frequency: Frequency)
				    -> impl Stream<Item=U,Error=()>
	where T: Iterator<Item=U>
{
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

fn construct_file_iterator<T>(file: &str, delim: u8) -> Result<impl Iterator<Item=T>,()> 
	where T: DeserializeOwned
{
	let f = match File::open(file) {
		Ok(f) => f,
		Err(_) => return Err(()),
	};

	Ok(BufReader::new(f)
		.split(delim)
		.filter_map(|x| match x {
			Ok(val) => bincode::deserialize(&val).ok(),
			_ => None
		})
	)
}

/* Must use type annotation on function to declare what to 
 * parse CSV entries as 
 */
pub fn construct_file_iterator_skip_newline<T>(file: &str, skip_val: usize, delim: char) -> Result<impl Iterator<Item=T>,()>
	where T: FromStr
{
	let f = match File::open(file) {
		Ok(f) => f,
		Err(_) => return Err(()),
	};

	Ok(BufReader::new(f)
		.lines()
		.filter_map(Result::ok)
		.flat_map(move |line: String| {
			line.split(delim)
				.skip(skip_val)
				.filter_map(|item: &str| item.parse::<T>().ok())
				.collect::<Vec<T>>()
				.into_iter()
		})
	)
}

pub fn construct_file_iterator_int(file: &str, skip_val: usize, delim: char, scl:i32) -> Result<impl Iterator<Item=u32>,()>
{
	let f = match File::open(file) {
		Ok(f) => f,
		Err(_) => return Err(()),
	};

	Ok(BufReader::new(f)
		.lines()
		.filter_map(Result::ok)
		.flat_map(move |line: String| {
			line.split(delim)
				.skip(skip_val)
				.filter_map(|item: &str| item.parse::<f32>().ok())
				.map(|x| (x*scl as f32).ceil().abs() as u32)
				.collect::<Vec<u32>>()
				.into_iter()
		})
	)
}

pub fn construct_file_iterator_int_signed(file: &str, skip_val: usize, delim: char, scl:i32) -> Result<impl Iterator<Item=i32>,()>
{
	let f = match File::open(file) {
		Ok(f) => f,
		Err(_) => return Err(()),
	};

	Ok(BufReader::new(f)
		.lines()
		.filter_map(Result::ok)
		.flat_map(move |line: String| {
			line.split(delim)
				.skip(skip_val)
				.filter_map(|item: &str| item.parse::<f32>().ok())
				.map(|x| (x*scl as f32).ceil() as i32)
				.collect::<Vec<i32>>()
				.into_iter()
		})
	)
}



pub fn construct_file_client<T>(file: &str, delim: u8, amount: Amount, 
						 run_period: RunPeriod, frequency: Frequency)
						 -> Result<impl Stream<Item=T,Error=()>,()> 
	where T: DeserializeOwned
{
	let producer = construct_file_iterator::<T>(file, delim)?;
	Ok(client_from_iter(producer, amount, run_period, frequency))
}

/* An example of how to combine an iterator and IterClient constructor */
pub fn construct_file_client_skip_newline<T>(file: &str, skip_val: usize, delim: char, amount: Amount, run_period: RunPeriod,
				   		 frequency: Frequency)
				    	 -> Result<impl Stream<Item=T,Error=()>,()>
	where T: FromStr,
{
	let producer = construct_file_iterator_skip_newline::<T>(file, skip_val, delim)?;
	Ok(client_from_iter(producer, amount, run_period, frequency))
}


/* First approach at enabling a framework for random generation 
 * Failed because f32 does not implement From<f64>
 * This lack of implementation prevents coverting f64 values 
 * which is the only type supported by rusts normal distribution
 * So this framework won't work for a f32 database setting with a 
 * normal distribution generator client
 */
pub struct BasicItemGenerator<T,U,V>
	where V: From<T>,
		  U: Distribution<T>,
{
	rng: SmallRng,
	dist: U,
	phantom1: PhantomData<T>,
	phantom2: PhantomData<V>,
}

impl<T,U,V> BasicItemGenerator<T,U,V> 
	where V: From<T>,
		  U: Distribution<T>,
{
	pub fn new(dist: U) -> BasicItemGenerator<T,U,V> {
		BasicItemGenerator {
			rng: SmallRng::from_entropy(),
			dist: dist,
			phantom1: PhantomData,
			phantom2: PhantomData,
		}
	}
}

impl<T,U,V> Iterator for BasicItemGenerator<T,U,V> 
	where V: From<T>,
		  U: Distribution<T>,
{
	type Item = V;

	fn next(&mut self) -> Option<V> {
		Some(self.dist.sample(&mut self.rng).into())
	}
}

pub fn construct_gen_client<T,U,V>(dist: U, 
		amount: Amount, run_period: RunPeriod, frequency: Frequency) 
			-> impl Stream<Item=V,Error=()>
		where V: From<T>,
			  U: Distribution<T>,

{
	let producer = BasicItemGenerator::new(dist);
	client_from_iter(producer, amount, run_period, frequency)
}

pub struct NormalDistItemGenerator<V>
	where V: From<f32>,
{
	rng: SmallRng,
	dist: Normal,
	phantom: PhantomData<V>,
}

impl<V> NormalDistItemGenerator<V> 
	where V: From<f32>,
{
	pub fn new(mean: f64, std: f64) -> NormalDistItemGenerator<V> {
		NormalDistItemGenerator {
			rng: SmallRng::from_entropy(),
			dist: Normal::new(mean,std),
			phantom: PhantomData,
		}
	}
}

impl<V> Iterator for NormalDistItemGenerator<V> 
	where V: From<f32>,
{
	type Item = V;

	fn next(&mut self) -> Option<V> {
		Some((self.dist.sample(&mut self.rng) as f32).into())
	}
}



pub fn construct_normal_gen_client<T>(mean: f64, std: f64, 
	amount: Amount, run_period: RunPeriod, frequency: Frequency)
		-> impl Stream<Item=T,Error=()> 
	where T: From<f32>,
{
	let producer = NormalDistItemGenerator::<T>::new(mean,std);
	client_from_iter(producer, amount, run_period, frequency)
}

