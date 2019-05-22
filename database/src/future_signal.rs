extern crate tokio;

use crate::client::{construct_file_client,Amount,RunPeriod,Frequency};
use std::sync::RwLock;
use crate::buffer_pool::{SegmentBuffer,VDBufferPool};
use crate::fake_client::construct_stream;
use crate::segment::Segment;
use std::time::SystemTime;
use std::sync::Arc;
use std::time::Duration;
use std::mem;
use tokio::io;
use tokio::prelude::*;
use tokio::runtime::{Builder,Runtime};

pub type SignalId = u64;

pub struct Signal<T,U,F> 
	where T: Copy + Send,
	      U: Stream,
	      F: Fn(usize,usize) -> bool,
{
	timestamp: Option<SystemTime>,
	prev_seg_offset: Option<SystemTime>,
	seg_size: usize,
	signal_id: SignalId,
	data: Vec<T>,
	time_lapse: Vec<Duration>,
	signal: U,
	buffer: Arc<RwLock<SegmentBuffer<T> + Send + Sync>>,
	split_decider: F
}

/* Fix the buffer to not reuqire broad locking it */
impl<T,U,F> Signal<T,U,F> 
	where T: Copy + Send,
		  U: Stream,
		  F: Fn(usize,usize) -> bool,
{

	pub fn new(signal_id: u64, signal: U, seg_size: usize, 
		buffer: Arc<RwLock<SegmentBuffer<T> + Send + Sync>>,
		split_decider: F) 
		-> Signal<T,U,F> 
	{
		Signal {
			timestamp: None,
			prev_seg_offset: None,
			seg_size: seg_size,
			signal_id: signal_id,
			data: Vec::with_capacity(seg_size),
			time_lapse: Vec::with_capacity(seg_size),
			signal: signal,
			buffer: buffer,
			split_decider: split_decider,
		}
	}
}

/* Currently just creates the segment and writes it to a buffer,
   Potential improvements:
   		1. Allow a method to be passed, that will be immediately applied to data
   		2. Allow optional collection of time data for each value
   		3. Allow a function that when passed a segment returns a boolean
   			to indicate that it should be written immediately to file or buffer pool
   		4. Allow function that determines what method to apply,
   			Like a hashmap from signal id to a method enum that should
   				be applied for that signal
   		5. Allow early return/way for user to kill a signal without 
   			having the signal neeed to exhaust the stream
 */
impl<T,U,F> Future for Signal<T,U,F> 
	where T: Copy + Send,
		  U: Stream<Item=T,Error=()>,
		  F: Fn(usize,usize) -> bool,
{
	type Item  = ();
	type Error = ();

	fn poll(&mut self) -> Poll<(),()> {
		loop {
			match self.signal.poll() {
				Ok(Async::NotReady) => return Ok(Async::NotReady),
				Ok(Async::Ready(None)) => return Ok(Async::Ready(())),
				Err(_) => {
					/* Implement an error log to indicate a dropped value */
					/* Continue to run and silence the error for now */
				}
				Ok(Async::Ready(Some(value))) => {

					let cur_time    = SystemTime::now();
					if let None = self.timestamp {
						self.timestamp = Some(cur_time);
					};

					/* case where the value reaches split size */
					if (self.split_decider)(self.data.len(), self.seg_size) {
						let data = mem::replace(&mut self.data, Vec::with_capacity(self.seg_size));
						let time_lapse = mem::replace(&mut self.time_lapse, Vec::with_capacity(self.seg_size));
						let old_timestamp = mem::replace(&mut self.timestamp, Some(cur_time));
						let prev_seg_offset = mem::replace(&mut self.prev_seg_offset, old_timestamp);
						let dur_offset = match prev_seg_offset {
							Some(t) => match old_timestamp.unwrap().duration_since(t) {
								Ok(d) => Some(d),
								Err(_) => panic!("Hard Failure, since messes up implicit chain"),
							}
							None => None,
						};

						let seg = Segment::new(None,old_timestamp.unwrap(),self.signal_id,
											   data, time_lapse, dur_offset);

						match self.buffer.write() {
							Ok(mut buf) => match buf.put(seg) {
								Ok(()) => (),
								Err(_) => panic!("Failed to put segment in buffer"),
							},
							Err(_)  => panic!("Failed to acquire buffer write lock"),
						}; /* Currently panics if can't get it */
					}

					/* Always add the newly received data  */
					self.data.push(value);
					match cur_time.duration_since(self.timestamp.unwrap()) {
						Ok(d)  => self.time_lapse.push(d),
						Err(_) => self.time_lapse.push(Duration::default()),
					}
				}
			}	
		}
	}
}

/* may need to add enter().expect("Failure message for panic")
 * If it seems possible that someone will try to call run from within
 * the main executor running everything
 */
pub fn construct_runtime(num_threads: usize) -> io::Result<Runtime> {
	Builder::new()
		.core_threads(num_threads)
		.build()
}

pub fn execute_runtime<F>(mut rt: Runtime, init_future: F)
	where F: Future<Item = (), Error = ()> + Send + 'static
{
	rt.spawn(init_future);
	rt.shutdown_on_idle().wait().unwrap();
}

#[test]
fn run_dual_signals() {
	let buffer: Arc<RwLock<VDBufferPool<f32>>>  = Arc::new(RwLock::new(VDBufferPool::new()));
	let client1 = match construct_file_client::<f32>(
						"../UCRArchive2018/Ham/Ham_TEST", 1, ',',
						 Amount::Unlimited, RunPeriod::Indefinite, None)
	{
		Ok(x) => x,
		Err(_) => panic!("Failed to create client1"),
	};
	let client2 = match construct_file_client::<f32>(
						"../UCRArchive2018/Fish/Fish_TEST", 1, ',',
						 Amount::Unlimited, RunPeriod::Indefinite, None) 
	{
		Ok(x) => x,
		Err(_) => panic!("Failed to create client2"),
	};

	let sig1 = Signal::new(1, client1, 400, buffer.clone(), |i,j| i >= j);
	let sig2 = Signal::new(2, client2, 600, buffer.clone(), |i,j| i >= j);

	let mut rt = match Builder::new().build() {
		Ok(rt) => rt,
		_ => panic!("Failed to build runtime"),
	};

	rt.spawn(sig1);
	rt.spawn(sig2);
	match rt.shutdown_on_idle().wait() {
		Ok(_) => (),
		Err(_) => panic!("Failed to shutdown properly"),
	}

	let mut buf = match Arc::try_unwrap(buffer) {
		Ok(lock) => match lock.into_inner() {
			Ok(buf) => buf,
			Err(_)  => panic!("Failed to get value in lock"),
		},
		Err(_)   => panic!("Failed to get inner Arc value"),
	};

	let mut seg: &Segment<f32> = match buf.back() {
		Some(seg) => seg,
		None => panic!("Buffer did not store any value"),
	};

	let seg_sig1 = seg.get_signal();

	let mut counter1 = 1;
	while let Some(key) = seg.get_prev_key() {
		seg = match buf.get(key) {
			Some(seg) => seg,
			None  => panic!(format!("Failed to get and remove segment from buffer, {}", counter1)),
		};
		counter1 += 1;
	}

	assert!(counter1 == 113 || counter1 == 135);

	let mut idx = buf.last_idx();

	loop {
		match buf.idx_get(idx) {
			Some(pot_seg) => {
				if pot_seg.get_signal() != seg_sig1 {
					seg = pot_seg;
					break;
				}
			}
			None => panic!("Failed to find a segment from one of the clients"),
		}
		idx -= 1;
	}

	let mut counter2 = 1;
	while let Some(key) = seg.get_prev_key() {
		seg = match buf.get(key) {
			Some(seg) => seg,
			None => panic!(format!("Failed to get and remove segment from buffer, {}", counter2)),
		};
		counter2 += 1;
	}

	match counter1 {
		113 => assert!(counter2 == 135),
		135 => assert!(counter2 == 113),
		_   => panic!("Incorrect number of segments produced"),
	}

	assert!(buf.len() == (113 + 135));
}
