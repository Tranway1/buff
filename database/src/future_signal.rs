extern crate tokio;

use std::str::FromStr;
use serde::Serialize;
use serde::de::DeserializeOwned;
use crate::file_handler::FileManager;
use crate::client::{construct_file_client_skip_newline,Amount,RunPeriod,Frequency};
use std::sync::{Arc,Mutex};
use crate::buffer_pool::{SegmentBuffer,ClockBuffer};
use crate::segment::{Segment,SegmentKey};
use std::time::SystemTime;
use std::time::{Duration,Instant};
use std::{mem, thread};
use tokio::prelude::*;
use tokio::runtime::{Builder,Runtime};
use crate::query::{Count, Max, Sum, Average};
use ndarray::Array2;
use nalgebra::Matrix2;
use crate::kernel::Kernel;
use rustfft::FFTnum;
use num::Float;
use ndarray_linalg::Lapack;
use std::ptr::null;
use futures::sync::oneshot;

pub type SignalId = u64;
const DEFAULT_BATCH_SIZE: usize = 50;

pub struct BufferedSignal<T,U,F,G> 
	where T: Copy + Send,
	      U: Stream,	
	      F: Fn(usize,usize) -> bool,
	      G: Fn(&mut Segment<T>)
{
	start: Option<Instant>,
	timestamp: Option<SystemTime>,
	prev_seg_offset: Option<SystemTime>,
	seg_size: usize,
	signal_id: SignalId,
	data: Vec<T>,
	time_lapse: Vec<Duration>,
	signal: U,
	buffer: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
	split_decider: F,
	compress_func: G,
	compress_on_segmentation: bool,
	compression_percentage: f64,
	segments_produced: u32,
	kernel: Option<Kernel<T>>
}

/* Fix the buffer to not reuqire broad locking it */
impl<T,U,F,G> BufferedSignal<T,U,F,G> 
	where T: Copy + Send+ FFTnum + Float + Lapack,
		  U: Stream,
		  F: Fn(usize,usize) -> bool,
		  G: Fn(&mut Segment<T>)
{

	pub fn new(signal_id: u64, signal: U, seg_size: usize, 
		buffer: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
		split_decider: F, compress_func: G, 
		compress_on_segmentation: bool, dict: Option<Array2<T>>)
		-> BufferedSignal<T,U,F,G> 
	{
		let mut kernel:Option<Kernel<T>>= match dict {
			// The division was valid
			Some(x) => {
				let mut kernel_dict = Kernel::new(x.clone(), 1, 4, DEFAULT_BATCH_SIZE);
				kernel_dict.dict_pre_process();
				Some(kernel_dict)
			},
			// The division was invalid
			None    => None,
		};


		BufferedSignal {
			start: None,
			timestamp: None,
			prev_seg_offset: None,
			seg_size: seg_size,
			signal_id: signal_id,
			data: Vec::with_capacity(seg_size),
			time_lapse: Vec::with_capacity(seg_size),
			signal: signal,
			buffer: buffer,
			split_decider: split_decider,
			compress_func: compress_func,
			compress_on_segmentation: compress_on_segmentation,
			compression_percentage: 0.0,
			segments_produced: 0,
			kernel: kernel,
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
impl<T,U,F,G> Future for BufferedSignal<T,U,F,G> 
	where T: Copy + Send+ FFTnum+ Float+Lapack,
		  U: Stream<Item=T,Error=()>,
		  F: Fn(usize,usize) -> bool,
		  G: Fn(&mut Segment<T>)
{
	type Item  = Option<SystemTime>;
	type Error = ();

	fn poll(&mut self) -> Poll<Option<SystemTime>,()> {
		let mut batch_vec: Vec<T> = Vec::new();
		let mut bsize = 0;
		loop {
			match self.signal.poll() {
				Ok(Async::NotReady) => return Ok(Async::NotReady),
				Ok(Async::Ready(None)) => {
					let elapse: Duration = self.start.unwrap().elapsed();
					if self.compress_on_segmentation {
						let percentage = self.compression_percentage / (self.segments_produced as f64);
						println!("Signal: {}\n Segments produced: {}\n Compression percentage: {}\n Time: {:?}", self.signal_id, self.segments_produced, percentage, elapse);
					} else {
						//self.buffer.lock().unwrap().flush();
						println!("Signal: {}\n Segments produced: {}\n Data points in total {} \n Time: {:?}\n Throughput: {:?} points/second", self.signal_id, (self.segments_produced as usize)/self.seg_size, self.segments_produced, elapse, (self.segments_produced as f64) / ((elapse.as_nanos() as f64) / (1_000_000_000 as f64)));
					}
					
					return Ok(Async::Ready(self.prev_seg_offset))
				}
				Err(e) => {
					println!("The client signal produced an error: {:?}", e);
					/* Implement an error log to indicate a dropped value */
					/* Continue to run and silence the error for now */
					return Err(e);
				}
				Ok(Async::Ready(Some(value))) => {

					let cur_time    = SystemTime::now();
					if let None = self.timestamp {
						self.start = Some(Instant::now());
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
						//todo: adjust logics here to fix kernel method.
						if bsize<DEFAULT_BATCH_SIZE{
							//batch_vec.extend(&data);
							//bsize= bsize+1;
						}
						else {
							bsize = 0;
							let belesize = batch_vec.len();
							println!("vec for matrix length: {}", belesize);
							let mut x = Array2::from_shape_vec((DEFAULT_BATCH_SIZE,self.seg_size),mem::replace(&mut batch_vec, Vec::with_capacity(belesize))).unwrap();
							println!("matrix shape: {} * {}", x.rows(), x.cols());
							match &self.kernel{
								Some(kn) => kn.run(x),
								None => (),
							};
							println!("new vec for matrix length: {}", batch_vec.len());
						}

						let mut seg = Segment::new(None,old_timestamp.unwrap(),self.signal_id,
											   data, Some(time_lapse), dur_offset);
						
						if self.compress_on_segmentation {
							let before = self.data.len() as f64;
							(self.compress_func)(&mut seg);
							let after = self.data.len() as f64;
							self.compression_percentage += after/before;
						}


						match self.buffer.lock() {
							Ok(mut buf) => match buf.put(seg) {
								Ok(()) => (),
								Err(e) => panic!("Failed to put segment in buffer: {:?}", e),
							},
							Err(_)  => panic!("Failed to acquire buffer write lock"),
						}; /* Currently panics if can't get it */

					}

					/* Always add the newly received data  */
					self.data.push(value);
					self.segments_produced += 1;
					match cur_time.duration_since(self.timestamp.unwrap()) {
						Ok(d)  => self.time_lapse.push(d),
						Err(_) => self.time_lapse.push(Duration::default()),
					}
				}
			}	
		}
	}
}

pub struct NonStoredSignal<T,U,F,G> 
	where T: Copy + Send,
	      U: Stream,
	      F: Fn(usize,usize) -> bool,
	      G: Fn(&mut Segment<T>)
{
	timestamp: Option<SystemTime>,
	prev_seg_offset: Option<SystemTime>,
	seg_size: usize,
	signal_id: SignalId,
	data: Vec<T>,
	time_lapse: Vec<Duration>,
	signal: U,
	split_decider: F,
	compress_func: G,
	compress_on_segmentation: bool,
	compression_percentage: f64,
	segments_produced: u64,
}

/* Fix the buffer to not reuqire broad locking it */
impl<T,U,F,G> NonStoredSignal<T,U,F,G> 
	where T: Copy + Send,
		  U: Stream,
		  F: Fn(usize,usize) -> bool,
		  G: Fn(&mut Segment<T>)
{

	pub fn new(signal_id: u64, signal: U, seg_size: usize, 
		split_decider: F, compress_func: G, compress_on_segmentation: bool) 
		-> NonStoredSignal<T,U,F,G> 
	{
		NonStoredSignal {
			timestamp: None,
			prev_seg_offset: None,
			seg_size: seg_size,
			signal_id: signal_id,
			data: Vec::with_capacity(seg_size),
			time_lapse: Vec::with_capacity(seg_size),
			signal: signal,
			split_decider: split_decider,
			compress_func: compress_func,
			compress_on_segmentation: compress_on_segmentation,
			segments_produced: 0,
			compression_percentage: 0.0,
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
impl<T,U,F,G> Future for NonStoredSignal<T,U,F,G> 
	where T: Copy + Send,
		  U: Stream<Item=T,Error=()>,
		  F: Fn(usize,usize) -> bool,
		  G: Fn(&mut Segment<T>)
{
	type Item  = Option<SystemTime>;
	type Error = ();

	fn poll(&mut self) -> Poll<Option<SystemTime>,()> {
		loop {
			match self.signal.poll() {
				Ok(Async::NotReady) => return Ok(Async::NotReady),
				Ok(Async::Ready(None)) => {
					if self.compress_on_segmentation {
						let percentage = self.compression_percentage / (self.segments_produced as f64);
						println!("Signal {} produced {} segments with a compression percentage of {}", self.signal_id, self.segments_produced, percentage);
					} else {
						println!("Signal {} produced {} segments", self.signal_id, (self.segments_produced as usize)/self.seg_size);
					}
					
					return Ok(Async::Ready(self.prev_seg_offset))
				}
				Err(e) => {
					println!("The client signal produced an error: {:?}", e);
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

						let mut seg = Segment::new(None,old_timestamp.unwrap(),self.signal_id,
											   data, Some(time_lapse), dur_offset);

						if self.compress_on_segmentation {
							let before = self.data.len() as f64;
							(self.compress_func)(&mut seg);
							let after = self.data.len() as f64;
							self.compression_percentage += after/before;
						}
					}

					/* Always add the newly received data  */
					self.data.push(value);
					self.segments_produced += 1;
					match cur_time.duration_since(self.timestamp.unwrap()) {
						Ok(d)  => self.time_lapse.push(d),
						Err(_) => self.time_lapse.push(Duration::default()),
					}
				}
			}	
		}
	}
}

pub struct StoredSignal<T,U,F,G,V> 
	where T: Copy + Send + Serialize + DeserializeOwned,
	      U: Stream,
	      F: Fn(usize,usize) -> bool,
	      G: Fn(&mut Segment<T>),
	      V: AsRef<[u8]>,
{
	timestamp: Option<SystemTime>,
	prev_seg_offset: Option<SystemTime>,
	seg_size: usize,
	signal_id: SignalId,
	data: Vec<T>,
	time_lapse: Vec<Duration>,
	signal: U,
	fm: Arc<Mutex<FileManager<Vec<u8>,V> + Send + Sync>>,
	split_decider: F,
	compress_func: G,
	compress_on_segmentation: bool,
	compression_percentage: f64,
	segments_produced: u64,
}

/* Fix the buffer to not reuqire broad locking it */
impl<T,U,F,G,V> StoredSignal<T,U,F,G,V> 
	where T: Copy + Send + Serialize + DeserializeOwned,
		  U: Stream,
		  F: Fn(usize,usize) -> bool,
		  G: Fn(&mut Segment<T>),
		  V: AsRef<[u8]>,
{

	pub fn new(signal_id: u64, signal: U, seg_size: usize, 
		fm: Arc<Mutex<FileManager<Vec<u8>,V> + Send + Sync>>,
		split_decider: F, compress_func: G, 
		compress_on_segmentation: bool) 
		-> StoredSignal<T,U,F,G,V> 
	{
		StoredSignal {
			timestamp: None,
			prev_seg_offset: None,
			seg_size: seg_size,
			signal_id: signal_id,
			data: Vec::with_capacity(seg_size),
			time_lapse: Vec::with_capacity(seg_size),
			signal: signal,
			fm: fm,
			split_decider: split_decider,
			compress_func: compress_func,
			compress_on_segmentation: compress_on_segmentation,
			compression_percentage: 0.0,
			segments_produced: 0,
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
impl<T,U,F,G,V> Future for StoredSignal<T,U,F,G,V> 
	where T: Copy + Send + Serialize + DeserializeOwned + FromStr,
		  U: Stream<Item=T,Error=()>,
		  F: Fn(usize,usize) -> bool,
		  G: Fn(&mut Segment<T>),
		  V: AsRef<[u8]>,
{
	type Item  = Option<SystemTime>;
	type Error = ();

	fn poll(&mut self) -> Poll<Option<SystemTime>,()> {
		loop {
			match self.signal.poll() {
				Ok(Async::NotReady) => return Ok(Async::NotReady),
				Ok(Async::Ready(None)) => {
					if self.compress_on_segmentation {
						let percentage = self.compression_percentage / (self.segments_produced as f64);
						println!("Signal {} produced {} segments with a compression percentage of {}", self.signal_id, self.segments_produced, percentage);
					} else {
						println!("Signal {} produced {} segments", self.signal_id, (self.segments_produced as usize)/self.seg_size);
					}
					
					return Ok(Async::Ready(self.prev_seg_offset))
				}
				Err(e) => {
					println!("The client signal produced an error: {:?}", e);
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

						let mut seg = Segment::new(None,old_timestamp.unwrap(),self.signal_id,
											   data, Some(time_lapse), dur_offset);
						
						if self.compress_on_segmentation {
							let before = self.data.len() as f64;
							(self.compress_func)(&mut seg);
							let after = self.data.len() as f64;
							self.compression_percentage += after/before;
						}

						match self.fm.lock() {
							Ok(fm) => {
								let key_bytes = seg.get_key().convert_to_bytes().expect("The segment key should be byte convertible");
								let seg_bytes = seg.convert_to_bytes().expect("The segment should be byte convertible");
								match fm.fm_write(key_bytes, seg_bytes) {
									Ok(()) => (),
									Err(e) => panic!("Failed to put segment in buffer: {:?}", e),
								}
							}
							Err(_)  => panic!("Failed to acquire buffer write lock"),
						}; /* Currently panics if can't get it */
					}

					/* Always add the newly received data  */
					self.data.push(value);
					self.segments_produced += 1;
					match cur_time.duration_since(self.timestamp.unwrap()) {
						Ok(d)  => self.time_lapse.push(d),
						Err(_) => self.time_lapse.push(Duration::default()),
					}
				}
			}	
		}
	}
}

#[test]
fn run_dual_signals() {
	let mut db_opts = rocksdb::Options::default();
	db_opts.create_if_missing(true);
	let fm = match rocksdb::DB::open(&db_opts, "../rocksdb") {
		Ok(x) => x,
		Err(e) => panic!("Failed to create database: {:?}", e),
	};

	let buffer: Arc<Mutex<ClockBuffer<f32,rocksdb::DB>>>  = Arc::new(Mutex::new(ClockBuffer::new(50,fm)));
	let client1 = match construct_file_client_skip_newline::<f32>(
						"../UCRArchive2018/Ham/Ham_TEST", 1, ',',
						 Amount::Unlimited, RunPeriod::Indefinite, Frequency::Immediate)
	{
		Ok(x) => x,
		Err(_) => panic!("Failed to create client1"),
	};
	let client2 = match construct_file_client_skip_newline::<f32>(
						"../UCRArchive2018/Fish/Fish_TEST", 1, ',',
						 Amount::Unlimited, RunPeriod::Indefinite, Frequency::Immediate) 
	{
		Ok(x) => x,
		Err(_) => panic!("Failed to create client2"),
	};

	let sig1 = BufferedSignal::new(1, client1, 400, buffer.clone(), |i,j| i >= j, |_| (), false,None);
	let sig2 = BufferedSignal::new(2, client2, 600, buffer.clone(), |i,j| i >= j, |_| (), false,None);

	let mut rt = match Builder::new().build() {
		Ok(rt) => rt,
		_ => panic!("Failed to build runtime"),
	};

	let (seg_key1,seg_key2) = match rt.block_on(sig1.join(sig2)) {
		Ok((Some(time1),Some(time2))) => (SegmentKey::new(time1,1),SegmentKey::new(time2,2)),
		_ => panic!("Failed to get the last system time for signal1 or signal2"),
	};

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



	let mut seg1: &Segment<f32> = match buf.get(seg_key1).unwrap() {
		Some(seg) => seg,
		None => panic!("Buffer lost track of the last value"),
	};


	let mut counter1 = 1;
	while let Some(key) = seg1.get_prev_key() {
		seg1 = match buf.get(key).unwrap() {
			Some(seg) => seg,
			None  => panic!(format!("Failed to get and remove segment from buffer, {}", counter1)),
		};
		counter1 += 1;
	}

	assert!(counter1 == 113 || counter1 == 135);

	let mut seg2: &Segment<f32> = match buf.get(seg_key2).unwrap() {
		Some(seg) => seg,
		None => panic!("Buffer lost track of the last value"),
	};

	let mut counter2 = 1;
	while let Some(key) = seg2.get_prev_key() {
		seg2 = match buf.get(key).unwrap() {
			Some(seg) => seg,
			None  => panic!(format!("Failed to get and remove segment from buffer, {}", counter1)),
		};
		counter2 += 1;
	}

	match counter1 {
		113 => assert!(counter2 == 135),
		135 => assert!(counter2 == 113),
		_   => panic!("Incorrect number of segments produced"),
	}

}





#[test]
fn run_single_signals() {
    let mut db_opts = rocksdb::Options::default();
    db_opts.create_if_missing(true);
    let fm = match rocksdb::DB::open(&db_opts, "../rocksdb") {
        Ok(x) => x,
        Err(e) => panic!("Failed to create database: {:?}", e),
    };

    let buffer: Arc<Mutex<ClockBuffer<f32,rocksdb::DB>>>  = Arc::new(Mutex::new(ClockBuffer::new(50,fm)));
    let client1 = match construct_file_client_skip_newline::<f32>(
        "../UCRArchive2018/Kernel/randomwalkdatasample1k-1k", 1, ',',
        Amount::Unlimited, RunPeriod::Indefinite, Frequency::Immediate)
        {
            Ok(x) => x,
            Err(_) => panic!("Failed to create client1"),
        };
    let client2 = match construct_file_client_skip_newline::<f32>(
		"../UCRArchive2018/Kernel/randomwalkdatasample1k-1k", 1, ',',
        Amount::Unlimited, RunPeriod::Indefinite, Frequency::Immediate)
        {
            Ok(x) => x,
            Err(_) => panic!("Failed to create client2"),
        };
	let start = Instant::now();
    let sig1 = BufferedSignal::new(1, client1, 1000, buffer.clone(), |i,j| i >= j, |_| (), false, None);
    let sig2 = BufferedSignal::new(2, client2, 1000, buffer.clone(), |i,j| i >= j, |_| (), false,None);

    let mut rt = match Builder::new().build() {
        Ok(rt) => rt,
        _ => panic!("Failed to build runtime"),
    };


//	let handle1 = thread::spawn( move || {
//		println!("Run ingestion demon 1" );
//		oneshot::spawn(sig1, &rt1.executor());
//	});
//
//	let handle2 = thread::spawn( move || {
//		println!("Run ingestion demon 2" );
//		oneshot::spawn(sig2, &rt2.executor());
//	});

	let (seg_key1,seg_key2) = match rt.block_on(sig2.join(sig1)) {
		Ok((Some(time1),Some(time2))) => (SegmentKey::new(time1,1),SegmentKey::new(time2,2)),
		_ => panic!("Failed to get the last system time for signal1 or signal2"),
	};
	let duration = start.elapsed();
//	handle2.join().unwrap();
//	handle1.join().unwrap();
	println!("Time elapsed in ingestion function() is: {:?}", duration);
//
//	let (seg_key1) = match rt.block_on(sig1) {
//        Ok((Some(time1))) => (SegmentKey::new(time1,1)),
//        _ => panic!("Failed to get the last system time for signal1 or signal2"),
//    };



    let mut buf = match Arc::try_unwrap(buffer) {
        Ok(lock) => match lock.into_inner() {
            Ok(buf) => buf,
            Err(_)  => panic!("Failed to get value in lock"),
        },
        Err(_)   => panic!("Failed to get inner Arc value"),
    };


    /* Add query test */
//    let count = Max::run(&buf);
//    println!("total count: {}", count);

//    let mut seg1: &Segment<f32> = match buf.get(seg_key1).unwrap() {
//        Some(seg) => seg,
//        None => panic!("Buffer lost track of the last value"),
//    };
//
//
//    let mut counter1 = 1;
//    while let Some(key) = seg1.get_prev_key() {
//        seg1 = match buf.get(key).unwrap() {
//            Some(seg) => seg,
//            None  => panic!(format!("Failed to get and remove segment from buffer, {}", counter1)),
//        };
//        counter1 += 1;
//    }
//
//    assert!(counter1 == 113 || counter1 == 135);
//
//    let mut seg2: &Segment<f32> = match buf.get(seg_key2).unwrap() {
//        Some(seg) => seg,
//        None => panic!("Buffer lost track of the last value"),
//    };
//
//    let mut counter2 = 1;
//    while let Some(key) = seg2.get_prev_key() {
//        seg2 = match buf.get(key).unwrap() {
//            Some(seg) => seg,
//            None  => panic!(format!("Failed to get and remove segment from buffer, {}", counter1)),
//        };
//        counter2 += 1;
//    }
//
//    match counter1 {
//        113 => assert!(counter2 == 135),
//        135 => assert!(counter2 == 113),
//        _   => panic!("Incorrect number of segments produced"),
//    }

}

