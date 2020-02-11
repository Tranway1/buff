use std::fmt::Debug;
use serde::{Serialize,Deserialize};
use serde::de::DeserializeOwned;
use rocksdb::DBVector;
use crate::file_handler::{FileManager};
use std::collections::vec_deque::Drain;
use std::collections::VecDeque;
use std::collections::hash_map::{HashMap,Entry};

use crate::segment;

use segment::{Segment,SegmentKey};

/* 
 * Overview:
 * This is the API for a buffer pool iplementation. 
 * The goal is to construct a trait for immutable buffers and then 
 * provide an extension to make mutable (element-wise) buffers. 
 * This constructs a general framework that makes future
 * implementations easier to integrate.
 *
 * Design Choice:
 * The buffer design is broken up into two parts
 * 1. A trait that must be implemented to create an immutable 
 *    (element-wise) buffer.
 * 2. A trait that can then extend the buffer to be mutable
 *    at an element-wise level.
 *
 * Current Implementations:
 * There are two basic implementations created to mainly show
 * how these traits could be implemented as well as create a 
 * structure that would make the implementation easy.
 */


/***************************************************************
 **************************Buffer_Pool**************************
 ***************************************************************/

pub trait SegmentBuffer<T: Copy + Send> {

	/* If the index is valid it will return: Ok(ref T)
	 * Otherwise, it will return: Err(()) to indicate 
	 * that the index was invalid
	 */
	fn get(&mut self, key: SegmentKey) -> Result<Option<&Segment<T>>,BufErr>;

	/* If the index is valid it will return: Ok(mut ref T)
	 * Otherwise, it will return: Err(()) to indicate 
	 * that the index was invalid
	 */
	fn get_mut(&mut self, key: SegmentKey) -> Result<Option<&Segment<T>>,BufErr>;

	/* If the buffer succeeds it will return: Ok(index)
	 * Otherwise, it will return: Err(BufErr) to indicate 
	 * that some failure occured and the item couldn't be
	 * buffered.
	 */
	fn put(&mut self, item: Segment<T>) -> Result<(), BufErr>;


	/* Will return a Drain iterator that gains ownership over
	 * the items in the buffer, but will leave the allocation
	 * of the buffer untouched
	 */
	fn drain(&mut self) -> Drain<Segment<T>>;

	/* Will copy the buffer and collect it into a vector */
	fn copy(&self) -> Vec<Segment<T>>;

	/* Will lock the buffer and write everything to disk */
	fn persist(&self) -> Result<(),BufErr>;

	/* Will empty the buffer essentially clear */
	fn flush(&mut self);

	/* Returns true if the number of items in the buffer divided by 
	 * the maximum number of items the buffer can hold exceeds
	 * the provided threshold
	 */
	fn exceed_threshold(&self, threshold: f32) -> bool;

	/* Returns true if the number of items in the buffer exceeds
	 * the provided batchsize
	 */
	fn exceed_batch(&self, batchsize: usize) -> bool;

	/* Remove the segment from the buffer and return it */
	fn remove_segment(&mut self) -> Result<Segment<T>, BufErr>;


	/* Signal done*/
	fn is_done(&self)->bool;
}


/* Designed to carry error information 
 * May be unnecessary
 */
#[derive(Debug)]
pub enum BufErr {
	NonUniqueKey,
	FailedSegKeySer,
	FailedSegSer,
	FileManagerErr,
	FailPut,
	ByteConvertFail,
	GetFail,
	GetMutFail,
	EvictFailure,
	BufEmpty,
	UnderThresh,
	RemoveFailure,
	CantGrabMutex,
}


/***************************************************************
 ************************VecDeque_Buffer************************
 ***************************************************************/
/* Look into fixed vec deque or concurrent crates if poor performance */

#[derive(Debug)]
pub struct ClockBuffer<T,U> 
	where T: Copy + Send,
		  U: FileManager<Vec<u8>,DBVector> + Sync + Send,
{
	hand: usize,
	tail: usize,
	buffer: HashMap<SegmentKey,Segment<T>>,
	clock: Vec<(SegmentKey,bool)>,
	clock_map: HashMap<SegmentKey,usize>,
	file_manager: U,
	buf_size: usize,
	done: bool,
}


impl<T,U> SegmentBuffer<T> for ClockBuffer<T,U> 
	where T: Copy + Send + Serialize + DeserializeOwned + Debug,
		  U: FileManager<Vec<u8>,DBVector> + Sync + Send,
{

	fn get(&mut self, key: SegmentKey) -> Result<Option<&Segment<T>>,BufErr> {		
		if self.retrieve(key)? {
			match self.buffer.get(&key) {
				Some(seg) => Ok(Some(seg)),
				None => Err(BufErr::GetFail),
			}
		} else {
			Ok(None)
		}
	}

	fn get_mut(&mut self, key: SegmentKey) -> Result<Option<&Segment<T>>,BufErr> {
		if self.retrieve(key)? {
			match self.buffer.get_mut(&key) {
				Some(seg) => Ok(Some(seg)),
				None => Err(BufErr::GetMutFail),
			}
		} else {
			Ok(None)
		}
	}


	fn is_done(&self)->bool{
		self.done
	}

	#[inline]
	fn put(&mut self, seg: Segment<T>) -> Result<(), BufErr> {
		let seg_key = seg.get_key();
		self.put_with_key(seg_key, seg)
	}


	fn drain(&mut self) -> Drain<Segment<T>> {
		unimplemented!()
	}

	fn copy(&self) -> Vec<Segment<T>> {
		self.buffer.values().map(|x| x.clone()).collect()
	}

	/* Write to file system */
	fn persist(&self) -> Result<(),BufErr> {
		for (seg_key,seg) in self.buffer.iter() {
			let seg_key_bytes = match seg_key.convert_to_bytes() {
				Ok(bytes) => bytes,
				Err(_) => return Err(BufErr::FailedSegKeySer),
			};
			let seg_bytes = match seg.convert_to_bytes() {
				Ok(bytes) => bytes,
				Err(_) => return Err(BufErr::FailedSegSer),
			};

			match self.file_manager.fm_write(seg_key_bytes,seg_bytes) {
				Err(_) => return Err(BufErr::FileManagerErr),
				_ => (),
			}
		}

		Ok(())
	}

	fn flush(&mut self) {
		self.buffer.clear();
		self.clock.clear();
		self.done = true;
	}


	fn exceed_threshold(&self, threshold: f32) -> bool {
		//println!("{}full, threshold:{}",(self.buffer.len() as f32 / self.buf_size as f32), threshold);
		return (self.buffer.len() as f32 / self.buf_size as f32) >= threshold;
	}

	fn remove_segment(&mut self) -> Result<Segment<T>,BufErr> {
		let mut counter = 0;
		loop {
			if let (seg_key,false) = self.clock[self.tail] {
				let seg = match self.buffer.remove(&seg_key) {
					Some(seg) => seg,
					None => {
						//println!("Failed to get segment from buffer.");
						self.update_tail();
						return Err(BufErr::EvictFailure)
					},
				};
				match self.clock_map.remove(&seg_key) {
					None => panic!("Non-unique key panic as clock map and buffer are desynced somehow"),
					_ => (),
				}
				//println!("fetch a segment from buffer.");
				return Ok(seg);
			} else {
				self.clock[self.tail].1 = false;
			} 

			self.update_tail();
			counter += 1;
			if counter > self.clock.len() {
				return Err(BufErr::BufEmpty); 
			}
		}
	}

	fn exceed_batch(&self, batchsize: usize) -> bool {
		return self.buffer.len() >= batchsize;
	}
}


impl<T,U> ClockBuffer<T,U> 
	where T: Copy + Send + Serialize + DeserializeOwned + Debug,
		  U: FileManager<Vec<u8>,DBVector> + Sync + Send,
{
	pub fn new(buf_size: usize, file_manager: U) -> ClockBuffer<T,U> {
		ClockBuffer {
			hand: 0,
			tail: 0,
			buffer: HashMap::with_capacity(buf_size),
			clock: Vec::with_capacity(buf_size),
			clock_map: HashMap::with_capacity(buf_size),
			file_manager: file_manager,
			buf_size: buf_size,
			done: false,
		}
	}

	/* Assumes that the segment is in memory and will panic otherwise */
	#[inline]
	fn update(&mut self, key: SegmentKey) {
		let key_idx: usize = *self.clock_map.get(&key).unwrap();
		self.clock[key_idx].1 = false;
	}

	#[inline]
	fn update_hand(&mut self) {
		self.hand = (self.hand + 1) % self.buf_size;
	}

	#[inline]
	fn update_tail(&mut self) {
		self.tail = (self.tail + 1) % self.clock.len();
	}

	fn put_with_key(&mut self, key: SegmentKey, seg: Segment<T>) -> Result<(), BufErr> {
		let slot = if self.buffer.len() >= self.buf_size {
			let slot = self.evict_no_saving()?;
			self.clock[slot] = (key,true);
			slot
		} else {
			let slot = self.hand;
			self.clock.push((key,true));
			self.update_hand();
			slot
		};

		
		match self.clock_map.insert(key,slot) {
			None => (),
			_ => return Err(BufErr::NonUniqueKey),
		}
		match self.buffer.entry(key) {
			Entry::Occupied(_) => panic!("Non-unique key panic as clock map and buffer are desynced somehow"),
			Entry::Vacant(vacancy) => {
				vacancy.insert(seg);
				Ok(())
			}
		}
	}

	/* Gets the segment from the filemanager and places it in */
	fn retrieve(&mut self, key: SegmentKey) -> Result<bool,BufErr> {
		if let Some(_) = self.buffer.get(&key) {
			println!("reading from the buffer");
			self.update(key);
			return Ok(true);
		}
		println!("reading from the file_manager");
		match key.convert_to_bytes() {
			Ok(key_bytes) => {
				match self.file_manager.fm_get(key_bytes) {
					Err(_) => Err(BufErr::FileManagerErr),
					Ok(None) => Ok(false),
					Ok(Some(bytes)) => {
						match Segment::convert_from_bytes(&bytes) {
							Ok(seg) => {
								self.put_with_key(key,seg)?;
								Ok(true)
							}
							Err(()) => Err(BufErr::ByteConvertFail),
						}
					}
				}				
			}
			Err(_) => Err(BufErr::ByteConvertFail)
		}
	}

	fn evict(&mut self) -> Result<usize,BufErr> {
		loop {
			if let (seg_key,false) = self.clock[self.hand] {
				let seg = match self.buffer.remove(&seg_key) {
					Some(seg) => seg,
					None => return Err(BufErr::EvictFailure),
				};
				match self.clock_map.remove(&seg_key) {
					None => panic!("Non-unique key panic as clock map and buffer are desynced somehow"),
					_ => (),
				}

				/* Write the segment to disk */
				let seg_key_bytes = match seg_key.convert_to_bytes() {
					Ok(bytes) => bytes,
					Err(()) => return Err(BufErr::FailedSegKeySer)
				};
				let seg_bytes = match seg.convert_to_bytes() {
					Ok(bytes) => bytes,
					Err(()) => return Err(BufErr::FailedSegSer),
				};

				match self.file_manager.fm_write(seg_key_bytes,seg_bytes) {
					Ok(()) => {
						self.tail = self.hand + 1;
						return Ok(self.hand)
					}
					Err(_) => return Err(BufErr::FileManagerErr),
				}

			} else {
				self.clock[self.hand].1 = false;
			} 

			self.update_hand();
		}
	}

	fn evict_no_saving(&mut self) -> Result<usize,BufErr> {
		loop {
			if let (seg_key,false) = self.clock[self.hand] {
				self.buffer.remove(&seg_key);
				self.clock_map.remove(&seg_key);
				return Ok(self.hand)
			} else {
				self.clock[self.hand].1 = false;
			}

			self.update_hand();
		}
	}
}


#[derive(Debug)]
pub struct NoFmClockBuffer<T> 
	where T: Copy + Send,
{
	hand: usize,
	tail: usize,
	buffer: HashMap<SegmentKey,Segment<T>>,
	clock: Vec<(SegmentKey,bool)>,
	clock_map: HashMap<SegmentKey,usize>,
	buf_size: usize,
	done: bool,
}


impl<T> SegmentBuffer<T> for NoFmClockBuffer<T> 
	where T: Copy + Send + Debug,
{

	fn get(&mut self, _key: SegmentKey) -> Result<Option<&Segment<T>>,BufErr> {		
		unimplemented!()
	}

	fn get_mut(&mut self, _key: SegmentKey) -> Result<Option<&Segment<T>>,BufErr> {
		unimplemented!()
	}

	fn is_done(&self)->bool{
		self.done
	}

	#[inline]
	fn put(&mut self, seg: Segment<T>) -> Result<(), BufErr> {
		let seg_key = seg.get_key();
		self.put_with_key(seg_key, seg)
	}


	fn drain(&mut self) -> Drain<Segment<T>> {
		unimplemented!()
	}

	fn copy(&self) -> Vec<Segment<T>> {
		self.buffer.values().map(|x| x.clone()).collect()
	}

	/* Write to file system */
	fn persist(&self) -> Result<(),BufErr> {
		unimplemented!()
	}

	fn flush(&mut self) {
		self.buffer.clear();
		self.clock.clear();
		self.done = true;
	}

	fn exceed_threshold(&self, threshold: f32) -> bool {
		return (self.buffer.len() as f32 / self.buf_size as f32) > threshold;
	}

	fn remove_segment(&mut self) -> Result<Segment<T>,BufErr> {
		let mut counter = 0;
		loop {
			if let (seg_key,false) = self.clock[self.hand] {
				let seg = match self.buffer.remove(&seg_key) {
					Some(seg) => seg,
					None => return Err(BufErr::EvictFailure),
				};
				match self.clock_map.remove(&seg_key) {
					None => panic!("Non-unique key panic as clock map and buffer are desynced somehow"),
					_ => (),
				}

				return Ok(seg);
			} else {
				self.clock[self.hand].1 = false;
			} 

			self.update_hand();
			counter += 1;
			if counter >= self.clock.len() {
				return Err(BufErr::BufEmpty);
			}
		}
	}

	fn exceed_batch(&self, batchsize: usize) -> bool {
		return self.buffer.len() >= batchsize;
	}
}


impl<T> NoFmClockBuffer<T> 
	where T: Copy + Send + Debug,
{
	pub fn new(buf_size: usize) -> NoFmClockBuffer<T> {
		NoFmClockBuffer {
			hand: 0,
			tail: 0,
			buffer: HashMap::with_capacity(buf_size),
			clock: Vec::with_capacity(buf_size),
			clock_map: HashMap::with_capacity(buf_size),
			buf_size: buf_size,
			done:false,
		}
	}

	/* Assumes that the segment is in memory and will panic otherwise */
	#[inline]
	fn update(&mut self, key: SegmentKey) {
		let key_idx: usize = *self.clock_map.get(&key).unwrap();
		self.clock[key_idx].1 = false;
	}

	#[inline]
	fn update_hand(&mut self) {
		self.hand = (self.hand + 1) % self.buf_size;
	}

	fn put_with_key(&mut self, key: SegmentKey, seg: Segment<T>) -> Result<(), BufErr> {
		let slot = if self.buffer.len() >= self.buf_size {
			let slot = self.evict()?;
			self.clock[slot] = (key,true);
			slot
		} else {
			let slot = self.hand;
			self.clock.push((key,true));
			self.update_hand();
			slot
		};

		
		match self.clock_map.insert(key,slot) {
			None => (),
			_ => return Err(BufErr::NonUniqueKey),
		}
		match self.buffer.entry(key) {
			Entry::Occupied(_) => panic!("Non-unique key panic as clock map and buffer are desynced somehow"),
			Entry::Vacant(vacancy) => {
				vacancy.insert(seg);
				Ok(())
			}
		}
	}

	fn evict(&mut self) -> Result<usize,BufErr> {
		loop {
			if let (seg_key,false) = self.clock[self.hand] {
				let _seg = match self.buffer.remove(&seg_key) {
					Some(seg) => seg,
					None => return Err(BufErr::EvictFailure),
				};
				match self.clock_map.remove(&seg_key) {
					None => panic!("Non-unique key panic as clock map and buffer are desynced somehow"),
					_ => (),
				}
			} else {
				self.clock[self.hand].1 = false;
			} 

			self.update_hand();
		}
	}
}