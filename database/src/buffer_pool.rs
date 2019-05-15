use crate::file_handler::{FileManager,RocksFM};
use std::collections::vec_deque::Drain;
use std::collections::VecDeque;
use std::collections::hash_map::HashMap;

use crate::segment;

use segment::{Segment,SegmentKey};

const BUFFERSIZE: usize = 500;

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

pub trait SegmentBuffer<T: Clone + Send> {

	/* If the index is valid it will return: Ok(ref T)
	 * Otherwise, it will return: Err(()) to indicate 
	 * that the index was invalid
	 */
	fn get(&self, key: SegmentKey) -> Option<&Segment<T>>;

	/* If the index is valid it will return: Ok(mut ref T)
	 * Otherwise, it will return: Err(()) to indicate 
	 * that the index was invalid
	 */
	fn get_mut(&mut self, key: SegmentKey) -> Option<&mut Segment<T>>;

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

	/* Will evict some number of items from the buffer freeing
	   There location */
	fn evict(&self, num_evict: u32);

	/* Will lock the buffer and write everything to disk */
	fn persist(&self);

	/* Will empty the buffer essentially clear */
	fn flush(&self);
}


/* Designed to carry error information 
 * May be unnecessary
 */
pub struct BufErr {
	info: u32,
}

/***************************************************************
 ************************VecDeque_Buffer************************
 ***************************************************************/

#[derive(Debug)]
pub struct VDBufferPool<T:Send> {
	buffer: VecDeque<Segment<T>>,
	map: HashMap<SegmentKey,usize>
}

impl<T: Copy + Send> SegmentBuffer<T> for VDBufferPool<T> {

	fn get(&self, key: SegmentKey) -> Option<&Segment<T>> {
		match self.map.get(&key) {
			Some(idx) => self.buffer.get(*idx),
			None => None,
		}
	}

	fn get_mut(&mut self, key: SegmentKey) -> Option<&mut Segment<T>> {
		match self.map.get(&key) {
			Some(idx) => self.buffer.get_mut(*idx),
			None => None,
		}
	}

	fn put(&mut self, seg: Segment<T>) -> Result<(), BufErr> {
		let seg_key = seg.get_key();
		self.buffer.push_back(seg);
		let idx = self.buffer.len() - 1;
		if let Some(_) = self.map.insert(seg_key,idx) {
			return Err(BufErr { info: 0 });
		}

		Ok(())
	}


	fn drain(&mut self) -> Drain<Segment<T>> {
		self.buffer.drain(..)
	}

	fn copy(&self) -> Vec<Segment<T>> {
		Vec::from(self.buffer.clone())
	}

	/* Currently not implemented, need to change */
	fn evict(&self, _num_evict: u32) {
		unimplemented!() 
	}

	fn persist(&self) {
		unimplemented!()
	}

	fn flush(&self) {
		unimplemented!()
	}
}


impl<T: Send> VDBufferPool<T> {
	pub fn new() -> VDBufferPool<T> {
		VDBufferPool {
			buffer: VecDeque::with_capacity(BUFFERSIZE),
			map: HashMap::new(),
		}
	}

	pub fn is_empty(&self) -> bool {
		self.buffer.is_empty()
	}

	pub fn len(&self) -> usize {
		self.buffer.len()
	}

	pub fn back(&mut self) -> Option<&Segment<T>> {
		self.buffer.back()
	}

	pub fn idx_get(&mut self, idx: usize) -> Option<&Segment<T>> {
		self.buffer.get(idx)
	}

	pub fn last_idx(&self) -> usize {
		self.buffer.len() - 1
	}

}