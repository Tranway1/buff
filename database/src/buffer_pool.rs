use std::cell::Cell;
use std::collections::vec_deque::Drain;
use std::collections::VecDeque;

use crate::segment;

use segment::{Segment, IncompleteSegment};

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

pub trait SegmentBuffer<'a,T: Clone> {

	/* If the index is valid it will return: Ok(ref T)
	 * Otherwise, it will return: Err(()) to indicate 
	 * that the index was invalid
	 */
	fn get(&self, idx: usize) -> Result<&T, ()>;

	/* If the buffer succeeds it will return: Ok(index)
	 * Otherwise, it will return: Err(BufErr) to indicate 
	 * that some failure occured and the item couldn't be
	 * buffered.
	 */
	fn put(&self, item: T) -> Result<usize, BufErr>;


	/* Will return a Drain iterator that gains ownership over
	 * the items in the buffer, but will leave the allocation
	 * of the buffer untouched
	 */
	fn drain(&self, start: u32, end: u32) -> Drain<'a, T>;

	/* Will copy the buffer and collect it into a vector */
	fn copy(&self, start: u32, end: u32) -> Vec<T>;

	/* Will evict some number of items from the buffer freeing
	   There location */
	fn evict(&self, num_evict: u32);

	/* Will lock the buffer and write everything to disk */
	fn persist(&self);

	/* Will empty the buffer */
	fn flush(&self);
}

pub trait MutableBuffer<T> {
	/* If the index is valid it will return: Ok(mut ref T)
	 * Otherwise, it will return: Err(()) to indicate 
	 * that the index was invalid
	 */
	fn get_mut(&mut self, idx: usize) -> Result<&mut T, ()>;
}


/* Designed to carry error information 
 * May be unnecessary
 */
pub struct BufErr {
	info: u32,
}

/***************************************************************
 ***********************Immutable_Buffer************************
 ***************************************************************/

pub struct VDBufferPool<T> {
	buffer: VecDeque<Segment<T>>,
}

impl<'a,T: Clone> SegmentBuffer<'a,Segment<T>> for VDBufferPool<T> {
	fn get(&self, idx: usize) -> Result<&Segment<T>, ()> {
		unimplemented!()
	}

	fn put(&self, item: Segment<T>) -> Result<usize, BufErr> {
		unimplemented!()
	}

	fn drain(&self, start: u32, end: u32) -> Drain<'a, Segment<T>> {
		unimplemented!()
	}

	fn copy(&self, start: u32, end: u32) -> Vec<Segment<T>> {
		unimplemented!()
	}

	fn evict(&self, num_evict: u32) {
		unimplemented!()
	}

	fn persist(&self) {
		unimplemented!()
	}

	fn flush(&self) {
		unimplemented!()
	}
}


impl<T> VDBufferPool<T> {
	pub fn new() -> VDBufferPool<T> {
		VDBufferPool {
			buffer : VecDeque::with_capacity(BUFFERSIZE)
		}
	}
}


/***************************************************************
 ************************Mutable_Buffer*************************
 ***************************************************************/

pub struct VDMutBufferPool<T> {
	buffer: VecDeque<IncompleteSegment<T>>,
}

impl<T> VDMutBufferPool<T> {
	pub fn new() -> VDMutBufferPool<T> {
		unimplemented!()
	}
}

impl<'a,T: Clone> SegmentBuffer<'a,IncompleteSegment<T>> for VDBufferPool<T> {
	fn get(&self, idx: usize) -> Result<&IncompleteSegment<T>, ()> {
		unimplemented!()
	}

	fn put(&self, item: IncompleteSegment<T>) -> Result<usize, BufErr> {
		unimplemented!()
	}

	fn drain(&self, start: u32, end: u32) -> Drain<'a, IncompleteSegment<T>> {
		unimplemented!()
	}

	fn copy(&self, start: u32, end: u32) -> Vec<IncompleteSegment<T>> {
		unimplemented!()
	}

	fn evict(&self, num_evict: u32) {
		unimplemented!()
	}

	fn persist(&self) {
		unimplemented!()
	}

	fn flush(&self) {
		unimplemented!()
	}
}

impl<T> MutableBuffer<IncompleteSegment<T>> for VDMutBufferPool<T>
{
	fn get_mut(&mut self, idx: usize) -> Result<&mut IncompleteSegment<T>, ()> {
		unimplemented!()
	}
}