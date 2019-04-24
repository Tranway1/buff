use rocksdb::DBVector;
use crate::dictionary::Dictionary;
use crate::dictionary;
use crate::methods;

use dictionary::DictionaryId;
use methods::Methods;

/* 
 * Overview:
 * This is the API for a constructing a unit of time data. 
 * The goal is to construct two structures for immutable and 
 * mutable memory units. Furthermore, allow the mutable unit
 * to transfer to an immutable one
 *
 * Design Choice:
 * This was implemented with two structs instead of two traits.
 * This was done as
 * 1. There will likely only ever need to be a single type of segment
 * 2. Simpler implementation as fewer generics to keep track off
 * 3. Simpler memory model as heap allocation or generic restriction
 *    are not required
 *
 * Current Implementations:
 * There are two basic implementations created to be further fleshed
 * out and provide all neccessary functionality required by these
 * memory units.
 * Note: Mutable Segments may be unnecessary if futures implementation
 *       is chosen instead.
 * Note: Mutable Segments are entirely volatile and currently are not
 *       planned to be saved to disk. Must become immutable first.
 */


/***************************************************************
 **********************Dictionary_Tracker***********************
 ***************************************************************/

type SegmentId = u32;
const SEGMENTSIZE: usize = 500;

#[derive(Clone)]
pub struct Segment<T> {
	dictionary_id: Option<DictionaryId>,
	method: Option<Methods>,
	metadata: u32,
	timestamp: u32,
	segment_id: SegmentId,
	data: [T; SEGMENTSIZE],
}

impl<T> Segment<T> {

	/* Construction Methods */
	pub fn new() -> Segment<T> {
		unimplemented!()
	}

	pub fn convert_from_bytes(bytes: DBVector) -> Segment<T> {
		unimplemented!()
	}

	pub fn convert_to_bytes(&self) -> Vec<u8> {
		unimplemented!()
	}
	

	/* Compressing and Decompression Methods */
	pub fn compress(&mut self, method: Methods, dict: Option<Dictionary<T>>) {
		unimplemented!()
	}

	pub fn compress_and_retain(&self, method: Methods, dict: Option<Dictionary<T>>) -> Segment<T> {
		unimplemented!()
	}

	pub fn decompress(&mut self) {
		unimplemented!()
	}

	pub fn decompress_and_retain(&self) -> Segment<T> {
		unimplemented!()
	}

	pub fn recompress(&self, method: Methods, new_dict: Option<Dictionary<T>>) {
		unimplemented!()
	}
}


#[derive(Clone)]
pub struct IncompleteSegment<T> {
	metadata: u32,
	data: Vec<T>
}


impl<T> IncompleteSegment<T> {

	pub fn new() -> IncompleteSegment<T> {
		unimplemented!()
	}

	pub fn make_segment(&self) -> Segment<T> {
		unimplemented!()
	}
}