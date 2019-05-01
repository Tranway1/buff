use rocksdb::{DB, ColumnFamilyDescriptor, Options};

use crate::segment;
use crate::dictionary;

use segment::Segment;
use dictionary::Dictionary;

const FILEPATH: &str = "../timedb";
const SEGMENT: u32 = 0;
const DICTIONARY: u32 = 1;

/* 
 * Overview:
 * This is the API for a constructing a file manager. 
 * This will be the main unit the Buffer will interact with.
 *
 * Design Choice:
 * A trait was chosen so as to enable changing file handlers with
 * relative ease, enabling more freedom in the future as well
 * as an easier time comparing performances with different managers.
 *
 * Current Implementations:
 * The current implementation uses RocksDB to implement the file manager.
 */

/* Think about truncating results when memory is gotten too large */

pub trait FileManager<T> {

	fn write_segment(key: &[T], value: &[T]);

	fn get_segment(key: &[T]) -> Option<Vec<T>>;

	fn write_dictionary(key: &[T], value: &[T]);

	fn get_dictionary(key: &[T]) -> Option<Vec<T>>;
}



pub struct RDBFileManager {
	file: DB,
} 

impl RDBFileManager {
	pub fn new() -> RDBFileManager {
		unimplemented!()
	}
}

impl FileManager<u8> for RDBFileManager{

	fn write_segment(key: &[u8], value: &[u8]) {
		unimplemented!()
	}

	fn get_segment(key: &[u8]) -> Option<Vec<u8>> {
		unimplemented!()
	}

	fn write_dictionary(key: &[u8], value: &[u8]) {
		unimplemented!()
	}

	fn get_dictionary(key: &[u8]) -> Option<Vec<u8>> {
		unimplemented!()
	}
}