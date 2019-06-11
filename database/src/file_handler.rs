use std::time::SystemTime;
use std::fmt::Debug;

use serde::{Serialize,Deserialize};

use rocksdb::{Options,DBVector};

use segment::{Segment,random_f32signal};
use crate::segment;

extern crate compression;
use compression::prelude::*;

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

pub trait FileManager<T,U> 
	where T: AsRef<[u8]>,
		  U: AsRef<[u8]>,
{

	/* Takes the bytes representing a segment or dictionary and
	 * its key and writes it to the file controlled by the FileManager
	 * Will return =>
	 *   Ok(()): Indicating the operation succesfully completed
	 *   Err(e): Indiacting an error prevented the bytes from being written
	 */
	fn fm_write(&self, key: T, value: T) -> Result<(),Error>;

	/* Takes the bytes representing its key and gets
	 * the bytes representing the segment or dictionary from
	 * the file controlled by the FileManager. 
	 * Will return =>
	 *   Ok(Some(x): where x is a DBVvector holding the bytes
	 *   Ok(None): Indicating that the segment was not present in the file
	 *   Err(e): Indicating some failure
	 */
	fn fm_get(&self, key: T) -> Result<Option<U>,Error>;

	/* add batch write support */
}

/* Error enum used by the FileManager to wrap rocksdb errors */
#[derive(Debug)]
pub enum Error {
	DbError(rocksdb::Error),
	ColumnError(&'static str),
}


impl<T,U> FileManager<T,U> for rocksdb::DB
	where T: AsRef<[u8]>,
	      U: AsRef<[u8]> + From<DBVector>,
{
	#[inline]
	fn fm_write(&self, key: T, value: T) -> Result<(),Error> {
		match self.put(key,value) {
			Err(e) => Err(Error::DbError(e)),
			Ok(_)  => Ok(()),
		}
	}

	#[inline]
	fn fm_get(&self, key: T) -> Result<Option<U>,Error> {
		match self.get(key) {
			Ok(Some(x)) => Ok(Some(x.into())),
			Ok(None)    => Ok(None),
			Err(e)      => Err(Error::DbError(e)),
		}

	}
}

/***************************************************************
 ****************************Testing****************************
 ***************************************************************/

const FILEPATH: &str = "../rocksdb";

fn read_write_validate<'a,T:Send>(fm: &FileManager<Vec<u8>,DBVector>, seg: &Segment<T>) -> DBVector
	where T: Clone + Serialize + Deserialize<'a> + Debug + PartialEq
{
	let seg_key = seg.get_key();

	let seg_bytes = match seg.convert_to_bytes() {
		Ok(x) => x,
		Err(e) => panic!("Failed to serialize segment: {:?}", e),
	};

	let seg_key_bytes = match seg_key.convert_to_bytes() {
		Ok(x) => x,
		Err(e) => panic!("Failed to serialize segment key: {:?}", e),
	};

	let compressed_seg = seg_bytes.iter().cloned()
								  .encode(&mut BZip2Encoder::new(9), Action::Finish)
								  .collect::<Result<Vec<_>, _>>()
								  .unwrap();

	let compressed_key = seg_key_bytes.iter().cloned()
								  	  .encode(&mut BZip2Encoder::new(9), Action::Finish)
								  	  .collect::<Result<Vec<_>, _>>()
								  	  .unwrap();	

	match fm.fm_write(compressed_key.clone(),compressed_seg) {
		Ok(_) => (),
		Err(e) => panic!("Failed to put segment into file: {:?}", e),
	};

	match fm.fm_get(compressed_key) {
		Ok(Some(x)) => x,
		Ok(None)    => panic!("Failed to find segment in file."),
		Err(e)      => panic!("Experienced error trying to get segment: {:?}", e),
	}
}

#[test]
fn read_write_test() {
	let mut db_opts = Options::default();
	db_opts.create_if_missing(true);
	let fm = match rocksdb::DB::open(&db_opts, FILEPATH) {
		Ok(x) => x,
		Err(e) => panic!("Failed to create database: {:?}", e),
	};

	let sizes: Vec<usize> = vec![10,100,1024,5000];
	let segs: Vec<Segment<f32>> = sizes.into_iter().map(move |x| {
		Segment::new(None, SystemTime::now(), x as u64, 
			random_f32signal(x), None, None,
		)}).collect();

	for seg in segs {
		let inflated_seg_bytes = read_write_validate(&fm, &seg).as_ref().iter()
        .cloned()
        .decode(&mut BZip2Decoder::new())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

		match Segment::convert_from_bytes(&inflated_seg_bytes) {
			Ok(x) => assert_eq!(seg, x),
			Err(e) => panic!("Failed to convert bytes to segment {:?}", e),
		}
	}

	let _ = rocksdb::DB::destroy(&db_opts, FILEPATH);
}