use std::time::{Duration,SystemTime};
use std::fmt::Debug;

use serde::{Serialize,Deserialize};

use rocksdb::{DB, ColumnFamily, Options,DBVector};

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

	/* Takes the bytes representing a segment and its key and writes
	 * it to the file controlled by the FileManager
	 * Ok(()): Indicating the operation succesfully completed
	 * Err(e): Indiacting an error prevented the bytes from being written
	 */
	fn write_segment(&self, key: T, value: T) -> Result<(),Error>;

	/* Takes the bytes representing its key and gets
	 * the bytes representing the segment from the file controlled 
	 * by the FileManager. Will return 
	 * Ok(Some(x): where x is a DBVvector holding the bytes
	 * Ok(None): Indicating that the segment was not present in the file
	 * Err(e): Indicating some failure
	 */
	fn get_segment(&self, key: T) -> Result<Option<U>,Error>;

	/* Same as write_segment, but for the bytes of a dictionary */
	fn write_dictionary(&self, key: T, value: T) -> Result<(),Error>;

	/* Same as get_segment, but for the bytes of a dictionary */
	fn get_dictionary(&self, key: T) -> Result<Option<U>,Error>;
}

/* Error enum used by the FileManager to wrap rocksdb errors */
#[derive(Debug)]
pub enum Error {
	DbError(rocksdb::Error),
	ColumnError(&'static str),
}

/* Used to implicitly convert rocksdb errors when necessary to a 
 * more general type
 */
impl From<rocksdb::Error> for Error {
	fn from(error: rocksdb::Error) -> Self {
		Error::DbError(error)
	}
}

/* A struct designed to take a database reference and create 
 * an object to facilitate all write and get operations
 * Due to lifetime constraints the database must first be created
 * and then this object can be freely used in any scope contained
 * within the scope that created the database (Compiler warning if not adhered to)
 */
pub struct RocksFM<'a> {
	db: &'a DB,
	seg_cf: ColumnFamily<'a>,
	dic_cf: ColumnFamily<'a>, 
}

const SEGMENTCF: &str = "segment";
const DICTIONARYCF: &str = "dictionary";

impl<'a> RocksFM<'a> {

	/* Helper function to construct the RocksDB FileManager */
	fn new(db: &'a DB) -> Result<RocksFM<'a>,Error> {
		let get_fc_handle = |db: &'a DB, cf_name: &str| {
			match db.cf_handle(cf_name) {
				Some(cf) => Ok(cf),
				None => Err(Error::ColumnError("Failed to get column family handle.")),
			}
		};

		let seg_cf: ColumnFamily<'a> = get_fc_handle(db, SEGMENTCF)?;

		let dic_cf: ColumnFamily<'a> = get_fc_handle(db, DICTIONARYCF)?;

		Ok(RocksFM {
				db: db,
				seg_cf: seg_cf,
				dic_cf: dic_cf,
			})
	}
	
	/* Helper function to construct a rocksdb object */
	pub fn create_db(path: &str, db_opts: &mut Options) -> Result<DB,rocksdb::Error> {
		let column_families = vec![SEGMENTCF,DICTIONARYCF];
		db_opts.create_missing_column_families(true);
		db_opts.create_if_missing(true);
		
		DB::open_cf(&db_opts, path, column_families)
	}
}


impl<'a,T> FileManager<T,DBVector> for RocksFM<'a> 
	where T: AsRef<[u8]>,
{

	fn write_segment(&self, key: T, value: T) -> Result<(),Error> {
		match self.db.put_cf(self.seg_cf,key,value) {
			Err(e) => Err(Error::DbError(e)),
			Ok(_)  => Ok(()),
		}
	}

	fn get_segment(&self, key: T) -> Result<Option<DBVector>,Error> {
		match self.db.get_cf(self.seg_cf, key) {
			Ok(Some(x)) => Ok(Some(x)),
			Ok(None)    => Ok(None),
			Err(e)      => Err(Error::DbError(e)),
		}

	}

	fn write_dictionary(&self, key: T, value: T) -> Result<(),Error> {
		match self.db.put_cf(self.dic_cf,key,value) {
			Err(e) => Err(Error::DbError(e)),
			Ok(_)  => Ok(()),
		}
	}

	fn get_dictionary(&self, key: T) -> Result<Option<DBVector>,Error> {
		match self.db.get_cf(self.dic_cf, key) {
			Ok(Some(x)) => Ok(Some(x)),
			Ok(None)    => Ok(None),
			Err(e)      => Err(Error::DbError(e)),
		}
	}

}

/***************************************************************
 ****************************Testing****************************
 ***************************************************************/

const FILEPATH: &str = "../rocksdb";

fn read_write_validate<'a,T:Send>(rfm: &RocksFM, seg: &Segment<T>) -> DBVector
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

	match rfm.write_segment(&compressed_key,&compressed_seg) {
		Ok(_) => (),
		Err(e) => panic!("Failed to put segment into file: {:?}", e),
	};

	match rfm.get_segment(&compressed_key) {
		Ok(Some(x)) => x,
		Ok(None)    => panic!("Failed to find segment in file."),
		Err(e)      => panic!("Experienced error trying to get segment: {:?}", e),
	}
}

#[test]
fn read_write_test() {
	let mut db_opts = Options::default();
	let rocks_db = match RocksFM::create_db(FILEPATH, &mut db_opts) {
		Ok(x) => x,
		Err(e) => panic!("Failed to create database: {:?}", e),
	};

	let rfm = match RocksFM::new(&rocks_db) {
		Ok(x) => x,
		Err(e) => panic!("Failed to construct file manager: {:?}", e),
	};

	let sizes: Vec<usize> = vec![10,100,1024,5000];
	let segs: Vec<Segment<f32>> = sizes.into_iter().map(move |x| {
		Segment::new(None, SystemTime::now(), x as u64, 
			random_f32signal(x), vec![], None,
		)}).collect();

	for seg in segs {
		let inflated_seg_bytes = read_write_validate(&rfm, &seg).as_ref().iter()
        .cloned()
        .decode(&mut BZip2Decoder::new())
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

		match Segment::convert_from_bytes(&inflated_seg_bytes) {
			Ok(x) => assert_eq!(seg, x),
			Err(e) => panic!("Failed to convert bytes to segment {:?}", e),
		}
	}

	DB::destroy(&db_opts, FILEPATH);
}