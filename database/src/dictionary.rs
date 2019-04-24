use std::slice::Chunks;
use std::collections::HashMap;
use std::sync::RwLock;
use rocksdb::DBVector;


use crate::methods::{MethodHolder, Methods, MethodImplementError};

/* 
 * Overview:
 * This is the API for the dictionary iplementation. This is broken
 * into two parts.
 * 1. The dicitionary tracker which keeps track of a mapping
 * 	  from an Id to the dicitionary. Currently, This will
 *    likely interact with a dicitionary buffer to extract it.
 *    This enables a lightweight method for compressed signals to 
 *    keep track of the dictionary used in the process if one was used
 * 2. The dictionary which unlike the tracker is not a trait, but
 *    a structure. The reason for making this a structure was that 
 *    	(1) I do not forsee much change in the dictionary or a 
 *			need for mutiple implementations
 *      (2) Allowing multiple dicitionaries complicates memory
 *          management and so on as they aren't uniform. So 
 *          heap allocation, Box, would be required to handle this.
 *    The dictionary should provide all necessary features that any 
 *    future compression method or exisiting compression method will
 *    require.
 *
 * Design Choice:
 * Mainly discussed above, but the tracker is a trait as it doesn't
 * increase code complexity much and allows for easier revision and
 * implementation of new designs.
 *
 * Current Implementations:
 * I have made one basic stuctures that theroertically could 
 * implement the tracker traits as well the base for the dictionary 
 * structure
 */

/***************************************************************
 **********************Dictionary_Tracker***********************
 ***************************************************************/


/* This is not a primary interface, just a tool to help obtain the 
 * dictionary regardless of whether in-memory or not. Will interact
 * with a specialized dictionary buffer. 
 */
pub trait DictionaryTracker<T> {
	fn get(&self, dictionary_id: DictionaryId) -> Option<Dictionary<T>>;

	fn add(&mut self, dictionary: Dictionary<T>, write_through: bool) -> Result<DictionaryId, DictTrackerError>;

	fn remove(&mut self, dictionary_id: DictionaryId) -> Result<(), DictTrackerError>;

	fn delete(&mut self, dictionary_id: DictionaryId) -> Result<(), DictTrackerError>;
}

pub struct BasicDictionaryTracker<T> {
	tracker: RwLock<HashMap<DictionaryId,Dictionary<T>>>
}

impl<T> BasicDictionaryTracker<T> {
	pub fn new() -> BasicDictionaryTracker<T> {
		unimplemented!()
	}
}

impl<T> DictionaryTracker<T> for BasicDictionaryTracker<T> {

	fn get(&self, dictionary_id: DictionaryId) -> Option<Dictionary<T>> {
		unimplemented!()
	}

	fn add(&mut self, dictionary: Dictionary<T>, write_through: bool) -> Result<DictionaryId, DictTrackerError> {
		unimplemented!()
	}

	fn remove(&mut self, dictionary_id: DictionaryId) -> Result<(), DictTrackerError> {
		unimplemented!()
	}

	fn delete(&mut self, dictionary_id: DictionaryId) -> Result<(), DictTrackerError> {
		unimplemented!()
	}
}


/***************************************************************
 **************************Dictionary***************************
 ***************************************************************/


pub type DictionaryId = u32; /* Type alias for dictionary id */


pub struct Dictionary<T> {
	meta_data: u32, /* temp holder for metadata */
	id: DictionaryId,
	num_items: u32,
	entry_size: u32, /* Indicates the size of each entry in items */ 
	methods: RwLock<MethodHolder>, /* Described in methods.rs */
	items: Vec<T>, /* Flattened 2-D array to hold each entry */ 
}

impl<T> Dictionary<T> {

	pub fn new() -> Dictionary<T> {
		unimplemented!()
	}

	pub fn convert_from_bytes(bytes: DBVector) -> Dictionary<T> {
		unimplemented!()
	}

	pub fn convert_to_bytes(&self) -> Vec<u8> {
		unimplemented!()
	}

	pub fn get_item_iter(&self) -> Chunks<T> {
		unimplemented!()
	}

	pub fn get_item(&self, idx: u32) -> &[T] {
		unimplemented!()
	}

	pub fn is_implemented(&self, method_type: Methods) -> bool {
		unimplemented!()
	}

	pub fn implement_method(&self, method_type: Methods) -> Result<(),MethodImplementError> {
		unimplemented!()
	}
}


/***************************************************************
 *************************Error/Helpers**************************
 ***************************************************************/


pub struct DictTrackerError {
	stuff: u32,
}