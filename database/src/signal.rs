/* DEPRECATED DO NOT USE OR IMPLEMENT, USE future_signal INSTEAD */

use std::net::{TcpStream};
use std::collections;

/* 
 * Overview:
 * This is the API for a signal iplementation using
 * only standard rust. The goal is to construct a tracker
 * who can keep track of the signals registered to provide
 * a method to listen to them. The goal of a signal was to construct
 * a black box that could record once regsitered.
 *
 * Design Choice:
 * The signal tracker and signal are both implemented as 
 * traits as I wanted to make it easy to create different 
 * varaiants of either. Since this created little additional
 * overhead and allows for easier revision and new implementation
 *
 * Current Implementations:
 * I have made two basic stuctures that theroertically could 
 * implement these traits as well as provide the additional 
 * methods I would expect necessary
 */


/***************************************************************
 ************************Signal_Tracker*************************
 ***************************************************************/

/* Trait to describe minimal behavior of a signal tracker.
 * Main interface for working with signals.
 */
pub trait SignalTracker<T: Signal> {

	/* Takes a hostname and creates a tracker entry for that hostname
	 * Returns an Id to track signal for later use, wrapped in a Result
	 * Will return Err if signal tracker fails to make a proper connection
	 * and propogate the error up to be handled by the caller.SignalId
	 * Otherwise will return an Ok with the associated SignalId.
	 */
	fn register(signal_hostname: &str) -> Result<SignalId, TcpStream>;


	/* If the signal is regsitered it will return: Some(SignalId)
	 * Otherwise it will return: None
	 */
	fn get_signal_id(signal_hostname: &str) -> Option<SignalId>;


	/* If the SignalId is asscoiated with some regsitered signal: 
	 * 		it will return: Some(Signal)
	 * Otherwise: 
	 *		it will return: None
	 */
	fn get_signal(signal_id: SignalId) -> Option<T>;


	/* Takes a signal id and will remove it from the tracker 
	 * If it succeds it will return Ok(())
	 * Otherwise:
	 * 	If it fails and was absent it will return Err(Absent)
	 *  If it fails and was present it will return Err(Present)
	 * Note: may change return value, unsure of how much info I will need
	 */
	fn unregister(signal_id: SignalId) -> Result<(),SignalPresent>;

	/* Will remove the signal associated with the id from the 
	 * listener ensuring that data from this source will be 
	 * temporarily ignored
	 */
	fn pause(signal_id: SignalId) -> Result<(),SignalPresent>;

	/* Will currently block until the read IO completes
	 * currently returning the bytes as a vec, may change
	 * to an array to avoid heap allocation.
	 */ 
	fn record(signal_id: SignalId) -> Vec<u8>;
}

/* A Simple Implementation of Signal Tracker */

/* A serial signal tracker for n signal 1 thread model*/
pub struct SerialSignalTracker<T: Signal> {
	tracker:  collections::HashMap<String,SignalId>,
	id_name_map: collections::HashMap<SignalId,String>,
	listener: collections::HashMap<SignalId,T>
}


impl<T: Signal> SerialSignalTracker<T> {
	pub fn new() -> SerialSignalTracker<T> {
		unimplemented!()
	}
}


impl<T: Signal> SignalTracker<T> for SerialSignalTracker<T> {

	fn register(signal_name: &str) -> Result<SignalId, TcpStream> {
		unimplemented!()
	}

	fn get_signal_id(signal_name: &str) -> Option<SignalId> {
		unimplemented!()
	}

	fn get_signal(signal_id: SignalId) -> Option<T> {
		unimplemented!()
	}

	fn unregister(signal_id: SignalId) -> Result<(),SignalPresent> {
		unimplemented!()
	}

	fn pause(signal_id: SignalId) -> Result<(),SignalPresent> {
		unimplemented!()
	}

	fn record(signal_id: SignalId) -> Vec<u8> {
		unimplemented!()
	}
}



/***************************************************************
 ****************************Signal*****************************
 ***************************************************************/

pub type SignalId = u64; /* Type alias to describe signals */
const RECBUFFSIZE: usize = 512; /* Currently fixing size of record buffer */


/* The simple Signal abstraction must only support record */
pub trait Signal {
	fn record(&self) -> [u8; RECBUFFSIZE];
}

/* A simple signal implementation */
pub struct TcpSignal {
	signal_id: SignalId,
	stream: TcpStream, /* The connected stream of data */
	interval: f64 /* Additional information to indicate how frequently to check */
}

impl TcpSignal {
	
	pub fn new(signal_name: &str) -> Result<TcpSignal,TcpStream> {
		unimplemented!()
	}

	/* Meta-data functions etc */
}


/* Required methods for signal that will be used by signal tracker */
impl Signal for TcpSignal {

	/* Function called by tracker to record the signal */
	fn record(&self) -> [u8; RECBUFFSIZE] {
		unimplemented!()
	}

}


/***************************************************************
 *************************Hellper/Error*************************
 ***************************************************************/

pub enum SignalPresent {
	Present,
	Absent
}