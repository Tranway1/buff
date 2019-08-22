use rocksdb::DBVector;
use crate::buffer_pool::BufErr;
use serde::Serialize;
use serde::de::DeserializeOwned;
use crate::segment::Segment;
use std::sync::{Arc,Mutex};
use crate::buffer_pool::{SegmentBuffer,ClockBuffer};
use crate::file_handler::{FileManager};
use crate::file_handler;


pub struct CompressionDemon<T,U,F> 
	where T: Copy + Send + Serialize + DeserializeOwned,
	      U: FileManager<Vec<u8>,DBVector> + Sync + Send,
		  F: Fn(&mut Segment<T>)
{
	seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
	comp_seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
	file_manager: Option<U>,
	comp_threshold: f32,
	uncomp_threshold: f32,
	compress_func: F,
}

impl<T,U,F> CompressionDemon<T,U,F> 
	where T: Copy + Send + Serialize + DeserializeOwned,
		  U: FileManager<Vec<u8>,DBVector> + Sync + Send,
		  F: Fn(&mut Segment<T>)
{
	pub fn new(seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
			   comp_seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
			   file_manager: Option<U>,
			   comp_threshold: f32, uncomp_threshold: f32, compress_func: F)
			   -> CompressionDemon<T,U,F>
	{
		CompressionDemon {
			seg_buf: seg_buf,
			comp_seg_buf: comp_seg_buf,
			file_manager: file_manager,
			comp_threshold: comp_threshold,
			uncomp_threshold: uncomp_threshold,
			compress_func: compress_func,
		}
	}

	fn get_seg_from_uncomp_buf(&self) -> Result<Segment<T>,BufErr>
	{
		match self.seg_buf.lock() {
			Ok(mut buf) => {
				if buf.exceed_threshold(self.uncomp_threshold) {
					buf.remove_segment()
				} else {
					Err(BufErr::UnderThresh)
				}
			}
			Err(_) => Err(BufErr::RemoveFailure)
		}

	}

	fn put_seg_in_comp_buf(&self, seg: Segment<T>) -> Result<(),BufErr>
	{
		match self.comp_seg_buf.lock() {
			Ok(mut buf) => buf.put(seg),
			Err(_) => Err(BufErr::CantGrabMutex),
		}
	}

	pub fn run(&self)
	{
		loop {
			if let Ok(mut seg) = self.get_seg_from_uncomp_buf() {
				match &self.file_manager {
					Some(fm) => {
						let key_bytes = match seg.get_key().convert_to_bytes() {
							Ok(bytes) => bytes,
							Err(_) => continue, /* silence failure to byte convert */ 
						};
						let seg_bytes = match seg.convert_to_bytes() {
							Ok(bytes) => bytes,
							Err(_) => continue, /* silence failure to byte convert */
						};
						match fm.fm_write(key_bytes, seg_bytes) {
							Ok(()) => continue,
							Err(_) => continue, /* currently silence error from fialed write */
						}
					}
					None => {
						(self.compress_func)(&mut seg);
						match self.put_seg_in_comp_buf(seg) {
							Ok(()) => continue,
							Err(_) => continue, /* Silence the failure to put the segment in */
						}
					}
				}
			}
		}
	}

}