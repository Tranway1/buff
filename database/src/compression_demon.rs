use rocksdb::DBVector;
use crate::buffer_pool::BufErr;
use serde::Serialize;
use serde::de::DeserializeOwned;
use crate::segment::Segment;
use std::sync::{Arc,Mutex};
use crate::buffer_pool::{SegmentBuffer,ClockBuffer};
use crate::file_handler::{FileManager};
use crate::file_handler;
use crate::methods::compress::CompressionMethod;
use std::any::Any;
use crate::buffer_pool::BufErr::BufEmpty;

pub struct CompressionDemon<T,U,F> 
	where T: Copy + Send + Serialize + DeserializeOwned,
	      U: FileManager<Vec<u8>,DBVector> + Sync + Send,
		  F: CompressionMethod<T>
{
	seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
	comp_seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
	file_manager: Option<U>,
	comp_threshold: f32,
	uncomp_threshold: f32,
	processed: usize,
	compress_method: F,
}

impl<T,U,F> CompressionDemon<T,U,F> 
	where T: Copy + Send + Serialize + DeserializeOwned,
		  U: FileManager<Vec<u8>,DBVector> + Sync + Send,
		  F: CompressionMethod<T>
{
	pub fn new(seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
			   comp_seg_buf: Arc<Mutex<SegmentBuffer<T> + Send + Sync>>,
			   file_manager: Option<U>,
			   comp_threshold: f32, uncomp_threshold: f32, compress_method: F)
			   -> CompressionDemon<T,U,F>
	{
		CompressionDemon {
			seg_buf: seg_buf,
			comp_seg_buf: comp_seg_buf,
			file_manager: file_manager,
			comp_threshold: comp_threshold,
			uncomp_threshold: uncomp_threshold,
			processed: 0,
			compress_method: compress_method,
		}
	}

	fn get_seg_from_uncomp_buf(&self) -> Result<Vec<Segment<T>>,BufErr>
	{
		match self.seg_buf.lock() {
			Ok(mut buf) => {
				//println!("Lock aquired");
				if buf.exceed_threshold(self.uncomp_threshold) && buf.exceed_batch(self.compress_method.get_batch()) {
					let batch_size = self.compress_method.get_batch();
					//println!("Get segment for compression batch size:{}",batch_size);
					let mut segs = Vec::with_capacity(batch_size);
					loop {
						match buf.remove_segment(){
							Ok(seg) => {
								segs.push(seg);
								if segs.len() == batch_size{
									break;
								}
							},
							Err(_) => continue,
						}
					}
					Ok(segs)
				} else {
					Err(BufErr::UnderThresh)
				}
			}
			Err(_) => Err(BufErr::RemoveFailure)
		}

	}

	fn is_done_buf(&self) -> bool{
		self.seg_buf.lock().unwrap().is_done()
	}

	fn put_seg_in_comp_buf(&self, seg: Segment<T>) -> Result<(),BufErr>
	{
		match self.comp_seg_buf.lock() {
			Ok(mut buf) => buf.put(seg),
			Err(_) => Err(BufErr::CantGrabMutex),
		}
	}

	fn put_segs_in_comp_buf(&self, segs: Vec<Segment<T>>) -> Result<(),BufErr>
	{
		match self.comp_seg_buf.lock() {
			Ok(mut buf) => {
				for seg in segs{
					match buf.put(seg) {
						Ok(_) => continue,
						Err(_) => return Err(BufErr::FailPut),
					}
				}
				return Ok(());
			}
			Err(_) => Err(BufErr::CantGrabMutex),
		}
	}

	pub fn get_processed(&self) -> usize{
		self.processed
	}

	pub fn run(&mut self)
	{
		loop {
			match self.get_seg_from_uncomp_buf() {
				Ok(mut segs) => {
					self.processed = self.processed + segs.len();
					println!("segment processed {}", self.processed);
					match &self.file_manager {
						Some(fm) => {
							for mut seg in segs{
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
						}
						None => {
							self.compress_method.run_compress(&mut segs);
							match self.put_segs_in_comp_buf(segs) {
								Ok(()) => continue,
								Err(_) => continue, /* Silence the failure to put the segment in */
							}
						}
					}
				},
				Err(BufErr::BufEmpty) => {
					println!("Buffer is empty. Compression process exit.");
					break
				},
				Err(e)=>{
					//println!("No segment is fetched, try again.");
					//println!("Application error: {:?}",e);
					continue
				},
			}
		}
	}

}