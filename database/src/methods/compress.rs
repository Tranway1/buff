use crate::kernel::Kernel;
use crate::segment::Segment;
use ndarray::Array2;

pub trait CompressionMethod<T> {


	fn get_segments(&self);

	fn get_batch(&self) -> usize;

	fn run_compress(&self, mut segs: Vec<Segment<T>>);

	fn run_decompress(&self, mut segs: Vec<Segment<T>>);
}
