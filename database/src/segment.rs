//use std::convert::TryFrom;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;
use crate::buffer_pool::SegmentBuffer;
use std::time::SystemTime;
use std::ops::Sub;
use num::FromPrimitive;
use std::ops::Div;
use std::ops::Add;
use crate::bincode;
use serde::{Serialize, Deserialize};

use rustfft::{FFTplanner,FFTnum};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use std::time::{Duration};
use crate::future_signal::SignalId;
use num::Num;
use std::cmp::Ord;

/* Currently plan to move methods into this file */
use crate::methods;
use methods::Methods;
use Methods::{Fourier};

extern crate rand;
use rand::{SeedableRng};
use rand::rngs::{StdRng};
use rand::distributions::{Normal, Distribution};
use crate::methods::compress::CompressionMethod;


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
 ***********************Segment Structure***********************
 ***************************************************************/

/* Time stamps currently represented by Duration, will 
 * switch to a DateTime object instead.
 * Segment currently uniquely identified by SignalId and timestamp
 * since Signals are 1-n with threads so they shouldn't be able to produce
 * more than one in an instant.
 * Currently, There is an implicit linked list between segments from the 
 * same Signal. This is achieved by having each thread produce a unique
 * key defined by the SignalId it was produced from and its timestamp at creation.
 * To get the segment that was recorded before the current segment was, 
 * subtract the prev_seg_offset from the current segments timestamp and 
 * use those value to produce a key.
 */
#[derive(Clone,Serialize,Deserialize,Debug,PartialEq)]
pub struct Segment<T> {
	method: Option<Methods>,
	timestamp: SystemTime,
	signal: SignalId,
	data: Vec<T>,
	time_lapse: Option<Vec<Duration>>,
	prev_seg_offset: Option<Duration>,
	//next_seg_offset: Option<Duration>,
}

impl<T> Segment<T> {
	pub fn new(method: Option<Methods>, timestamp: SystemTime, signal: SignalId,
	    data: Vec<T>, time_lapse: Option<Vec<Duration>>, next_seg_offset: Option<Duration>) -> Segment<T> {
		
		Segment {
			method: method,
			timestamp: timestamp,
			signal: signal,
			data: data,
			time_lapse: time_lapse,
			prev_seg_offset: next_seg_offset
		}
	}

	pub fn get_key(&self) -> SegmentKey {
		SegmentKey::new(self.timestamp,self.signal)
	}

	pub fn get_prev_key(&self) -> Option<SegmentKey> {
		match self.prev_seg_offset {
			Some(time) => Some(SegmentKey::new(self.timestamp-time,self.signal)),
			None       => None, 
		}
	}

	pub fn get_signal(&self) -> SignalId {
		self.signal
	}

	pub fn get_data(&self) ->  &Vec<T>
		where T: Clone
	{
		&self.data
	}
}

impl<'a,T> Segment<T> 
	where T: Serialize + Deserialize<'a>
{
	pub fn convert_to_bytes(&self) -> Result<Vec<u8>,()> {
		match bincode::serialize(self) {
			Ok(seg) => Ok(seg),
			Err(_)  => Err(())
		}
	}

	pub fn convert_from_bytes(bytes: &'a [u8]) -> Result<Segment<T>,()> {
		match bincode::deserialize(bytes) {
			Ok(seg) => Ok(seg),
			Err(_)  => Err(())
		}
	}
}


/*impl<'a,T,U> TryFrom<U> for Segment<T>
	where T: Serialize + Deserialize<'a>,
		  U: AsRef<[u8]>
{
	type Error = ();

	fn try_from(value: U) -> Result<Self, Self::Error> {
		let bytes = value.as_ref();
		Segment::convert_from_bytes(bytes)
	}
}
*/
/***************************************************************
 *******************Segment Key Implementation******************
 ***************************************************************/ 

#[derive(Serialize,Deserialize,Debug,PartialEq,Eq,Hash,Copy,Clone)]
pub struct SegmentKey {
	timestamp: SystemTime,
	signal: SignalId,
}

impl<'a> SegmentKey {
	pub fn new(timestamp: SystemTime, signal: SignalId) -> SegmentKey{
		SegmentKey {
			timestamp: timestamp,
			signal: signal,
		}
	}

	pub fn convert_to_bytes(&self) -> Result<Vec<u8>,()> {
		match bincode::serialize(self) {
			Ok(key) => Ok(key),
			Err(_)  => Err(())
		}
	}

	pub fn convert_from_bytes(bytes: &'a [u8]) -> Result<SegmentKey,()> {
		match bincode::deserialize(bytes) {
			Ok(key) => Ok(key),
			Err(_)  => Err(())
		}
	}
}


/***************************************************************
 ************************Segment Verifier***********************
 ***************************************************************/


pub struct SegmentIter<T> 
	where T: Copy + Send
{
	buffer: Arc<Mutex<SegmentBuffer<T>>>,
	cur_seg_key: SegmentKey,
}

impl<T> SegmentIter<T> 
	where T: Copy + Send,
{
	pub fn new(s_id: SignalId, timestamp: SystemTime, buffer: Arc<Mutex<SegmentBuffer<T>>>) -> SegmentIter<T> {
		SegmentIter {
			buffer: buffer,
			cur_seg_key: SegmentKey::new(timestamp, s_id),
		}
	}

	pub fn get_last_n(s_id: SignalId, timestamp: SystemTime, buffer: Arc<Mutex<SegmentBuffer<T>>>, n: usize) -> Vec<Segment<T>> {
		SegmentIter::new(s_id, timestamp, buffer).take(n).collect()
	}
}

impl<T> Iterator for SegmentIter<T> 
	where T: Copy + Send,
{
	type Item = Segment<T>;

	fn next(&mut self) -> Option<Segment<T>> {
		if let Ok(mut buf) = self.buffer.lock() {
			if let Ok(Some(seg)) = buf.get(self.cur_seg_key) {
				self.cur_seg_key = match seg.get_prev_key() {
					Some(key) => key,
					_ => return None,
				};
				return Some(seg.clone());
			}
		}

		None
	}
}


/***************************************************************
 ********************Fourier Implementation*********************
 ***************************************************************/

/* Takes any segment whose entries support FFTnum (f32 and f64 currently)
 * and applys the fourier transform using FFT algorithms
 * Returns a segment containing ComplexDef<T> entries 
 * Which is a Segmentable wrapper for Complex<T>
 * Note: This decision described below
 */
impl<'a,T: FFTnum + Serialize + Deserialize<'a>> Segment<T> {
	
	pub fn fourier_compress(&self) -> Segment<Complex<T>> {
		let size = self.data.len();
		let mut planner = FFTplanner::new(false);
		let fft = planner.plan_fft(size);

		let mut input: Vec<Complex<T>> = self.data.iter()
											 .map(|x| Complex::new(*x,Zero::zero()))
											 .collect();

		let mut output: Vec<Complex<T>> = vec![Complex::zero(); size];

		fft.process(&mut input, &mut output);

		Segment {
			method: Some(Fourier),
			timestamp: self.timestamp,
			signal: self.signal,
			data: output,
			time_lapse: self.time_lapse.clone(),
			prev_seg_offset: self.prev_seg_offset,
		}

	}
}


/* Applys the inverse fourier transfrom to the result of the
 * forward operation defined above. Will return a segment
 * just containing the real part of the resulting data.
 * Like above, this will likely be changed to accept Complex<T>
 * instead, and the ComplexDef coversion will be hidden behind 
 * calls to convert to and from bytes.
 */
impl<'a,T: FFTnum + Serialize + Deserialize<'a>> Segment<Complex<T>> {
	
	pub fn fourier_decompress(&self) -> Segment<T> {
		let mut planner = FFTplanner::new(true);
		let size = self.data.len();
		let fft = planner.plan_fft(size);

		let mut output: Vec<Complex<T>> = vec![Complex::zero(); size];

		fft.process(&mut self.data.clone(), &mut output);

		let output = output.iter().map(|c| c.re).collect();

		Segment {
			method: None,
			timestamp: self.timestamp,
			signal: self.signal,
			data: output,
			time_lapse: self.time_lapse.clone(),
			prev_seg_offset: self.prev_seg_offset,
		}
	}
}

/* A wrapper for the Complex structure in fftrust::num_complex
 * Due to Rust's No Orphan rule it is currently impossible to implement a
 * trait(Serialize and Deserialize) from crate A(serde) 
 * for a sturucture(Complex) in Crate B(rustfft)
 * in a new crate C(this)
 * So, to deal with this a wrapper struct to hold the Complex parameters
 * was created and that structure can implement Seilaize and Deserialize
 * Note: May change this, so that instead of returning a ComplexDef, I 
 * define a from and to bytes function for a Segment<Complex<T>> that does
 * the mapping to a ComplexDef<T> only then. Given that there will be a lot
 * of anaylsis this seems like a better option.
 */
#[derive(Clone,Serialize,Deserialize,PartialEq,Debug)]
pub struct ComplexDef<T> 
{
	re: T,
	im: T,
}

impl<'a,T> ComplexDef<T> 
	where T: Copy + Num + Serialize + Deserialize<'a>
{
	#[inline]
	fn new(re: T, im: T) -> ComplexDef<T> {
		ComplexDef {
			re: re,
			im: im,
		}
	}

	#[inline]
	fn to_complex(&self) -> Complex<T> {
		Complex::new(self.re,self.im)
	}

	#[inline]
	fn from_complex(c: &Complex<T>) -> ComplexDef<T> {
		ComplexDef::new(c.re,c.im)
	}

	pub fn convert_from_bytes(bytes: &'a [u8]) -> Result<Segment<Complex<T>>,()> {
		let deserialized_data: Result<Segment<ComplexDef<T>>,_> = bincode::deserialize(bytes);
		match deserialized_data {
			Ok(seg) => Ok(Segment::new(seg.method,seg.timestamp,seg.signal,
						seg.data.iter().map(|x| ComplexDef::to_complex(x)).collect(),
						seg.time_lapse,seg.prev_seg_offset)),
			Err(e)  => {
				println!("{:?}", e);
				Err(())
			}
		}
	}

	pub fn convert_to_bytes(seg: &Segment<Complex<T>>) -> Result<Vec<u8>,()> {
		let persitable_data: Segment<ComplexDef<T>> = 
			Segment::new(
				seg.method.clone(), seg.timestamp, seg.signal,
				seg.data.iter().map(|x| ComplexDef::from_complex(x)).collect(),
				seg.time_lapse.clone(), seg.prev_seg_offset
			);
		match bincode::serialize(&persitable_data) {
			Ok(seg) => Ok(seg),
			Err(e)  => {
				println!("{:?}", e);
				Err(())
			}
		}
	}
}

/***************************************************************
 **********************PAA  Implementation**********************
 ***************************************************************/

/* Performs paa compression on the data carried by the segment */
pub fn paa_compress_and_retain<T>(seg: &Segment<T>, chunk_size: usize) -> Segment<T> 
	where T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive
{

	let zero = T::zero();
	let paa_data = seg.data.chunks(chunk_size)
						   .map(|x| {
						   		x.iter().fold(zero, |sum, &i| sum + i) / FromPrimitive::from_usize(x.len()).unwrap()
						   })
						   .collect();

	Segment {
		method: None,
		timestamp: seg.timestamp,
		signal: seg.signal,
		data: paa_data,
		time_lapse: seg.time_lapse.clone(),
		prev_seg_offset: seg.prev_seg_offset,
	}	
}

/* Performs paa compression on the data carried by the segment */
pub fn paa_compress<T>(seg: &mut Segment<T>, chunk_size: usize)
	where T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive
{

	let zero = T::zero();
	let paa_data = seg.data.chunks(chunk_size)
						   .map(|x| {
								x.iter().fold(zero, |sum, &i| sum + i) / FromPrimitive::from_usize(x.len()).unwrap()
						   })
						   .collect();
	println!("paa finished");
	seg.data = paa_data	
}

#[derive(Clone)]
pub struct PAACompress {
	chunksize: usize,
	batchsize: usize
}

impl PAACompress {
	pub fn new(chunksize: usize, batchsize: usize) -> Self {
		PAACompress { chunksize, batchsize }
	}
}

impl<T> CompressionMethod<T> for PAACompress
	where T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive{
	fn get_segments(&self) {
		unimplemented!()
	}

	fn get_batch(&self) -> usize {
		self.batchsize
	}

	fn run_compress(&self, segs: &mut Vec<Segment<T>>) {
		for seg in segs {
			paa_compress(seg,self.chunksize);
		}
	}

	fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
		unimplemented!()
	}
}

/* Performs Fourier compression on the data carried by the segment */
pub fn fourier_compress<'a,T>(seg: &mut Segment<T>)
	where T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive+FFTnum+Deserialize<'a>+Serialize
{
	seg.fourier_compress();
}


#[derive(Clone)]
pub struct FourierCompress {
	chunksize: usize,
	batchsize: usize
}

impl FourierCompress {
	pub fn new(chunksize: usize, batchsize: usize) -> Self {
		FourierCompress { chunksize, batchsize }
	}
}

impl<'a,T> CompressionMethod<T> for FourierCompress
	where T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive+FFTnum+Deserialize<'a>+Serialize {
	fn get_segments(&self) {
		unimplemented!()
	}

	fn get_batch(&self) -> usize {
		self.batchsize
	}

	fn run_compress(&self, segs: &mut Vec<Segment<T>>) {
		for seg in segs {
			fourier_compress(seg);
		}
	}

	fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
		unimplemented!()
	}
}
/***************************************************************
 ****************************Testing****************************
 ***************************************************************/
/* The seed for the random number generator used to generate */
const RNG_SEED: [u8; 32] = [1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8,
    4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/* Test Helper Functions */
pub fn random_f32signal(length: usize) -> Vec<f32> {
	let mut sig = Vec::with_capacity(length);
	let normal_dist = Normal::new(-10.0,10.0);
	let mut rngs: StdRng = SeedableRng::from_seed(RNG_SEED);
	for _ in 0..length {
		sig.push(normal_dist.sample(&mut rngs) as f32);
	}

	return sig;
}

pub fn random_f32complex_signal(length: usize) -> Vec<Complex<f32>> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(-10.0, 10.0);
    let mut rngs: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        sig.push(Complex{re: (normal_dist.sample(&mut rngs) as f32),
                         im: (normal_dist.sample(&mut rngs) as f32)});
    }
    return sig;
}

pub fn compare_vectors<T>(vec1: &[T], vec2: &[T]) -> bool 
	where T: Sub<T, Output = T> + Copy + Num + PartialOrd + FromPrimitive
{
    assert_eq!(vec1.len(), vec2.len());
    let mut sse = T::zero();
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        sse = sse + (a - b);
    }
    let size = FromPrimitive::from_usize(vec1.len()).unwrap();
    let err_margin = FromPrimitive::from_f32(0.1).unwrap();
    return (sse / size) < err_margin;
}


pub fn error_rate<T>(actual: &[T], saved: &[T]) -> T
	where T: Sub<T, Output = T> + Copy + Num + PartialOrd + FromPrimitive
{
	assert_eq!(actual.len(), saved.len());
	let mut sse = T::zero();
	for (&a, &b) in actual.iter().zip(saved.iter()) {
		sse = sse +  (a - b) / a * FromPrimitive::from_f32(1.0).unwrap();
	}
	let size = FromPrimitive::from_usize(actual.len()).unwrap();
	return sse / size;
}

/* Main Segment Test Functions */
#[test]
fn test_fourier_compression() {
	let data: Vec<f32> = vec![-0.62195737,-0.62067724,-0.61668396,-0.61101182,-0.60376876,-0.59526654,-0.58647653,-0.57789937,-0.57014665,-0.56427365,
						-0.55999256,-0.55476112,-0.5474581,-0.53943256,-0.53062456,-0.52216732,-0.51320042,-0.50285746,-0.49217873,-0.48166489,
						-0.47088723,-0.45836476,-0.44452618,-0.42874193,-0.41364421,-0.40055812,-0.38835942,-0.37925162,-0.37483863,-0.37323772,
						-0.3745808,-0.37744086,-0.3785561,-0.37838821,-0.37688923,-0.37435596,-0.37161582,-0.37097426,-0.3726831,-0.37674233,
						-0.38282819,-0.39020916,-0.39782997,-0.40271964,-0.40482421,-0.405139,-0.40320531,-0.40209607,-0.40450643,-0.40913228,
						-0.41437871,-0.41877371,-0.42068341,-0.42061446,-0.42003286,-0.4193853,-0.42044358,-0.42355246,-0.42679925,-0.4299621,
						-0.43338876,-0.4370133,-0.44144728,-0.44607612,-0.45027926,-0.45334617,-0.45363398,-0.45104074,-0.44520372,-0.4346539,
						-0.42063844,-0.40501009,-0.38813158,-0.37023676,-0.35256679,-0.33583219,-0.31876181,-0.301712419,-0.285571409,-0.269301489,
						-0.252180139,-0.234528159,-0.216570389,-0.200285479,-0.186971539,-0.177240169,-0.171933779,-0.170434799,-0.172398469,-0.177057289,
						-0.183673789,-0.189627729,-0.192997439,-0.191966139,-0.186671749,-0.180990619,-0.176229859,-0.172713249,-0.169943139,-0.166990149,
						-0.162721049,-0.156848049,-0.150066669,-0.141633409,-0.131245489,-0.119916199,-0.108499969,-0.096340251,-0.083158226,-0.0697573499,
						-0.0566292889,-0.0421851239,-0.0246110888,-0.00551708969,0.0134300093,0.0317595293,0.0477386503,0.0608007564,0.0721150524,
						0.0812768144,0.0866851324,0.0873476814,0.0823770654,0.0708738974,0.0549607324,0.0371918313,0.0218962453,0.0146591733,0.0183856353,
						0.0337321863,0.0588940544,0.0913829334,0.128563622,0.166442832,0.202007612,0.234331602,0.263414802,0.289128292,0.310380822,0.328569442,
						0.349893922,0.381669282,0.428752232,0.494686332,0.579342693,0.679117763,0.788387363,0.897219263,0.996553624,1.0867472,1.176626,1.2697036,
						1.3568063,1.4195176,1.448301,1.4431026,1.4088599,1.354336,1.2914928,1.2396521,1.2183036,1.2402487,1.3010323,1.392572,1.4882728,1.5569621,
						1.5809457,1.5417264,1.4510622,1.3434204,1.2592887,1.2392504,1.308863,1.4664207,1.69350111,1.95796901,2.21981371,2.44155481,2.60004781,
						2.70220331,2.77060771,2.83122341,2.90104291,2.98069571,3.05830081,3.13801951,3.20854351,3.24397941,3.24490281,3.20795291,3.14083761,
						3.03872121,2.91427891,2.78453021,2.65306371,2.50789361,2.36291831,2.22699681,2.09469991,1.96810511,1.84871441,1.73805081,1.63397961,
						1.5329544,1.4369058,1.3438402,1.2517728,1.1667387,1.0916968,1.0273756,0.977096874,0.941478123,0.915758633,0.896268903,0.880832413,
						0.868909533,0.860542233,0.859582883,0.868261973,0.885608163,0.911318663,0.940626703,0.970837134,0.998975974,1.0235442,1.0439424,
						1.0587703,1.0681119,1.0672365,1.0484123,1.0117593,0.957343323,0.890260993,0.815183113,0.738666213,0.675694093,0.633386893,
						0.605628793,0.581363323,0.554492613,0.526977352,0.503404402,0.481036632,0.455886752,0.428959082,0.404936442,0.390477292,0.385476692,
						0.381834172,0.370426942,0.344398662,0.301051172,0.246725162,0.195436082,0.156378682,0.133657152,0.122189962,0.115189721,0.109604531,
						0.106177861,0.103632591,0.0969711285,0.0844906264,0.0694588614,0.0565256674,0.0481073983,0.0438772793,0.0397221083,0.0355699353,0.0310190333,
						0.0208319703,0.00367165393,-0.0196854428,-0.0483129509,-0.076451792,-0.099395171,-0.110973289,-0.109843059,-0.098052085,-0.076247931,-0.0483339369,
						-0.0181624798,0.00440915173,0.0223969043,0.0412840443,0.0596855154,0.0740847114,0.0820682754,0.0848563774,0.0860855404,0.0901237904,0.0970670635,
						0.104337111,0.108894011,0.116050141,0.123610992,0.128074952,0.136460242,0.159829332,0.216715602,0.266553662,0.230644112,0.115204711,-0.0106615868,
						-0.109396359,-0.192958469,-0.274164179,-0.32414614,-0.35466836,-0.36840501,-0.37406516,-0.36417189,-0.34753622,-0.33574525,-0.34106962,-0.37276104,
						-0.3892798,-0.37503649,-0.3220116,-0.245809479,-0.179728469,-0.145464799,-0.166654379,-0.197965059,-0.219271549,-0.226451659,-0.199269169,
						-0.135427639,-0.0517695979,0.0319873743,0.101132291,0.174609272,0.243100632,0.296035592,0.342887692,0.385953372,0.430619952,0.494926172,0.582319663,
						0.712509023,0.854045653,0.924326803,0.965812553,1.0276874,1.0600174,1.0359468,0.988531084,0.828377133,0.489161092,0.103101951,-0.30778628,-0.46598857,
						-0.58364046,-0.64754195,-0.69489171,-0.733349531,-0.757540061,-0.777293611,-0.793491581,-0.811923031,-0.826016431,-0.833856091,-0.842181431,-0.849118701,
						-0.855282511,-0.860055261,-0.862576541,-0.856694541,-0.859611561,-0.869705691,-0.876448091,-0.880480351,-0.882506971,-0.887078861,-0.892025491,-0.896123701,
						-0.898342191,-0.900878461,-0.902203561,-0.903051981,-0.904532971,-0.904344101,-0.905980991,-0.906679511,-0.902932061,-0.901750871,-0.904892731,-0.907297091,
						-0.911940931,-0.914738021,-0.917277291,-0.919381861,-0.920488111,-0.920200301,-0.918383541,-0.912744381,-0.915376591,-0.916413881,-0.917370231,-0.917220331,
						-0.916063121,-0.917834921,-0.919181001,-0.919088061,-0.919151021,-0.919702641,-0.920440141,-0.920973781,-0.920757921,-0.920299241,-0.920212301,-0.919591721,
						-0.918872211,-0.917712001,-0.916587761,-0.916494831,-0.917055451,-0.916857581,-0.916344931,-0.915748341,-0.915226691,-0.914594121,-0.913667751,-0.913278021,
						-0.912501551,-0.909458621,-0.907524931,-0.905270471,-0.902011691,-0.900314841,-0.898713931,-0.896756271,-0.895665011,-0.893953181,-0.889552171,-0.884101881,
						-0.880213531,-0.873809891,-0.865526531,-0.856361771,-0.840271731,-0.822256991,-0.813463981,-0.806298861,-0.799391561,-0.800153041,-0.809887411,-0.821756331,
						-0.833412401,-0.842238391,-0.845599101,-0.845946861,-0.845137411,-0.842520201,-0.838814721,-0.832471041,-0.826625021,-0.818299691,-0.809308811,-0.796495531,
						-0.782447101,-0.773327311,-0.759641631,-0.745428301,-0.72934425,-0.70699747,-0.67561783,-0.65552252,-0.61660601,-0.59196579,-0.56767033,-0.54149815,-0.52006575,
						-0.49335094,-0.46228909,-0.41341037,-0.35229997,-0.262643019,-0.178226499,-0.0736187209,0.0838820404,0.277598142,0.514676722,0.841804993,1.2105839,1.5876971,
						2.39163871,2.82621381,3.21871261,3.47780221,3.69865281,3.64230621,3.48208031,3.02960141,2.63782211,2.18579591,1.77726711,1.1683966,0.729171683,0.388690502,
						0.0450794603,-0.153076619,-0.289552699,-0.41678607,-0.49278732,-0.56761337,-0.62557891,-0.6634731,-0.69534141,-0.7216365,-0.747484901,-0.765124891,-0.783643281,
						-0.797577801,-0.808415421,-0.818935251,-0.830987051,-0.839309381,-0.847916521,-0.855198561,-0.863080201,-0.869840591,-0.876349161,-0.882662861,-0.888248061,
						-0.893890221,-0.899700261,-0.903696541,-0.907312081,-0.911200431,-0.914621101,-0.917136391,-0.920083381,-0.923000401,-0.924916091,-0.928594591,-0.932018261,
						-0.934893301,-0.938107111,-0.941129051,-0.944043071,-0.946063691,-0.947907441,-0.949688231,-0.950962361,-0.952008651,-0.953171851,-0.954431001,-0.956025911,
						-0.957141151,-0.958265391,-0.959446581,-0.960726711,-0.962213701,-0.963790621,-0.964902871,-0.964471161,-0.965019791,-0.966054081,-0.966018111,-0.966335891,
						-0.967106371,-0.967484111,-0.967454131,-0.967816881,-0.968638321,-0.969141981,-0.969540711,-0.970344161,-0.970377141];
	let init_seg = Segment::new(None, SystemTime::now(),0,data, Some(vec![]), None);
	let compressed_seg = init_seg.fourier_compress();
	let decompressed_seg = compressed_seg.fourier_decompress();
	println!("decompressed {:?}", decompressed_seg.data.as_slice());

	assert!(compare_vectors(init_seg.data.as_slice(), decompressed_seg.data.as_slice()));
	assert_eq!(compressed_seg.method, Some(Fourier));
	assert_eq!(decompressed_seg.method, None)
}

#[test]
fn test_complex_segment_byte_conversion() {
	let sizes = vec![10,100,1024,5000];
	let segs: Vec<Segment<Complex<f32>>> = sizes.into_iter().map(move |x|
		Segment{
			method: None,
			timestamp: SystemTime::now(),
			signal: 0,
			data: random_f32complex_signal(x),
			time_lapse: Some(vec![]),
			prev_seg_offset: None,
		}).collect();

	let mut converted_segs: Vec<Segment<Complex<f32>>> = segs.iter().map({|seg|
		if let Ok(bytes) = ComplexDef::convert_to_bytes(seg) {
			match ComplexDef::convert_from_bytes(&bytes) {
				Ok(new_seg) => new_seg,
				_           => panic!("Failed to convert bytes into segment"),
			}
		} else {
			panic!("Failed to convert segment into bytes");
		}
	}).collect();
	
	for (seg1,seg2) in segs.iter().zip(converted_segs.iter_mut()) {
		assert_eq!(seg1, seg2);
	}
	
}

#[test]
fn test_segment_byte_conversion() {
	let sizes = vec![10,100,1024,5000];
	let segs: Vec<Segment<f32>> = sizes.into_iter().map(move |x| 
		Segment {
			method: None,
			timestamp: SystemTime::now(),
			signal: 0,
			data: random_f32signal(x),
			time_lapse: Some(vec![]),
			prev_seg_offset: None,
		}).collect();

	let mut converted_segs: Vec<Segment<f32>> = segs.iter().map({|seg|
		if let Ok(bytes) = seg.convert_to_bytes() {
			match Segment::convert_from_bytes(&bytes) {
				Ok(new_seg) => new_seg,
				_           => panic!("Failed to convert bytes into segment"),
			}
		} else {
			panic!("Failed to convert segment into bytes");
		}
	}).collect();
	
	for (seg1,seg2) in segs.iter().zip(converted_segs.iter_mut()) {
		assert_eq!(seg1, seg2);
	}
}

#[test]
fn test_paa_compression() {
	let data = vec![-8.267001490320215, -4.701408995824961, -3.9473912522030634, 1.50407251209921, -4.999423104642167, -0.28289749385261587, -0.6753507278963333, -5.326739149145712,
		-2.1362597150259894, -3.314403760401026, 3.420589457671861, -1.712699288745334, -8.183436090626452, 0.15183842041441586, -2.2802280274023357, 3.279496512365787, 5.247330227129956, -3.9719581135031152,
		8.01152964472265, -5.48099695985327, 8.170989770876538, -9.129530005554134, -6.593045254202177, -0.8350828824329959, -4.91309022394743, -6.445900409398706, -6.4329687402629565,
		-6.611464654075149, -3.1816580161523467, -9.887218622891062, 6.962306141809812, 5.091221080349031, -8.24772280500376, -5.096972967331386, 3.085634629993324, 4.232512030886422, 6.530522943413565, 2.984618610720876,
		8.153663784677978, 2.6321383973434553, -9.41199132494699, 0.0869897646583766, -4.667797464065007, 7.607496844933294, -5.438272397674817, -9.084890536432288, 3.1874721901461953,
		0.3798605807927693, -2.025513677453528, 6.706578138886233, 0.33282338747066653, -4.306610162957492, -9.827484921553733, 7.807651814568477, -2.006265477115674, -8.505813371648042, -0.9447616205646128, -2.506428732611883,
		0.4254133276500802, -9.992707546838307, 8.893505303435681, 3.156886237175014, 0.015536746660293588, 5.655429001755028, -7.418224745449836, 4.863129452185156, -2.4838357061064187, 3.9137354423611157,
		2.2397954007396237, 5.883220460412373, -6.215086648360739, 7.425055753593313, -7.69143693714661, 0.5710216632051797, 4.320316365240407, 9.072729037257837, -7.220428131285665, -8.77069028948311, 1.9955530846071703,
		-6.188036231770518, 9.095302934925396, 5.584025322601583, -2.995158134634841, 4.92671046289562, -9.571007616192517, 2.7724560669537226, -1.8522905017334796, -2.8380095163670322,
		6.334988114262327, 7.264121425542719, 1.874283061574129, 9.74422127363868, -9.672811184063907, -8.898637556200882, 6.603350224689084, -0.628918759685682, 8.513223771426471, -8.041579967785776, 8.921911750563325, -9.157191639192238];
	let seg1 = Segment {
				method: None,
				timestamp: SystemTime::now(),
				signal: 0,
				data: data,
				time_lapse: Some(vec![]),
				prev_seg_offset: None,
			};

	let seg2 = seg1.clone();
	let seg3 = seg1.clone();

	let paa_seg1 = paa_compress_and_retain(&seg1,3);
	let paa_seg2 = paa_compress_and_retain(&seg2,7);
	let paa_seg3 = paa_compress_and_retain(&seg3,10);

	compare_vectors(&paa_seg3.data, &vec![-3.2146803177212875, -0.15185342178258382, -4.585896903804042, 2.6327921846859317, -1.2660067881155765, -2.952418330360052, 2.4719177593168036,
								   -1.2701002334142735, 1.8721138558253496, -0.07421490250367953]);
	compare_vectors(&paa_seg2.data, &vec![-3.052771507520021, -2.443015732265462, 1.8537375791908872, -5.851583167124793, -1.610630079889484, 2.1726363152505264, -1.434520637107626,
								   -1.3998743703356522, -0.13607946929910497, 1.8076070436995775, 0.037453014643388904,
									0.5211010213059002, 0.5692201477199811, -0.3401645997117159, -0.11763994431445646] );
	compare_vectors(&paa_seg1.data, &vec![-5.638600579449413, -1.2594160287985243, -2.712783197356012, -0.5355045304914997, -3.437275232538124, 1.5182895419975424, 3.567174151915306, -5.519219380729769, -5.930653124536364,
								   -6.560113764372853, 1.268601472385028, 0.7403912311827868, 5.889601779604139, -2.2309543876483864, -0.8328576722688433, -1.839185921831108,
									1.671295949634457, -2.1088144233142496, -3.818946823109443, -4.024574317266704, 4.0219760957569965, 1.0334445694967827, 1.2232317123314402,
									2.3643965218816487, -0.9333663029003411, -2.306129794503646, 1.6342732625873495, 2.5051925502874544, -2.8836140169907583, 3.5870333411460043, 0.6485643837163005, -0.9747353637324933, 3.1311851847346737, -9.157191639192238]);
	let error_rate = error_rate(&paa_seg3.data, &vec![-3.2146803177212875, -0.15185342178258382, -4.585896903804042, 2.6327921846859317, -1.2660067881155765, -2.952418330360052, 2.4719177593168036,
													  -1.2701002334142735, 1.8721138558253496, -0.07421490250367953]);
	assert_eq!(0.0, error_rate);
}

#[test]
fn test_implicit_key_list() {
	let data: Vec<f32> = vec![1.0,2.0,3.0];
	let timestamp = SystemTime::now();
	let sig0seg1 = Segment::new(None,timestamp,0,data.clone(),None,None);
	let sig0seg2 = Segment::new(None,timestamp + Duration::new(15,0),0,data.clone(),None,Some(Duration::new(15,0)));
	let sig0seg3 = Segment::new(None,timestamp + Duration::new(18,0),0,data.clone(),None,Some(Duration::new(3,0)));
	let sig0seg4 = Segment::new(None,timestamp + Duration::new(34,0),0,data.clone(),None,Some(Duration::new(16,0)));

	let sig1seg1 = Segment::new(None,timestamp + Duration::new(10,0),0,data.clone(),None,None);
	let sig1seg2 = Segment::new(None,timestamp + Duration::new(15,0),0,data.clone(),None,Some(Duration::new(5,0)));
	let sig1seg3 = Segment::new(None,timestamp + Duration::new(35,0),0,data.clone(),None,Some(Duration::new(20,0)));
	let sig1seg4 = Segment::new(None,timestamp + Duration::new(135,0),0,data.clone(),None,Some(Duration::new(100,0)));

	assert_eq!(sig0seg4.get_prev_key(), Some(sig0seg3.get_key()));
	assert_eq!(sig0seg3.get_prev_key(), Some(sig0seg2.get_key()));
	assert_eq!(sig0seg2.get_prev_key(), Some(sig0seg1.get_key()));
	assert_eq!(sig0seg1.get_prev_key(), None);

	assert_eq!(sig1seg4.get_prev_key(), Some(sig1seg3.get_key()));
	assert_eq!(sig1seg3.get_prev_key(), Some(sig1seg2.get_key()));
	assert_eq!(sig1seg2.get_prev_key(), Some(sig1seg1.get_key()));
	assert_eq!(sig1seg1.get_prev_key(), None);
}

#[test]
fn test_key_byte_conversion() {
	let key = SegmentKey::new(SystemTime::now(),0);

	if let Ok(bytes) = key.convert_to_bytes() {
		match SegmentKey::convert_from_bytes(&bytes) {
			Ok(new_key) => assert_eq!(key, new_key),
			_           => panic!("Failed to convert bytes into key"),
		} 
	} else {
		panic!("Failed to convert key into bytes");
	}
}
