//use std::convert::TryFrom;
use std::sync::Arc;
use std::sync::Mutex;
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
use num::Num;

/* Currently plan to move methods into this file */
use crate::methods;
use methods::Methods;
use Methods::{Fourier};

extern crate rand;
use rand::{SeedableRng};
use rand::rngs::{StdRng};
use rand::distributions::{Normal, Distribution};
use crate::methods::compress::CompressionMethod;

pub type SignalId = u64;
pub type DictionaryId = u32; /* Type alias for dictionary id */
const DEFAULT_BATCH_SIZE: usize = 50;

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

	pub fn get_byte_size(&self) -> Result<usize,()> {
		match bincode::serialized_size(self) {
			Ok(bsize) => Ok(bsize as usize),
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
	let paa_data:Vec<T> = seg.data.chunks(chunk_size)
						   .map(|x| {
								x.iter().fold(zero, |sum, &i| sum + i) / FromPrimitive::from_usize(x.len()).unwrap()
						   })
						   .collect();
//	println!("paa finished with length {}",paa_data.len());
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

	fn run_compress<'a>(&self, segs: &mut Vec<Segment<T>>) {
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

	fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
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
		sse = sse + ( (a - b) / a * FromPrimitive::from_f32(1.0).unwrap() );
	}
	let size = FromPrimitive::from_usize(actual.len()).unwrap();
	return sse / size;
}

/* Main Segment Test Functions */
#[test]
fn test_fourier_on_segment() {
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
	let size = compressed_seg.data.len();
	println!("decompressed {:?}", compressed_seg.data.len());
	let decompressed_seg = compressed_seg.fourier_decompress();
	println!("decompressed {:?}", decompressed_seg.data.iter().map(|&x|{x/size as f32 }).collect::<Vec<_>>());

	assert!(compare_vectors(init_seg.data.as_slice(), decompressed_seg.data.as_slice()));
	assert_eq!(compressed_seg.method, Some(Fourier));
	assert_eq!(decompressed_seg.method, None)
}


#[test]
fn test_fourier_compression() {
	let data: Vec<f64> = vec![-0.55999256,-0.55476112,-0.5474581,-0.53943256,-0.53062456,-0.52216732,-0.51320042,-0.50285746,-0.49217873,-0.48166489];
	let fres = fft_ifft_ratio(&data,0.4);
	println!("decompressed {:?}",fres );
	let mut sse = 0.0;
	for (&a, &b) in data.as_slice().iter().zip(fres.as_slice().iter()) {
		sse = sse + ( (a - b) / a );
		// println!("{},{},cur err: {}",a, b, (a - b) / a);
	}

	println!("err rate: {}",sse);
	// assert!(compare_vectors(data.as_slice(), decompressed_seg.data.as_slice()));
	// assert_eq!(compressed_seg.method, Some(Fourier));
	// assert_eq!(decompressed_seg.method, None)
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
fn test_paa_compression_int() {
	let data = vec![
		1391, 1756, 532, 1912, 569, 382, 1520, 1238, 795, 1742, 1867, 665, 1972, 1342, 1205, 1717,
		722, 1182, 961, 1623, 470, 8, 321, 789, 1448, 1022, 783, 941, 1204, 196, 866, 1555, 1058,
		476, 24, 1478, 201, 803, 1070, 939, 1955, 1465, 1943, 705, 650, 561, 1754, 1802, 1896, 886,
		786, 987, 936, 485, 1193, 443, 1127, 293, 1076, 1049, 121, 1219, 1367, 910, 372, 695, 1934,
		1948, 1898, 98, 651, 482, 1746, 1092, 49, 1947, 117, 1293, 1820, 28, 1946, 1921, 1486, 215,
		310, 1823, 598, 120, 718, 1718, 805, 1653, 266, 1024, 1390, 1572, 1122, 794, 1632, 1890,
		1534, 926, 1006, 979, 226, 1380, 1297, 307, 1221, 1982, 220, 1801, 138, 173, 1260, 1371,
		1335, 1860, 899, 1209, 1399, 427, 221, 568, 352, 610, 60, 331, 1265, 1179, 1529, 438, 426,
		2004, 1745, 817, 1310, 1042, 3, 1992, 1162, 1779, 1872, 1275, 182, 771, 5, 222, 755, 1775,
		1735, 643, 499, 897, 767, 129, 1167, 1576, 1834, 1702, 1226, 1884, 796, 1336, 813, 1724,
		143, 211, 1475, 1885, 1816, 1150, 1410, 823, 669, 1283, 518, 125, 1770, 1249, 533, 1396,
		270, 1175, 124, 753, 164, 854, 711, 498, 723, 420, 496, 1978, 721, 1579, 594, 697, 309, 954,
		1425, 1012, 611, 931, 1125, 1593, 1725, 1825, 1018, 1776, 1925, 1322, 1877, 1014, 1422, 72,
		1413, 504, 1519, 1404, 296, 186, 1291, 41, 874, 35, 1759, 1301, 500, 1560, 586, 1189, 1272,
		1979, 1998, 826, 1098, 234, 1203, 375, 575, 742, 1269, 535, 1812, 810, 1636, 1220, 429, 671,
		1712, 1259, 233, 1899, 712, 1580, 660, 1302, 1038, 445, 1800, 578, 1728, 1964, 1950, 1197,
		154, 461, 424, 906, 565, 816, 1940, 1826, 1803, 797, 1256, 1105, 183, 273, 1187, 1444, 78,
		1723, 646, 1870, 1103, 298, 389, 1211, 1967, 165, 677, 1300, 636, 142, 637, 383, 52, 962,
		715, 1583, 843, 250, 243, 390, 513, 787, 1445, 1491, 1703, 1100, 1842, 1191, 1464, 269, 188,
		1726, 1185, 1864, 1811, 1498, 335, 1041, 552, 1313, 1598, 1365, 733, 699, 760, 750, 189,
		430, 1692, 528, 1959, 924, 1016, 548, 972, 262, 726, 462, 1449, 872, 7, 1273, 404, 988, 394,
		1783, 245, 807, 223, 1124, 1696, 1607, 185, 435, 707, 1282, 539, 1347, 358, 48, 1882, 100,
		1698, 1648, 237, 1061, 1684, 1321, 942, 69, 773, 1442, 455, 1647, 1471, 1277, 1377, 1855,
		704, 740, 81, 1857, 1522, 1328, 956, 680, 493, 1905, 23, 1771, 1457, 811, 902, 741, 1274,
		304, 1411, 951, 341, 675, 1044, 759, 582, 1530, 1683, 1575, 1429, 1456, 1276, 1320, 1731,
		208, 1976, 1916, 105, 1455, 989, 622, 1370, 674, 1987, 1089, 836, 330, 456, 1231, 673, 764,
		1078, 315, 413, 1466, 272, 305, 419, 1392, 1288, 153, 454, 1094, 421, 647, 889, 261, 700,
		316, 82, 328, 197, 915, 373, 1975, 691, 1241, 301, 20, 439, 346, 1074, 1004, 1309, 56, 415,
		90, 42, 1630, 1051, 311, 1093, 730, 1517, 1989, 664, 1699, 1797, 319, 1056, 6, 265, 410,
		279, 1436, 449, 1871, 1687, 1539, 135, 1111, 377, 299, 1040, 1148, 927, 688, 1532, 76, 503,
		869, 925, 784, 522, 839, 336, 1206, 958, 754, 1133, 282, 1886, 846, 1734, 67, 842, 1348,
		834, 1186, 1248, 1781, 1991, 1707, 827, 727, 1611, 645, 1318, 1962, 144, 1994, 458, 1484,
		1299, 1144, 891, 1030, 1550, 1071, 1400, 1806, 566, 1874, 275, 969, 155, 1740, 1401, 1767, 1665, 1599, 1937, 1689, 918, 1315, 1388, 1114, 1020, 1620, 446, 1581, 519, 1633, 724, 479, 1526, 769, 107, 232, 1381, 1345, 1531, 683, 1496, 670, 15, 252, 737, 995, 820, 678, 1883, 666, 1236, 837, 1533, 343, 708, 1738, 1654, 574, 1813, 469, 1369, 907, 437, 388, 1201, 524, 58, 1218, 547, 692, 1920, 46, 1831, 1933, 193, 103, 1488, 1688, 847, 672, 398, 473, 1157, 689, 957, 423, 1096, 749, 433, 676, 1911, 2006, 1568, 1052, 587, 1017, 276, 911, 1705, 1303, 1706, 1850, 814, 1788, 157, 1545, 663, 26, 1174, 150, 1667, 278, 1851, 698, 564, 983, 831, 1641, 1819, 772, 1863, 765, 1543, 1119, 1172, 1762, 1476, 1402, 531, 1657, 1424, 947, 1239, 1502, 990, 101, 1354, 738, 1713, 1108, 228, 240, 557, 865, 16, 523, 145, 1281, 1963, 777, 260, 1230, 1541, 1945, 920, 1112, 1569, 1838, 997, 1270, 1080, 1511, 1792, 257, 158, 371, 195, 277, 139, 992, 356, 589, 403, 1351, 1503, 1586, 1773, 1213, 1225, 913, 529, 113, 1880, 1763, 656, 776, 360, 1374, 975, 285, 483, 1799, 1675, 1332, 114, 728, 821, 1169, 1861, 481, 768, 1067, 18, 583, 559, 1427, 861, 1324, 545, 1892, 1968, 400, 658, 543, 2003, 409, 1142, 638, 829, 932, 448, 187, 801, 134, 1638, 690, 757, 119, 89, 505, 597, 1895, 489, 184, 57, 1951, 171, 320, 1490, 1897, 300, 1136, 1312, 1251, 682, 177, 1650, 460, 668, 895, 21, 1050, 735, 344, 1469, 444, 935, 491, 1588, 879, 1431, 1571, 2008, 1084, 422, 1487, 80, 1856, 1784, 1873, 973, 434, 1736, 1518, 1868, 248, 877, 1628, 654, 1497, 137, 364, 1383, 2007, 1973, 982, 1741, 818, 1966, 702, 800, 780, 1081, 178, 71, 1337, 526, 231, 1082, 1137, 1180, 1840, 884, 1553, 1668, 111, 1063, 1557, 1663, 93, 901, 1504, 1495, 1821, 1750, 116, 1671, 603, 1985, 649, 452, 799, 833, 1364, 1612, 1188, 1414, 1013, 701, 1680, 1141, 1565, 592, 1953, 1104, 432, 36, 32, 1544, 1363, 1477, 1673, 1409, 751, 736, 1184, 1514, 904, 530, 1701, 599, 508, 1660, 635, 1341, 1753, 1778, 550, 1441, 53, 1672, 1794, 1512, 181, 985, 414, 495, 1156, 1995, 249, 974, 1567, 374, 1686, 1761, 1460, 179, 576, 881, 1015, 148, 1481, 1343, 258, 194, 720, 88, 1027, 1965, 1935, 878, 31, 1854, 1235, 641, 1578, 1600, 1462, 1116, 948, 572, 1747, 1516, 1382, 1222, 714, 1066, 933, 944, 102, 984, 1791, 630, 1385, 349, 864, 955, 1626, 1356, 487, 1165, 632, 648, 1604, 431, 1690, 1311, 345, 588, 1634, 844, 1200, 1314, 207, 1327, 1398, 369, 1903, 1603, 418, 1489, 1591, 11, 160, 61, 1090, 739, 1419, 407, 928, 411, 644, 1160, 743, 1173, 198, 1470, 474, 1421, 280, 560, 1941, 1338, 710
	];
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

	let paa_seg3 = paa_compress_and_retain(&seg3,100);
	println!("PAA compressed data: {:?}", paa_seg3.data);

	let error_rate = error_rate(&paa_seg3.data, &vec![1055, 968, 1032, 976, 863, 1057, 1036, 877, 1105, 979]);
	assert_eq!(0, error_rate);
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

