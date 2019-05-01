use crate::bincode;
use serde::{Serialize, Deserialize};

use rustfft::{FFTplanner,FFTnum};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use std::time::{Duration};
use crate::signal::SignalId;
use num::Num;

/* Currently plan to move methods into this file */
use crate::methods;
use methods::Methods;
use Methods::{Fourier};

extern crate rand;
use rand::{SeedableRng};
use rand::rngs::{StdRng};
use rand::distributions::{Normal, Distribution};


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
/* Traits an object must met in order to become the data type of a segment */
pub trait Segmentable<'a>: Serialize + Deserialize<'a> + Clone {}

/* Implementing the trait for some basic primitives */
impl<'a> Segmentable<'a> for f64 {}
impl<'a> Segmentable<'a> for f32 {}

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
	timestamp: Duration,
	signal: SignalId,
	data: Vec<T>,
	prev_seg_offset: Option<Duration>,
}


impl<'a,T> Segment<T> 
	where T: Segmentable<'a>
{
	/* Construction Methods */
	pub fn new(method: Option<Methods>, timestamp: Duration, signal: SignalId,
	    data: Vec<T>, next_seg_offset: Option<Duration>) -> Segment<T> {
		
		Segment {
			method: method,
			timestamp: timestamp,
			signal: signal,
			data: data,
			prev_seg_offset: next_seg_offset
		}
	}

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

	pub fn get_key(&self) -> SegmentKey {
		SegmentKey::new(self.timestamp,self.signal)
	}

	pub fn get_prev_key(&self) -> Option<SegmentKey> {
		match self.prev_seg_offset {
			Some(time) => Some(SegmentKey::new(self.timestamp-time,self.signal)),
			None       => None, 
		}
	}

}

/***************************************************************
 *******************Segment Key Implementation******************
 ***************************************************************/ 

#[derive(Serialize,Deserialize,Debug,PartialEq)]
pub struct SegmentKey {
	timestamp: Duration,
	signal: SignalId,
}

impl<'a> SegmentKey {
	pub fn new(timestamp: Duration, signal: SignalId) -> SegmentKey{
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
impl<'a,T: FFTnum + Segmentable<'a>> Segment<T> {
	
	pub fn fourier_compress(&self) -> Segment<ComplexDef<T>> {
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
			data: output.into_iter().map(|c| ComplexDef::from_complex(c)).collect(),
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
impl<'a,T: FFTnum + Segmentable<'a>> Segment<ComplexDef<T>> {
	
	pub fn fourier_decompress(&self) -> Segment<T> {
		let mut planner = FFTplanner::new(true);
		let size = self.data.len();
		let fft = planner.plan_fft(size);

		let mut input:  Vec<Complex<T>> = self.data.iter().map(|c| c.to_complex()).collect();
		let mut output: Vec<Complex<T>> = vec![Complex::zero(); size];

		fft.process(&mut input, &mut output);

		let output = output.iter().map(|c| c.re).collect();

		Segment {
			method: None,
			timestamp: self.timestamp,
			signal: self.signal,
			data: output,
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
#[derive(Clone,Serialize,Deserialize)]
pub struct ComplexDef<T> 
{
	re: T,
	im: T,
}

impl<T> ComplexDef<T> 
	where T: Clone + Num
{
	#[inline]
	fn new(re: T, im: T) -> ComplexDef<T> {
		ComplexDef {
			re: re,
			im: im,
		}
	}

	#[inline]
	fn to_complex(&self) -> Complex<T>{
		Complex::new(self.re.clone(),self.im.clone())
	}

	#[inline]
	fn from_complex(c: Complex<T>) -> ComplexDef<T> {
		ComplexDef::new(c.re.clone(),c.im.clone())
	}
}

/***************************************************************
 ****************************Testing****************************
 ***************************************************************/
/* The seed for the random number generator used to generate */
const RNG_SEED: [u8; 32] = [1, 9, 1, 0, 1, 1, 4, 3, 1, 4, 9, 8,
    4, 1, 4, 8, 2, 8, 1, 2, 2, 2, 6, 1, 2, 3, 4, 5, 6, 7, 8, 9];

/* Test Helper Functions */
fn random_f32signal(length: usize) -> Vec<f32> {
	let mut sig = Vec::with_capacity(length);
	let normal_dist = Normal::new(-10.0,10.0);
	let mut rngs: StdRng = SeedableRng::from_seed(RNG_SEED);
	for _ in 0..length {
		sig.push(normal_dist.sample(&mut rngs) as f32);
	}

	return sig;
}

fn random_f32complex_signal(length: usize) -> Vec<Complex<f32>> {
    let mut sig = Vec::with_capacity(length);
    let normal_dist = Normal::new(-10.0, 10.0);
    let mut rngs: StdRng = SeedableRng::from_seed(RNG_SEED);
    for _ in 0..length {
        sig.push(Complex{re: (normal_dist.sample(&mut rngs) as f32),
                         im: (normal_dist.sample(&mut rngs) as f32)});
    }
    return sig;
}

fn compare_vectors(vec1: &[f32], vec2: &[f32]) -> bool {
    assert_eq!(vec1.len(), vec2.len());
    let mut sse = 0f32;
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        sse = sse + (a - b);
    }
    return (sse / vec1.len() as f32) < 0.1f32;
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
	let init_seg = Segment::new(None, Duration::new(5,0),0,data, None);
	let compressed_seg = init_seg.fourier_compress();
	let decompressed_seg = compressed_seg.fourier_decompress();

	assert!(compare_vectors(init_seg.data.as_slice(), decompressed_seg.data.as_slice()));
	assert_eq!(compressed_seg.method, Some(Fourier));
	assert_eq!(decompressed_seg.method, None)
}


#[test]
fn test_segment_byte_conversion() {
	let sizes = vec![10,100,1024,5000];
	let segs: Vec<Segment<f32>> = sizes.into_iter().map(move |x| 
		Segment {
			method: None,
			timestamp: Duration::new(5,0),
			signal: 0,
			data: random_f32signal(x),
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
fn test_implicit_key_list() {
	let data: Vec<f32> = vec![1.0,2.0,3.0];
	let sig0seg1 = Segment::new(None,Duration::new(0,0),0,data.clone(),None);
	let sig0seg2 = Segment::new(None,Duration::new(15,0),0,data.clone(),Some(Duration::new(15,0)));
	let sig0seg3 = Segment::new(None,Duration::new(18,0),0,data.clone(),Some(Duration::new(3,0)));
	let sig0seg4 = Segment::new(None,Duration::new(34,0),0,data.clone(),Some(Duration::new(16,0)));

	let sig1seg1 = Segment::new(None,Duration::new(10,0),0,data.clone(),None);
	let sig1seg2 = Segment::new(None,Duration::new(15,0),0,data.clone(),Some(Duration::new(5,0)));
	let sig1seg3 = Segment::new(None,Duration::new(35,0),0,data.clone(),Some(Duration::new(20,0)));
	let sig1seg4 = Segment::new(None,Duration::new(135,0),0,data.clone(),Some(Duration::new(100,0)));

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
	let key = SegmentKey::new(Duration::new(5,0),0);

	if let Ok(bytes) = key.convert_to_bytes() {
		match SegmentKey::convert_from_bytes(&bytes) {
			Ok(new_key) => assert_eq!(key, new_key),
			_           => panic!("Failed to convert bytes into key"),
		} 
	} else {
		panic!("Failed to convert key into bytes");
	}
}
