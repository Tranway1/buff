use crate::kernel::Kernel;
use crate::segment::Segment;
use ndarray::Array2;
extern crate flate2;
extern crate tsz;

use std::vec::Vec;
use tsz::{DataPoint, Encode, Decode, StdEncoder, StdDecoder};
use tsz::stream::{BufferedReader, BufferedWriter};
use tsz::decode::Error;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io;
use std::io::prelude::*;
use flate2::write::ZlibEncoder;
use serde::{Serialize, Deserialize};
use self::flate2::read::{ZlibDecoder, DeflateDecoder};
use self::flate2::write::DeflateEncoder;
use parity_snappy as snappy;
use parity_snappy::{compress, decompress};
use std::time::SystemTime;

pub trait CompressionMethod<T> {


    fn get_segments(&self);

	fn get_batch(&self) -> usize;

    fn run_compress<'a>(&self, segs: &mut Vec<Segment<T>>);

	fn run_decompress(&self, segs: &mut Vec<Segment<T>>);
}


#[derive(Clone)]
pub struct GZipCompress {
    chunksize: usize,
    batchsize: usize
}

impl GZipCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        GZipCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let mut e = GzEncoder::new(Vec::new(), Compression::fast());
        e.write_all(seg.convert_to_bytes().unwrap().as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a Gz Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        let mut gz = GzDecoder::new(&bytes[..]);
        let mut s = String::new();
        gz.read_to_string(&mut s)?;
        Ok(s)
    }
}

impl<'a, T> CompressionMethod<T> for GZipCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}


#[derive(Clone)]
pub struct DeflateCompress {
    chunksize: usize,
    batchsize: usize
}

impl DeflateCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        DeflateCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let mut e = DeflateEncoder::new(Vec::new(), Compression::fast());
        e.write_all(seg.convert_to_bytes().unwrap().as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a deflte Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        let mut df = DeflateDecoder::new(&bytes[..]);
        let mut s = String::new();
        df.read_to_string(&mut s)?;
        Ok(s)
    }
}

impl<'a, T> CompressionMethod<T> for DeflateCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}



#[derive(Clone)]
pub struct ZlibCompress {
    chunksize: usize,
    batchsize: usize
}

impl ZlibCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        ZlibCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let mut e = ZlibEncoder::new(Vec::new(), Compression::fast());
        e.write_all(seg.convert_to_bytes().unwrap().as_slice()).unwrap();
        let bytes = e.finish().unwrap();
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a zl Encoded vector of bytes and returns a string or error
    // todo: fix later for decompression
    fn decode_reader(bytes: Vec<u8>) -> io::Result<String> {
        let mut zl = ZlibDecoder::new(&bytes[..]);
        let mut s = String::new();
        zl.read_to_string(&mut s)?;
        Ok(s)
    }
}

impl<'a, T> CompressionMethod<T> for ZlibCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}




#[derive(Clone)]
pub struct SnappyCompress {
    chunksize: usize,
    batchsize: usize
}

impl SnappyCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        SnappyCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>)
        where T: Serialize + Deserialize<'a>{
        let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a snappy Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode_reader(bytes: Vec<u8>) -> Vec<u8> {
        let mut snappy = decompress(bytes.as_slice());
        let mut s = snappy.unwrap();
        s
    }
}

impl<'a, T> CompressionMethod<T> for SnappyCompress
    where T: Serialize + Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}

#[derive(Clone)]
pub struct GorillaCompress {
    chunksize: usize,
    batchsize: usize
}

impl GorillaCompress {
    pub fn new(chunksize: usize, batchsize: usize) -> Self {
        GorillaCompress { chunksize, batchsize }
    }

    // Compress a sample string and print it after transformation.
    fn encode<'a,T>(&self, seg: &mut Segment<T>) -> Vec<u8>
        where T: Serialize + Clone+ Copy+Into<f64> + Deserialize<'a>{

        let w = BufferedWriter::new();

        // 1482892260 is the Unix timestamp of the start of the stream
        let mut encoder = StdEncoder::new(0, w);

        let mut actual_datapoints = Vec::new();

        let mut t =0;
        for val in seg.get_data(){
            let v = (*val).into();
            let dp = DataPoint::new(t, v);
            t=t+1;
            actual_datapoints.push(dp);
        }
        for dp in &actual_datapoints {
            encoder.encode(*dp);
        }
        let bytes = encoder.close();
        bytes.to_vec()

        //let bytes = compress(seg.convert_to_bytes().unwrap().as_slice());
        //println!("{}", decode_reader(bytes).unwrap());
    }

    // Uncompresses a snappy Encoded vector of bytes and returns a string or error
    // Here &[u8] implements BufRead
    fn decode(&self, bytes: Vec<u8>) -> Vec<DataPoint> {
        let r = BufferedReader::new(bytes.into_boxed_slice());
        let mut decoder = StdDecoder::new(r);

        let mut expected_datapoints:Vec<DataPoint> = Vec::new();

        let mut done = false;
        loop {
            if done {
                break;
            }

            match decoder.next() {
                Ok(dp) => expected_datapoints.push(dp),
                Err(err) => {
                    if err == Error::EndOfStream {
                        done = true;
                    } else {
                        panic!("Received an error from decoder: {:?}", err);
                    }
                }
            };
        }
        expected_datapoints
    }

}

impl<'a, T> CompressionMethod<T> for GorillaCompress
    where T: Serialize + Clone+ Copy+Into<f64>+ Deserialize<'a>{
    fn get_segments(&self) {
        unimplemented!()
    }

    fn get_batch(&self) -> usize {
        self.batchsize
    }

    fn run_compress<'b>(&self, segs: &mut Vec<Segment<T>>) {
        for seg in segs {
            self.encode(seg);
        }
    }

    fn run_decompress(&self, segs: &mut Vec<Segment<T>>) {
        unimplemented!()
    }
}


#[test]
fn test_zlib_compress() {
	let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
	e.write_all(b"foo");
	e.write_all(b"bar");
	let compressed_bytes = e.finish();
}

#[test]
fn test_Grilla_seg_compress() {
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
    let mut seg1 = Segment::new(None,SystemTime::now(),0,data.clone(),None,None);
    let comp = GorillaCompress::new(10,10);
    let compressed = comp.encode(&mut seg1);
    let decompress = comp.decode(compressed);
    println!("expected datapoints: {:?}", decompress);

}

#[test]
fn test_Grilla_compress() {
    const DATA: &'static str = "1482892270,1.76
1482892280,7.78
1482892288,7.95
1482892292,5.53
1482892310,4.41
1482892323,5.30
1482892334,5.30
1482892341,2.92
1482892350,0.73
1482892360,-1.33
1482892370,-1.78
1482892390,-12.45
1482892401,-34.76
1482892490,78.9
1482892500,335.67
1482892800,12908.12
";
    let w = BufferedWriter::new();

    // 1482892260 is the Unix timestamp of the start of the stream
    let mut encoder = StdEncoder::new(1482892260, w);

    let mut actual_datapoints = Vec::new();

    for line in DATA.lines() {
        let substrings: Vec<&str> = line.split(",").collect();
        let t = substrings[0].parse::<u64>().unwrap();
        let v = substrings[1].parse::<f64>().unwrap();
        let dp = DataPoint::new(t, v);
        actual_datapoints.push(dp);
    }

    for dp in &actual_datapoints {
        encoder.encode(*dp);
    }

    let bytes = encoder.close();
    let r = BufferedReader::new(bytes);
    let mut decoder = StdDecoder::new(r);

    let mut expected_datapoints = Vec::new();

    let mut done = false;
    loop {
        if done {
            break;
        }

        match decoder.next() {
            Ok(dp) => expected_datapoints.push(dp),
            Err(err) => {
                if err == Error::EndOfStream {
                    done = true;
                } else {
                    panic!("Received an error from decoder: {:?}", err);
                }
            }
        };
    }

    println!("actual datapoints: {:?}", actual_datapoints);
    println!("expected datapoints: {:?}", expected_datapoints);
}

