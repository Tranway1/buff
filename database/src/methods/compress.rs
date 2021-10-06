use crate::segment::{Segment, PAACompress, FourierCompress, fourier_compress, paa_compress};

extern crate flate2;
extern crate tsz;
use log::{info, trace, warn};

extern crate bitpacking;
use bitpacking::{BitPacker4x, BitPacker};
use std::vec::Vec;
use croaring::Bitmap;
use tsz::{DataPoint,StdEncoder, StdDecoder};
use tsz::stream::{BufferedReader, BufferedWriter};
use tsz::decode::Error;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::{io, env, mem};
use std::io::prelude::*;
use flate2::write::ZlibEncoder;
use serde::{Serialize, Deserialize};
use self::flate2::read::{ZlibDecoder, DeflateDecoder};
use self::flate2::write::DeflateEncoder;
use parity_snappy as snappy;
use parity_snappy::{compress, decompress};
use std::time::{SystemTime, Instant};
use crate::client::{construct_file_client_skip_newline, construct_file_iterator_skip_newline, construct_file_iterator_int, construct_file_iterator_int_signed};
use self::bitpacking::BitPacker1x;
use std::str::FromStr;
use num::{FromPrimitive, Num, Float};
use rustfft::FFTnum;
use std::hash::Hash;
use std::borrow::Borrow;
use crate::methods::prec_double::{PrecisionBound, FIRST_ONE, get_precision_bound};
use std::any::Any;
use std::collections::HashMap;
use crate::compress::split_double::SplitBDDoubleCompress;

use std::path::Path;
use std::fmt::Debug;
use crate::compress::PRECISION_MAP;
use self::tsz::{Encode, Decode};
use std::slice::Iter;
use my_bit_vec::BitVec;

pub const TYPE_LEN:usize = 8usize;
pub const SCALE: f64 = 1.0f64;
pub const PRED: f64 = 9.15f64;
pub const PRECISION:i32 = 100000;
pub const PREC_DELTA:f64 = 0.000005f64;
// pub const TEST_FILE:&str = "../taxi/dropoff_latitude-fulltaxi-1k.csv";
pub const TEST_FILE:&str = "../UCRArchive2018/Kernel/randomwalkdatasample1k-40k";


pub trait CompressionMethod<T> {


    fn get_segments(&self);

	fn get_batch(&self) -> usize;

    fn run_compress<'a>(&self, segs: &mut Vec<Segment<T>>);

	fn run_decompress(&self, segs: &mut Vec<Segment<T>>);
}
