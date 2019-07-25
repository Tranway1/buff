use std::time::SystemTime;
use crate::buffer_pool::SegmentBuffer;
use crate::segment::Segment;
use num::FromPrimitive;
use std::ops::Div;
use std::ops::Add;
use num::Num;

pub struct Sum<T> {
    star_timestamp: SystemTime,
    end_timestamp: SystemTime,
}

impl <T> Sum<T> {
    pub fn new(start: SystemTime, end: SystemTime) -> Sum<T>
    {
        Sum{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    /*
    *   run query on whole signal.
    */

    pub fn run<T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive>(signals: &SegmentBuffer<T>) -> T{
        let zero = U::zero();
        signals.copy().iter().map(|x|x.get_data().iter().fold(zero, |sum, &i| sum + i)).sum()
    }
}