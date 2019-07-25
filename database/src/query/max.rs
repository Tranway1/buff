use std::time::SystemTime;
use crate::buffer_pool::SegmentBuffer;
use crate::segment::Segment;
use num::FromPrimitive;
use std::ops::Div;
use std::ops::Add;
use num::Num;
use num_traits::real::Real;

pub struct Max<T> {
    star_timestamp: SystemTime,
    end_timestamp: SystemTime,
}


impl<T> Max<T>{
    pub fn new(start: SystemTime, end: SystemTime) -> Max<T>
    {
        Max{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    /*
    *   run query on whole signal.
    */

    pub fn run<T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive>(signals: &SegmentBuffer<T>) -> T{
        signals.copy().iter().map(|x|x.get_data().max()).max()
    }

}