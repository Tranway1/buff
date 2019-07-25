use std::time::SystemTime;
use crate::buffer_pool::SegmentBuffer;
use crate::query::sum::Sum;
use crate::query::count::Count;
use crate::segment::Segment;
use num::FromPrimitive;
use std::ops::Div;
use std::ops::Add;
use num::Num;

pub struct Average<T> {
    star_timestamp: SystemTime,
    end_timestamp: SystemTime,
}

impl <T> Average<T> {
    pub fn new(start: SystemTime, end: SystemTime) -> Average<T>
    {
        Average{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    pub fn run< T: Num + Div + Copy + Add<T, Output = T> + FromPrimitive>(&self, signals: &SegmentBuffer<T>) -> T{
        let sum : T = Sum::new(self.star_timestamp, self.end_timestamp);
        let count = Count::new(self.star_timestamp, self.end_timestamp);
        (sum as T)/count

    }
}