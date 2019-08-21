
use std::time::SystemTime;
use std::fmt;
use crate::buffer_pool::SegmentBuffer;
use crate::segment::Segment;
use num::{FromPrimitive, abs, Signed};
use std::ops::Div;
use std::ops::Add;
use num::Num;
use std::cmp::Ord;

/* will contain a struct containing all of statistics for each block*/


//
///* An enum holding every supported aggregation query
// */
//#[derive(Clone,Debug,Serialize,Deserialize, PartialEq)]
//pub enum Query {
//	Max,
//	Min,
//	Count,
//	Average,
//    Sum,
//}
//
//impl fmt::Display for Query {
//    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//        match self {
//            Query::Max => write!(f, "Max"),
//            Query::Min => write!(f, "{}", format!("Kernel w/ DictionaryId")),
//            Query::Count=> write!(f, "{}", format!("Sparse Learning w/ DictionaryId")),
//            Query::Average  => write!(f, "{}", format!("Deep Learning w/ file")),
//            Query::Sum  => write!(f, "{}", format!("Deep Learning w/ file "))
//        }
//    }
//}


pub struct Count {
    star_timestamp: Option<SystemTime>,
    end_timestamp: Option<SystemTime>,
}
impl Count {
    pub fn new(start: Option<SystemTime>, end: Option<SystemTime>) -> Count
    {
        Count{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    /*
    *   run query on whole signal.
    */

    pub fn run<T: Copy + Send + Num + Add>(signals: &SegmentBuffer<T>) -> usize{
        signals.copy().iter().map(|x|x.get_data().len()).sum()
    }

}


pub struct Min {
    star_timestamp: Option<SystemTime>,
    end_timestamp: Option<SystemTime>,
}


impl Min {
    pub fn new(start: Option<SystemTime>, end: Option<SystemTime>) -> Min
    {
        Min{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    /*
    *   run query on whole signal.
    */

    pub fn run<T: Num + Copy + Send + FromPrimitive + PartialOrd>(signals: &SegmentBuffer<T>) -> T {

        signals.copy().iter().map(|s| s.get_data().iter().fold(None, |min, x| match min {
            None => Some(x),
            Some(y) => Some(if x < y { x } else { y }),
        })).fold(None, |min, x| match min {
            None => Some(x),
            Some(y) => Some(if x < y { x } else { y }),
        }).unwrap().unwrap().to_owned()

    }
}

pub struct Max {
    star_timestamp: Option<SystemTime>,
    end_timestamp: Option<SystemTime>,
}


impl Max {
    pub fn new(start: Option<SystemTime>, end: Option<SystemTime>) -> Max
    {
        Max{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    /*
    *   run query on whole signal.
    */

    pub fn run<T: Num + Copy + Send + FromPrimitive + PartialOrd>(signals: &SegmentBuffer<T>) -> T {

        signals.copy().iter().map(|s| s.get_data().iter().fold(None, |min, x| match min {
            None => Some(x),
            Some(y) => Some(if x > y { x } else { y }),
        })).fold(None, |min, x| match min {
            None => Some(x),
            Some(y) => Some(if x > y { x } else { y }),
        }).unwrap().unwrap().to_owned()

    }
}

pub struct Sum {
    star_timestamp: Option<SystemTime>,
    end_timestamp: Option<SystemTime>,
}

impl Sum {
    pub fn new(start: Option<SystemTime>, end: Option<SystemTime>) -> Sum
    {
        Sum{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    /*
    *   run query on whole signal.
    */

    pub fn run<T: Num + Div + Copy + Send + Add<T, Output = T> + Signed + FromPrimitive>(signals: &SegmentBuffer<T>) -> T{
        let zero = T::zero();
        signals.copy().iter().map(|x|x.get_data().iter().fold(zero, |sum, &i| sum + i)).fold(zero, |sum, i| sum + i)
    }
}


pub struct Average {
    star_timestamp: Option<SystemTime>,
    end_timestamp: Option<SystemTime>,
}

impl Average {
    pub fn new(start: Option<SystemTime>, end: Option<SystemTime>) -> Average
    {
        Average{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    pub fn run< T:  Num + Div + Copy + Send + Add<T, Output = T> + Signed + FromPrimitive>(signals: &SegmentBuffer<T>) -> T{
        let sum  = Sum::run(signals);
        let count = Count::run(signals);
        (sum)/FromPrimitive::from_usize(count).unwrap()
    }
}