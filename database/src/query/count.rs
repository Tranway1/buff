use std::time::SystemTime;
use crate::buffer_pool::SegmentBuffer;

pub struct Count {
    star_timestamp: SystemTime,
    end_timestamp: SystemTime,
}
impl Count {
    pub fn new(start: SystemTime, end: SystemTime) -> Count
    {
        Count{
            star_timestamp: start,
            end_timestamp: end,
        }
    }

    /*
    *   run query on whole signal.
    */

    fn run<T: Copy + Send>(signals: &SegmentBuffer<T>) -> u32{
        signals.copy().iter().map(|x|x.get_data().len()).sum()
    }

}