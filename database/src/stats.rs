use std::time::SystemTime;
use std::default::Default;
use num::Num;

#[derive(Clone,Serialize,Deserialize,Debug,PartialEq)]
pub struct Stats<T> {
    star_timestamp: SystemTime,
    end_timestamp: SystemTime,
    max: T,
    min: T,
    count: usize,
    avg: T,
    sum: T,
}


impl<T> Stats<T>{
    pub fn new(star_timestamp: SystemTime,
               end_timestamp: SystemTime,
               max: T,
               min: T,
               count: usize,
               avg: T,sum: T) -> Stats<T> {

        Stats {
            star_timestamp: star_timestamp,
            end_timestamp: end_timestamp,
            max: max,
            min: min,
            count: count,
            avg: avg,
            sum: sum
        }
    }

    pub fn get_interval(&self) -> (SystemTime, SystemTime) {
        (self.star_timestamp,self.end_timestamp)
    }

    pub fn get_max(&self) -> &T {
        &self.max
    }

    pub fn get_min(&self) -> &T {
        &self.min
    }

    pub fn get_avg(&self) -> &T {
        &self.avg
    }

    pub fn get_sum(&self) -> &T {
        &self.sum
    }

    pub fn get_count(&self) -> &usize {
        &self.count
    }
}

impl<T> Default for Stats<T>
    where T: Num + Default {
    fn default() -> Self {
        Stats{
            star_timestamp: SystemTime::now(),
            end_timestamp: SystemTime::now(),
            max: Default::default(),
            min: Default::default(),
            count: Default::default(),
            avg: Default::default(),
            sum: Default::default()
        }
    }
}