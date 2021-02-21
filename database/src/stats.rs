use std::time::SystemTime;
use std::default::Default;
use num::Num;
use crate::compress::btr_array::{f_max, f_min};

#[derive(Clone,Serialize,Deserialize,Debug,PartialEq)]
pub struct Stats<T> {
    star_timestamp: i64,
    end_timestamp: i64,
    max: T,
    min: T,
    count: usize,
    avg: T,
    sum: T,
}


impl<T> Stats<T>{
    pub fn new(star_timestamp: i64,
               end_timestamp: i64,
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

    pub fn get_interval(&self) -> (i64, i64) {
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
            star_timestamp: 0i64,
            end_timestamp: 1i64,
            max: Default::default(),
            min: Default::default(),
            count: Default::default(),
            avg: Default::default(),
            sum: Default::default()
        }
    }
}

pub fn merge_adjacent(a: & Stats<f64>,b: & Stats<f64>) -> Stats<f64>{
    Stats::new(a.star_timestamp,b.end_timestamp,f_max(a.max,b.max),f_min(a.min,b.min),a.count+b.count,(a.sum+b.sum)/((a.count+b.count) as f64),a.sum+b.sum)
}