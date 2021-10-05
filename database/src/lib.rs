#[macro_use]
extern crate serde_derive;
extern crate bincode;
#[macro_use] extern crate futures;
extern crate toml_loader;
#[macro_use] extern crate queues;
#[macro_use] extern crate lazy_static;
#[macro_use] extern crate itertools;


#[cfg(test)]
extern crate quickcheck;
#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;


pub mod segment;
pub mod methods;
pub mod simd;
pub mod client;
mod query;
pub mod compress;
pub mod pscan;
pub mod outlier;
