#[macro_use]
extern crate serde_derive;
extern crate bincode;

mod buffer_pool;
mod dictionary;
mod file_handler;
mod segment;
mod signal;
mod methods;