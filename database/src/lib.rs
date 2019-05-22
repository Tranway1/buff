#[macro_use]
extern crate serde_derive;
extern crate bincode;
#[macro_use]
extern crate futures;

mod buffer_pool;
mod dictionary;
mod file_handler;
mod segment;
mod methods;
mod future_signal;
mod client;
mod fake_client;