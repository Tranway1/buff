use std::mem;


use tsz::stream::Write;
use tsz::{DataPoint, Bit};

use tsz::stream::Read;
use tsz::decode::{Decode, Error};
use crate::methods::gorilla_encoder::{END_MARKER, END_MARKER_LEN};


// this is the code revised from tsz gorilla library.
//      by Chunwei Liu

pub trait SepDecode {
    fn next_val(&mut self) -> Result<f64, Error>;
    fn next_stp(&mut self) -> Result<u64, Error>;
}

/// GorillaDecoder
///
/// GorillaDecoder is used to decode `DataPoint`s
#[derive(Debug)]
pub struct GorillaDecoder<T: Read> {
    time: u64, // current time
    delta: u64, // current time delta
    value_bits: u64, // current float value as bits
    xor: u64, // current xor

    leading_zeroes: u32, // leading zeroes
    trailing_zeroes: u32, // trailing zeroes

    first: bool, // will next DataPoint be the first DataPoint decoded
    done: bool,

    r: T,
}

impl<T> GorillaDecoder<T>
    where T: Read
{
    /// new creates a new GorillaDecoder which will read bytes from r
    pub fn new(r: T) -> Self {
        GorillaDecoder {
            time: 0,
            delta: 0,
            value_bits: 0,
            xor: 0,
            leading_zeroes: 0,
            trailing_zeroes: 0,
            first: true,
            done: false,
            r: r,
        }
    }

    fn read_initial_timestamp(&mut self) -> Result<u64, Error> {
        self.r
            .read_bits(64)
            .map_err(|_| Error::InvalidInitialTimestamp)
            .map(|time| {
                self.time = time;
                //println!("start timestamp:{}, ",time);
                time
            })
    }

    fn read_first_timestamp(&mut self) -> Result<u64, Error> {
        self.read_initial_timestamp()?;

        // sanity check to confirm that the stream contains more than just the initial timestamp
        let control_bit = self.r.peak_bits(1)?;
        if control_bit == 1 {
            return self.r
                .read_bits(END_MARKER_LEN)
                .map_err(|err| Error::Stream(err))
                .and_then(|marker| if marker == END_MARKER {
                    Err(Error::EndOfStream)
                } else {
                    Err(Error::InvalidEndOfStream)
                });
        }

        // stream contains datapoints so we can throw away the control bit
        self.r.read_bit()?;

        self.r
            .read_bits(14)
            .map(|delta| {
                self.delta = delta;
                self.time += delta;
            })?;

        Ok(self.time)
    }

    fn read_next_timestamp(&mut self) -> Result<u64, Error> {
        let mut control_bits = 0;
        for _ in 0..4 {
            let bit = self.r.read_bit()?;

            if bit == Bit::One {
                control_bits += 1;
            } else {
                break;
            }
        }

        let size = match control_bits {
            0 => {
                self.time += self.delta;
                return Ok(self.time);
            }
            1 => 7,
            2 => 9,
            3 => 12,
            4 => {
                return self.r
                    .read_bits(32)
                    .map_err(|err| Error::Stream(err))
                    .and_then(|dod| if dod == 0 {
                        Err(Error::EndOfStream)
                    } else {
                        Ok(dod)
                    });
            }
            _ => unreachable!(),
        };

        let mut dod = self.r.read_bits(size)?;

        // need to sign extend negative numbers
        if dod > (1 << (size - 1)) as u64 {
            let mask = u64::max_value() << size as u64;
            dod = dod | mask;
        }

        // by performing a wrapping_add we can ensure that negative numbers will be handled correctly
        self.delta = self.delta.wrapping_add(dod);
        self.time = self.time.wrapping_add(self.delta);

        Ok(self.time)
    }

    fn read_first_value(&mut self) -> Result<u64, Error> {
        self.read_initial_timestamp()?;

        // Check if it reach the end.
        let control_bit = self.r.peak_bits(1)?;
        if control_bit == 1 {
            return self.r
                .read_bits(END_MARKER_LEN)
                .map_err(|err| Error::Stream(err))
                .and_then(|marker| if marker == END_MARKER {
                    Err(Error::EndOfStream)
                } else {
                    Err(Error::InvalidEndOfStream)
                });
        }
        self.r.read_bit()?;

        self.r
            .read_bits(64)
            .map_err(|err| Error::Stream(err))
            .map(|bits| {
                self.value_bits = bits;
//                let value = unsafe { mem::transmute::<u64, f64>(bits) };
//                println!("{}, ",value);
                self.value_bits
            })
    }

    fn read_next_value(&mut self) -> Result<u64, Error> {


        let contol_bit = self.r.read_bit()?;

        if contol_bit == Bit::Zero {
//            let value = unsafe { mem::transmute::<u64, f64>(self.value_bits) };
//            println!("{}, ",value);
            return Ok(self.value_bits);
        }

        let zeroes_bit = self.r.read_bit()?;

        if zeroes_bit == Bit::One {
            // check the end
            let try_control_bit = self.r.peak_bits(2)?;
            if try_control_bit == 3 {
                let ind = self.r.peak_bits(END_MARKER_LEN-2)?;
                if ind == 12884901888{
                    return Err(Error::EndOfStream);
                }
            }
            self.leading_zeroes = self.r.read_bits(6).map(|n| n as u32)?;
            let significant_digits = self.r.read_bits(6).map(|n| (n + 1) as u32)?;
            self.trailing_zeroes = 64 - self.leading_zeroes - significant_digits;
        }

        let size = 64 - self.leading_zeroes - self.trailing_zeroes;
        self.r
            .read_bits(size)
            .map_err(|err| Error::Stream(err))
            .map(|bits| {
                self.value_bits ^= bits << self.trailing_zeroes as u64;
                //println!("{}, ",unsafe { mem::transmute::<u64, f64>(self.value_bits) });
                self.value_bits
            })
    }
}

impl<T> SepDecode for GorillaDecoder<T>
    where T: Read
{
    fn next_val(&mut self) -> Result<f64, Error> {
        if self.done {
            return Err(Error::EndOfStream);
        }
        let value_bits;

        if self.first {
            self.first = false;
            value_bits = self.read_first_value()
                .map_err(|err| {
                    if err == Error::EndOfStream {
                        self.done = true;
                    }
                    err
                })?;;
        } else {
            value_bits = self.read_next_value()
                .map_err(|err| {
                    if err == Error::EndOfStream {
                        self.done = true;
                    }
                    err
                })?;;
        }

        let value = unsafe { mem::transmute::<u64, f64>(value_bits) };

        Ok(value)
    }

    fn next_stp(&mut self) -> Result<u64, Error> {
        if self.done {
            return Err(Error::EndOfStream);
        }

        let time;

        if self.first {
            self.first = false;
            time = self.read_first_timestamp()
                .map_err(|err| {
                    if err == Error::EndOfStream {
                        self.done = true;
                    }
                    err
                })?;;
        } else {
            time = self.read_next_timestamp()
                .map_err(|err| {
                    if err == Error::EndOfStream {
                        self.done = true;
                    }
                    err
                })?;;
        }

        Ok(time)
    }
}
