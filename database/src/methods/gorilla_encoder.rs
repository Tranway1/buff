use std::mem;


use tsz::stream::Write;
use tsz::{Bit};

// this is the code revised from tsz gorilla library.
//      by Chunwei Liu

pub trait SepEncode {
    fn encode_float(&mut self, dp: f64);
    fn encode_integer(&mut self, dp: u64);
    fn close(self) -> Box<[u8]>;
}

// END_MARKER relies on the fact that when we encode the delta of delta for a number that requires
// more than 12 bits we write four control bits 1111 followed by the 32 bits of the value. Since
// encoding assumes the value is greater than 12 bits, we can store the value 0 to signal the end
// of the stream

/// END_MARKER is a special bit sequence used to indicate the end of the stream
pub const END_MARKER: u64 = 0b111100000000000000000000000000000000;

/// END_MARKER_LEN is the length, in bits, of END_MARKER
pub const END_MARKER_LEN: u32 = 36;

/// GorillaEncoder
///
/// GorillaEncoder is used to encode `DataPoint`s
#[derive(Debug)]
pub struct GorillaEncoder<T: Write> {
    time: u64, // current time
    delta: u64, // current time delta
    value_bits: u64, // current float value as bits

    // store the number of leading and trailing zeroes in the current xor as u32 so we
    // don't have to do any conversions after calling `leading_zeros` and `trailing_zeros`
    leading_zeroes: u32,
    trailing_zeroes: u32,

    first: bool, // will next DataPoint be the first DataPoint encoded

    w: T,
}

impl<T> GorillaEncoder<T>
    where T: Write
{
    /// new creates a new GorillaEncoder whose starting timestamp is `start` and writes its encoded
    /// bytes to `w`
    pub fn new(start: u64, w: T) -> Self {
        let mut e = GorillaEncoder {
            time: start,
            delta: 0,
            value_bits: 0,
            leading_zeroes: 64, // 64 is an initial sentinel value
            trailing_zeroes: 64, // 64 is an intitial sentinel value
            first: true,
            w: w,
        };

        // write timestamp header
        e.w.write_bits(start, 64);

        e
    }

    fn write_first_int(&mut self, time: u64) {
        self.delta = time - self.time;
        self.time = time;

        // write one control bit so we can distinguish a stream which contains only an initial
        // timestamp, this assumes the first bit of the END_MARKER is 1
        self.w.write_bit(Bit::Zero);

        // store the first delta with 14 bits which is enough to span just over 4 hours
        // if one wanted to use a window larger than 4 hours this size would increase
        self.w.write_bits(self.delta, 14);

        self.first = true
    }

    fn write_first_float(&mut self, value_bits: u64) {
        self.value_bits = value_bits;

        // write one control bit so we can distinguish a stream which contains only an initial
        // timestamp, this assumes the first bit of the END_MARKER is 1
        self.w.write_bit(Bit::Zero);
        // store the first value exactly
        self.w.write_bits(self.value_bits, 64);
        self.first = true
    }

    fn write_next_timestamp(&mut self, time: u64) {
        let delta = time - self.time; // current delta
        let dod = delta.wrapping_sub(self.delta) as i32; // delta of delta
        // store the delta of delta using variable length encoding
        match dod {
            0 => {
                self.w.write_bit(Bit::Zero);
            }
            -63...64 => {
                self.w.write_bits(0b10, 2);
                self.w.write_bits(dod as u64, 7);
            }
            -255...256 => {
                self.w.write_bits(0b110, 3);
                self.w.write_bits(dod as u64, 9);
            }
            -2047...2048 => {
                self.w.write_bits(0b1110, 4);
                self.w.write_bits(dod as u64, 12);
            }
            _ => {
                self.w.write_bits(0b1111, 4);
                self.w.write_bits(dod as u64, 32);
            }
        }

        self.delta = delta;
        self.time = time;
    }

    fn write_next_value(&mut self, value_bits: u64) {
        let xor = value_bits ^ self.value_bits;
        self.value_bits = value_bits;

        if xor == 0 {
            // if xor with previous value is zero just store single zero bit
            self.w.write_bit(Bit::Zero);
        } else {
            self.w.write_bit(Bit::One);

            let leading_zeroes = xor.leading_zeros();
            let trailing_zeroes = xor.trailing_zeros();

            // if (leading_zeroes + trailing_zeroes) >= (self.leading_zeroes + self.trailing_zeroes+16){
            //
            //
            //     // if the number of leading and trailing zeroes in this xor are not less than the
            //     // leading and trailing zeroes in the previous xor then we store a control bit and
            //     // use 6 bits to store the number of leading zeroes and 6 bits to store the number
            //     // of significant digits before storing the significant digits themselves
            //
            //     self.w.write_bit(Bit::One);
            //     self.w.write_bits(leading_zeroes as u64, 6);
            //
            //     // if significant_digits is 64 we cannot encode it using 6 bits, however since
            //     // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
            //     // significant_digits can always be expressed with 6 bits or less
            //     let significant_digits = 64 - leading_zeroes - trailing_zeroes;
            //     self.w.write_bits((significant_digits - 1) as u64, 6);
            //     self.w.write_bits(xor.wrapping_shr(trailing_zeroes), significant_digits);
            //
            //     // finally we need to update the number of leading and trailing zeroes
            //     self.leading_zeroes = leading_zeroes;
            //     self.trailing_zeroes = trailing_zeroes;
            // }
            // else
            if leading_zeroes >= self.leading_zeroes && trailing_zeroes >= self.trailing_zeroes {
                // if the number of leading and trailing zeroes in this xor are >= the leading and
                // trailing zeroes in the previous xor then we only need to store a control bit and
                // the significant digits of this xor
                self.w.write_bit(Bit::Zero);
                self.w.write_bits(xor.wrapping_shr(self.trailing_zeroes),
                                  64 - self.leading_zeroes - self.trailing_zeroes);
            } else {

                // if the number of leading and trailing zeroes in this xor are not less than the
                // leading and trailing zeroes in the previous xor then we store a control bit and
                // use 6 bits to store the number of leading zeroes and 6 bits to store the number
                // of significant digits before storing the significant digits themselves

                self.w.write_bit(Bit::One);
                self.w.write_bits(leading_zeroes as u64, 6);

                // if significant_digits is 64 we cannot encode it using 6 bits, however since
                // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
                // significant_digits can always be expressed with 6 bits or less
                let significant_digits = 64 - leading_zeroes - trailing_zeroes;
                self.w.write_bits((significant_digits - 1) as u64, 6);
                self.w.write_bits(xor.wrapping_shr(trailing_zeroes), significant_digits);

                // finally we need to update the number of leading and trailing zeroes
                self.leading_zeroes = leading_zeroes;
                self.trailing_zeroes = trailing_zeroes;
                //todo: add logics to avoid 0 leading and 0 trailing cases.
            }

        }
    }

    fn write_first_float_with_print(&mut self, value_bits: u64) {
        self.value_bits = value_bits;

        // write one control bit so we can distinguish a stream which contains only an initial
        // timestamp, this assumes the first bit of the END_MARKER is 1
        self.w.write_bit(Bit::Zero);
        println!("write first value start with label:{:?}", Bit::Zero);

        // store the first value exactly
        self.w.write_bits(self.value_bits, 64);
        println!("write first value:{:b}", self.value_bits);
        self.first = true
    }

    fn write_next_value_with_print(&mut self, value_bits: u64) {
        let xor = value_bits ^ self.value_bits;
        self.value_bits = value_bits;

        if xor == 0 {
            // if xor with previous value is zero just store single zero bit
            self.w.write_bit(Bit::Zero);
            println!("write value start with label:{:?}", Bit::Zero);
        } else {
            self.w.write_bit(Bit::One);
            println!("write value start with label:{:?}", Bit::One);

            let leading_zeroes = xor.leading_zeros();
            let trailing_zeroes = xor.trailing_zeros();

            // if (leading_zeroes + trailing_zeroes) >= (self.leading_zeroes + self.trailing_zeroes+16){
            //
            //
            //     // if the number of leading and trailing zeroes in this xor are not less than the
            //     // leading and trailing zeroes in the previous xor then we store a control bit and
            //     // use 6 bits to store the number of leading zeroes and 6 bits to store the number
            //     // of significant digits before storing the significant digits themselves
            //
            //     self.w.write_bit(Bit::One);
            //     self.w.write_bits(leading_zeroes as u64, 6);
            //
            //     // if significant_digits is 64 we cannot encode it using 6 bits, however since
            //     // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
            //     // significant_digits can always be expressed with 6 bits or less
            //     let significant_digits = 64 - leading_zeroes - trailing_zeroes;
            //     self.w.write_bits((significant_digits - 1) as u64, 6);
            //     self.w.write_bits(xor.wrapping_shr(trailing_zeroes), significant_digits);
            //
            //     // finally we need to update the number of leading and trailing zeroes
            //     self.leading_zeroes = leading_zeroes;
            //     self.trailing_zeroes = trailing_zeroes;
            // }
            // else
            // println!("leading_zeroes:{}, trailling zeros:{}; self.leading_zeroes:{}, self.trailling zeros:{}", leading_zeroes,trailing_zeroes,self.leading_zeroes, self.trailing_zeroes);
            if leading_zeroes >= self.leading_zeroes && trailing_zeroes >= self.trailing_zeroes {
                // if the number of leading and trailing zeroes in this xor are >= the leading and
                // trailing zeroes in the previous xor then we only need to store a control bit and
                // the significant digits of this xor
                self.w.write_bit(Bit::Zero);
                // println!("{:?}", Bit::Zero);
                let diff =xor.wrapping_shr(self.trailing_zeroes);
                // println!("diff part: {:b} with length:{}", diff, 64 - self.leading_zeroes - self.trailing_zeroes);
                self.w.write_bits(diff, 64 - self.leading_zeroes - self.trailing_zeroes);
                // self.w.write_bits(xor.wrapping_shr(self.trailing_zeroes),
                //                   64 - self.leading_zeroes - self.trailing_zeroes);
            } else {

                // if the number of leading and trailing zeroes in this xor are not less than the
                // leading and trailing zeroes in the previous xor then we store a control bit and
                // use 6 bits to store the number of leading zeroes and 6 bits to store the number
                // of significant digits before storing the significant digits themselves

                self.w.write_bit(Bit::One);
                self.w.write_bits(leading_zeroes as u64, 6);
                println!("{:?}", Bit::One);
                println!("leading 0s: {}", leading_zeroes);
                // if significant_digits is 64 we cannot encode it using 6 bits, however since
                // significant_digits is guaranteed to be at least 1 we can subtract 1 to ensure
                // significant_digits can always be expressed with 6 bits or less
                let significant_digits = 64 - leading_zeroes - trailing_zeroes;
                self.w.write_bits((significant_digits - 1) as u64, 6);
                let diff = xor.wrapping_shr(trailing_zeroes);
                self.w.write_bits(diff, significant_digits);
                // self.w.write_bits(xor.wrapping_shr(trailing_zeroes), significant_digits);
                println!("significant_digits length -1: {:?}", (significant_digits - 1));
                println!("diff part: {:b} with length:{}", diff, significant_digits);
                // finally we need to update the number of leading and trailing zeroes
                self.leading_zeroes = leading_zeroes;
                self.trailing_zeroes = trailing_zeroes;
            }

        }
    }
}

impl<T> SepEncode for GorillaEncoder<T>
    where T: Write
{
    fn encode_float(&mut self, val: f64) {
        let value_bits = unsafe { mem::transmute::<f64, u64>(val) };

        if self.first {
            self.write_first_float(value_bits);
            self.first = false;
            return;
        }
        self.write_next_value(value_bits)
    }

    fn encode_integer(&mut self, ts: u64) {
        if self.first {
            self.write_first_int(ts);
            self.first = false;
            return;
        }

        self.write_next_timestamp(ts);
    }

    fn close(mut self) -> Box<[u8]> {
        self.w.write_bits(END_MARKER, 36);
        self.w.close()
    }
}