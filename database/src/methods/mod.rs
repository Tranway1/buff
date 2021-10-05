pub mod compress;
pub mod bit_packing;
pub mod prec_double;
use std::fmt;


pub type DictionaryId = u32; /* Type alias for dictionary id */
/*
 * Overview:
 * This is the API for compression methods using
 * The goal is to construct a general framework that
 * each compression method must implement. Furthermore,
 * this file acts as a hub for each of the compression methods.
 *
 * Design Choice:
 * The two main goals were to construct
 * 1. A struct to hold all of the compression methods. Where
 *    each method is a struct wrapped in an option. The method
 *    struct should contain everything, besides the dictionary,
 *    needed to compress.
 *	  Note: This implementation unless optimized (not sure) will be
 *    space expensive. I assumed the number of dictionaries as well
 *    as the parameters in each method would be relatively small.
 *    Or that each dicitonary is heavily implemented. Making
 *    an unoptimized space usage probably acceptable.
 * 2. An enum to contain every compression method fully implemented
 *    so that Segments can indicate which compression method should
 *    be used.
 *
 * Current Implementations:
 * I have files for each of the three listed however,
 * they have not been implemented or heavily explored.
 */


/* An enum holding every supported compression format
 * Used to indicate what method of compression should be used
 */
#[derive(Clone,Debug,Serialize,Deserialize, PartialEq)]
pub enum Methods {
    Fourier,
    Kernel (DictionaryId),
    SparseLearning (DictionaryId),
    DeepLearning (String),
}

impl fmt::Display for Methods {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Methods::Fourier => write!(f, "Fourier"),
            Methods::Kernel (id) => write!(f, "{}", format!("Kernel w/ DictionaryId {:?}", id)),
            Methods::SparseLearning (id) => write!(f, "{}", format!("Sparse Learning w/ DictionaryId {:?}", id)),
            Methods::DeepLearning (file) => write!(f, "{}", format!("Deep Learning w/ file {:?}", file))
        }
    }
}

/* Structures to be fleshed out later to provide error information
 * when methods are not properly implemented/used
 */
pub struct MethodUsageError {
    err_msg: &'static str,
    err_num: u8,
}

pub struct MethodImplementError {
    err_msg: &'static str,
    err_num: u8,
}