mod kernel;
mod deep_learning;
mod fourier;

use kernel::KernelMethod;
use deep_learning::DeepLearningMethod;

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
#[derive(Clone)]
pub enum Methods {
	Fourier,
	Kernel,
	DeepLearning,
}

/* A struct to be held by each dictionary so it can keep track of 
 * what methods it supports as well as the parameters that the 
 * method will need.
 * Methods that are dictionary independent are not present.
 */
pub struct MethodHolder {
	kernel: Option<KernelMethod>,
	deep_learning: Option<DeepLearningMethod>
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