use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{SCALE, SnappyCompress, GZipCompress, BPDoubleCompress, TEST_FILE};
use crate::compress::gorilla::{GorillaCompress, GorillaBDCompress};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::time::{Instant, SystemTime};
use std::mem;
use crate::segment::Segment;
use crate::compress::sprintz::SprintzDoubleCompress;
use crate::compress::split_double::SplitBDDoubleCompress;
use crate::compress::buff_slice::BuffSliceCompress;

pub mod influx_bench;
pub mod tsbs;

pub const BENCH_DATA:&str = "/home/cc/float_comp/benchmark/";


fn read_a_file(path: &str) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(path).unwrap();

    let mut data = Vec::new();
    file.read_to_end(&mut data);

    return Ok(data);
}

fn write_file(path: &str, data: &[u8]) {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(path).unwrap();

    file.write_all(data);
}


fn prepare_benchmark_data(file:&str, comp:&str, scl:usize){
    let file_iter = construct_file_iterator_skip_newline::<f64>(file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap().collect();
    let mut compressed:Vec<u8> = Vec::new();
    let mut outfile = file.to_owned();
    let len= file_vec.len();
    let org_size = len * mem::size_of::<f64>() ;
    let start = Instant::now();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);

    match comp{
        "buff" => {
            outfile.push_str(".buff");
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            compressed = comp.byte_fixed_encode(&mut seg);
        },
        "buff-major" => {
            outfile.push_str(".buff-major");
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            compressed = comp.buff_encode_majority(&mut seg);
        },
        "gorilla" => {
            outfile.push_str(".gorilla");
            let comp = GorillaCompress::new(10,10);
            compressed = comp.encode(&mut seg);
        },
        "gorillabd" => {
            outfile.push_str(".gorillabd");
            let comp = GorillaBDCompress::new(10,10,scl);
            compressed = comp.encode(&mut seg);
        },

        "snappy" => {
            outfile.push_str(".snappy");
            let comp = SnappyCompress::new(10,10);
            compressed = comp.encode(&mut seg);
        },

        "gzip" => {
            outfile.push_str(".gzip");
            let comp = GZipCompress::new(10,10);
            compressed = comp.encode(&mut seg);
        },

        "fixed" => {
            outfile.push_str(".fixed");
            let comp = SplitBDDoubleCompress::new(10,10,scl);
            compressed = comp.fixed_encode(&mut seg);
        },
        "sprintz" => {
            outfile.push_str(".sprintz");
            let comp = SprintzDoubleCompress::new(10,10,scl);
            compressed = comp.encode(&mut seg);
        },
        "buff-slice" => {
            outfile.push_str(".buff-slice");
            let comp = BuffSliceCompress::new(10,10,scl);
            compressed= comp.buff_slice_encode(&mut seg);
        },
        _ => {panic!("Compression not supported yet.")}
    }
    let duration = start.elapsed();
    write_file(&outfile, &compressed);
    let comp_size = compressed.len();
    println!("\ncompression profiling:{:?},{},{},{}",file,len,comp_size as f64/ org_size as f64,1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0);

}

pub fn get_comp_file(file:&str, compression: &str) ->Vec<u8>
{
    let mut path = BENCH_DATA.to_owned();
    path.push_str(file);
    path.push_str(".");
    path.push_str(compression);
    let binary = read_a_file(&path).unwrap();
    binary
}

pub fn get_csv_file(f:&str) ->Vec<usize>
{
    let mut file = BENCH_DATA.to_owned();
    file.push_str(f);

    let file_iter = construct_file_iterator_skip_newline::<usize>(&file, 0, ',');
    let file_vec: Vec<usize> = file_iter.unwrap().collect();
    file_vec
}

#[test]
fn test_compresse_bench_data(){
    let mut encodings: [&str; 9] = ["buff", "buff-major","gorilla","gorillabd","snappy","gzip","fixed","sprintz","buff-slice"];
    let mut files: [&str; 1] =["dt_cur_load.csv"];
    let scl = 10000;
    let mut outfile = BENCH_DATA.clone().to_owned();
    for &f in files.iter(){
        for &enc in &encodings {
            outfile = BENCH_DATA.clone().to_owned();
            outfile.push_str(f);
            println!("compressing : {}.{}", outfile, enc);
            prepare_benchmark_data(&outfile,enc,scl);
        }
    }
}

