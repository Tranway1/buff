use std::{env, fs};
use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{ SCALE, SplitDoubleCompress, test_split_compress_on_file, BPDoubleCompress, test_BP_double_compress_on_file, SprintzDoubleCompress, test_sprintz_double_compress_on_file, SplitBDDoubleCompress, test_splitbd_compress_on_file, GorillaBDCompress, test_grillabd_compress_on_file, GorillaCompress, test_grilla_compress_on_file, GZipCompress, SnappyCompress, PRED};
use std::time::{SystemTime, Instant};
use crate::segment::Segment;
use std::path::Path;
use std::rc::Rc;
use parquet::file::properties::WriterProperties;
use parquet::schema::parser::parse_message_type;
use parquet::basic::{Encoding, Compression};
use crate::methods::parquet::{DICTPAGE_LIM, USE_DICT};
use parquet::file::writer::{SerializedFileWriter, FileWriter};
use parquet::column::writer::ColumnWriter;
use std::fs::File;
use croaring::Bitmap;
use parquet::file::reader::{SerializedFileReader, FileReader};
use parquet::record::RowAccessor;
use parity_snappy::compress;

pub fn run_bpsplit_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq= compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in bpsplit compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in bpsplit decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in bpsplit range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in bpsplit equal filter function() is: {:?}", duration4);
    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}


pub fn run_bp_double_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64>= file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = BPDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in bp_double compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in bp_double decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in bp_double range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(&comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in bp_double equal filter function() is: {:?}", duration4);

    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_sprintz_double_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64>= file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SprintzDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in sprintz_double compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in sprintz_double decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in sprintz_double range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in sprintz equal filter function() is: {:?}", duration4);

    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_splitbd_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SplitBDDoubleCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.offset_encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in splitbd compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in splitbd decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in splitbd range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in splitbd equal filter function() is: {:?}", duration4);

    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_gorillabd_encoding_decoding(test_file:&str, scl:usize,pred: f64 ) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GorillaBDCompress::new(10,10,scl);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in gorillabd compress function() is: {:?}", duration1);
    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in gorillabd decompress function() is: {:?}", duration2);
    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in gorillabd range filter function() is: {:?}", duration3);
    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in gorillabd equal filter function() is: {:?}", duration4);
    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_gorilla_encoding_decoding(test_file:&str, scl:usize, pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GorillaCompress::new(10,10);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in gorilla compress function() is: {:?}", duration1);

    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in gorilla decompress function() is: {:?}", duration2);

    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in gorilla range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in gorilla equal filter function() is: {:?}", duration4);

    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_gzip_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = GZipCompress::new(10,10);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in gzip compress function() is: {:?}", duration1);
    //test_grilla_compress_on_file::<f64>(TEST_FILE);
    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in gzip decompress function() is: {:?}", duration2);
    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in gzip range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in gzip equal filter function() is: {:?}", duration4);

    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}

pub fn run_snappy_encoding_decoding(test_file:&str, scl:usize,pred: f64) {
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let mut seg = Segment::new(None,SystemTime::now(),0,file_vec.clone(),None,None);
    let org_size = seg.get_byte_size().unwrap();
    let comp = SnappyCompress::new(10,10);
    let start1 = Instant::now();
    let compressed = comp.encode(&mut seg);
    let duration1 = start1.elapsed();
    let comp_cp = compressed.clone();
    let comp_eq = compressed.clone();
    let comp_size = compressed.len();
    println!("Time elapsed in snappy compress function() is: {:?}", duration1);
    //test_grilla_compress_on_file::<f64>(TEST_FILE);
    let start2 = Instant::now();
    comp.decode(compressed);
    let duration2 = start2.elapsed();
    println!("Time elapsed in snappy decompress function() is: {:?}", duration2);
    let start3 = Instant::now();
    comp.range_filter(comp_cp,pred);
    let duration3 = start3.elapsed();
    println!("Time elapsed in snappy range filter function() is: {:?}", duration3);

    let start4 = Instant::now();
    comp.equal_filter(comp_eq,pred);
    let duration4 = start4.elapsed();
    println!("Time elapsed in snappy equal filter function() is: {:?}", duration4);

    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration4.as_nanos() as f64 / 1024.0/1024.0
    )
}


pub fn run_parquet_write_filter(test_file:&str, scl:usize,pred: f64, enc:&str){
    let path = Path::new("target/debug/examples/sample.parquet");
    let file_iter = construct_file_iterator_skip_newline::<f64>(test_file, 0, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();
    let org_size=file_vec.len()*8;
    let mut comp = Compression::UNCOMPRESSED;
    let mut dictpg_lim:usize = 20;
    let mut use_dict = false;
    match enc {
        "dict" => {
            use_dict = true;
            dictpg_lim = 200000000;
        },
        "plain" => {},
        "pqgzip" => {comp=Compression::GZIP},
        "pqsnappy" => {comp=Compression::SNAPPY},
        _ => {panic!("Compression not supported by parquet.")}
    }
    // profile encoding
    let start = Instant::now();
    let message_type = "
      message schema {
        REQUIRED DOUBLE b;
      }
    ";
    let schema = Rc::new(parse_message_type(message_type).unwrap());
    let props = Rc::new(WriterProperties::builder()
        .set_encoding(Encoding::PLAIN)
        .set_compression(comp)
        .set_dictionary_pagesize_limit(dictpg_lim) // change max page size to avoid fallback to plain and make sure dict is used.
        .set_dictionary_enabled(use_dict)
        .build());
    let file = fs::File::create(&path).unwrap();
    let mut writer = SerializedFileWriter::new(file, schema, props).unwrap();
    let mut row_group_writer = writer.next_row_group().unwrap();
    while let Some(mut col_writer) = row_group_writer.next_column().unwrap() {
        // ... write values to a column writer
        match col_writer {
            ColumnWriter::DoubleColumnWriter(ref mut typed) => {
                typed.write_batch(&file_vec, None, None).unwrap();
            }
            _ => {
                unimplemented!();
            }
        }
        row_group_writer.close_column(col_writer).unwrap();
    }
    writer.close_row_group(row_group_writer).unwrap();
    writer.close().unwrap();
    let duration = start.elapsed();
    println!("Time elapsed in parquet compress function() is: {:?}", duration);


    let bytes = fs::read(&path).unwrap();
    let comp_size= bytes.len();
    println!("file size: {}",comp_size);
    //println!("read: {:?}",str::from_utf8(&bytes[0..4]).unwrap());
    let file = File::open(&path).unwrap();

    // profile decoding
    let start1 = Instant::now();
    let mut expected_datapoints:Vec<f64> = Vec::new();
    let reader = SerializedFileReader::new(file).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    while let Some(record) = iter.next() {
        expected_datapoints.push( record.get_double(0).unwrap());
    }

    let duration1 = start1.elapsed();
    let num = expected_datapoints.len();
    println!("Time elapsed in parquet scan {} items is: {:?}",num, duration1);

    let file2 = File::open(&path).unwrap();

    // profile decoding
    let start2 = Instant::now();
    let reader = SerializedFileReader::new(file2).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    let mut i = 0;
    let mut res = Bitmap::create();
    while let Some(record) = iter.next() {
        if (record.get_double(0).unwrap()>pred){
            res.add(i);
        }
        i+=1;
    }
    res.run_optimize();
    let duration2 = start2.elapsed();
    println!("Number of qualified items:{}", res.cardinality());
    println!("Time elapsed in parquet filter is: {:?}",duration2);

    let file3 = File::open(&path).unwrap();

    // profile decoding
    let start3 = Instant::now();
    let reader = SerializedFileReader::new(file3).unwrap();
    let rg_meta = reader.metadata().row_group(0).clone();
    let colmeta =  rg_meta.column(0).encodings();
    println!("column encodings: {:?}", colmeta.as_slice());
    let mut iter = reader.get_row_iter(None).unwrap();
    let mut i = 0;
    let mut res = Bitmap::create();
    while let Some(record) = iter.next() {
        if (record.get_double(0).unwrap()==pred){
            res.add(i);
        }
        i+=1;
    }
    res.run_optimize();
    let duration3 = start3.elapsed();
    println!("Number of qualified items for equal:{}", res.cardinality());
    println!("Time elapsed in parquet filter is: {:?}",duration3);

    println!("Performance:{},{},{},{},{},{},{},{}", test_file, scl, pred,
             comp_size as f64/ org_size as f64,
             1000000000.0 * org_size as f64 / duration.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration1.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration2.as_nanos() as f64 / 1024.0/1024.0,
             1000000000.0 * org_size as f64 / duration3.as_nanos() as f64 / 1024.0/1024.0
    )
}