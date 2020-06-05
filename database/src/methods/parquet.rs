use std::fs;
use std::path::Path;
use std::rc::Rc;
use std::str;


use croaring::Bitmap;
use parquet::file::properties::WriterProperties;
use parquet::file::writer::{FileWriter, SerializedFileWriter};
use parquet::schema::parser::parse_message_type;
use parquet::column::writer::ColumnWriter;
use parquet::basic::{Encoding, Compression};
use crate::client::construct_file_iterator_skip_newline;
use crate::methods::compress::{TEST_FILE, SCALE, PRED};
use std::fs::File;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::borrow::BorrowMut;
use std::time::Instant;
use parquet::record::RowAccessor;

pub const USE_DICT: bool = false;
pub const DICTPAGE_LIM: usize = 2000000;
#[test]
fn test_parquet_write_scan(){
    let path = Path::new("target/debug/examples/sample.parquet");
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

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
        .set_compression(Compression::GZIP)
        .set_dictionary_pagesize_limit(DICTPAGE_LIM) // change max page size to avoid fallback to plain and make sure dict is used.
        .set_dictionary_enabled(USE_DICT)
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
    println!("file size: {}",bytes.len());
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
    let mut i = 0;
    while let Some(record) = iter.next() {
        // if (i<10){
        //     println!("{}", record.get_double(0).unwrap());
        // }
        expected_datapoints.push(record.get_double(0).unwrap());
        // i+=1;
    }
    let duration1 = start1.elapsed();
    let num = expected_datapoints.len();
    println!("Number of scan items:{}", num);
    println!("Time elapsed in parquet decompress {} items is: {:?}",num, duration1);

}



#[test]
fn test_parquet_write_filter(){
    let path = Path::new("target/debug/examples/sample.parquet");
    let file_iter = construct_file_iterator_skip_newline::<f64>(TEST_FILE, 1, ',');
    let file_vec: Vec<f64> = file_iter.unwrap()
        .map(|x| (x*SCALE))
        .collect();

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
        .set_compression(Compression::GZIP)
        .set_dictionary_pagesize_limit(DICTPAGE_LIM) // change max page size to avoid fallback to plain and make sure dict is used.
        .set_dictionary_enabled(USE_DICT)
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
    println!("file size: {}",bytes.len());
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
    let mut i = 0;
    let mut res = Bitmap::create();
    while let Some(record) = iter.next() {
        // if (i<10){
        //     println!("{}", record.get_double(0).unwrap());
        // }
        if (record.get_double(0).unwrap()>PRED){
            res.add(i);
        }
        i+=1;
    }
    res.run_optimize();
    let duration1 = start1.elapsed();
    println!("Number of qualified items:{}", res.cardinality());
    let num = expected_datapoints.len();
    println!("Time elapsed in parquet filter {} items is: {:?}",num, duration1);

}