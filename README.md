# TimeSeriesDB
A rust implementation of a time series focused database. There are three modules included in this repo:

### Ingestion
There are several basic structures implemented for the TimeSeriesDB. Two buffer pools are designed for ingestion and compression purpose respectively.
We use clients to generate configurable data streaming.
For each signal from the clients, the system buffers and process those data as the unit of the segment. All the ingested segments are saved in the raw buffer pool. 
The compression thread fetches segments from the raw buffer pool, compress and save the compressed segment into the compressed buffer pool, then materialize into storage when necessary.

Use following command to run a ingestion experiment with 2 gzip compression threads:
```
run --package time_series_start --bin time_series_start ./test_configs/config-single.toml f64 gzip 2
```

### Compression
There are comprehensive compression methods implemented in the methods folder. Those methods include byte-oriented compression methods (e.g. deflate, gzip, snappy, zlib),
Gorilla, Fourier, PAA for Double data type. FCM/DFCM, Bit-Packing, Delta-BP are implemented for the Integer data type as well.

All those compression methods implement the trait
```
trait CompressionMethod<T>
```
Use the following Cargo command to run compression experiments. This command will run gorilla compression on the f64 data segment extracted from the input file.
```
run --release --package time_series_start --bin compress ../UCRArchive2018/Kernel/randomwalkdatasample1k-10k f64 gorilla
```

### Query
Some basic aggregation query is implemented under the query folder. 

### Issues
Run 
```
sudo apt install make clang pkg-config libssl-dev
```
to solve the error "failed to run custom build command for `librocksdb-sys v5.18.3`"
