# BUFF
Rust implementation of float compression for VLDB 2021 paper [Decomposed Bounded Floats for Fast Compression and Queries](http://vldb.org/pvldb/vol14/p2586-liu.pdf) [video](https://youtu.be/Krn98iPY99o). All BUFF-related implementations are included in the `database` folder. As BUFF query relies on bit-vector operations, we included a popular bit-vector implementation `roaring-rs` bitmap and our own implementation `bit-vec`.



## Compile
This compile uses the cargo nightly channel.

Quick start:
```
cd ./database
cargo +nightly run --release  --package buff --bin comp_profiler ./data/randomwalkdatasample1k-1k buff-simd 10000 1.1509
```
The code sample above parses a CSV file, encodes file with BUFF, and executes equality, range, aggregation, materialization queries on BUFF encoded data chunk with SIMD enabled. To use the scalar version, please replace "buff-simd" with "buff".

## BUFF highlights
BUFF not only reduces storage requirements but also enhances query performance:

-	Fast compression throughput and better compression ratio.
-	Query-friendly compression (fast filtering and aggregation execution).
-	Adaptive SIMD acceleration.
-	Supporting materialization with variable precision as requested.
-	Approximate query processing.
-	Outlier handling.
