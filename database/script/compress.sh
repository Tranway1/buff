cd /mnt/hdd-2T-3/chunwei/TimeSeriesDB/database;
for comp in grilla zlib paa fourier snappy deflate gzip bp;
do 
	for type in u32 f32 f64;
		do 
			for file in $(ls /mnt/hdd-2T-3/chunwei/TimeSeriesDB/UCRArchive2018/*/*);
			    do
			         cargo run --package time_series_start --bin compress $file $type $comp
			    done

		done

done
			