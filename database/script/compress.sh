cd /mnt/hdd-2T-3/chunwei/TimeSeriesDB/database;
#dir=$1
for comp in fcm dfcm grilla zlib paa fourier snappy deflate gzip bp deltabp;
do
	for type in i32 f32 f64;
		do
			for file in $(ls /mnt/hdd-2T-3/chunwei/timeseries_dataset/*/*/*);
			    do
if [[ $type == "u32" || $type == "i32" ]]
then
                for scl in 1 10 100;
                do
                  cargo run --package time_series_start --bin compress $file $type $comp $scl
                done
else
                cargo run --package time_series_start --bin compress $file $type $comp
              fi
			    done

		done

done
			
