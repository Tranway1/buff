cd /mnt/hdd-2T-3/chunwei/TimeSeriesDB/database;
#dir=$1
#for comp in ofsgorilla gorilla gorillabd splitbd split bp zlib paa fourier snappy deflate gzip deltabp;
for comp in gorilla gorillabd;
do
	for type in i32 f64;
		do
			for file in /mnt/hdd-2T-3/chunwei/UCI-all.csv /mnt/hdd-2T-3/chunwei/UCR-all.csv ;
			    do
if [[ $type == "u32" || $type == "i32" ]]
then
                for scl in 100;
                do
                  cargo run --release --package time_series_start --bin compress $file $type $comp $scl
                done
else
                cargo run --release --package time_series_start --bin compress $file $type $comp
              fi
			    done

		done

done
			
