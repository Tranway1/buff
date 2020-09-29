cd /Users/chunwei/research/TimeSeriesDB/database;
#dir=$1
#for comp in ofsgorilla gorilla gorillabd splitbd split bp zlib paa fourier snappy deflate gzip deltabp;

for file in $(ls /Users/chunwei/research/TimeSeriesDB/UCRArchive2018);
  do
    for prec in -1 5 4 3 2 1 0;
		  do
		    cargo +nightly run --package time_series_start --bin knn ../UCRArchive2018/${file}/${file}_TRAIN ../UCRArchive2018/${file}/${file}_TEST $prec
	    done
	done
echo "all done"
			
