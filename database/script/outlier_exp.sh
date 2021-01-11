cd /home/cc/TimeSeriesDB/database;
#dir=$1
#for comp in gorilla gorillabd splitdouble byteall bytedec bp bpsplit gzip snappy sprintz plain dict pqgzip pqsnappy;
TIME=$1
for comp in byte sparse;
#for comp in gorilla gorillabd;
do
  for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95;
		do
		  echo $i
#			for file in $(ls /mnt/hdd-2T-3/chunwei/timeseries_dataset/*/*/*);
      for i in $(seq 1 $TIME);
			    do
            cargo +nightly run --release --package time_series_start --bin outlier_exp 100000000 $ratio $comp
			    done

		done

done
echo "outlier experiments done!"
python ./script/python/outlier_logparser.py new.out performance.csv $TIME
			
