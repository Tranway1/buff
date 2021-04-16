cd /home/cc/TimeSeriesDB/database;
#dir=$1
#for comp in gorilla gorillabd splitdouble byteall bytedec bp bpsplit gzip snappy sprintz plain dict pqgzip pqsnappy;
TIME=$3
SCL=$2
for comp in gorilla buff sprintz gzip snappy dict;
#for comp in gorilla gorillabd;
do
  for i in $(seq 1 $TIME);
		do
		  echo $i
#			for file in $(ls /mnt/hdd-2T-3/chunwei/timeseries_dataset/*/*/*);
      for R in 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
			    do

            cargo +nightly run --release  --package time_series_start --bin predict $1 $comp $SCL $R

			    done

		done

done
echo "CR prediction done!"
python ./script/python/crparser.py new.out predict.csv $TIME
			
