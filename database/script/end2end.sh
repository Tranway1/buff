cd /home/cc/TimeSeriesDB/database;
TIME=5
BENCH=influx

for query in max max_groupby;
		do
			for comp in gorilla gorillabd snappy gzip fixed sprintz scaled-slice buff-slice buff buff-major;
			    do
			      for i in $(seq 1 $TIME);
      		    do
      		      echo $i
                cargo +nightly run --release --package time_series_start --bin bench $BENCH $comp $query 0.1 >> new.out
			        done
			    done

done

BENCH=tsbs

for query in project single range range-new;
		do
			for comp in gorilla gorillabd snappy gzip fixed sprintz scaled-slice buff-slice buff buff-major;
			    do
			      for i in $(seq 1 $TIME);
      		    do
      		      echo $i
                cargo +nightly run --release --package time_series_start --bin bench $BENCH $comp $query 0.1 >> new.out
			        done
			    done
done
echo "Float compression done!"
cd /home/cc/TimeSeriesDB/database
python ./script/python/e2e_parser.py new.out performance.csv $TIME
mv performance.csv end2end-performance.csv
rm new.out