nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c2-current.csv 100000 3.59375 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c2-current.csv 100000 172.625 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c2-current.csv 100000 269.2344 5
mv performance.csv ts120rpm-c2-performance.csv
rm nohup.out

nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c5-voltage.csv 1000000 6.572917 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c5-voltage.csv 1000000 107.6354 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c5-voltage.csv 1000000 197.4896 5
mv performance.csv ts120rpm-c5-performance.csv
rm nohup.out

nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv 10000 294.5538 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv 10000 295.7114 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv 10000 296.0503 5
mv performance.csv ts120rpm-c8-performance.csv
rm nohup.out

nohup bash script/float_compress.sh /home/cc/float_comp/signal/pmu_p1_L1ANG 1000000 16.054514 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/pmu_p1_L1ANG 1000000 316.050659 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/pmu_p1_L1ANG 1000000 359.056244 5
mv performance.csv pmu_p1_L1ANG-performance.csv
rm nohup.out

nohup bash script/float_compress.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7360.449707 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7511.229980 5
nohup bash script/float_compress.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7533.481445 5
mv performance.csv pmu_p1_L1MAG-performance.csv
rm nohup.out

nohup bash script/float_compress.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k 10000 1.1509 5
nohup bash script/float_compress.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k 10000 4.1582 5
nohup bash script/float_compress.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k 10000 9.1517 5
mv performance.csv cbf-performance.csv
rm nohup.out

nohup bash script/float_compress.sh /home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv 1000000 39.753071 5
nohup bash script/float_compress.sh /home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv 1000000 40.759209 5
nohup bash script/float_compress.sh /home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv 1000000 41.75766 5
mv performance.csv dropoff_latitude-performance.csv
rm nohup.out

echo "done"