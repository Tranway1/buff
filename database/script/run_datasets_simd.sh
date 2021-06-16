bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k 10000 1.1509 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k 10000 4.1582 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k 10000 9.1517 5 >> new.out
mv performance.csv cbf-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv 1000000 39.753071 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv 1000000 40.759209 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv 1000000 41.75766 5 >> new.out
mv performance.csv dropoff_latitude-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv 10000 295.7114 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv 10000 294.5538 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv 10000 296.0503 5 >> new.out
mv performance.csv ts120rpm-c8-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7360.449707 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7445.229980 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7533.481445 5 >> new.out
mv performance.csv pmu_p1_L1MAG-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/vm_cpu_readings-file-19-20_c4-avg.csv 1000000 20.774518 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/vm_cpu_readings-file-19-20_c4-avg.csv 1000000 2.891158 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/vm_cpu_readings-file-19-20_c4-avg.csv 1000000 80.980372 5 >> new.out
mv performance.csv cpu19-20-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/Stocks-c1-open.csv 1000 9.528 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/Stocks-c1-open.csv 1000 41.198 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/Stocks-c1-open.csv 1000 1669.83 5 >> new.out
mv performance.csv stocks-open-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCR-all.csv 100000 1.1886 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCR-all.csv 100000 8.8574 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCR-all.csv 100000 13.909 5 >> new.out
mv performance.csv UCR-all-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/city_temperature_c8.csv 10 0.4 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/city_temperature_c8.csv 10 80.6 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/city_temperature_c8.csv 10 99.4 5 >> new.out
mv performance.csv city_temperature_c8-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCI-all.csv 100000 1.12878 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCI-all.csv 100000 4.36088 5 >> new.out
bash script/float_compress_simd.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCI-all.csv 100000 13.2222 5 >> new.out
mv performance.csv UCI-all-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c2-current.csv 100000 172.625 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c2-current.csv 100000 3.59375 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c2-current.csv 100000 269.2344 5 >> new.out
mv performance.csv ts120rpm-c2-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c5-voltage.csv 1000000 6.572917 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c5-voltage.csv 1000000 107.6354 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/time_series_120rpm-c5-voltage.csv 1000000 197.4896 5 >> new.out
mv performance.csv ts120rpm-c5-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1ANG 1000000 16.054514 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1ANG 1000000 316.050659 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1ANG 1000000 359.056244 5 >> new.out
mv performance.csv pmu_p1_L1ANG-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/gas-array-all-c2-Humidity.txt 10000 29.6800 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/gas-array-all-c2-Humidity.txt 10000 50.4095 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/gas-array-all-c2-Humidity.txt 10000 76.4429 5 >> new.out
mv performance.csv gas-array-all-c2-Humidity-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/gas-array-all-c3-temperature.txt 10000 13.7998 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/gas-array-all-c3-temperature.txt 10000 25.1800 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/gas-array-all-c3-temperature.txt 10000 27.4056 5 >> new.out
mv performance.csv gas-array-all-c3-temperature-simd-performance.csv
rm new.out

bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7360.449707 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7445.229980 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 1000000 7533.481445 5 >> new.out
mv performance.csv pmu_p1_L1MAG-performance-freq.csv
rm new.out


bash script/float_compress_simd.sh /home/cc/float_comp/signal/Household_power_consumption_c3_voltage.csv 100 230.71 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/Household_power_consumption_c3_voltage.csv 100 241.84 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/Household_power_consumption_c3_voltage.csv 100 249.91 5 >> new.out
mv performance.csv house-voltage-simd-performance.csv
rm new.out


bash script/float_compress_simd.sh /home/cc/float_comp/signal/Household_power_consumption_c4_global_intensity.csv 10 0.8 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/Household_power_consumption_c4_global_intensity.csv 10 1.6 5 >> new.out
bash script/float_compress_simd.sh /home/cc/float_comp/signal/Household_power_consumption_c4_global_intensity.csv 10 40.4 5 >> new.out
mv performance.csv house-intensity-simd-performance.csv
rm new.out

echo "done"