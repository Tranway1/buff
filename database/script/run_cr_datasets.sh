bash script/cr_predict.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/randomwalkdatasample1k-40k 4 1 >> new.out
mv predict.csv cbf-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/TimeSeriesDB/taxi/dropoff_latitude-fulltaxi-1k.csv 6 1 >> new.out
mv predict.csv dropoff_latitude-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/time_series_120rpm-c8-supply-voltage.csv 4 1 >> new.out
mv predict.csv ts120rpm-c8-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 6 1 >> new.out
mv predict.csv pmu_p1_L1MAG-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/vm_cpu_readings-file-19-20_c4-avg.csv 6 1 >> new.out
mv predict.csv cpu19-20-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/Stocks-c1-open.csv 3 1 >> new.out
mv predict.csv stocks-open-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCR-all.csv 5 1 >> new.out
mv predict.csv UCR-all-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/city_temperature_c8.csv 1 1 >> new.out
mv predict.csv city_temperature_c8-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/TimeSeriesDB/UCRArchive2018/Kernel/UCI-all.csv 5 1 >> new.out
mv predict.csv UCI-all-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/time_series_120rpm-c2-current.csv 5 1 >> new.out
mv predict.csv ts120rpm-c2-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/time_series_120rpm-c5-voltage.csv 6 1 >> new.out
mv predict.csv ts120rpm-c5-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/pmu_p1_L1ANG 6 1 >> new.out
mv predict.csv pmu_p1_L1ANG-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/gas-array-all-c2-Humidity.txt 4 1 >> new.out
mv predict.csv gas-array-all-c2-Humidity-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/gas-array-all-c3-temperature.txt 4 1 >> new.out
mv predict.csv gas-array-all-c3-temperature-predict.csv
rm new.out

bash script/cr_predict.sh /home/cc/float_comp/signal/pmu_p1_L1MAG 6 1 >> new.out
mv predict.csv pmu_p1_L1MAG-predict-freq.csv
rm new.out

echo "done"