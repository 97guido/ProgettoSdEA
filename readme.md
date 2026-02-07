

## Testing with Nsight Compute
```
for t in 20000000 40000000 60000000 80000000 100000000 ; do 
    for f in $(ls histogram_stacking_cuda*.py); do 
        nv-nsight-cu-cli --target-processes all --set full -o "$t"_"$f"_report -k "compute.*" -f python3 $car_urban_night_penno_small_loop_data.h5 0 $t 200000 512
    done
done
```