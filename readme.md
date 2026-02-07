#Ottimizzazione CUDA per rappresentazioni stacked di eventi generati da una videocamera neuromorfica
## Simone Guidi


## Testing with Nsight Compute
Eseguo Nsight Compute su tutti e tre i codici cuda con un numero di eventi diverso
```
for t in 20000000 40000000 60000000 80000000 100000000 ; do 
    for f in $(ls histogram_stacking_cuda*.py); do 
        nv-nsight-cu-cli --target-processes all --set full -o "$t"_"$f"_report -k "compute.*" -f python3 $car_urban_night_penno_small_loop_data.h5 0 $t 200000 512
    done
done
```
Utilizzando `report_stats_extractor.py` salvo tutte le metriche dei file di report su MongoDB 
