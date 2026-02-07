# Ottimizzazione CUDA per rappresentazioni stacked di eventi generati da una videocamera neuromorfica
Simone Guidi
Matricola: 981961

### Implementazioni
CPU Naive => histogram_stacking_cpu.py  
CPU Naive C++ => histogram_stacking_cpu_cpp.py  
CUDA Naive => histogram_stacking_cuda_naive.py  
CUDA Ottimizzata => histogram_stacking_cuda_16bit.py  
CUDA con Ordinamento => histogram_stacking_cuda_with_sort.py  


## Testing with Nsight Compute
Eseguo Nsight Compute su tutti e tre i codici cuda con un numero di eventi diverso  
(I risultati sono salvati nella cartella NSComputeReports)
```
for t in 20000000 40000000 60000000 80000000 100000000 ; do 
    for f in $(ls histogram_stacking_cuda*.py); do 
        nv-nsight-cu-cli --target-processes all --set full -o "$t"_"$f"_report -k "compute.*" -f python3 $car_urban_night_penno_small_loop_data.h5 0 $t 200000 512
    done
done
```
Utilizzando `report_stats_extractor.py` salvo tutte le metriche dei file di report su MongoDB, utilizzate poi per i grafici
