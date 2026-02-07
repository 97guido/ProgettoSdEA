import time
from typing import Any
import numpy as np
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit
import h5py
import sys
import hashlib
import pymongo
from datetime import datetime
from pathlib import Path


kernel_code = """
__global__ void compute_histogram_stacking(unsigned int* d_input_x, 
                        unsigned int* d_input_y, char* d_input_p, 
                        unsigned int* d_output,
                        unsigned int height, unsigned int width, 
                        unsigned int lastEventIdx, 
                        unsigned int eventsPerStack) {

    int eventIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (eventIdx < lastEventIdx) {
        int stackId = eventIdx / eventsPerStack;
        
        unsigned int resX = d_input_x[eventIdx];
        unsigned int resY = d_input_y[eventIdx];
        signed char resP = d_input_p[eventIdx];

        unsigned int idx = (stackId * height * width * 2) + 
                                 (resY * width * 2) + 
                                 (resX * 2) + (1 - resP);

        atomicAdd(&d_output[idx], 1);
    }
}
"""

class HistogramStackingCUDA():
    
    def __init__(self):
        self.functionName = 'histogram_stacking_cuda_naif.py'
        # Utilizzo mongodb per salvare i risultati e i parametri di esecuzione
        self.logToDb = Path('.db_connection_string').is_file()
        if (self.logToDb):
            client = pymongo.MongoClient(Path('.db_connection_string').read_text(encoding='utf-8'))
            db = client["sdea"]
            self.collezione = db["esecuzioni"]

    def run(self, dataset_path, start_event, end_event, events_per_stack, threads_per_block, resolution=(720, 1280)):
        h, w = resolution
        total_events = end_event - start_event
        total_stacks = (total_events + events_per_stack - 1) // events_per_stack

        mod = compiler.SourceModule(kernel_code)
        func = mod.get_function("compute_histogram_stacking")

        with h5py.File(dataset_path, 'r') as hf:
            ds = hf['/prophesee/left']
            
            datasetP = ds['p'][start_event:end_event] # polarity
            datasetX = ds['x'][start_event:end_event] # x coordinate
            datasetY = ds['y'][start_event:end_event] # y coordinate

            
            start_t = time.time()
            x_gpu = gpuarray.to_gpu(np.array(datasetX, dtype=np.uint32))
            y_gpu = gpuarray.to_gpu(np.array(datasetY, dtype=np.uint32))
            p_gpu = gpuarray.to_gpu(np.array(datasetP, dtype=np.int8))
            
            res_gpu = gpuarray.zeros((total_stacks, h, w, 2), dtype=np.uint32)

            grid_size = ((total_events + threads_per_block - 1) // threads_per_block, 1)

            func(x_gpu, y_gpu, p_gpu, res_gpu,
                    np.int32(h), np.int32(w),
                    np.int32(total_events), np.int32(events_per_stack),
                    block=(threads_per_block, 1, 1), grid=grid_size)

            res_cpu = res_gpu.get()

            end_t = time.time()
            
            # Calcolo il checksum del risultato per validarne l'accuratezza
            checksum = hashlib.md5(bytes(res_cpu.clip(0, 255).astype(np.uint8))).hexdigest()
            newExecution = {"file": self.functionName, "tot_events": total_events, "events_per_stack": events_per_stack, "interval": f"{start_event}-{end_event}", "execution_time": end_t - start_t, "checksum": checksum, "start_timestamp": datetime.now().isoformat()}
            if (self.logToDb): self.collezione.insert_one(newExecution)

            # Stampo i risultati e i parametri di esecuzione
            for a,v in newExecution.items():
                BOLD = '\033[31m'
                RESET = '\033[0m'
                print(f"{BOLD}{a}{RESET}: {v}")

            return res_cpu

def main():
    args = sys.argv

    # Ottengo i parametri di esecuzione dagli argomenti
    datasetPath = args[1] if len(args) > 4 else 0
    startEvent = int(args[2]) if len(args) > 4 else 0
    endEvent = int(args[3]) if len(args) > 4 else 5000000
    eventsPerStack = int(args[4]) if len(args) > 4 else 200000
    threads_per_block = int(args[5]) if len(args) > 5 else 512

    HistogramStackingCUDA().run(datasetPath, startEvent, endEvent, eventsPerStack, threads_per_block)

# main()

