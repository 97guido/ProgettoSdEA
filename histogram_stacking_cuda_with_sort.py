from datetime import datetime
import os
import time
import h5py
from pycuda import driver, compiler, gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import hashlib
import pymongo
from pathlib import Path
import numpy as np
import math


kernel_code = """
__global__ void compute_sorted_histogram(const unsigned int* index_array, 
                                unsigned int* d_output,
                                int height, int width,
                                int total_events) {

    int eventIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (eventIdx < total_events) {
        unsigned int idx = index_array[eventIdx];
        if (idx != 0xFFFFFFFF) atomicAdd(&d_output[idx], 1); // Ignoro gli elementi di padding
    }
}
"""


kernel_sorted_indexes = """
__global__ void compute_indices_padded(unsigned int *x, unsigned int *y, unsigned char *p, 
                                        int tot_events, int events_per_stack, int padded_eps, 
                                        unsigned int frame_size, unsigned int w, 
                                        unsigned int *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total_padded_size = (tot_events / events_per_stack) * padded_eps;

    if (i < total_padded_size) {
        int s = i / padded_eps;
        int idx_in_stack = i % padded_eps;
        int original_idx = s * events_per_stack + idx_in_stack;

        if (idx_in_stack < events_per_stack && original_idx < tot_events) {
            out[i] = s * frame_size + (x[original_idx] + y[original_idx] * w) * 2 + (1 - p[original_idx]);
        } else {
            out[i] = 0xFFFFFFFF;
        }
    }
}

__global__ void bitonic_sort_step(unsigned int *out, int j, int k, int total_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total_size) {
        int ixj = i ^ j;
        if (ixj > i) {
            if ((i & k) == 0) {
                if (out[i] > out[ixj]) {
                    unsigned int tmp = out[i]; out[i] = out[ixj]; out[ixj] = tmp;
                }
            } else {
                if (out[i] < out[ixj]) {
                    unsigned int tmp = out[i]; out[i] = out[ixj]; out[ixj] = tmp;
                }
            }
        }
    }
}
"""

def compute_and_sort_indices(self, x, y, p, events_per_stack, h, w):

    tot_events = len(x)
    num_stacks = tot_events // events_per_stack
    
    # Trova la potenza di 2 successiva per il sorting
    padded_eps = 2**(math.ceil(math.log2(events_per_stack)))
    total_padded_size = num_stacks * padded_eps

    # Allocazione GPU
    d_x = cuda.to_device(x.astype(np.uint32))
    d_y = cuda.to_device(y.astype(np.uint32))
    d_p = cuda.to_device(p.astype(np.uint8))
    d_out = cuda.mem_alloc(total_padded_size * 4) # 4 bytes per uint32

    threads_per_block = 512
    grid_size = ((total_padded_size + threads_per_block - 1) // threads_per_block, 1)

    # Calcolo con Padding
    compute_func = self.mod2.get_function("compute_indices_padded")
    compute_func(d_x, d_y, d_p, np.int32(tot_events), np.int32(events_per_stack), 
                 np.int32(padded_eps), np.uint32(h*w*2), np.uint32(w), d_out,
                 block=(threads_per_block, 1, 1), grid=grid_size)

    # Sort Bitonico sulla dimensione padded
    sort_func = self.mod2.get_function("bitonic_sort_step")
    steps_k = int(math.log2(padded_eps))
    for k_idx in range(1, steps_k + 1):
        k = np.int32(1 << k_idx)
        for j_idx in range(k_idx - 1, -1, -1):
            j = np.int32(1 << j_idx)
            sort_func(d_out, j, k, np.int32(total_padded_size), block=(threads_per_block, 1, 1), grid=grid_size)

    return d_out, total_padded_size

class HistogramStackingCUDAwithSort:

    def __init__(self):
        self.functionName = 'histogram_stacking_cuda_with_sort.py'
        # Utilizzo mongodb per salvare i risultati e i parametri di esecuzione
        self.logToDb = Path('.db_connection_string').is_file()
        if (self.logToDb):
            client = pymongo.MongoClient(Path('.db_connection_string').read_text(encoding='utf-8'))
            db = client["sdea"]
            self.collezione = db["esecuzioni"]

        self.mod = compiler.SourceModule(kernel_code)
        self.mod2 = compiler.SourceModule(kernel_sorted_indexes)
        self.func = self.mod.get_function("compute_sorted_histogram")

    def run(self, dataset_path, start_event, end_event, events_per_stack, threads_per_block, resolution=(720, 1280)):

        h, w = resolution
        total_events = end_event - start_event
        total_stacks = (total_events + events_per_stack - 1) // events_per_stack

        with h5py.File(dataset_path, 'r') as hf:
            ds = hf['/prophesee/left']
            start_t = time.time()

            res_gpu = gpuarray.zeros((total_stacks, h, w, 2), dtype=np.uint32)

            # Calcolo gli indici e li riordino
            indexes, paded_size = compute_and_sort_indices(self, ds['x'][start_event:end_event], ds['y'][start_event:end_event], ds['p'][start_event:end_event], events_per_stack, h, w)

            grid_size = ((paded_size + threads_per_block - 1) // threads_per_block, 1)

            self.func(indexes, res_gpu,
                    np.int32(h), np.int32(w),
                    np.int32(paded_size),
                    block=(threads_per_block, 1, 1), grid=grid_size)
            
            res_cpu = res_gpu.get() 

            end_t = time.time()

            checksum = hashlib.md5(bytes(res_cpu.clip(0,255).astype(np.uint8))).hexdigest()
            newExecution = {"file": self.functionName, "tot_events": total_events, "events_per_stack": events_per_stack, "interval": f"{start_event}-{end_event}", "threads_per_block": threads_per_block, "execution_time": end_t - start_t, "checksum": checksum, "start_timestamp": datetime.now().isoformat()}
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

    HistogramStackingCUDAwithSort().run(datasetPath, startEvent, endEvent, eventsPerStack, threads_per_block)

# main()