from datetime import datetime
import os
import time
import numpy as np
import h5py
from pycuda import driver, compiler, gpuarray
import pycuda.autoinit
import sys
from PIL import Image
import hashlib
import pymongo
from pathlib import Path

kernel_code_short = """
__device__ short atomicAddShort(unsigned short* address, unsigned short val) {
    unsigned int* base_addr = (unsigned int*)((size_t)address & ~3);
    unsigned int selectors[2] = {0, 16};
    unsigned int sel = ((size_t)address & 2) ? 1 : 0;
    unsigned int shift = selectors[sel];
    
    unsigned int old, assumed, updated;
    old = *base_addr;
    do {
        assumed = old;
        // Isola i 16 bit corretti, aggiungi il valore e ricomponi la word da 32 bit
        short current_val = (short)((assumed >> shift) & 0xFFFF);
        short new_val = current_val + val;
        updated = (assumed & ~(0xFFFF << shift)) | ((unsigned int)(unsigned short)new_val << shift);
        old = atomicCAS(base_addr, assumed, updated);
    } while (assumed != old);
    
    return (short)((old >> shift) & 0xFFFF);
}

__global__ void compute_histogram_stacking(const unsigned short* d_input_x, 
                                const unsigned short* d_input_y, 
                                const signed char* d_input_p, 
                                unsigned short* d_output,
                                int height, int width, 
                                int total_events, int eventsPerStack) {

    int eventIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (eventIdx < total_events) {
        int stackId = eventIdx / eventsPerStack;
        
        unsigned short resX = d_input_x[eventIdx];
        unsigned short resY = d_input_y[eventIdx];
        signed char resP = d_input_p[eventIdx];

        unsigned int idx = (stackId * height * width * 2) + 
                            (resY * width * 2) + 
                            (resX * 2) + (1 - resP);

        atomicAddShort(&d_output[idx], 1);
    }
}
"""


conv_kernel_code_short = """
__global__ void normalize_and_convert(unsigned short* d_input, 
    unsigned char* d_output, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (d_input[idx] > 255) d_input[idx] = 255;
        d_output[idx] = (unsigned char)d_input[idx];
    }
}
"""

class HistogramStackingCuda16Bit():

    def __init__(self):
        self.functionName = 'histogram_stacking_cuda_16bit.py'
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

        mod = compiler.SourceModule(kernel_code_short)
        func = mod.get_function("compute_histogram_stacking")

        with h5py.File(dataset_path, 'r') as hf:
            ds = hf['/prophesee/left']
            
            datasetP = ds['p'][start_event:end_event] # polarity
            datasetX = ds['x'][start_event:end_event] # x coordinate
            datasetY = ds['y'][start_event:end_event] # y coordinate

            start_t = time.time()
            x_gpu = gpuarray.to_gpu(np.array(datasetX, dtype=np.uint16))
            y_gpu = gpuarray.to_gpu(np.array(datasetY, dtype=np.uint16))
            p_gpu = gpuarray.to_gpu(np.array(datasetP, dtype=np.int8))
            
            res_gpu = gpuarray.zeros((total_stacks, h, w, 2), dtype=np.uint16)

            grid_size = ((total_events + threads_per_block - 1) // threads_per_block, 1)

            func(x_gpu, y_gpu, p_gpu, res_gpu,
                    np.int32(h), np.int32(w),
                    np.int32(total_events), np.int32(events_per_stack),
                    block=(threads_per_block, 1, 1), grid=grid_size)

            # Aspetto che la GPU finisca
            driver.Context.synchronize()

            # Normalizzo il risultato a 1 byte, in modo da velocizzare il trasferimento in memoria host
            res_uint8_gpu = gpuarray.empty((total_stacks, h, w, 2), dtype=np.uint8)
            norm_func = compiler.SourceModule(conv_kernel_code_short).get_function("normalize_and_convert")
            size = total_stacks * h * w * 2
            norm_func(res_gpu, res_uint8_gpu, np.int32(size),
                    block=(512, 1, 1), grid=((size // 512) + 1, 1))

            # Copio il risultato normalizzato sulla RAM dell'host
            res_cpu= np.empty((total_stacks, h, w, 2), dtype=np.uint8)
            driver.memcpy_dtoh(res_cpu, res_uint8_gpu.ptr)

            end_t = time.time()
            
            # Calcolo il checksum del risultato per validarne l'accuratezza
            checksum = hashlib.md5(bytes(res_cpu)).hexdigest()
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

    # Esempio di esecuzione (adatta i parametri)
    HistogramStackingCuda16Bit().run(datasetPath, startEvent, endEvent, eventsPerStack, threads_per_block)

# main()