from datetime import datetime
import hashlib
import os
import time
import h5py
import numpy as np
from PIL import Image
import subprocess
import sys
import ctypes
import pymongo
from pathlib import Path

class HistogramStackingCPP:

    def __init__(self):
        self.functionName = 'histogram_stacking_cpu_cpp.py'
        # Utilizzo mongodb per salvare i risultati e i parametri di esecuzione
        self.logToDb = Path('.db_connection_string').is_file()
        if (self.logToDb):
            client = pymongo.MongoClient(Path('.db_connection_string').read_text(encoding='utf-8'))
            db = client["sdea"]
            self.collezione = db["esecuzioni"]

        # Importo il codice cpp compilato
        self.histogram_stacking_cpp = ctypes.CDLL('./histogram_stacking.so')

        print("ok1")

        # Configura i tipi dei parametri
        self.histogram_stacking_cpp.histogramStack.argtypes = [
            ctypes.POINTER(ctypes.c_uint16), # x
            ctypes.POINTER(ctypes.c_uint16), # y
            ctypes.POINTER(ctypes.c_uint8),  # p
            ctypes.c_int,                   # num_events
            ctypes.c_int,                   # width
            ctypes.c_int,                   # height
        ]

        # Configura il tipo di ritorno della funzione getBitmap
        self.histogram_stacking_cpp.histogramStack.restype = ctypes.POINTER(ctypes.c_uint8)

    def run(self, datasetPath, startEvent=0, endEvent=5000000, eventsPerStack=200000, resolution=(720, 1280)):
            try:

                with h5py.File(datasetPath, 'r') as hf:

                    print("ok2")

                    dataset = hf['/prophesee/left']

                    totalEvents = endEvent - startEvent

                    X = dataset['x'][startEvent:endEvent]
                    Y = dataset['y'][startEvent:endEvent]
                    P = dataset['p'][startEvent:endEvent]

                    bitmap_array = []


                    startTime = time.time()
                    for s in range(0, endEvent-startEvent, eventsPerStack):
                        
                        subsetX = X[s:s+eventsPerStack]
                        subsetY = Y[s:s+eventsPerStack]
                        subsetP = P[s:s+eventsPerStack]

                        # Chiamata alla funzione C++
                        bitmap = self.histogram_stacking_cpp.histogramStack(
                            subsetX.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                            subsetY.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                            subsetP.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                            eventsPerStack,
                            resolution[1],
                            resolution[0]
                        )

                        # Copia i dati dalla memoria C++ a un array NumPy
                        bitmap_array.append(np.ctypeslib.as_array(bitmap, shape=(resolution[0], resolution[1], 2)))

                        
                    bitmap_array = np.array(bitmap_array)
                    endTime = time.time()

                    checksum = hashlib.md5(bytes(bitmap_array)).hexdigest()
                    newExecution = {"file": self.functionName, "tot_events": totalEvents, "events_per_stack": eventsPerStack, "interval": f"{startEvent}-{endEvent}", "execution_time": endTime - startTime, "checksum": checksum, "start_timestamp": datetime.now().isoformat()}
                                    # Se mongodb è abilitato salvo i parametri di esecuzione
                                    
                    if (self.logToDb): self.collezione.insert_one(newExecution)

                    # Stampo i risultati e i parametri di esecuzione
                    for a,v in newExecution.items():
                        BOLD = '\033[31m'
                        RESET = '\033[0m'
                        print(f"{BOLD}{a}{RESET}: {v}")

                        
                    
            except KeyError:
                print("Dataset not found in the file.")
            except IOError:
                print("Error opening file. Check the file path and permissions.") 

def main():
    args = sys.argv

    # Ottengo i parametri di esecuzione dagli argomenti
    datasetPath = args[1] if len(args) > 4 else 0
    startEvent = int(args[2]) if len(args) > 4 else 0
    endEvent = int(args[3]) if len(args) > 4 else 5000000
    eventsPerStack = int(args[4]) if len(args) > 4 else 200000

    HistogramStackingCPP().run(datasetPath, startEvent, endEvent, eventsPerStack)
    
# main()
