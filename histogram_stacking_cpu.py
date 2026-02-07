import time
import h5py
import numpy as np
import sys
import hashlib
import pymongo
from datetime import datetime
from pathlib import Path

class HistogramStackingCPU:

    def __init__(self):
        self.functionName = 'histogram_stacking_cpu.py'
        # Utilizzo mongodb per salvare i risultati e i parametri di esecuzione
        self.logToDb = Path('.db_connection_string').is_file()
        if (self.logToDb):
            client = pymongo.MongoClient(Path('.db_connection_string').read_text(encoding='utf-8'))
            db = client["sdea"]
            self.collezione = db["esecuzioni"]

    def run(self, datasetPath, startEvent=0, endEvent=5000000, eventsPerStack=200000, resolution=(720, 1280)):
        try:



            # Carico il dataset
            with h5py.File(datasetPath, 'r') as hf:
                dataset = hf['/prophesee/left']
                totalEvents = endEvent - startEvent
                totalStacks = totalEvents // eventsPerStack

                # Definisco l'array del risultato, inizializzando tutte le celle a 0 
                data = [0] * ((resolution[0]*resolution[1]*2) * totalStacks)
                # data = np.zeros(((resolution[0]*resolution[1]*2) * totalStacks), dtype=np.uint16)

                X = dataset['x'][startEvent:endEvent]
                Y = dataset['y'][startEvent:endEvent]
                P = dataset['p'][startEvent:endEvent]

                startTime = time.time()
                for eventID in range(0, endEvent-startEvent, 1):
                    # Linearizzo la posizione del pixel dell'evento, utilizzo il valore della polarità 
                    # indirizzare automaticamente al canale corrispondente
                    idx = (eventID // eventsPerStack) * (resolution[0]*resolution[1]*2) + \
                        (int(Y[eventID])*resolution[1] + int(X[eventID]))*2 + \
                        (1-int(P[eventID]))
                    if data[idx] < 255: data[idx] += 1

                endTime = time.time()

                checksum = hashlib.md5(bytes(data)).hexdigest()

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
            print(f"Error opening file {datasetPath}. Check the file path and permissions.") 
    

def main():

    args = sys.argv

    # Ottengo i parametri di esecuzione dagli argomenti
    datasetPath = args[1]
    startEvent = int(args[2]) if len(args) > 4 else 0
    endEvent = int(args[3]) if len(args) > 4 else 5000000
    eventsPerStack = int(args[4]) if len(args) > 4 else 200000

    HistogramStackingCPU().run(datasetPath, startEvent, endEvent, eventsPerStack)
    
# main()