import sys
import ncu_report
import pymongo
from pathlib import Path

client = pymongo.MongoClient(Path('.db_connection_string').read_text(encoding='utf-8'))
db = client["sdea"]
collezione = db["nsight_results"]

args = sys.argv
directory_path = args[1] # Current directory
nsight_reports_paths = list(Path(directory_path).glob("*.nsight-cuprof-report"))

for report_path in nsight_reports_paths:

    # Carica il file del report
    context = ncu_report.load_report(report_path)

    # Un file può contenere più sessioni/range; prendiamo la prima
    for range_idx in range(context.num_ranges()):
        report_range = context.range_by_idx(range_idx)
        
        # Ora puoi accedere alle azioni (i kernel eseguiti)
        for action_idx in range(report_range.num_actions()):
            action = report_range.action_by_idx(action_idx)
            newExecution = {"report_path": f"{report_path}",
                            "kernel_name": action.name(), 
                            "gpu_time": action.metric_by_name('gpu__time_duration').as_double(), 
                            "l2_transactions": action.metric_by_name('memory_l2_transactions_global').as_double(), 
                            "warp_stall_wait": action.metric_by_name('smsp__pcsamp_warp_stall_wait').as_double(), 
                            "memory_sol": action.metric_by_name('gpu__compute_memory_sol_pct').as_double(), 
                            "throughput": action.metric_by_name('dram__bytes_per_sec').as_double(), 
                            "occupancy": action.metric_by_name('sm__active_warps_avg_per_active_cycle_pct').as_double(), 
                            }
            collezione.insert_one(newExecution)
            


# Usata per trovare i nomi delle metriche
# metric_names = action.metric_names()
# filtered_metric = [word for word in metric_names if "warps_avg_per" in word]
# print(f"Metrics: {filtered_metric}")