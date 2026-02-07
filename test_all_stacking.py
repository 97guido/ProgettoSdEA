from histogram_stacking_cpu import HistogramStackingCPU
from histogram_stacking_cpu_cpp import HistogramStackingCPP
from histogram_stacking_cuda_naive import HistogramStackingCUDA
from histogram_stacking_cuda_16bit import HistogramStackingCuda16Bit
from histogram_stacking_cuda_with_sort import HistogramStackingCUDAwithSort

dataset = "car_urban_night_penno_small_loop_data.h5"
events_per_stack = 200000
threads_per_block = 512

for endEvent in range(1000000, 140000000, 2000000):

    # HistogramStackingCPU().run(dataset, 0, endEvent, events_per_stack) 
    HistogramStackingCPP().run(dataset, 0, endEvent, events_per_stack)
    HistogramStackingCUDA().run(dataset, 0, endEvent, events_per_stack, threads_per_block)
    HistogramStackingCuda16Bit().run(dataset, 0, endEvent, events_per_stack, threads_per_block)
    HistogramStackingCUDAwithSort().run(dataset, 0, endEvent, events_per_stack, threads_per_block)
