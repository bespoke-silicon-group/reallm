# %%
import numpy as np
import os, sys
sys.path.append('../')
sys.path.append('llm_serving_sim/')
from llm_serving_sim.simulator import Simulator
from llm_serving_sim.hardware_sim import HardwareSim
from llm_serving_sim.hardware import Hardware, H100
from llm_serving_sim.model import llama70b, llama405b, deepseekv2, deepseekv3
from llm_serving_sim.scheduler import Scheduler
import logging
import multiprocessing
# logging.basicConfig(level=logging.DEBUG)
# %%
model = llama70b
num_nodes = 8
io_algo = 'multishot'
scheduler_algo = 'mixed-sarathi'
max_ctx_len = 8192*4

workloads = ['code', 'conv']

code_request_rates = [1, 3, 5, 7, 8, 9, 11]
conv_request_rates = [0.4, 1, 2, 3, 4, 5, 6, 7]

# conv_request_rates = [3, 4, 5, 6,]
all_hw_node_names = ['H100',
                     'H100_fast_main', 'H100_more_compute1', 'H100_more_compute2', 'H100_more_l2']

# workloads = ['conv']
# request_rates = [11,]
# all_hw_node_names = ['H100',]



parallelism = (1, 8, 1, 1) # EP, TP, PP, CP
eval_hardware = Hardware(node=H100, 
                            num_nodes=num_nodes,
                            parallelism=parallelism,
                            io_algo=io_algo,
)

hardware_sim = HardwareSim(
    hardware=eval_hardware,
    method='llmcompass',
    scheduler_algo=scheduler_algo,
    max_ctx_len = max_ctx_len,
)

scheduler = Scheduler(
    algo=scheduler_algo,
    prefill_chunk=2048,
)

def run(trace, hw_node_name):
    workload = trace.split('/')[-1].split('_')[1]
    request_rate = trace.split('/')[-1].split('_')[2][:-4]

    print(f'======{workload}_{request_rate}_{hw_node_name} Begin =====\n')
    eval_hardware.node.name = hw_node_name
    hw_name = f"{num_nodes}-{eval_hardware.node.name}-{scheduler_algo}-{parallelism}"
    file_name = f"results/{model.name}/rr_{workload}_{request_rate}/{hw_name}/sim_results.csv"
    # if os.path.exists(file_name):
    #     print(f'======{workload}_{request_rate}_{hw_node_name} Exists =====\n')
    #     return

    if float(request_rate) < 5:
        end_reqs = 500
    elif float(request_rate) < 9:
        end_reqs = 1000
    else:
        end_reqs = 2000
    
    if workload == 'conv':
        if float(request_rate) < 5:
            end_reqs = 800
            print(f'set end_reqs to 800 for conv_{request_rate}')
        elif float(request_rate) < 7:
            end_reqs = 1500
            print(f'set end_reqs to 1500 for conv_{request_rate}')
        else:
            end_reqs = 2000
            print(f'set end_reqs to 2000 for conv_{request_rate}')

        end_reqs = 10000
        print(f'set end_reqs to 10000 for conv_{request_rate}')

    sim = Simulator(
        model = llama70b,
        trace=trace,
        scheduler=scheduler,
        hardware_sim=hardware_sim,
        end_time=500,
        start_reqs=0,
        end_reqs=end_reqs,
    )
    sim.run()
    print(f'======{workload}_{request_rate}_{hw_node_name} Done =====\n')

# run in parallel

all_inputs = []

for workload in workloads:
    if workload == 'code':
        request_rates = code_request_rates
    elif workload == 'conv':
        request_rates = conv_request_rates
    for request_rate in request_rates:
        trace = f'./traces/rr_{workload}_{request_rate}.csv'
        for hw_node_name in all_hw_node_names:
            all_inputs.append((trace, hw_node_name))


num_cores = 64
with multiprocessing.Pool(num_cores) as p:
    p.starmap(run, all_inputs)



# trace = f'./traces/rr_code_6.csv'
# 
# parallelism = (1, 8, 1, 1) # EP, TP, PP, CP
# eval_hardware = Hardware(node=H100, 
#                             num_nodes=num_nodes,
#                             parallelism=parallelism,
#                             io_algo=io_algo,
# )

# hardware_sim = HardwareSim(
#     hardware=eval_hardware,
#     method='llmcompass',
#     scheduler_algo='mixed-sarathi',
#     max_ctx_len = max_ctx_len,
# )

# scheduler = Scheduler(
#     algo='mixed-sarathi',
#     prefill_chunk=2048,
# )
# sim = Simulator(
#     model = llama70b,
#     trace=trace,
#     scheduler=scheduler,
#     hardware_sim=hardware_sim,
#     end_time=500,
#     start_reqs=0,
#     end_reqs=20,
# )
# sim.run()

# # %%
# from llm_serving_sim.hardware_sim import batch_interpolate_latency

# csv_name = 'H100_matmul_lat.csv'

# model = llama70b
# parallelism = (1, 8, 1, 1) # EP, TP, PP, CP

# for prefill_len in [16,]:
#     print(f'======prefill_len: {prefill_len}=====')
#     kernel_sizes = model.get_kernel_sizes(prefill_len, [], parallelism)
#     lat = batch_interpolate_latency(csv_name, kernel_sizes['matmul'].kernel_sizes)
#     print(f'lat: {lat}')
# %%
