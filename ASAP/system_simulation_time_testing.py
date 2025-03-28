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
import time
# %%
model = llama70b
num_nodes = 8
io_algo = 'multishot'
scheduler_algo = 'mixed-sarathi'
max_ctx_len = 8192*4

workloads = ['code', 'conv']

code_request_rates = [1, 2]
conv_request_rates = [1, 2]

# conv_request_rates = [3, 4, 5, 6,]
all_hw_node_names = ['H100',]



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

    end_reqs = 500

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
        start = time.time()
        run(trace, 'H100')
        end = time.time()
        print(f'Elapsed time: {end-start}')
