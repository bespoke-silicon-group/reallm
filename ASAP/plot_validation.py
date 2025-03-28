# %%
import json

json_name = 'h100x4-nvl-94gb-llama-3.1-70b-instruct-sharegpt-s1.0-m100'
file_name = f'real_trace_time/{json_name}.json'

with open(file_name, 'r') as f:
    trace = json.load(f)
    print(trace.keys())
for t in trace['chats']:
    print(t, trace['chats'][t])
# %%
num_traces_to_plot = 90
traces_sys1 = []
traces_sys2 = []

for t in trace['chats']:
    traces_sys1.append((trace['chats'][t]['start_time'], trace['chats'][t]['end_time']))
    if len(traces_sys1) == num_traces_to_plot:
        break

our_results_file = 'results/llama70b/h100x4-nvl-94gb-llama-3.1-70b-instruct-sharegpt-s1/4-H100-continuous-(1, 4, 1, 1)/sim_results.csv'
with open(our_results_file, 'r') as f:
    for line in f:
        trace = line.strip().split(',')
        if trace[0] == 'request_id':
            continue
        traces_sys2.append((float(trace[1]), float(trace[-2])))
        if len(traces_sys2) == num_traces_to_plot:
            break

import matplotlib.pyplot as plt
import numpy as np


# Example trace data: (start_time, end_time) for each trace
# System 1

# Number of traces
num_traces = max(len(traces_sys1), len(traces_sys2))

fig, ax = plt.subplots(figsize=(6, 6))

height = 0.4
offset = 0.4
# Plot traces for system 1
for i, (start, end) in enumerate(traces_sys1):
    ax.barh(y=i, width=end - start, left=start, height=height, color='tab:blue', label="4 x H100" if i == 0 else "")

# Plot traces for system 2
for i, (start, end) in enumerate(traces_sys2):
    ax.barh(y=i + offset, width=end - start, left=start, height=height, color='tab:orange', label="ReaLLM Sim" if i == 0 else "")

ax.set_xlabel("Time", fontsize=13)
ax.set_ylabel("Traces", fontsize=13)
ax.set_title("Trace-Drive LLM System Comparison Between GPUs and ReaLLM", fontsize=13)
ax.legend(prop={'size': 15})
ax.set_ylim(0, num_traces + 0.3)
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
# save to pdf
fig.savefig('trace_comparison.pdf', bbox_inches='tight')

# calculate the difference between the two systems
error_rates = []
for i in range(num_traces):
    if i >= len(traces_sys1) or i >= len(traces_sys2):
        continue
    time1 = traces_sys1[i][1] - traces_sys1[i][0]
    time2 = traces_sys2[i][1] - traces_sys2[i][0]
    error_rate = abs((time2 - time1)) / time1
    # print(f"Error rate for trace {i}: {error_rate:.2%}")
    error_rates.append(error_rate)

print(f"Average error rate: {np.mean(error_rates):.2%}")

# %%
# length = len(trace['chats'])
# all_traces = []
# for i in range(length):
#     req = trace['chats'][str(i)]
#     arrival_time = req['start_time']
#     input_len = req['prompt_tokens']
#     output_len = req['completion_tokens']
#     all_traces.append((arrival_time, input_len, output_len))
# # %%
# file_to_write = f'traces/{file_name.split("/")[-1][:-5]}.csv'
# f = open(file_to_write, 'w')
# f.write('request_id,request_type,application_id,arrival_timestamp,batch_size,prompt_size,token_size\n')

# for i, t in enumerate(all_traces):
#     f.write(f'{i},2,0,{t[0]},1,{t[1]},{t[2]}\n')
# f.close()
# %%
import sys
sys.path.append('../')
sys.path.append('llm_serving_sim/')
from llm_serving_sim.simulator import Simulator
from llm_serving_sim.hardware_sim import HardwareSim
from llm_serving_sim.hardware import Hardware, A100, H100
from llm_serving_sim.model import llama70b, llama405b, deepseekv2, deepseekv3
from llm_serving_sim.scheduler import Scheduler
import logging

# logging.basicConfig(level=logging.DEBUG)

scheduler_algo = 'continuous'
max_ctx_len = 8192
io_algo = 'ring'
num_nodes = 8
model = llama70b
parallelism = (1, 8, 1, 1) # EP, TP, PP, CP
eval_hardware = Hardware(node=A100, 
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
)

eval_hardware.node.name = 'A100'
hw_name = f"8A100-validation"

sim = Simulator(
    model = llama70b,
    trace=file_to_write,
    scheduler=scheduler,
    hardware_sim=hardware_sim,
    end_time=200,
    start_reqs=0,
    end_reqs=100,
)
sim.run()

# %%
