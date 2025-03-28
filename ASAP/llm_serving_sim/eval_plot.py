# %%
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import pandas as pd

from top_utils import get_slo_ms
from generate_trace import generate_code_traces, generate_conv_traces, download_azure_llm_traces, generate_traces
from model import Model, llama405, llama70, opt175
from simulator import Simulator
from hardware_sim import HardwareSim
from scheduler import Scheduler, SimKernel, LLMKernel
from hardware import Hardware, H100, Our_3D, System_Num_Nodes
from run import run_simulator
# %%
# Generate traces
random_seed = 0

requests_rates_8k = list(range(10, 30, 1))
requests_rates_32k = list(range(1, 15, 1))
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3]
np.random.seed(random_seed)
overwrite = True
# generate 8k traces if not exists
for req_rate in requests_rates_8k:
    trace_file = f'traces/rr_code_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_code_traces(
            max_requests=50000,
            end_time=500,
            request_rates=[req_rate],
            code_distributions_file="data/code_distributions.csv")
    trace_file = f'traces/rr_conv_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_conv_traces(
            max_requests=50000,
            end_time=500,
            request_rates=[req_rate],
            conv_distributions_file="data/conv_distributions.csv")

for req_rate in requests_rates_32k:
    trace_file = f'traces/rr_code_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_code_traces(
            max_requests=50000,
            end_time=1000,
            request_rates=[req_rate],
            code_distributions_file="data/code_distributions.csv")
    trace_file = f'traces/rr_conv_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_conv_traces(
            max_requests=50000,
            end_time=1000,
            request_rates=[req_rate],
            conv_distributions_file="data/conv_distributions.csv")

for req_rate in requests_rates_128k:
    trace_file = f'traces/rr_code_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_code_traces(
            max_requests=50000,
            end_time=5000,
            request_rates=[req_rate],
            code_distributions_file="data/code_distributions.csv")
    trace_file = f'traces/rr_conv_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_conv_traces(
            max_requests=50000,
            end_time=5000,
            request_rates=[req_rate],
            conv_distributions_file="data/conv_distributions.csv")

print("Generated 8k traces at different request rates")

# Generate long context traces with the same ratio
for max_len in [32, 128]:
    times = max_len // 8
    if max_len == 32:
        long_ctx_requests_rates = requests_rates_32k
    elif max_len == 128:
        long_ctx_requests_rates = requests_rates_128k
    for req_rate in long_ctx_requests_rates:
        for workload in ['code', 'conv']:
            trace_file = f'traces/rr_{workload}_{req_rate}.csv'
            long_trace_file = f'traces/rr_{workload}{max_len}k_{req_rate}.csv'
            if not os.path.exists(long_trace_file) or overwrite:
                with open(trace_file, 'r') as f:
                    with open(long_trace_file, 'w') as f2:
                        for line in f:
                            if line.startswith('request_id'):
                                f2.write(line)
                                continue
                            req_id, req_type, app_id, arrival_time, batch_size, prompt_size, token_size = line.strip().split(',')
                            f2.write(f'{req_id},{req_type},{app_id},{arrival_time},{batch_size},{int(prompt_size)*times},{int(token_size)*times}\n')
print("Generated long context traces at different request rates")
# %%
# Run simulation on 8k
# requests_rates_8k = list(range(10, 24, 1)) +
requests_rates_8k = [10, 13, 15, 17, 18, 19, 20, 21, 22, 23]
overwrite = False
alogs = ["mixed-sarathi", "mixed-sarathi-2048", "prefetch-mixed", "prefetch-mixed-2048"]
for algo in alogs:
    if "mixed-sarathi" in algo:
        hw_node = H100
    else:
        hw_node = Our_3D
    run_simulator(
        eval_model = llama70,
        hardware_node = hw_node,
        scheduler_algos = [algo],
        workloads = ['conv'],
        req_rates = requests_rates_8k,
        ctx_lens=['8k'],
        sim_method = "llmcompass",
        prefill_chunk=1024,
        num_nodes=16,
        overwrite=overwrite,
    )
    print(f"Finished simulation running 8k traces on conv with {algo}")
# Run simulation on 32k
requests_rates_32k = list(range(1, 8, 1))
overwrite = False
for algo in alogs:
    if "mixed-sarathi" in algo:
        hw_node = H100
    else:
        hw_node = Our_3D
    run_simulator(
        eval_model = llama70,
        hardware_node = hw_node,
        scheduler_algos = [algo],
        workloads = ['conv'],
        req_rates = requests_rates_32k,
        ctx_lens=['32k'],
        sim_method = "llmcompass",
        prefill_chunk=1024,
        num_nodes=32,
        end_time=1000,
        overwrite=overwrite,
    )
    print(f"Finished simulation running 32k traces on conv with {algo}")
# Run simulation on 128k
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
overwrite = False
for algo in ["mixed-sarathi-2048", "prefetch-mixed", "prefetch-mixed-2048"]:
    if "mixed-sarathi" in algo:
        hw_node = H100
    else:
        hw_node = Our_3D
    run_simulator(
        eval_model = llama70,
        hardware_node = hw_node,
        scheduler_algos = [algo],
        workloads = ['conv'],
        req_rates = requests_rates_128k,
        ctx_lens=['128k'],
        sim_method = "llmcompass",
        prefill_chunk=1024,
        num_nodes=64,
        end_time=2000,
        overwrite=overwrite,
    )
    print(f"Finished our simulation running 128k traces on conv with {algo}")
# %%
# Draw conv llama70B results
# 3 rows: TTFT, TBT, E2E
# 9 cols:
#   - P50: 8K, 32K, 128K
#   - P90: 8K, 32K, 128K
#   - P99: 8K, 32K, 128K
requests_rates_8k = list(range(10, 24, 1))
requests_rates_32k = list(range(1, 8, 1))
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
algos = ["mixed-sarathi", "mixed-sarathi-2048", "prefetch-mixed", "prefetch-mixed-2048"]

fig, axs = plt.subplots(3, 9, figsize=(20, 5))
fig.tight_layout(pad=0.3)

percentiles = [50, 90, 99]
task = 'conv'
slo_norm = {'8k': {'TTFT': 800, 'TBT': 100, 'E2E': 800 + 100 * 1000},
            '32k': {'TTFT': 3200, 'TBT': 100, 'E2E': 3200 + 100 * 4000},
            '128k': {'TTFT': 12800, 'TBT': 100, 'E2E': 12800 + 100 * 16000}}
slo_factor = {50: 0.8, 90: 1.0, 99: 2.0}

for i, p in enumerate(percentiles):
    for j, ctx_len in enumerate(['8k', '32k', '128k']):
        if ctx_len == '8k':
            workload = task
            requests_rates = requests_rates_8k
            num_node = 16
        elif ctx_len == '32k':
            requests_rates = requests_rates_32k
            workload = f'{task}{ctx_len}'
            num_node = 32
        elif ctx_len == '128k':
            requests_rates = requests_rates_128k
            workload = f'{task}{ctx_len}'
            num_node = 64
        # get results
        y = dict()
        for algo in algos:
            y[algo] = {'TTFT': [], 'TBT': [], 'E2E': []}
            if algo == "mixed-sarathi":
                hw_node = H100
            else:
                hw_node = Our_3D
            for req_rate in requests_rates:
                file_path = f"lc_results/llama3-70B/rr_{workload}_{req_rate}/{num_node}-{hw_node.name}-{algo}/sim_results.csv"
                if algo == "mixed-sarathi":
                    if not os.path.exists(file_path):
                        y[algo]['TTFT'].append(1e6)
                        y[algo]['TBT'].append(1e6)
                        y[algo]['E2E'].append(1e6)
                        continue
                ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p])
                y[algo]['TTFT'].append(ttft_p[p] / (slo_norm[ctx_len]['TTFT'] * slo_factor[p]))
                y[algo]['TBT'].append(tbt_p[p] / (slo_norm[ctx_len]['TBT'] * slo_factor[p]))
                y[algo]['E2E'].append(ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p]))
        for row, slo in enumerate(['TTFT', 'TBT', 'E2E']):
            col = j + i * 3
            ax = axs[row, col]
            colors = ['#a6611a', '#dfc27d', '#80cdc1', '#018571']
            markers = ['x', 's', '^', 'o']
            for algo in algos:
                ax.plot(requests_rates, y[algo][slo], label=algo,
                        color=colors.pop(0), marker=markers.pop(0), linestyle='-', markersize=5)
            if col == 0:
                ax.set_ylabel(f'{slo}P{p}')
            if row == 0:
                ax.set_title(f'{ctx_len} P{p}')
            if row == 2:
                ax.set_xlabel('Request rate')

            if slo == 'TTFT':
                ax.set_ylim(0, 2.0)
            elif slo == 'TBT':
                ax.set_ylim(0, 1.2)
            elif slo == 'E2E':
                ax.set_ylim(0, 1.5)
            # if slo == 'TTFT':
            #     if ctx_len == '8k':
            #         ax.set_ylim([0, 800 * 1.5])
            #     elif ctx_len == '32k':
            #         ax.set_ylim([0, 3200 * 1.5])
            #     elif ctx_len == '128k':
            #         ax.set_ylim([0, 12800 * 1.5])
            # elif slo == 'TBT':
            #     ax.set_ylim([0, 1000])
            # elif slo == 'E2E':
            #     if ctx_len == '8k':
            #         ax.set_ylim([0, 800 + 100 * 
ax.legend()

fig.savefig('llama70B_conv.pdf', bbox_inches='tight')
# %%
# Ratio Exploration, Generate traces
random_seed = 0
requests_rates_32k = list(range(1, 8, 1))
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2]

# Generate synthetic traces
if not os.path.exists('traces/synthetic'):
    os.makedirs('traces/synthetic')
for ctx_len in ['32k', '128k']:
    if ctx_len == '32k':
        ratio_exp = [4, 6, 8]
    else:
        ratio_exp = [6, 8, 10]
    for exp in ratio_exp:
        ratio = 2**exp
        if ctx_len == '32k':
            long_ctx_requests_rates = requests_rates_32k
        elif ctx_len == '128k':
            long_ctx_requests_rates = requests_rates_128k
        for req_rate in long_ctx_requests_rates:
            trace_file = f'traces/synthetic/{ctx_len}_{ratio}_{req_rate}.csv'
            if not os.path.exists(trace_file):
                generate_traces(
                    max_requests=50000,
                    end_time=600,
                    request_rates=[req_rate],
                    pt_distributions_file=f'data/synthetic_{ctx_len}_{ratio}.csv',
                    trace_filename_template=f'traces/synthetic/{ctx_len}_{ratio}_{{}}.csv')
                print(f'Generated {trace_file}')

# %%
# Run simulation on 32k
algos = ["mixed-sarathi", "prefetch-mixed", "prefetch-thread"]
requests_rates_32k = list(range(1, 8, 1))
overwrite = False
for algo in alogs:
    if "mixed-sarathi" in algo:
        hw_node = H100
    else:
        hw_node = Our_3D
    for ratio in [16, 64, 256]:
        workload = f'synthetic_32k_{ratio}'
        run_simulator(
            eval_model = llama70,
            hardware_node = hw_node,
            scheduler_algos = [algo],
            workloads = [workload],
            req_rates = requests_rates_32k,
            ctx_lens=['32k'],
            sim_method = "llmcompass",
            prefill_chunk=1024,
            num_nodes=32,
            end_time=1000,
            overwrite=overwrite,
        )
        print(f"Finished simulation running 32k traces on {workload} with {algo}")
# Run simulation on 128k
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
overwrite = False
for algo in alogs:
    if "mixed-sarathi" in algo:
        hw_node = H100
    else:
        hw_node = Our_3D
    for ratio in [64, 256, 1024]:
        workload = f'synthetic_128k_{ratio}'
        run_simulator(
            eval_model = llama70,
            hardware_node = hw_node,
            scheduler_algos = [algo],
            workloads = [workload],
            req_rates = requests_rates_128k,
            ctx_lens=['128k'],
            sim_method = "llmcompass",
            prefill_chunk=1024,
            num_nodes=64,
            end_time=2000,
            overwrite=overwrite,
        )
        print(f"Finished simulation running 128k traces on {workload} with {algo}")