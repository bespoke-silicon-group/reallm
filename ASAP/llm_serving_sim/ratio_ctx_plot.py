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
# Ratio Exploration, Generate traces
random_seed = 0

requests_rates_16k = list(range(1, 15, 2))
requests_rates_32k = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
requests_rates_64k = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0,]
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

ratio_requests_rates_16k = {16: [5, 6, 7, 8], 32: [7, 8, 9],
                      64: [7, 8, 9], 128: [7, 8, 9, 9.5],
                      256: [5, 7, 9, 9.5, 10]}
ratio_requests_rates_32k = {16: [2.5, 3.0, 3.5, 4.0], 32: [3.5, 4.0, 5.0, 6.0],
                      64: [4.0, 5.0, 6.0, 7.0], 128: [6.0, 7.0, 8.0, 9.0],
                        256: [8.0, 9.0, 10.0]}
ratio_requests_rates_64k = {16: [1.5, 2.0, 2.5], 32: [2.5, 3.0, 3.5],
                        64: [3.5, 4.0, 5.0, 6.0], 128: [5.0, 6.0, 7.0, 8.0],
                            256: [7.0]}   
ratio_requests_rates_128k = {16: [0.7, 0.9, 1.4, 1.5], 32: [1.0, 1.1, 1.4, 1.6, 1.8],
                        64: [1.4, 1.6, 1.8, 2.0, 2.2], 128: [2.0, 2.2, 2.4],
                            256: [2.4, 2.6, 2.8]}

for ratio in [16, 32, 64, 128, 256]:
    new_requests_rates_16k = ratio_requests_rates_16k[ratio]
    new_requests_rates_32k = ratio_requests_rates_32k[ratio]
    new_requests_rates_64k = ratio_requests_rates_64k[ratio]
    new_requests_rates_128k = ratio_requests_rates_128k[ratio]
    requests_rates_16k.extend(new_requests_rates_16k)
    requests_rates_32k.extend(new_requests_rates_32k)
    requests_rates_64k.extend(new_requests_rates_64k)
    requests_rates_128k.extend(new_requests_rates_128k)
requests_rates_16k = sorted(requests_rates_16k)
requests_rates_32k = sorted(requests_rates_32k)
requests_rates_64k = sorted(requests_rates_64k)
requests_rates_128k = sorted(requests_rates_128k)

# Generate synthetic traces
if not os.path.exists('traces/synthetic'):
    os.makedirs('traces/synthetic')
for ctx_len in ['16k', '32k', '64k', '128k']:
    if ctx_len == '32k':
        ratio_exp = [4, 5, 6, 7, 8]
    else:
        ratio_exp = [4, 5, 6, 7, 8]
    for exp in ratio_exp:
        ratio = 2**exp
        if ctx_len == '32k':
            long_ctx_requests_rates = requests_rates_32k
        elif ctx_len == '128k':
            long_ctx_requests_rates = requests_rates_128k
        elif ctx_len == '64k':
            long_ctx_requests_rates = requests_rates_64k
        elif ctx_len == '16k':
            long_ctx_requests_rates = requests_rates_16k
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
# Run simulator
overwrite = False
prefill_chunk = 2048
algos = ['prefetch-mixed', 'mixed-sarathi']
ctx_lens = ['16k', '32k', '64k', '128k']
ratios = [16, 32, 64, 128, 256]

# Llama 70
for ctx_len in ctx_lens:
    for algo in algos:
        if "mixed-sarathi" in algo:
            hw_node = H100
        else:
            hw_node = Our_3D
        if ctx_len == '16k':
            num_node = 32
            req_rates = requests_rates_16k
        elif ctx_len == '32k':
            num_node = 32
            req_rates = requests_rates_32k
        elif ctx_len == '64k':
            num_node = 64
            req_rates = requests_rates_64k
        elif ctx_len == '128k':
            num_node = 64
            req_rates = requests_rates_128k
        for ratio in ratios:
            workload = f'synthetic_{ctx_len}_{ratio}'
            run_simulator(
                eval_model = llama70,
                hardware_node = hw_node,
                scheduler_algos = [algo],
                workloads = [workload],
                req_rates = req_rates,
                ctx_lens=[ctx_len],
                sim_method = "llmcompass",
                prefill_chunk=prefill_chunk,
                num_nodes=num_node,
                end_time=1000,
                overwrite=overwrite,
            )
            print(f"Finished simulation running Llama70B {ctx_len} traces on {workload} with {algo}")

# opt 175
for ctx_len in ctx_lens:
    for algo in algos:
        if "mixed-sarathi" in algo:
            hw_node = H100
        else:
            hw_node = Our_3D
        if ctx_len == '16k':
            num_node = 48
            req_rates = requests_rates_16k
        elif ctx_len == '32k':
            num_node = 48
            req_rates = requests_rates_32k
        elif ctx_len == '64k':
            num_node = 96
            req_rates = requests_rates_64k
        elif ctx_len == '128k':
            num_node = 96
            req_rates = requests_rates_128k
        for ratio in ratios:
            workload = f'synthetic_{ctx_len}_{ratio}'
            run_simulator(
                eval_model = opt175,
                hardware_node = hw_node,
                scheduler_algos = [algo],
                workloads = [workload],
                req_rates = req_rates,
                ctx_lens=[ctx_len],
                sim_method = "llmcompass",
                prefill_chunk=prefill_chunk,
                num_nodes=num_node,
                end_time=1000,
                overwrite=overwrite,
            )
            print(f"Finished simulation running OPT-175B {ctx_len} traces on {workload} with {algo}")

# %%
# Fig: ratio vs ctx Draw ratio explore
###########################################################
requests_rates_16k = list(range(1, 15, 2))
requests_rates_32k = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
requests_rates_64k = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0,]
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

algos = ["mixed-sarathi",  "prefetch-mixed"]

fig, axs = plt.subplots(4, 4, figsize=(7, 5))
fig.subplots_adjust(hspace=0.32, wspace=0.35)

slo_norm = {'8k':   {'TTFT': 400,  'TBT': 100, 'E2E': 400  + 100 * 250},
            '32k':  {'TTFT': 1600, 'TBT': 100, 'E2E': 1600 + 100 * 1000},
            '128k': {'TTFT': 6400, 'TBT': 100, 'E2E': 6400 + 100 * 4000}}
slo_factor = {50: 1.0, 90: 1.0, 99: 1.0}
p = 90

ratios = [16, 32, 64, 128]
for j, ctx_len in enumerate(['32k', '128k']):
    if ctx_len == '32k':
        requests_rates = requests_rates_32k
        workload = f'{ctx_len}'
        num_node = 32
    elif ctx_len == '128k':
        requests_rates = requests_rates_128k
        workload = f'{ctx_len}'
        num_node = 64
    for row, ratio in enumerate(ratios):
        # get results
        y = dict()
        for algo in algos:
            y[algo] = {'TTFT': [], 'TBT': [], 'E2E': []}
            if "mixed-sarathi" in algo:
                hw_node = H100
            else:
                hw_node = Our_3D
            for req_rate in requests_rates:
                file_path = f"lc_results/llama3-70B/{workload}_{ratio}_{req_rate}/{num_node}-{hw_node.name}-{algo}/sim_results.csv"
                ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p])
                y[algo]['TTFT'].append(ttft_p[p] / (slo_norm[ctx_len]['TTFT'] * slo_factor[p]))
                y[algo]['TBT'].append(tbt_p[p] / (slo_norm[ctx_len]['TBT'] * slo_factor[p]))
                y[algo]['E2E'].append(ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p]))

        for i, slo in enumerate(['TTFT', 'E2E']):
            col = i * 2 + j
            ax = axs[row, col]
            colors = ['#a6611a', '#018571', '#80cdc1',]
            markers = ['x', 'o', 'o']
            for algo in algos:
                labels = {'mixed-sarathi': 'Baseline', 
                          'mixed-sarathi-2048': 'Baseline-2048', 
                          'prefetch-mixed': 'Ours',
                            'prefetch-thread': 'Prefetch-Thread'}
                ax.plot(requests_rates, y[algo][slo], label=labels[algo],
                        color=colors.pop(0), marker=markers.pop(0), linestyle='-', 
                        markersize=3)
            if col == 0:
                ax.set_ylabel(f'{ratio}:1', fontsize=12)
            if row == 0:
                ax.set_title(f'{ctx_len} {slo}')
            if row == 0:
                if col ==  2:
                    ax.legend(loc='lower center',
                              bbox_to_anchor=(-0.1, 1.35), ncol=3,
                              fontsize='large')
            if row == 3:
                ax.set_xlabel('Request/Sec')

            if slo == 'TTFT':
                ax.set_ylim(0, 2.0)
            elif slo == 'TBT':
                ax.set_ylim(0, 1.0)
            elif slo == 'E2E':
                if ratio == 16:
                    ax.set_ylim(0, 2)
                elif ratio == 32:
                    ax.set_ylim(0, 1)
                elif ratio == 64:
                    ax.set_ylim(0, 1)
                elif ratio == 128:
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(0, 1)

            ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

fig.savefig('ratio_explore.pdf', bbox_inches='tight')


# %%
algos = ["mixed-sarathi",  "prefetch-mixed"]
requests_rates_16k = {16: [3, 5, 6, 7, 8], 32: [5, 7, 8, 9],
                      64: [5, 7, 8, 9], 128: [5, 7, 8, 9, 9.5],
                      256: [5, 7, 9, 9.5, 10]}
requests_rates_32k = {16: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 
                      32: [2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
                      64: [2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0], 
                      128: [2.5, 3.0, 3.5, 6.0, 7.0, 8.0, 9.0],
                        256: [3.0, 3.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}
requests_rates_64k = {16: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], 
                      32: [0.5, 1.0, 2.0, 2.5, 3.0, 3.5],
                        64: [1.0, 1.5, 2.0, 3.5, 4.0, 5.0, 6.0], 
                        128: [0.5, 0.7, 1.0, 1.5,  5.0, 6.0, 7.0, 8.0],
                            256: [2.0, 7.0, 8.0, 9.0]}   
requests_rates_128k = {16: [0.1, 0.5, 0.7, 0.9, 1.4, 1.5], 
                       32: [0.5, 1.0, 1.1, 1.4, 1.6, 1.8],
                        64: [0.5, 1.0, 1.4, 1.6, 1.8, 2.0, 2.2], 
                        128: [1.0, 1.4, 2.0, 2.2, 2.4],
                        256: [0.5, 0.9, 1.5, 2.4, 2.6, 2.8]}

slo_norm = {'8k':   {'TTFT': 400,  'TBT': 100, 'E2E': 400 + 100 * 250},
            '16k':  {'TTFT': 800,  'TBT': 100, 'E2E': 800 + 100 * 500},
            '32k':  {'TTFT': 1600, 'TBT': 100, 'E2E': 1600 + 100 * 1000},
            '64k':  {'TTFT': 3200, 'TBT': 100, 'E2E': 3200 + 100 * 2000},
            '128k': {'TTFT': 6400, 'TBT': 100, 'E2E': 6400 + 100 * 4000}}
slo_factor = {50: 1.0, 90: 1.0, 99: 1.0}
p = 90

speedups = dict()
for model in [llama70, opt175]:
    speedups[model.name] = dict()
    for j, ctx_len in enumerate(['16k', '32k', '64k', '128k']):
        speedups[model.name][ctx_len] = dict()
        ratios = [16, 32, 64, 128, 256]
        if ctx_len == '16k':
            workload = f'{ctx_len}'
            num_node = 32
            if model == opt175:
                num_node = 48
            requests_rates = requests_rates_16k
        elif ctx_len == '32k':
            workload = f'{ctx_len}'
            num_node = 32
            if model == opt175:
                num_node = 48
            requests_rates = requests_rates_32k
        elif ctx_len == '64k':
            workload = f'{ctx_len}'
            num_node = 64
            requests_rates = requests_rates_64k
            if model == opt175:
                num_node = 96
        elif ctx_len == '128k':
            workload = f'{ctx_len}'
            num_node = 64
            requests_rates = requests_rates_128k
            if model == opt175:
                num_node = 96
        for col, ratio in enumerate(ratios):
            # get results
            y = dict()
            max_req_rates = dict()
            for algo in algos:
                max_req_rates[algo] = 0
                y[algo] = {'TTFT': [], 'TBT': [], 'E2E': []}
                if "mixed-sarathi" in algo:
                    hw_node = H100
                else:
                    hw_node = Our_3D
                last_req_rate = 0
                last_e2e = 0
                for req_rate in requests_rates[ratio]:
                    # file_path = f"lc_results/llama3-70B/syn/{workload}_{ratio}_{req_rate}/{num_node}-{hw_node.name}-{algo}/sim_results.csv"
                    file_path = f"lc_results/{model.name}/{workload}_{ratio}_{req_rate}/{num_node}-{hw_node.name}-{algo}/sim_results.csv"
                    ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p])
                    y[algo]['TTFT'].append(ttft_p[p] / (slo_norm[ctx_len]['TTFT'] * slo_factor[p]))
                    y[algo]['TBT'].append(tbt_p[p] / (slo_norm[ctx_len]['TBT'] * slo_factor[p]))
                    e2e = ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p])
                    y[algo]['E2E'].append(e2e)
                    # find when E2E is greater than 1
                    if e2e > 1:
                        if last_req_rate == 0:
                            print(f'!! {algo} max_req_rate: {req_rate}')
                            raise ValueError("First req_rate is already greater than 1")
                        else:
                            # interpolate
                            max_req_rate = last_req_rate + (req_rate - last_req_rate) * (1 - last_e2e) / (e2e - last_e2e)
                            # print(f'!! {algo} max_req_rate: {max_req_rate}')
                            max_req_rates[algo] = max_req_rate
                            break
                    else:
                        # print(f'{algo}: {req_rate} {e2e}')
                        last_req_rate = req_rate
                        last_e2e = e2e


            speedups[model.name][ctx_len][ratio] = max_req_rates['prefetch-mixed'] / max_req_rates['mixed-sarathi'] 
# Speed up heatmap
fig, axes = plt.subplots(1, 2, figsize=(6, 2.7))
import seaborn as sns
import pandas as pd
for ax, model in zip(axes, [llama70, opt175]):
    speedup_df = pd.DataFrame(speedups[model.name])
    sns.heatmap(speedup_df, annot=True, fmt=".2f", 
                ax=ax, cmap='viridis')
    # remove cbar
    cbar = ax.collections[0].colorbar
    cbar.remove()
    ax.set_xlabel('Context Length', fontsize=10)
    ax.set_title(f'{model.name} E2E Speedup', fontsize=10)
axes[0].set_ylabel('Input-Output Ratio', fontsize=10)
fig.savefig('ratio_vs_ctx.pdf', bbox_inches='tight')



# %% 