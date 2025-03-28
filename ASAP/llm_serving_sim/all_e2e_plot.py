# %%
import os
import matplotlib.pyplot as plt
import numpy as np
from hardware import H100, Our_3D
from top_utils import get_slo_ms
from generate_trace import generate_code_traces, generate_conv_traces
from run import run_simulator
from model import llama70, opt175
# %%
# Figure 13: All E2E
llama_requests_rates_8k = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
llama_requests_rates_32k = list(range(1, 10, 1))
llama_requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0, 2.2, 2.4]

opt_requests_rates_8k = [4, 6, 8, 10, 11, 12, 13, 14]
opt_requests_rates_32k = [1.0, 1.4, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
opt_requests_rates_128k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.75, 0.85, 0.95]
algos = ["mixed-sarathi", "prefetch-mixed"]

random_seed = 0

requests_rates_8k = list(set(llama_requests_rates_8k + opt_requests_rates_8k))
requests_rates_32k = list(set(llama_requests_rates_32k + opt_requests_rates_32k))
requests_rates_128k = list(set(llama_requests_rates_128k + opt_requests_rates_128k))
np.random.seed(random_seed)

# %%
overwrite = False
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

print("Generated 8k traces at all different request rates")

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
# Run simulation
overwrite = False
algos = ["mixed-sarathi", "prefetch-mixed"]
ctx_lens = ['8k', '32k', '128k']
workloads = ['conv', 'code']
prefill_chunk = 2048

# Run simulation on llama70
model = llama70
for workload in workloads:
    for ctx_len in ctx_lens:
        if ctx_len == '8k':
            requests_rates = llama_requests_rates_8k
            num_node = 16
            end_time = 500
        elif ctx_len == '32k':
            requests_rates = llama_requests_rates_32k
            num_node = 32
            end_time = 1000
        elif ctx_len == '128k':
            requests_rates = llama_requests_rates_128k
            num_node = 64
            end_time = 2000
        for algo in algos:
            if "mixed-sarathi" in algo:
                hw_node = H100
            else:
                hw_node = Our_3D
            run_simulator(
                eval_model = llama70,
                hardware_node = hw_node,
                scheduler_algos = [algo],
                workloads = [workload],
                req_rates = requests_rates,
                ctx_lens=[ctx_len],
                sim_method = "llmcompass",
                prefill_chunk=prefill_chunk,
                num_nodes=num_node,
                overwrite=overwrite,
                end_time=end_time
            )
            print(f"Finished simulation running llama70B on {ctx_len} traces on {workload} with {algo}")

# Run simulation on opt175
model = opt175
for workload in workloads:
    for ctx_len in ctx_lens:
        if ctx_len == '8k':
            requests_rates = opt_requests_rates_8k
            num_node = 24
            end_time = 500
        elif ctx_len == '32k':
            requests_rates = opt_requests_rates_32k
            num_node = 48
            end_time = 1000
        elif ctx_len == '128k':
            requests_rates = opt_requests_rates_128k
            num_node = 96
            end_time = 2000
        for algo in algos:
            if "mixed-sarathi" in algo:
                hw_node = H100
            else:
                hw_node = Our_3D
            run_simulator(
                eval_model = opt175,
                hardware_node = hw_node,
                scheduler_algos = [algo],
                workloads = [workload],
                req_rates = requests_rates,
                ctx_lens=[ctx_len],
                sim_method = "llmcompass",
                prefill_chunk=prefill_chunk,
                num_nodes=num_node,
                overwrite=overwrite,
                end_time=end_time
            )
            print(f"Finished simulation running on {ctx_len} traces on {workload} with {algo}")

# %%
# Draw Figure
fig, axs = plt.subplots(3, 6, figsize=(14, 5.5))
fig.subplots_adjust(hspace=0.5, wspace=0.34)

percentiles = [50, 90, 99]
slo_norm = {'8k':   {'TTFT': 400,  'TBT': 100, 'E2E': 400  + 100 * 250},
            '32k':  {'TTFT': 1600, 'TBT': 100, 'E2E': 1600 + 100 * 1000},
            '128k': {'TTFT': 6400, 'TBT': 100, 'E2E': 6400 + 100 * 4000}}
slo_factor = {50: 1.0, 90: 1.0, 99: 1.0}

for row, p in enumerate(percentiles):
    for i, model in enumerate(['llama3-70B', 'opt-175B']):
        for j, ctx_len in enumerate(['8k', '32k', '128k']):
            col = j + i * 3
            ax = axs[row, col]
            colors = ['#dfc27d','#80cdc1', '#a6611a','#018571', ]
            markers = ['x', 'x', 'o', 'o']
            linestyles = ['--', '--', '-', '-']
            for task in ['code', 'conv']:
                if ctx_len == '8k':
                    workload = task
                    if model == 'llama3-70B':
                        num_node = 16
                        requests_rates = llama_requests_rates_8k
                    elif model == 'opt-175B':
                        num_node = 24
                        requests_rates = opt_requests_rates_8k
                elif ctx_len == '32k':
                    workload = f'{task}{ctx_len}'
                    if model == 'llama3-70B':
                        num_node = 32
                        requests_rates = llama_requests_rates_32k
                    elif model == 'opt-175B':
                        num_node = 48
                        requests_rates = opt_requests_rates_32k
                elif ctx_len == '128k':
                    workload = f'{task}{ctx_len}'
                    if model == 'llama3-70B':
                        num_node = 64
                        requests_rates = llama_requests_rates_128k
                    elif model == 'opt-175B':
                        num_node = 96
                        requests_rates = opt_requests_rates_128k
                y = dict()
                for algo in algos:
                    # if model == 'llama3-70B' and 'conv'in workload:
                    #     algo = f'{algo}-2048'
                    y[algo] = {'TTFT': [], 'TBT': [], 'E2E': []}
                    if "mixed-sarathi" in algo:
                        hw_node = H100
                    else:
                        hw_node = Our_3D
                    for req_rate in requests_rates:
                        file_path = f"lc_results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_node.name}-{algo}/sim_results.csv"
                        ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p])
                        y[algo]['TTFT'].append(ttft_p[p] / (slo_norm[ctx_len]['TTFT'] * slo_factor[p]))
                        y[algo]['TBT'].append(tbt_p[p] / (slo_norm[ctx_len]['TBT'] * slo_factor[p]))
                        y[algo]['E2E'].append(ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p]))

                        if model == 'llama3-70B' and 'conv'in workload:
                            if req_rate == 20:
                                print(f'{model} {ctx_len} {task} {algo} {p} {ete_p[p]}')

                    labels = {'mixed-sarathi': 'Baseline', 
                              'mixed-sarathi-2048': 'Baseline', 
                              'prefetch-mixed': 'Ours',
                              'prefetch-mixed-2048': 'Ours'}
                    ax.plot(requests_rates, y[algo]['E2E'], label=f'{task.capitalize()}, {labels[algo]}',
                            markersize = 4,
                            linestyle=linestyles.pop(0), color=colors.pop(0), marker=markers.pop(0))
            # draw horizontal line 1
            ax.axhline(y=1, color='r', linestyle='--', linewidth=0.5)
            ax.set_ylabel(f'P{p} E2E')
            if row == 0:
                # large, bold title
                ax.set_title(f'{model}, {ctx_len.capitalize()}',
                                fontsize='large',
                                fontweight='bold')
                if col == 3:
                    ax.legend(loc='lower center',
                              bbox_to_anchor=(-0.2, 1.24), ncol=4,
                              fontsize='large')
            ax.set_xlabel('Request/Sec')
            
            ax.set_ylim(0, 2.0)

            ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

fig.savefig('all_ete.pdf', bbox_inches='tight')
# %%
# Fig 14: throuput improvement
algos = ["mixed-sarathi", "prefetch-mixed"]
llama_requests_rates_8k = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
llama_requests_rates_32k = list(range(1, 10, 1))
llama_requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0, 2.2, 2.4]

opt_requests_rates_8k = [4, 6, 8, 10, 11, 12, 13, 14]
opt_requests_rates_32k = [1.0, 1.4, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
opt_requests_rates_128k = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35]
max_req_rates = dict()
p=90
for row, model in enumerate(['llama3-70B', 'opt-175B']):
    max_req_rates[model] = dict()
    for j, ctx_len in enumerate(['8k', '32k', '128k']):
        max_req_rates[model][ctx_len] = dict()
        colors = ['#dfc27d','#80cdc1', '#a6611a','#018571', ]
        markers = ['x', 'x', 'o', 'o']
        linestyles = ['--', '--', '-', '-']
        for task in ['code', 'conv']:
            max_req_rates[model][ctx_len][task] = dict()
            if ctx_len == '8k':
                workload = task
                if model == 'llama3-70B':
                    num_node = 16
                    requests_rates = llama_requests_rates_8k
                elif model == 'opt-175B':
                    num_node = 24
                    requests_rates = opt_requests_rates_8k
            elif ctx_len == '32k':
                workload = f'{task}{ctx_len}'
                if model == 'llama3-70B':
                    num_node = 32
                    requests_rates = llama_requests_rates_32k
                elif model == 'opt-175B':
                    num_node = 48
                    requests_rates = opt_requests_rates_32k
            elif ctx_len == '128k':
                workload = f'{task}{ctx_len}'
                if model == 'llama3-70B':
                    num_node = 64
                    requests_rates = llama_requests_rates_128k
                elif model == 'opt-175B':
                    num_node = 96
                    requests_rates = opt_requests_rates_128k
            y = dict()
            for algo in algos:
                max_req_rates[model][ctx_len][task][algo] = dict()
                if model == 'llama3-70B' and 'conv'in workload:
                    algo = f'{algo}-2048'
                y[algo] = {'TTFT': [], 'TBT': [], 'E2E': []}
                if "mixed-sarathi" in algo:
                    hw_node = H100
                else:
                    hw_node = Our_3D
                last_req_rate = 0
                max_req_rate = 0
                for req_rate in requests_rates:
                    file_path = f"lc_results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_node.name}-{algo}/sim_results.csv"
                    ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p])
                    e2e = (ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p]))
                    y[algo]['E2E'].append(e2e)
                    # find when E2E is greater than 1
                    if e2e > 1:
                        if last_req_rate == 0:
                            raise ValueError("First req_rate is already greater than 1")
                        else:
                            # interpolate
                            max_req_rate = last_req_rate + (req_rate - last_req_rate) * (1 - last_e2e) / (e2e - last_e2e)
                            # print(f'!! {algo} max_req_rate: {max_req_rate}')
                            break
                    else:
                        # print(f'{algo}: {req_rate} {e2e}')
                        last_req_rate = req_rate
                        last_e2e = e2e
                max_req_rates[model][ctx_len][task][algo] = max_req_rate

improvements = dict()
for model in ['llama3-70B', 'opt-175B']:
    improvements[model] = dict()
    for ctx_len in ['8k', '32k', '128k']:
        improvements[model][ctx_len] = dict()
        for task in ['code', 'conv']:
            for algo in algos:
                if model == 'llama3-70B' and 'conv'in task:
                    algo = f'{algo}-2048'
                req_rate = max_req_rates[model][ctx_len][task][algo]
                if 'mixed-sarathi' in algo:
                    base_req_rate = req_rate
                else:
                    new_req_rate = req_rate
            improvements[model][ctx_len][task] = new_req_rate / base_req_rate
            print(f'{model} {ctx_len} {task}: {improvements[model][ctx_len][task]}')
fig, axs = plt.subplots(1, 2, figsize=(6, 2.2))
fig.subplots_adjust(hspace=0.38, wspace=0.34)
w = 0.2
offset = 0.1
colors = ['#dfc27d','#80cdc1', '#a6611a','#018571', ]
hatchs = ['/', '\\', '|', '-']
for row, model in enumerate(['llama3-70B', 'opt-175B']):
    ax = axs[row]
    for j, ctx_len in enumerate(['8k', '32k', '128k']):
        for k, task in enumerate(['code', 'conv']):
            if j == 0:
                ax.bar(0.7*j + k * w + offset, improvements[model][ctx_len][task], 
                       width=w, label=f'{task.capitalize()}',
                         color=colors[k], hatch=hatchs[k], edgecolor='black')
            else:
                ax.bar(0.7*j + k * w + offset, improvements[model][ctx_len][task], 
                       width=w,
                         color=colors[k], hatch=hatchs[k], edgecolor='black')
    ax.set_title(model)
    ax.legend(loc='best')
    ax.axhline(y=1, color='r', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 2.1)
    ax.set_xticks([0.2, 0.9, 1.6], ['8k', '32k', '128k'])
    ax.set_xlabel('Context Length')
    ax.set_ylabel('Throughput Improvement')
fig.savefig('throughput_improvement.pdf', bbox_inches='tight')
# %%
