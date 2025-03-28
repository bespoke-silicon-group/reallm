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

colors = ['#a6611a','#dfc27d','#80cdc1','#018571']
markers = ['o', 's', '^', 'v', 'D', 'P']

# %%
# Figure 3: latency and breakdown given different context lengths
# To show the bottleneck of mixed continuous batching when context length is long
###################################################################################
ratios = [4, 100]
ctx_lens = ['8k', '16k', '32k', '64k', '128k']
prefill_chunk = 512
num_decodes = [16, 32, 64]

hw = Hardware(H100, 64, io_algo='multishot')
hw_sim = HardwareSim(hw, 'llmcompass', 'mixed-sarathi')

all_latencies = dict()
for num_decode in num_decodes:
    all_latencies[num_decode] = []
    lat_breakdown = []
    norm = None
    for ctx_len in ctx_lens:
        prefill_kernel = LLMKernel(phase='prefill', model=llama70, n=prefill_chunk, l_start=0, l_end=llama70.num_layers-1)
        decode_kernel = LLMKernel(phase='decode', model=llama70, n=num_decode, l_start=0, l_end=llama70.num_layers-1, ctx=[int(ctx_len[:-1]) * 1000]*num_decode)
        sim_kernel = SimKernel(prefill_kernel, decode_kernel)
        latency, _ = hw_sim.run(sim_kernel)
        perf = hw_sim.kernels_perf[-1]
        fc_lat = perf.prefill_fc_mem_latency + perf.decode_fc_mem_latency + perf.prefill_fc_compute_latency + perf.decode_fc_compute_latency
        attn_lat = perf.prefill_attn_mem_latency + perf.decode_attn_mem_latency + perf.prefill_attn_compute_latency + perf.decode_attn_compute_latency
        io_lat = perf.prefill_io_latency + perf.decode_io_latency
        # print(f'Context Length: {ctx_len}, Prefill Chunk: {prefill_chunk}, Num Decode: {num_decode}')
        # print(f'Latency: {latency}, FC Latency: {fc_lat} ({fc_lat/latency:.3}), Attn Latency: {attn_lat} ({attn_lat/latency:.3}), IO Latency: {io_lat} ({io_lat/latency:.3})')
        if norm is None:
            norm = latency
        all_latencies[num_decode].append(latency/norm)
        breakdown = {'Attention': attn_lat/latency, 'Linear': fc_lat/latency, 'Other': io_lat/latency}
        lat_breakdown.append(breakdown)

colors = ['#a6611a','#dfc27d','#80cdc1','#018571']
markers = ['o', 'x', 's', '^', 'v', 'D', 'P']
fig, axes = plt.subplots(1, 2, figsize=(6, 2))
fig.subplots_adjust(wspace=0.35)
# latency
ax = axes[0]
for i, num_decode in enumerate(num_decodes):
    ax.plot(ctx_lens, all_latencies[num_decode], label=f'{num_decode} Decode Tasks', marker=markers[i], color=colors[i])
ax.set_xlabel('Context Length (Tokens)')
ax.set_ylabel('Normalized Latency')
ax.legend()
# breakdown stacked bar
ax = axes[1]
bottom = [0] * len(ctx_lens)
for i, key in enumerate(['Attention', 'Linear', 'Other']):
    ax.bar(ctx_lens, [lat_breakdown[j][key] for j in range(len(ctx_lens))], bottom=bottom, label=key, color=colors[i], edgecolor='k')
    bottom = [bottom[j] + lat_breakdown[j][key] for j in range(len(ctx_lens))]
ax.set_xlabel('Context Length (Tokens)')
ax.set_ylabel('Latency Breakdown')
ax.legend()

fig.savefig('mixed_continuous_bottleneck.pdf', bbox_inches='tight')
# %%
# Figure 4: more activated requests and tokens as ctx length increases, estimated from slo
###################################################################################
ttft_per_k = 400 / 100 # ms to 0.1 s
tbt = 100 / 100 # ms to 0.1 s
num_traces = 500
end_time = 100000 # 10000 s
req_rate = 1
period = int(1 / req_rate * 10) # 0.1 s
ctx_lens = ['8k', '16k', '24k', '32k', '40k', '48k', '56k', '64k']
ratios = [100]

max_tokens = dict()
max_reqs = dict()

for ctx_len in ctx_lens:
    for ratio in ratios:
        ctx = int(ctx_len[:-1]) * 1000
        input_len = int(ctx * ratio / (ratio + 1))
        output_len = ctx - input_len

        num_tokens = [0] * end_time
        num_reqs = [0] * end_time

        arrival_time = 0
        ttft = int(ttft_per_k * input_len / 1000)
        while arrival_time < end_time:
            finish_time = []
            for j in range(output_len):
                finish_time.append(int(arrival_time + ttft + j * tbt))
            t_start = arrival_time
            tokens = input_len
            for t_end in finish_time:
                if t_end >= end_time:
                    break
                for t in range(t_start, t_end):
                    num_tokens[t] += tokens
                    num_reqs[t] += 1
                t_start = t_end
                tokens += 1
            arrival_time += period
        print(f'Context Length: {ctx_len}, Ratio: {ratio}:1')
        print(f'Max tokens: {max(num_tokens)}')
        print(f'Max reqs: {max(num_reqs)}')
        max_tokens[(ctx_len, ratio)] = max(num_tokens)
        max_reqs[(ctx_len, ratio)] = max(num_reqs)
fig, axes = plt.subplots(1, 2, figsize=(6, 2))
fig.subplots_adjust(wspace=0.3)
ratios = [100]
x = [int(ctx_len[:-1]) * 1000 for ctx_len in ctx_lens]
# max reqs
ax = axes[0]
for ratio in ratios:
    ax.plot(x, [max_reqs[(ctx_len, ratio)] for ctx_len in ctx_lens], label=f'{ratio}:1', 
            marker=markers[0], color=colors[0])
ax.set_ylabel('Max Activated Requests')
# max tokens
ax = axes[1]
for ratio in ratios:
    ax.plot(x, [max_tokens[(ctx_len, ratio)] for ctx_len in ctx_lens], label=f'{ratio}:1',
            marker=markers[1], color=colors[1])
ax.set_ylabel('Max Tokens in KV Cache')

for ax in axes:
    ax.set_xlabel('Context Length (Tokens)')
    ax.set_ylim(0, )
    # add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('max_reqs_tokens.pdf', bbox_inches='tight')

# %%
# Fig 5: Memory and compute utilization over time, short and long context
# From kernel_perf.pkl
###################################################################################
mem_time = {'8K': {'Linear': 2.85, 'Attention': 0.79},
            '128K': {'Linear': 2.85, 'Attention': 15.71}}
compute_time = {'8K': {'Linear': 5.23, 'Attention': 0.06},
                '128K': {'Linear': 5.28, 'Attention': 0.44}}

mem_util = dict()
compute_util = dict()
time_ratio = dict()
for context in ['8K', '128K']:
    mem_util[context] = dict()
    compute_util[context] = dict()
    total_time = 0
    for layer in ['Linear', 'Attention']:
        portion_time = max(mem_time[context][layer], compute_time[context][layer])
        mem_util[context][layer] = mem_time[context][layer] / portion_time
        compute_util[context][layer] = compute_time[context][layer] / portion_time
        total_time += portion_time
    time_ratio[context] = dict()
    for layer in ['Linear', 'Attention']:
        portion_time = max(mem_time[context][layer], compute_time[context][layer])
        time_ratio[context][layer] = portion_time / total_time

fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))
fig.subplots_adjust(wspace=0.3)
for ax, context in zip(axes, ['8K', '128K']):
    x = [0, time_ratio[context]['Linear'], time_ratio[context]['Linear'], 1.0]
    y1 = [mem_util[context]['Linear'], mem_util[context]['Linear'], mem_util[context]['Attention'], mem_util[context]['Attention']]
    y2 = [compute_util[context]['Linear'], compute_util[context]['Linear'], compute_util[context]['Attention'], compute_util[context]['Attention']]
    ax.plot(x, y1, label='Memory', marker='o')
    ax.plot(x, y2, label='Compute', marker='o')
    ax.vlines(time_ratio[context]['Linear'], 0, 1.05, linestyles='dashed', colors='grey')
    ax.fill_between([0, time_ratio[context]['Linear']], 0, 1.05, color='orange', alpha=0.1, label='Linear')
    ax.fill_between([time_ratio[context]['Linear'], 1], 0, 1.05, color='blue', alpha=0.1, label='Attention')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{context} Context')
    ax.set_ylabel('Utilization')
    # remove xticks
    # ax.set_xticks([])
    ax.set_xlabel('Time')
    ax.legend()
plt.savefig('util.pdf', bbox_inches='tight')

# %%
# Figure 10: Plot chat and code distribution
###################################################################################
download_azure_llm_traces()

TRACE_NAMES = [
    "Coding",
    "Conversation",
]
TRACE_FILENAMES = [
    "data/code_distributions.csv",
    "data/conv_distributions.csv",
]

# Read all traces
df_traces = {}
for trace_name, trace_filename in zip(TRACE_NAMES, TRACE_FILENAMES):
    df_traces[trace_name] = pd.read_csv(trace_filename, parse_dates=["TIMESTAMP"])
for trace_name, df_trace in df_traces.items():
    df_trace['TotalTokens'] = df_trace['ContextTokens'] + df_trace['GeneratedTokens']
    df_trace['PrefillRatio'] = df_trace['ContextTokens'] / df_trace['GeneratedTokens']

code_ratio = df_traces["Coding"]['PrefillRatio'].tolist()
code_ratio_mean = np.mean(code_ratio)
code_ratio_median = np.median(code_ratio)
cov_ratio = df_traces["Conversation"]['PrefillRatio'].tolist()
cov_ratio_mean = np.mean(cov_ratio)
cov_ratio_median = np.median(cov_ratio)

code_total = df_traces["Coding"]['TotalTokens'].tolist()
code_total_mean = np.mean(code_total)
code_total_median = np.median(code_total)
cov_total = df_traces["Conversation"]['TotalTokens'].tolist()
cov_total_mean = np.mean(cov_total)
cov_total_median = np.median(cov_total)

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.))
fig.tight_layout()
ax = axes[1]
bins = 10**np.linspace(-1,3,100)
ax.hist(code_ratio, bins=bins, alpha=0.5, label=f"Code (median: {code_ratio_median:.1f})", weights=np.ones_like(code_ratio) / len(code_ratio))
ax.hist(cov_ratio, bins=bins, alpha=0.5, label=f"Conv (median: {cov_ratio_median:.1f})", weights=np.ones_like(cov_ratio) / len(cov_ratio))
ax.legend()
ax.set_xlabel("Input:Output Ratio")
ax.set_xscale('log')
# ax.set_ylabel("Density")

ax = axes[0]
bins = 10**np.linspace(1,4,100)
# ax.hist(code_ctx, bins=bins, alpha=0.5, label=f"Coding (avg: {code_ctx_median:.2f})", weights=np.ones_like(code_ctx) / len(code_ctx))
# ax.hist(cov_ctx,  bins=bins, alpha=0.5, label=f"Chat (avg: {cov_ctx_median:.2f})", weights=np.ones_like(cov_ctx) / len(cov_ctx))
ax.hist(code_total, bins=bins, alpha=0.5, label=f"Code (median: {code_total_median:.0f})", weights=np.ones_like(code_total) / len(code_total))
ax.hist(cov_total,  bins=bins, alpha=0.5, label=f"Conv (median: {cov_total_median:.0f})", weights=np.ones_like(cov_total) / len(cov_total))
ax.legend()
ax.set_xlabel("Context Length(Tokens)")
ax.set_xscale('log')
ax.set_ylabel("Density")

fig.savefig('chat_code_dist.pdf', bbox_inches='tight')
# %%
# Figure 11: Synthetic traces
###################################################################################
random_seed = 0

np.random.seed(random_seed)
# key: context length exponent, value: (mean, max)
ctx_exp = {
            '16k': (12, 14),
            '32k': (13, 15), 
            '64k': (14, 16),
           '128k': (15, 17)}
ctx_ratio_exp = {'16k': [4, 5, 6, 7, 8],
                 '32k': [4, 5, 6, 7, 8], 
                 '64k': [4, 5, 6, 7, 8],
                 '128k': [4, 5, 6, 7, 8]}

fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.))
fig.tight_layout()
all_ctx = dict()
all_ratio = dict()
# ctx len
ax = axes[0]
for ctx_len in ['16k', '32k', '64k', '128k']:
    mu = ctx_exp[ctx_len][0]
    maxx = ctx_exp[ctx_len][1]
    sigma = 1
    s = np.random.normal(mu, sigma, 10000)
    np.clip(s, 0, maxx, out=s)
    print(f'Num of {ctx_len}: {len([x for x in s if x == maxx])}')
    ctxx = 2**s
    ctxx = ctxx.astype(int)
    ax.hist(ctxx, bins=2**np.linspace(8, 17.5, 300), 
            alpha=0.5, 
            label=f'{2**maxx/1024:.0f}k/{2**mu/1024:.0f}k',
            weights=np.ones_like(s) / len(s))
    all_ctx[ctx_len] = ctxx
# scale legend
ax.legend(loc='upper left', fontsize=9)
ax.set_xlabel("Context Length (Tokens)")
ax.set_xscale('log')
ax.set_ylabel("Density")
# ratio
ax = axes[1]
all_ratio_exps = [4, 5, 6,7, 8]
for ratio_exp in all_ratio_exps:
    mu = ratio_exp
    sigma = 1
    s = np.random.normal(mu, sigma, 10000)
    ratioo = 2**s
    ax.hist(ratioo, bins=10**np.linspace(0.2, 4.2, 300),
            alpha=0.5, label=f'{2**ratio_exp}:1', weights=np.ones_like(s) / len(s))
    all_ratio[ratio_exp] = ratioo
ax.legend(loc='upper right', fontsize=9, ncols=1,)
ax.set_xlabel("Input:Output Ratio")
ax.set_xscale('log')
fig.savefig('synthetic_dist.pdf', bbox_inches='tight')

# Generate distribution files
for ctx_len in ['16k', '32k', '64k', '128k']:
    for exp in ctx_ratio_exp[ctx_len]:
        ratio = 2**exp
        distribution_file = f'data/synthetic_{ctx_len}_{ratio}.csv'
        if not os.path.exists(distribution_file):
            with open(distribution_file, 'w') as f:
                f.write('TIMESTAMP,ContextTokens,GeneratedTokens\n')
                for i in range(10000):
                    total_ctx = all_ctx[ctx_len][i]
                    trace_ratio = all_ratio[exp][i]
                    ctx_tokens = int(total_ctx * trace_ratio / (trace_ratio + 1))
                    gen_tokens = total_ctx - ctx_tokens
                    f.write(f'{i},{ctx_tokens},{gen_tokens}\n')
            print(f'Generated {distribution_file}')

# %%
# Fig 17: Block size sweep for prefetch-mixed
###################################################################################
# LC Simulation: prefetch-mixed batching prefill chunk size sweep
run_simulator(
    eval_model = llama70,
    hardware_node = Our_3D,
    scheduler_algos = ["prefetch-mixed-384",  "prefetch-mixed-512",
                       "prefetch-mixed-1024", "prefetch-mixed-2048",
                       "prefetch-mixed-3072", "prefetch-mixed-4096"],
    workloads = ["conv"],
    req_rates = list(range(14, 21, 2)),
    ctx_lens=['8k'],
    sim_method = "llmcompass",
    prefill_chunk=None,
    # overwrite=True,
)
# in ms
colors = ['#a6611a','#dfc27d','#80cdc1','#018571']
markers = ['o', 'x', 's', '^', 'v', 'D', 'P']
std_slo ={'ttft': {'50': 134, '90': 400, '99':  800},
          'tbt':  {'50':  40, '90': 100,  '99':  200},
          'ete':  {'50': 134, '90': 400 + 100 * 250, '99':  800}}
slo_norm = {'8k':   {'TTFT': 400,  'TBT': 100, 'E2E': 400 + 100 * 250},
            '16k':  {'TTFT': 800,  'TBT': 100, 'E2E': 800 + 100 * 500},
            '32k':  {'TTFT': 1600, 'TBT': 100, 'E2E': 1600 + 100 * 1000},
            '64k':  {'TTFT': 3200, 'TBT': 100, 'E2E': 3200 + 100 * 2000},
            '128k': {'TTFT': 6400, 'TBT': 100, 'E2E': 6400 + 100 * 4000}}
model = 'llama3-70B'
workload = 'conv'
algo = 'prefetch-mixed'
chunck_sizes = [384, 512, 1024, 2048, 3072, 4096]

req_rates = [14, 16, 18, 20]
result_dir = 'lc_results'
all_lat = dict()
for model in ['llama3-70B', 'opt-175B']:
    if model == 'llama3-70B':
        req_rates = [14, 16, 18, 20]
        num_nodes = 16
    elif model == 'opt-175B':
        req_rates = [10, 11, 12, 13]
        num_nodes = 24
    for req_rate in req_rates:
        for chunk_size in chunck_sizes:
            key = f'{model}_{req_rate}_{algo}-{chunk_size}'
            all_lat[key] = dict()
            file_path = f'./{result_dir}/{model}/rr_{workload}_{req_rate}/{num_nodes}-our-{algo}-{chunk_size}/sim_results.csv'
            ttft, tbt, ete= get_slo_ms(file_path)
            all_lat[key]['ttft'] = ttft
            all_lat[key]['tbt'] = tbt
            all_lat[key]['ete'] = ete
# first row two figures
# second row one figure
fig, axes = plt.subplots(2, 2, figsize=(6.2, 3.8))
fig.subplots_adjust(wspace=0.32, hspace=0.4)
for row, model in enumerate(['llama3-70B']):
    if model == 'llama3-70B':
        req_rates = [14, 16, 18, 20]
        num_nodes = 16
    elif model == 'opt-175B':
        req_rates = [10, 11, 12]
        num_nodes = 24
    for col, y_spec in enumerate(['ttft', 'tbt', 'ete']):
        percentile = '90'
        if col == 2:
            ax = axes[1, 0]
        else:
            ax = axes[row, col]
        # x: req_rate, y: lat
        x = chunck_sizes
        for i, req_rate in enumerate(req_rates):
            y = []
            for chunk_size in chunck_sizes:
                key = f'{model}_{req_rate}_{algo}-{chunk_size}'
                y.append(all_lat[key][y_spec][int(percentile)] / std_slo[y_spec][percentile])
            ax.plot(x, y, label=f'{req_rate} reqs/s', marker=markers[i], color=colors[i])
        ax.set_xscale('log')
        ax.set_xlabel('Prefill Chunk (Tokens)')
        ax.set_ylabel(f'P{percentile} {y_spec.upper()}', 
                     fontsize=10)
        ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)
    
    ax = axes[1, 0]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=11,
              loc='center left', bbox_to_anchor=(1.4, 0.5),)

    fig.delaxes(axes[1, 1])

    fig.savefig('prefetch_block_size.pdf', bbox_inches='tight')
# %%
# Fig 18: Draw Max 3D DRAM used
###################################################################################
requests_rates_16k = list(range(1, 15, 2))
requests_rates_32k = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
requests_rates_64k = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0,]
requests_rates_128k = [0.1, 0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

ctx_lens = ['32k', '128k']
ratios = [16, 128]
models = [llama70, opt175]

max_3d_dram = dict()
for ctx_len in ctx_lens:
    workload = f'{ctx_len}'
    max_3d_dram[ctx_len] = dict()
    if ctx_len == '16k':
        num_node = 32
        requests_rates = requests_rates_16k
    elif ctx_len == '32k':
        num_node = 32
        requests_rates = requests_rates_32k
    elif ctx_len == '64k':
        num_node = 64
        requests_rates = requests_rates_64k
    elif ctx_len == '128k':
        num_node = 64
        requests_rates = requests_rates_128k
    for ratio in ratios:
        max_3d_dram[ctx_len][ratio] = dict()
        for model in models:
            max_3d_dram[ctx_len][ratio][model.name] = []
            print(f"ctx: {ctx_len}, ratio: {ratio}, model: {model.name}")
            # get results
            y = dict()
            algo = 'prefetch-mixed'
            hw_node = Our_3D
            for req_rate in requests_rates:
                if model == llama70:
                    if ctx_len == '32k':
                        num_node = 32
                    elif ctx_len == '128k':
                        num_node = 64
                    file_path = f"lc_results/llama3-70B/{workload}_{ratio}_{req_rate}/{num_node}-{hw_node.name}-{algo}/kernel_perf.pkl"
                elif model == opt175:
                    if ctx_len == '32k':
                        num_node = 48
                    elif ctx_len == '128k':
                        num_node = 96
                    file_path = f"lc_results/opt-175B/{workload}_{ratio}_{req_rate}/{num_node}-{hw_node.name}-{algo}/kernel_perf.pkl"
                max_num_tokens = 0
                with open(file_path, 'rb') as f:
                    kernel_perfs = pickle.load(f)
                    for i, perf in enumerate(kernel_perfs):
                        kernel = perf.kernel
                        prefill_kernel = kernel.prefill_kernel
                        decode_kernel = kernel.decode_kernel
                        if prefill_kernel != None:
                            pn = prefill_kernel.n
                        else:
                            pn = 0
                        num_tokens = 0
                        if decode_kernel != None:
                            dn = decode_kernel.n
                            for n in decode_kernel.ctx:
                                num_tokens += n
                        else:
                            dn = 0
                        max_num_tokens = max(max_num_tokens, num_tokens)
                max_3d_dram[ctx_len][ratio][model.name].append(max_num_tokens)
                print(f"req_rate: {req_rate}, max_num_tokens: {max_num_tokens}")
max_3d_dram_GB = dict()
for model in models:
    max_3d_dram_GB[model.name] = dict()
    for ctx_len in ctx_lens:
        max_3d_dram_GB[model.name][ctx_len] = dict()
        for ratio in ratios:
            max_3d_dram_GB[model.name][ctx_len][ratio] = []
            for num_tokens in max_3d_dram[ctx_len][ratio][model.name]:
                byte = model.kv_cache_size_per_token_byte * num_tokens / llama70.num_layers
                GB = byte / 1024 / 1024 / 1024
                max_3d_dram_GB[model.name][ctx_len][ratio].append(GB)

fig, axs = plt.subplots(1, 2, figsize=(6, 1.5))
fig.subplots_adjust(hspace=0.4, wspace=0.27)
colors = ['#a6611a','#dfc27d','#018571','#80cdc1']
markers = ['x', '^', 'o', 's']
for i, ctx_len in enumerate(ctx_lens):
    if ctx_len == '16k':
        requests_rates = requests_rates_16k
    elif ctx_len == '32k':
        requests_rates = requests_rates_32k
    elif ctx_len == '64k':
        requests_rates = requests_rates_64k
    elif ctx_len == '128k':
        requests_rates = requests_rates_128k
    ax = axs[i]
    j = 0
    for model in models:
        for ratio in ratios:
            if model == llama70:
                ax.plot(requests_rates, max_3d_dram_GB[model.name][ctx_len][ratio], 
                        color = colors[j], markersize=3, marker=markers[j], label=f'{model.name} {ratio}:1')
            else:
                ax.plot(requests_rates, max_3d_dram_GB[model.name][ctx_len][ratio], 
                        color = colors[j], markersize=3, marker=markers[j], label=f'{model.name} {ratio}:1')
            j += 1
    
    ax.set_title(f'{ctx_len}')
    ax.set_xlabel('Request/Sec')
    ax.set_ylabel('3D DRAM Used(GB)')
    ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

axs[1].legend(loc='lower center', bbox_to_anchor=(-0.2, 1.2), ncol=2)
fig.savefig('max_3d_dram.pdf', bbox_inches='tight')

# %%
