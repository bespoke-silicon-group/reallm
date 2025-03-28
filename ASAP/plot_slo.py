# %%
import matplotlib.pyplot as plt

def find_min_trace_len(file_paths):
    min_len = float('inf')
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            min_len = min(min_len, len(f.readlines()))
    return min_len - 1

def get_slo_ms(file_path, percentiles=[50, 90, 99], min_len=None):
    ttft = dict()
    tbt = dict()
    ete = dict()
    with open(file_path, 'r') as f:
        i = 0
        for line in f:
            if line.startswith('request_id'):
                continue
            req_id, arrival_time, *finish_time = line.strip().split(',')
            finish_time = finish_time[:-1]
            if 'None' in finish_time:
                continue
            req_ttft = float(finish_time[0]) - float(arrival_time)
            req_ete = float(finish_time[-1]) - float(arrival_time)
            req_tbt = (float(finish_time[-1]) - float(finish_time[0])) / (len(finish_time) - 1)
            ttft[int(req_id)] = req_ttft * 1000
            tbt[int(req_id)] = req_tbt * 1000
            ete[int(req_id)] = req_ete * 1000
            if min_len and i >= min_len:
                break
            i += 1
        
    ttft_sorted = sorted(ttft.values())
    tbt_sorted = sorted(tbt.values())
    ete_sorted = sorted(ete.values())
    print(f'{file_path} {len(ete)}')
    ttft_p = dict()
    tbt_p = dict()
    ete_p = dict()
    for p in percentiles:
        ttft_p[p] = ttft_sorted[int(len(ttft_sorted) * p / 100)]
        tbt_p[p] = tbt_sorted[int(len(tbt_sorted) * p / 100)]
        ete_p[p] = ete_sorted[int(len(ete_sorted) * p / 100)]
    return ttft_p, tbt_p, ete_p
# %%
# hardware_name = '8-H100-mixed-sarathi-(1, 8, 1, 1)'
# ttft_p, tbt_p, ete_p = get_slo_ms(f'results/llama70b/rr_code_0.05/{hardware_name}/sim_results.csv')
# print(ttft_p, tbt_p, ete_p)
# %%
model = 'llama70b'
models = ['llama70b', 'deepseekv3']
algo = 'mixed-sarathi'
num_node = 8 
ctx_len = '8k'
parallelism = (1, 8, 1, 1)
workloads = ['code', 'conv']
# request_rates = [1, 5, 9]
code_request_rates = [3, 5, 7,8,  9]
conv_request_rates = [0.4, 1, 3,4, 5, 6, 7]
all_hw_node_names = ['H100',
                     'H100_fast_main', 'H100_more_compute1', 'H100_more_compute2',]

slo_norm_code = {'8k':   {'TTFT': 800,  'TBT': 30, 'E2E': 800  + 50 * 250},
            '32k':  {'TTFT': 1600, 'TBT': 100, 'E2E': 1600 + 100 * 1000},
            '128k': {'TTFT': 6400, 'TBT': 100, 'E2E': 6400 + 100 * 4000}}

slo_norm_conv = {'8k':   {'TTFT': 100,  'TBT': 30, 'E2E': 300  + 50 * 800}}

slo_factor = {50: 1.0, 90: 1.0, 99: 2.0}

percentiles = [50, 90]
fig, axes = plt.subplots(2, 2, figsize=(9, 5),)
# adjust the subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.19, hspace=0.3)

labels = {'H100': 'Baseline',
          'H100_fast_main': '2x HBM BW',
          'H100_more_compute1': '2x SA Height',
          'H100_more_compute2': '2x Cores',
          'H100_more_l2': '2x L2 Cache'}

workload = 'code'
for row, p in enumerate(percentiles):
    # for col, workload in enumerate(workloads):
    for col, model in enumerate(models):
        ax = axes[row, col]
        y = dict()
        markers = ['o', 's', 'D', '^', 'v']
        colors = ['b', 'g', 'r', 'c', 'm']
        if workload == 'code':
            slo_norm = slo_norm_code
        else:
            slo_norm = slo_norm_conv
        for hw_node_name in all_hw_node_names:
            y[hw_node_name] = {'TTFT': [], 'TBT': [], 'E2E': []}
            if workload == 'code':
                request_rates = code_request_rates
            else:
                request_rates = conv_request_rates

            if model == 'deepseekv3':
                request_rates = [1, 2, 3, 4, 6]
                num_node = 32
                parallelism = (32, 1, 1, 1)
            else:
                request_rates = [3, 5, 7, 8, 9]
                num_node = 8
                parallelism = (1, 8, 1, 1)

            for req_rate in request_rates:

                all_file_paths = [f"results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_hw_name}-{algo}-{parallelism}/sim_results.csv" for hw_hw_name in all_hw_node_names]

                min_len = find_min_trace_len(all_file_paths)
                print(min_len)

                file_path = f"results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_node_name}-{algo}-{parallelism}/sim_results.csv"
                # if workload == 'conv':
                #     min_len = 80
                ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p], min_len=min_len)
                # print(f'{hw_node_name} {workload} {req_rate} {p} {ttft_p[p]} {tbt_p[p]} {ete_p[p]}')
                y[hw_node_name]['TTFT'].append(ttft_p[p] / (slo_norm[ctx_len]['TTFT'] * slo_factor[p]))
                y[hw_node_name]['TBT'].append(tbt_p[p] / (slo_norm[ctx_len]['TBT'] * slo_factor[p]))
                y[hw_node_name]['E2E'].append(ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p]))

            ax.plot(request_rates, y[hw_node_name]['E2E'], label=f'{labels[hw_node_name]}', markersize = 4,
                    color=colors.pop(0), marker=markers.pop(0))
            # draw horizontal line 1
        ax.axhline(y=1, color='r', linestyle='--', linewidth=0.5)
        ax.set_ylabel(f'P{p} E2E', fontsize='large')
        if row == 0:
            # large, bold title
            if model == 'llama70b':
                ax.set_title(f'Llama3-70B on 8 Nodes',)
            else:
                ax.set_title(f'DeepSeek v3 on 32 Nodes',)
            if col == 3:
                ax.legend(loc='lower center',
                          bbox_to_anchor=(-0.2, 1.24), ncol=4,
                          fontsize='large')
        ax.set_xlabel('Request/Sec', fontsize='large')
        ax.legend(loc='upper left')
        
        ax.set_ylim(0, 2.0)

        ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

fig.savefig('all_ete.pdf', bbox_inches='tight')
# %%

# %%
model = 'deepseekv3'
algo = 'mixed-sarathi'
num_node = 32 
ctx_len = '8k'
parallelism = (32, 1, 1, 1)
# workloads = ['code', 'conv']
workloads = ['code',]
# request_rates = [1, 5, 9]
code_request_rates = [0.5, 1, 2, 3, 4,  6, ]
conv_request_rates = [1, ]
all_hw_node_names = ['H100',
                     'H100_fast_main', 'H100_more_compute1', 'H100_more_compute2',]

slo_norm_code = {'8k':   {'TTFT': 800,  'TBT': 30, 'E2E': 800  + 50 * 250},
            '32k':  {'TTFT': 1600, 'TBT': 100, 'E2E': 1600 + 100 * 1000},
            '128k': {'TTFT': 6400, 'TBT': 100, 'E2E': 6400 + 100 * 4000}}

slo_norm_conv = {'8k':   {'TTFT': 100,  'TBT': 30, 'E2E': 300  + 50 * 500}}

slo_factor = {50: 1.0, 90: 1.0, 99: 2.0}

percentiles = [50, 90]
fig, axes = plt.subplots(2, 1, figsize=(5, 4),)
# adjust the subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.19, hspace=0.2)

labels = {'H100': 'Baseline',
          'H100_fast_main': '2x HBM BW',
          'H100_more_compute1': '2x SA Height',
          'H100_more_compute2': '2x Cores',
          'H100_more_l2': '2x L2 Cache'}

# for row, p in enumerate(percentiles):
p = 50
for row, metric in enumerate(['TTFT',  'E2E']):
    for col, workload in enumerate(workloads):
        # ax = axes[row, col]
        ax = axes[row]
        y = dict()
        markers = ['o', 's', 'D', '^', 'v']
        colors = ['b', 'g', 'r', 'c', 'm']
        if workload == 'code':
            slo_norm = slo_norm_code
        else:
            slo_norm = slo_norm_conv
        for hw_node_name in all_hw_node_names:
            y[hw_node_name] = {'TTFT': [], 'TBT': [], 'E2E': []}
            if workload == 'code':
                request_rates = code_request_rates
            else:
                request_rates = conv_request_rates
            for req_rate in request_rates:
                all_file_paths = [f"results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_hw_name}-{algo}-{parallelism}/sim_results.csv" for hw_hw_name in all_hw_node_names]
                min_len = find_min_trace_len(all_file_paths)
                file_path = f"results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_node_name}-{algo}-{parallelism}/sim_results.csv"
                ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p], min_len=min_len)
                # print(f'{hw_node_name} {workload} {req_rate} {p} {ttft_p[p]} {tbt_p[p]} {ete_p[p]}')
                y[hw_node_name]['TTFT'].append(ttft_p[p] / (slo_norm[ctx_len]['TTFT'] * slo_factor[p]))
                y[hw_node_name]['TBT'].append(tbt_p[p] / (slo_norm[ctx_len]['TBT'] * slo_factor[p]))
                y[hw_node_name]['E2E'].append(ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p]))

            print(y[hw_node_name][metric])
            ax.plot(request_rates, y[hw_node_name][metric], label=f'{labels[hw_node_name]}', markersize = 4,
                    color=colors.pop(0), marker=markers.pop(0))
            # draw horizontal line 1
        ax.axhline(y=1, color='r', linestyle='--', linewidth=0.5)
        ax.set_ylabel(f'Normalized {metric}', fontsize='large')
        if row == 0:
            # large, bold title
            if workload == 'conv':
                workload = 'chat'
            ax.set_title(f'DeepSeek v3 on Code Workload',
                            fontsize='large',
                            fontweight='bold')
            if col == 3:
                ax.legend(loc='lower center',
                          bbox_to_anchor=(-0.2, 1.24), ncol=4,
                          fontsize='large')
        ax.set_xlabel('Request/Sec', fontsize='large')
        ax.legend(loc='upper left')
        
        if metric == 'E2E':
            ax.set_ylim(0, 3.0)
        else:
            ax.set_ylim(0, 4.0)

        ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)

fig.savefig('deepseek_ete.pdf', bbox_inches='tight')

# %%

# %%
# %%
model = 'llama70b'
algo = 'mixed-sarathi'
num_node = 8
ctx_len = '8k'
parallelism = (1, 8, 1, 1)
# workloads = ['code', 'conv']
workloads = ['conv',]
# request_rates = [1, 5, 9]
# code_request_rates = [0.5, 1, 2, 3, 4,  6, ]
conv_request_rates = [0.4, 1, 2, 3, 4, 5]
all_hw_node_names = ['H100',
                     ]

slo_norm_code = {'8k':   {'TTFT': 800,  'TBT': 30, 'E2E': 800  + 50 * 250},
            '32k':  {'TTFT': 1600, 'TBT': 100, 'E2E': 1600 + 100 * 1000},
            '128k': {'TTFT': 6400, 'TBT': 100, 'E2E': 6400 + 100 * 4000}}

slo_norm_conv = {'8k':   {'TTFT': 100,  'TBT': 30, 'E2E': 300  + 50 * 500}}

slo_factor = {50: 1.0, 90: 1.0, 99: 2.0}

percentiles = [50, 90]
fig, axes = plt.subplots(1, 2, figsize=(7, 2),)
# adjust the subplots
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.29, hspace=0.2)

labels = {'H100': 'Baseline',
          'H100_fast_main': '2x HBM BW',
          'H100_more_compute1': '2x SA Height',
          'H100_more_compute2': '2x Cores',
          'H100_more_l2': '2x L2 Cache'}

# for row, p in enumerate(percentiles):
p = 50
for row, metric in enumerate(['TTFT',  'E2E']):
    for col, workload in enumerate(workloads):
        # ax = axes[row, col]
        ax = axes[row]
        y = dict()
        markers = ['o', 's', 'D', '^', 'v']
        colors = ['b', 'g', 'r', 'c', 'm']
        if workload == 'code':
            slo_norm = slo_norm_code
        else:
            slo_norm = slo_norm_conv
        for hw_node_name in all_hw_node_names:
            y[hw_node_name] = {'TTFT': [], 'TBT': [], 'E2E': []}
            if workload == 'code':
                request_rates = code_request_rates
            else:
                request_rates = conv_request_rates
            for req_rate in request_rates:
                all_file_paths = [f"results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_hw_name}-{algo}-{parallelism}/sim_results.csv" for hw_hw_name in all_hw_node_names]
                min_len = find_min_trace_len(all_file_paths)
                file_path = f"results/{model}/rr_{workload}_{req_rate}/{num_node}-{hw_node_name}-{algo}-{parallelism}/sim_results.csv"
                ttft_p, tbt_p, ete_p = get_slo_ms(file_path, percentiles=[p], min_len=min_len)
                # print(f'{hw_node_name} {workload} {req_rate} {p} {ttft_p[p]} {tbt_p[p]} {ete_p[p]}')
                y[hw_node_name]['TTFT'].append(ttft_p[p] / (slo_norm[ctx_len]['TTFT'] * slo_factor[p]))
                y[hw_node_name]['TBT'].append(tbt_p[p] / (slo_norm[ctx_len]['TBT'] * slo_factor[p]))
                y[hw_node_name]['E2E'].append(ete_p[p] / (slo_norm[ctx_len]['E2E'] * slo_factor[p]))

            print(y[hw_node_name][metric])
            ax.plot(request_rates, y[hw_node_name][metric], label=f'{labels[hw_node_name]}', markersize = 4,
                    color=colors.pop(0), marker=markers.pop(0))
            # draw horizontal line 1
        ax.axhline(y=1, color='r', linestyle='--', linewidth=0.5)
        ax.set_ylabel(f'Normalized {metric}', fontsize='large')
        if row == 0:
            ax.text(1.1, 1.1, 'Llama3-70B on Conversation Workload',
                    ha='center', va='center', 
                    fontsize='large', 
                    transform=ax.transAxes)
            # ax.set_title(f'Llama3-70B on Conversation Workload', loc = 'right')

                            # fontweight='bold',
                            

            if col == 3:
                ax.legend(loc='lower center',
                          bbox_to_anchor=(-0.2, 1.24), ncol=4,
                          fontsize='large')
        ax.set_xlabel('Request/Sec', fontsize='large')
        # ax.legend(loc='upper left')
        
        if metric == 'E2E':
            ax.set_ylim(0, 3.0)
        else:
            ax.set_ylim(0, 5.0)

        ax.grid(color = 'lightgray', linestyle = '--', linewidth = 0.5)
        ax.set_xticks([1, 2, 3, 4, 5])

fig.savefig('conv.pdf', bbox_inches='tight')
# %%

# %%
