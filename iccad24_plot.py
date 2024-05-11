# %%
import pickle, sys, math, functools
sys.path.append('micro_arch_sim/')
sys.path.append('chiplet_cloud_simulator_vlsi_numbers')
from structs.System import System
from structs.Mapping import Mapping
from structs.Performance import Performance
from matplotlib import pyplot as plt
from copy import deepcopy
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def split_sys(systems, key):
    sys_dict = dict()
    for _sys in systems:
        val = rgetattr(_sys, key)
        if isinstance(val, list):
            old_val = val
            val = str(old_val[0])
            for v in old_val[1: ]:
                val += ', '
                val += str(v)
        if val in sys_dict:
            sys_dict[val].append(_sys)
        else:
            sys_dict[val] = [_sys]
    return sys_dict

def get_batch_opt_sys(systems, max_batch=1024):
    batch_prefill_lat = dict()
    batch_prefill_tco = dict()
    batch_generate_lat = dict()
    batch_generate_tco = dict()
    batch_prefill_lat_sys = dict()
    batch_prefill_tco_sys = dict()
    batch_generate_lat_sys = dict()
    batch_generate_tco_sys = dict()
    batch = 1
    while batch <= max_batch:
        batch_prefill_lat[batch] = math.inf
        batch_prefill_tco[batch] = math.inf
        batch_generate_lat[batch] = math.inf
        batch_generate_tco[batch] = math.inf
        batch *= 2

    for _sys in systems:
        batch = 1
        while batch <= max_batch:
            if batch in _sys.batch_opt_prefill_lat:
                if _sys.batch_opt_prefill_lat[batch].prefill_latency < batch_prefill_lat[batch]:
                    batch_prefill_lat[batch] = _sys.batch_opt_prefill_lat[batch].prefill_latency
                    batch_prefill_lat_sys[batch] = _sys
            if batch in _sys.batch_opt_prefill_tco:
                if _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token < batch_prefill_tco[batch]:
                    batch_prefill_tco[batch] = _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token
                    batch_prefill_tco_sys[batch] = _sys
            if batch in _sys.batch_opt_generate_lat:
                if _sys.batch_opt_generate_lat[batch].generate_latency < batch_generate_lat[batch]:
                    batch_generate_lat[batch] = _sys.batch_opt_generate_lat[batch].generate_latency
                    batch_generate_lat_sys[batch] = _sys
            if batch in _sys.batch_opt_generate_tco:
                if _sys.batch_opt_generate_tco[batch].generate_tco_per_token < batch_generate_tco[batch]:
                    batch_generate_tco[batch] = _sys.batch_opt_generate_tco[batch].generate_tco_per_token
                    batch_generate_tco_sys[batch] = _sys
            batch *= 2
    return batch_prefill_lat_sys, batch_prefill_tco_sys, batch_generate_lat_sys, batch_generate_tco_sys

def get_opt_sys_batch(systems, target='generate_tco'):
    opt = math.inf
    for batch in systems:
        _sys = systems[batch]
        if target == 'generate_tco':
            if _sys.batch_opt_generate_tco[batch].generate_tco_per_token < opt:
                opt = _sys.batch_opt_generate_tco[batch].generate_tco_per_token
                opt_sys = _sys
                opt_batch = batch
        elif target == 'generate_lat':
            if _sys.batch_opt_generate_lat[batch].generate_latency < opt:
                opt = _sys.batch_opt_generate_lat[batch].generate_latency
                opt_sys = _sys
                opt_batch = batch
        elif target == 'prefill_tco':
            if _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token < opt:
                opt = _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token
                opt_sys = _sys
                opt_batch = batch
        elif target == 'prefill_lat':
            if _sys.batch_opt_prefill_lat[batch].prefill_latency < opt:
                opt = _sys.batch_opt_prefill_lat[batch].prefill_latency
                opt_sys = _sys
                opt_batch = batch
    return opt_sys, opt_batch

def get_latency_breakdown(perf, stage):
    lat = perf._get_micro_batch_latency(stage)
    io_ratio = lat.communication / lat.total
    layer_compute = 0
    layer_mem = 0
    for k in ['atten_qkv', 'atten_matmul1', 'atten_matmul2', 'atten_fc', 'fc1', 'fc2']:
        mm = getattr(lat, k)
        if mm.block_ldst_time > mm.block_comp_time:
            layer_mem += mm.block_ldst_time
        else:
            layer_compute += mm.block_comp_time
    mem_ratio = (1 - io_ratio) * layer_mem / (layer_compute + layer_mem)
    compute_ratio = (1 - io_ratio) * layer_compute / (layer_compute + layer_mem)
    if stage == 'prefill':
        latency = perf.prefill_latency
    else:
        latency = perf.generate_latency
    t_io = io_ratio * latency
    t_compute = compute_ratio * latency
    t_mem = mem_ratio * latency
    return t_io, t_compute, t_mem

def get_kernel_latency(perf, stage):
    if stage == 'prefill':
        latency = perf.prefill_latency
    else:
        latency = perf.generate_latency 
    lat = perf._get_micro_batch_latency(stage)
    breakdown = {}
    breakdown['qkv_proj'] = lat.atten_qkv.time
    breakdown['sdp_attn'] = lat.atten_matmul1.time + lat.atten_matmul2.time
    breakdown['o_proj'] = lat.atten_fc.time
    breakdown['allreduce_1'] = lat.atten_communication2
    breakdown['ff_1'] = lat.fc1.time
    breakdown['ff_2'] = lat.fc2.time
    breakdown['allreduce_2'] = lat.fc_communication

    for key in breakdown:
        breakdown[key] = breakdown[key] / lat.total * latency
    return breakdown


# %%
test_names = ['tpuv4']
model = 'palm'

with open('outputs/tpuv4/palm.pkl', 'rb') as f:
    sys = pickle.load(f)
    eval_len_sys = split_sys(sys, 'eval_len')

# %%
eval_lens = ['20, 8', '60, 20', '128, 8']
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
sim_data = {}
breakdown = {}
for eval_len in eval_lens:
    _sys = eval_len_sys[eval_len][0]
    print(eval_len)
    for batch in batch_sizes:
        perf = _sys.batch_opt_generate_tco[batch]
        prefill_lat = perf.prefill_latency * 1000
        generate_lat = perf.generate_latency * 1000
        prefill_uti = perf.prefill_utilization
        generate_uti = perf.generate_utilization
        if eval_len not in sim_data:
            sim_data[eval_len] = {
                'prefill_lat': [],
                'prefill_uti': [],
                'generate_lat': [],
                'generate_uti': [],
            }
        sim_data[eval_len]['prefill_lat'].append(prefill_lat)
        sim_data[eval_len]['prefill_uti'].append(prefill_uti)
        sim_data[eval_len]['generate_lat'].append(generate_lat)
        sim_data[eval_len]['generate_uti'].append(generate_uti)
        print(f'Batch Size: {batch}, Prefill: {prefill_lat} ({prefill_uti}) Generate: {generate_lat} ({generate_uti})')

        if eval_len not in breakdown:
            breakdown[eval_len] = {
                'prefill_io': [],
                'prefill_compute': [],
                'prefill_mem': [],
                'generate_io': [],
                'generate_compute': [],
                'generate_mem': [],
            }
        for stage in ['prefill', 'generate']:
            t_io, t_compute, t_mem = get_latency_breakdown(perf, stage)
            total = t_io + t_compute + t_mem
            breakdown[eval_len][f'{stage}_io'].append(t_io / total)
            breakdown[eval_len][f'{stage}_compute'].append(t_compute / total)
            breakdown[eval_len][f'{stage}_mem'].append(t_mem / total)

# %%
tpuv4_data = {
    '20, 8': {
        'prefill_lat':  [34,   40,   58,   99,   186,  356,  668,  1366, 2785],
        'prefill_uti':  [0.14, 0.25, 0.34, 0.40, 0.42, 0.44, 0.47, 0.46, 0.45],
        'generate_lat': [255,  226,  234,  235,  265,  312,  415,  671,  1256],
        'generate_uti': [0.01, 0.02, 0.03, 0.07, 0.12, 0.20, 0.30, 0.37, 0.40],
    },
    '60, 20': {
        'prefill_lat':  [50,   80,   153,  270,  501,  985,  2041, 4167, 8349],
        'prefill_uti':  [0.29, 0.37, 0.39, 0.44, 0.47, 0.48, 0.46, 0.45, 0.45],
        'generate_lat': [640,  574,  602,  626,  717,  829,  1114, 1743, 3260],
        'generate_uti': [0.01, 0.02, 0.03, 0.06, 0.11, 0.19, 0.28, 0.36, 0.39],
    },
    '128, 8': {
        'prefill_lat':  [81,   149,  287,  536,  1056, 2202, 4479, 8913, 17766],
        'generate_lat': [258,  234,  253,  263,  317,  381,  431,  734,  1370],
        'prefill_uti':  [0.29, 0.37, 0.39, 0.44, 0.47, 0.48, 0.46, 0.45, 0.45],
        'generate_uti': [0.01, 0.02, 0.03, 0.06, 0.11, 0.19, 0.28, 0.36, 0.39],
    },
}
# Compare with the data from the simulator
fontsize = 14
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.1))
fig.subplots_adjust(wspace=0.17, hspace=0.3)
axs = [ax1, ax2]

eval_len = ['60, 20', '128, 8'] #  [[20, 8], [60, 20], [128, 8]]
batch_sizes = [16, 32, 64, 128, 256]
batch_start_index = 2
x = batch_sizes
x_labels = [str(i) for i in x]
x_ticks = [i for i in range(len(x_labels))]
bar_width = 0.4
bar1_x = [i - bar_width/2 for i in x_ticks]
bar2_x = [i + bar_width/2 for i in x_ticks]

colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
hatches = ['//', 'x', '\\\\', 'x', '-', '+', 'x', 'o', 'O', '.', '*']

prefill_error_rate = []
decoding_error_rate = []
all_error_rate = []
for ax, eval_len in zip(axs, eval_len):
    # normalize
    tpu_norm = {'Prefill': [], 'Decoding': []}
    sim_norm = {'Prefill': [], 'Decoding': []}
    for i in range(batch_start_index, len(batch_sizes) + batch_start_index):
        norm = tpuv4_data[eval_len]['prefill_lat'][i] + tpuv4_data[eval_len]['generate_lat'][i]
        tpuv4_prefill = tpuv4_data[eval_len]['prefill_lat'][i]
        tpuv4_decode = tpuv4_data[eval_len]['generate_lat'][i]
        sim_prefill = sim_data[eval_len]['prefill_lat'][i]
        sim_decode = sim_data[eval_len]['generate_lat'][i]
        tpu_norm['Prefill'].append(tpuv4_prefill/ norm)
        tpu_norm['Decoding'].append(tpuv4_decode / norm)
        sim_norm['Prefill'].append( sim_prefill / norm)
        sim_norm['Decoding'].append(sim_decode / norm)
        prefill_error_rate.append(abs(tpuv4_prefill - sim_prefill) / tpuv4_prefill)
        decoding_error_rate.append(abs(tpuv4_decode - sim_decode) / tpuv4_decode)
        all_error_rate.append(abs(tpuv4_prefill + tpuv4_decode - sim_prefill - sim_decode) / (tpuv4_prefill + tpuv4_decode))
    


    # stack prefill and generate
    bottom = [0 for _ in range(len(x))]
    for i, data_name in enumerate(['Prefill', 'Decoding']):
        ax.bar(bar1_x, tpu_norm[data_name], label=f'TPUv4 {data_name}', 
               edgecolor='black', color=colors[0], hatch = hatches[i],
               width=bar_width, bottom=bottom)
        bottom = [a + b for a, b in zip(bottom, tpu_norm[data_name])]
    bottom = [0 for _ in range(len(x))]
    for i, data_name in enumerate(['Prefill', 'Decoding']):
        ax.bar(bar2_x, sim_norm[data_name], label=f'ReaLLM {data_name}', 
               edgecolor='black', color=colors[2], hatch = hatches[i],
               width=bar_width, bottom=bottom)
        bottom = [a + b for a, b in zip(bottom, sim_norm[data_name])]
    
    ax.set_xlabel('Batch Size', fontsize=fontsize)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_ylabel('Normalized Latency', fontsize=fontsize)
    input_len = eval_len.split(', ')[0]
    output_len = eval_len.split(', ')[1]
    ax.set_title(f'Input Len: {input_len}, Output Len: {output_len}', fontsize=fontsize)
    if ax == axs[1]:
        ax.legend(fontsize=fontsize-2, loc='lower right')

print(f'Prefill Error Rate: {sum(prefill_error_rate) / len(prefill_error_rate):.1%}')
print(f'Decoding Error Rate: {sum(decoding_error_rate) / len(decoding_error_rate):.1%}')
print(f'All Error Rate: {sum(all_error_rate) / len(all_error_rate):.1%}')

fig.savefig(f'tpu_valid.pdf', format='pdf', bbox_inches='tight')
# %%
# Breakdown stacked bar
fontsize = 12
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
fig.subplots_adjust(wspace=0.2, hspace=0.3)
eval_len = '60, 20'
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
x = batch_sizes
x_labels = [str(i) for i in x]
x_ticks = [i for i in range(len(x_labels))]
bar_width = 0.5

for ax, stage in zip(axes, ['prefill', 'generate']):
    ax.set_xlabel('Batch Size', fontsize=fontsize)
    ax.set_xticks(x_ticks, x_labels)
    bottom = [0 for _ in range(len(x))]
    for i, hw in enumerate(['io', 'compute', 'mem']):
        data = breakdown[eval_len][stage + '_' + hw]
        ax.bar(x_ticks, data, label=hw, edgecolor='black', alpha=0.8, width=bar_width, bottom=bottom)
        for j in range(len(data)):
            bottom[j] += data[j]
    ax.legend(fontsize=fontsize-2, loc='best')
    ax.set_ylim(0, 1.1)


# %%
# v100 on gpt3
model = 'gpt3_1layer'

with open(f'outputs/v100/{model}.pkl', 'rb') as f:
    sys = pickle.load(f)
    eval_len_sys = split_sys(sys, 'eval_len')

eval_lens = ['20, 8', '60, 20', '128, 8']
batch_sizes = [4, 8, 16, 32, 64, 128, 256]
sim_data = {}
breakdown = {}
for batch in batch_sizes:
    for eval_len in eval_lens:
        _sys = eval_len_sys[eval_len][0]
        perf = _sys.batch_opt_generate_tco[batch]
        prefill_lat = perf.prefill_latency * 1000
        generate_lat = perf.generate_latency * 1000
        prefill_uti = perf.prefill_utilization
        generate_uti = perf.generate_utilization
        if eval_len not in sim_data:
            sim_data[eval_len] = {
                'prefill_lat': [],
                'prefill_uti': [],
                'generate_lat': [],
                'generate_uti': [],
            }
        sim_data[eval_len]['prefill_lat'].append(prefill_lat)
        sim_data[eval_len]['prefill_uti'].append(prefill_uti)
        sim_data[eval_len]['generate_lat'].append(generate_lat)
        sim_data[eval_len]['generate_uti'].append(generate_uti)

        if eval_len not in breakdown:
            breakdown[eval_len] = []
           
        breakdown[eval_len].append(get_kernel_latency(perf, 'prefill'))
        
for i, batch in enumerate(batch_sizes):
    for eval_len in eval_lens:
        for key in breakdown[eval_len][i]: 
            print(breakdown[eval_len][i][key] * 1e6, end=', ')
        print()

# %% 
colors = ['#d73027','#fc8d59','#fee08b','#ffffbf','#d9ef8b','#91cf60','#1a9850']
keys = ['qkv_proj', 'sdp_attn', 'o_proj', 'allreduce_1', 'ff_1', 'ff_2', 'allreduce_2']
v100_1 = [1079, 37, 425, 1385, 1493, 1456, 1324]
sim_1  = [1035,  7, 345, 1180, 1381, 1381, 1180]
v100_2 = [4742, 146, 1675, 7573, 6984, 6303, 8205]
sim_2  = [5177, 33,  1726, 5898, 6903, 6903, 5898]

data1 = {}
data2 = {}
data = [data1, data2]
error_rate = []
for i, key in enumerate(keys):
    data1[key] = [v100_1[i]/1000, sim_1[i]/1000]
    data2[key] = [v100_2[i]/1000, sim_2[i]/1000]
    error_rate.append(abs(v100_1[i] - sim_1[i]) / v100_1[i])
    error_rate.append(abs(v100_2[i] - sim_2[i]) / v100_2[i])

print(f'Error Rate: {sum(error_rate) / len(error_rate):.1%}')

fontsize = 12
fig, axes = plt.subplots(1, 2, figsize=(6, 2.6))
fig.subplots_adjust(wspace=0.3, hspace=0.3)
# stacked bar
x = [0, 1]
x_labels = ['V100 GPUs', 'ReaLLM']
x_ticks = [0, 1]
bar_width = 0.7

for j in range(2):
    ax = axes[j]
    bottom = [0, 0]
    for i, key in enumerate(keys):
        ax.bar(x, data[j][key], label=f'{key}',
                  edgecolor='black', color=colors[i], hatch = hatches[i],
                  width=bar_width, bottom=bottom)
        bottom = [a + b for a, b in zip(bottom, data[j][key])]
    ax.set_xticks(x_ticks, x_labels, fontsize=fontsize)
    ax.set_ylabel('Latency (ms)', fontsize=fontsize)

axes[0].set_title('Batch: 4, Input Len: 128', fontsize=fontsize)
axes[1].set_title('Batch: 128, Input Len: 20', fontsize=fontsize)
# reverse the legend order
handles, labels = axes[0].get_legend_handles_labels()
axes[1].legend(reversed(handles), reversed(labels), 
               fontsize=fontsize, 
               bbox_to_anchor=(1.0, 1.0), loc='upper left', 
               title='Kernel', title_fontsize=fontsize)

fig.savefig(f'gpu_valid.pdf', format='pdf', bbox_inches='tight')

# %%
# comparing with TPUv5p and B200 GPUs
from structs.TCO import TCO
import numpy as np
hatches = ['/', '\\']
colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
# batch_sizes = [8, 32, 128]
batch_sizes = [4, 16, 64]
# batch_sizes = [4, 16, 64, 256]
hardwares = ['tpuv5p_eval', 'b200_eval', 'iccad']
# models = ['llama2', 'gpt3', 'mtnlg','palm']
models = ['llama2', 'gpt3']
model_labels = {'llama2': 'Llama-2 70B', 
                'gpt3': 'GPT-3 175B', 
                'mtnlg': 'MT-NLG 530B',
                'palm': 'PaLM 540B'}
# model_srvs = {'llama2': 2, 'gpt3': 4, 'mtnlg': 8, 'palm': 8}
model_srvs = {'llama2': 1, 'gpt3': 1, 'mtnlg': 4, 'palm': 4}
# eval_lens = ['256, 1024', '1024, 256']
eval_lens = {
    'llama2': ['256, 64', '64, 256'],
    # 'gpt3': ['1024, 256', '256, 1024'],
    'gpt3': ['256, 64', '256, 64'],
    'mtnlg': ['256, 64', '64, 256'],
    # 'mtnlg': ['1024, 256', '256, 1024'],
    'palm': ['256, 64', '64, 256'],
}

tco_opt_latency = dict()
tco_opt_prefill_lat = dict()
tco_opt_generate_lat = dict()
tco_opt_tco = dict()
lat_opt_latency = dict()
lat_opt_tco = dict()

for hw in hardwares:
    for model in models:
        num_srvs = model_srvs[model]
        if hw == 'tpuv5p_eval':
            num_srvs *= 2
        with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
            all_sys = pickle.load(f)
            sys_eval_len = split_sys(all_sys, 'eval_len')
            for eval_len in eval_lens[model]:
                sys_srvs = split_sys(sys_eval_len[eval_len], 'num_servers')
                key = f'{hw}_{model}_{eval_len}'
                if hw == 'iccad':
                    if key not in tco_opt_latency:
                        tco_opt_latency[key] = []
                        tco_opt_prefill_lat[key] = []
                        tco_opt_generate_lat[key] = []
                        tco_opt_tco[key] = []
                        lat_opt_latency[key] = []
                        lat_opt_tco[key] = []
                    _, _, gen_lat, gen_tco = get_batch_opt_sys(sys_srvs[num_srvs], max_batch=128)

                    sys = gen_tco[batch_sizes[-1]]
                    area = sys.server.package.chip.area
                    tops = sys.server.package.chip.tops
                    gb = sys.server.package.dram / 1024 / 1024 / 1024
                    pkg_chips = sys.server.package.num_chips
                    print(f'TCO {model} {sys.num_servers} {pkg_chips=} {area=} {tops=:.1f} {gb=}')
                    for batch in batch_sizes:
                        perf = sys.batch_opt_generate_tco[batch]
                        tco_opt_latency[key].append(perf.prefill_latency + perf.generate_latency)
                        tco_opt_prefill_lat[key].append(perf.prefill_latency)
                        tco_opt_generate_lat[key].append(perf.generate_latency)
                        tco_opt_tco[key].append(perf.generate_tco_per_token)

                    sys = gen_lat[batch_sizes[-1]]
                    area = sys.server.package.chip.area
                    tops = sys.server.package.chip.tops
                    gb = sys.server.package.dram / 1024 / 1024 / 1024
                    pkg_chips = sys.server.package.num_chips
                    print(f'Lat {model} {sys.num_servers} {pkg_chips=} {area=} {tops=:.1f} {gb=}')
                    for batch in batch_sizes:
                        perf = sys.batch_opt_generate_lat[batch]
                        lat_opt_latency[key].append(perf.prefill_latency + perf.generate_latency)
                        lat_opt_tco[key].append(perf.generate_tco_per_token )

                else:
                    sys = sys_srvs[num_srvs][0]
                    if key not in tco_opt_latency:
                        tco_opt_latency[key] = []
                        tco_opt_prefill_lat[key] = []
                        tco_opt_generate_lat[key] = []
                        tco_opt_tco[key] = []
                        lat_opt_latency[key] = []
                        lat_opt_tco[key] = []
                    for batch in batch_sizes:
                        perf = sys.batch_opt_generate_tco[batch]
                        tco_opt_latency[key].append(perf.prefill_latency + perf.generate_latency)
                        tco_opt_prefill_lat[key].append(perf.prefill_latency)
                        tco_opt_generate_lat[key].append(perf.generate_latency)
                        tco_opt_tco[key].append(perf.generate_tco_per_token)
                        perf = sys.batch_opt_generate_lat[batch]
                        lat_opt_latency[key].append(perf.prefill_latency + perf.generate_latency)
                        lat_opt_tco[key].append(perf.generate_tco_per_token )

# plot
fig, axes = plt.subplots(2, 4, figsize=(10, 3.6))
fig.subplots_adjust(wspace=0.12, hspace=0.16)
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
colors = [colors[0], colors[1], colors[2]]
x = np.arange(len(batch_sizes) + 1)
width = 0.4
offset = [-0.2, 0.2]
batch_sizes_str = [str(batch) for batch in batch_sizes]
x_labels = batch_sizes_str + ['Mean']
# Row 0: Prefill Latency
for i, model in enumerate(models):
    for j, eval_len in enumerate(eval_lens[model]):
        ax = axes[0, i*2+j]
        norm_key = f'{hardwares[-1]}_{model}_{eval_len}'
        for k, hw in enumerate(hardwares[:-1]):
            key = f'{hw}_{model}_{eval_len}'
            all_lat =  []
            for l, batch in enumerate(batch_sizes):
                all_lat.append(lat_opt_latency[key][l] / lat_opt_latency[norm_key][l])
            # add mean
            all_lat.append(np.mean(all_lat))
            if 'tpu' in hw:
                label = 'Over TPUv5p'
            elif 'b200' in hw:
                label = 'Over B200 GPU'
            ax.bar(x + offset[k], all_lat, width, facecolor=colors[k], 
                   hatch=hatches[k],
                   edgecolor='k', label=label)
            for l in range(len(batch_sizes) + 1):
                ax.text(x[l] + offset[k]+0.04, all_lat[l] + 0.0, 
                        f'{all_lat[l]:.1f}', 
                        fontsize=10, ha='center', va='bottom')
            if k == 0:
                ax.set_ylim(0, max(all_lat) + 0.5)

        # title = f'{model_labels[model]}\n{eval_len.split(",")[0]} input, {eval_len.split(",")[1][1:]} output'
        ax.text(0.5, 1.15, model_labels[model],
                ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.text(0.5, 1.05, f'{eval_len.split(",")[0]} in, {eval_len.split(",")[1][1:]} out',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

        if i == 0 and j == 0:
            ax.set_ylabel('Latency\nSpeedup', fontsize=12)
            ax.legend(loc='lower center', 
                      bbox_to_anchor=(2.2, 1.17), 
                      ncol=2,
                      fontsize=12)
# Row: TCO OPT Inference TCO
for i, model in enumerate(models):
    for j, eval_len in enumerate(eval_lens[model]):
        ax = axes[-1, i*2+j]
        norm_key = f'{hardwares[-1]}_{model}_{eval_len}'
        for k, hw in enumerate(hardwares[:-1]):
            all_tco = []
            for l, batch in enumerate(batch_sizes):
                all_tco.append(tco_opt_tco[f'{hw}_{model}_{eval_len}'][l] / tco_opt_tco[norm_key][l])
            # add mean
            all_tco.append(np.mean(all_tco))
            ax.bar(x + offset[k], all_tco, width, facecolor=colors[k],
                   hatch=hatches[k],
                   edgecolor='k')
            for l in range(len(batch_sizes) + 1):
                ax.text(x[l] + offset[k]+0.04, all_tco[l] + 0.0, 
                        f'{all_tco[l]:.1f}',
                        fontsize=10, ha='center', va='bottom')
            if k == 0:
                ax.set_ylim(0, max(all_tco) + 0.5)
        if i == 0 and j == 0:
            ax.set_ylabel('TCO/Request\nImprovement', fontsize=12)

for ax in axes.flatten():
    ax.set_xticks(x, x_labels, fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
for ax in axes[-1]:
    ax.set_xlabel('Batch Size', fontsize=12)

fig.savefig('new_design.pdf', format='pdf', bbox_inches='tight')

# %%
# allreduce algorithms comparison
def get_allreduce_time(p, a, B, N, algo='Ring'):
    if algo == 'Ring':
        return 2 * (p-1) * a + 2 * (p-1) / p * N / B
    elif algo == '2D Ring':
        # return 4 * (math.sqrt(p) -1) * a + 4 * (math.sqrt(p)-1) / math.sqrt(p) * N / B
        return 4 * (math.sqrt(p) -1) * a + 2 * (math.sqrt(p)-1) / math.sqrt(p) * N / B
    elif algo == 'Two Tree':
        if p == 1:
            return 0
        return 4 * math.log2(p) * a + 2 * N / B + 4 * math.sqrt(2 * math.log2(p) * a * N / B)
    elif algo == 'Linear Pipeline':
        # return 2 * p * a + 2 * N / B + 2 * math.sqrt(math.log2(p) * a * N / B)
        return 2 * p * a + 2 * N / B + 2 * math.sqrt(math.log2(p) * a * N / B)
    elif algo == 'Rabenseifner':
        return 2 * math.log2(p) * a + 2 * (p-1) / p * N * math.log2(p) / B
    elif algo == 'Hypercube Ring':
        # return 8 * (p**0.25 -1) * a + 8 * (p**0.25-1) / p**0.25 * N / B
        return 8 * (p**0.25 -1) * a + 2 * (p**0.25-1) / p**0.25 * N / B
    elif algo == 'Hierarchical':
        local_p = 4
        t_local_ar = get_allreduce_time(local_p, a, B, N, algo='2D Ring')
        if p == local_p:
            return t_local_ar
        else:
            p_global = math.ceil(p / local_p)
            # sqrt_p = math.ceil(p_global ** (1/2))
            # t_global = 4 * (sqrt_p - 1) * a + (3 * a + N / B / sqrt_p) 
            t_global = get_allreduce_time(p_global, a, B, N, algo='2D Ring')
            t_local_bc = 0.5 * get_allreduce_time(local_p, a, B, N, algo='Two Tree')
            return t_local_ar + t_global + t_local_bc
    else:
        raise ValueError('Invalid algorithm')

algos = ['Ring', '2D Ring', 'Two Tree', 'Hierarchical']
a = 3e-8 # 1e-8 s = 10 ns
B = 300e9 

message_kb = []
nodes = 32
bw1 = {algo: [] for algo in algos}
for i in range(1, 11):
    message_kb.append(2**i)
    byte = 2**i * 1000
    for algo in algos:
        bw1[algo].append(byte / get_allreduce_time(nodes, a, B, byte, algo) / 1e9)

num_nodes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
byte = 128 * 1000
bw2 = {algo: [] for algo in algos}
for nodes in num_nodes:
    for algo in algos:
        bw2[algo].append(byte / get_allreduce_time(nodes, a, B, byte, algo) / 1e9)

# plot
fig, axes = plt.subplots(1, 2, figsize=(8, 1.9))
fig.subplots_adjust(wspace=0.23, hspace=0.3)
ax1, ax2 = axes
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
for i, algo in enumerate(algos):
    ax1.plot(message_kb, bw1[algo], color=colors[i],marker=markers[i], label=algo)
    ax2.plot(num_nodes, bw2[algo],  color=colors[i],marker=markers[i], label=algo)

ax1.set_xlabel('Message Size (KB)', fontsize=12)
ax1.set_xscale('log')
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax2.set_xlabel('Number of Nodes', fontsize=12)
ax2.set_xscale('log')
ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)

# ax1.legend(loc='best',
ax1.legend(loc = 'best', ncol=2, 
           handletextpad=0.2, columnspacing=0.6, fontsize=9)
ax2.legend(loc = 'best', ncol=2, 
           handletextpad=0.2, columnspacing=0.6, fontsize=9)

fig.savefig('allreduce_algo.pdf', format='pdf', bbox_inches='tight')