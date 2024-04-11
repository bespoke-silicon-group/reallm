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
    t_io = lat.communication
    t_compute = 0
    t_mem = 0
    for k in ['atten_qkv', 'atten_matmul1', 'atten_matmul2', 'atten_fc', 'fc1', 'fc2']:
        mm = getattr(lat, k)
        if mm.block_ldst_time > mm.block_comp_time:
            t_mem += (mm.block_ldst_time * lat.num_layers)
        else:
            t_compute += (mm.block_comp_time * lat.num_layers)
    return t_io, t_compute, t_mem

def get_tco_breakdown(perf, stage):
    srv_tco, tco_per_token = perf._get_tco(stage)

    tco_per_Mtoken = tco_per_token * 1e6
    capex = srv_tco.fix_part
    opex = srv_tco.power_part

    return tco_per_Mtoken, capex, opex

# %%
# TPUv5p
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
with open('outputs/tpuv5p/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    eval_len_sys = split_sys(sys, 'eval_len')

results = dict()
for eval_len in eval_len_sys:
    results[eval_len] = dict()
    print('input, output:', eval_len)
    sys = eval_len_sys[eval_len][0]
    for stage in ['prefill', 'generate']:
        results[eval_len][stage] = {'t_io': [], 't_compute': [], 't_mem': [], 't_total': [], 
                                    'tco_per_Mtoken': [], 'capex': [], 'opex': []}
        for batch in batch_sizes:
            perf = sys.batch_opt_generate_tco[batch]
            t_io, t_compute, t_mem = get_latency_breakdown(perf, stage)
            results[eval_len][stage]['t_io'].append(t_io)
            results[eval_len][stage]['t_compute'].append(t_compute)
            results[eval_len][stage]['t_mem'].append(t_mem)
            results[eval_len][stage]['t_total'].append(t_io + t_compute + t_mem)

            tco_per_Mtoken, capex, opex = get_tco_breakdown(perf, stage)
            results[eval_len][stage]['tco_per_Mtoken'].append(tco_per_Mtoken)
            results[eval_len][stage]['capex'].append(capex)
            results[eval_len][stage]['opex'].append(opex)

# %%
# TPUv5p Performance
fig, axes = plt.subplots(3, 2, figsize=(10, 7))
fig.subplots_adjust(wspace=0.15, hspace=0.3)

features =  ['Prefill Latency', 'Generation Latency', 'Prefill vs. Generation']
eval_lens = ['256, 64', '64, 256']

batch_sizes_str = [str(batch) for batch in batch_sizes]

for row, feature in enumerate(features):
    for col, eval_len in enumerate(eval_lens):
        num_input, num_output = eval_len.split(',')
        task_name = f'{num_input} Input Tokens, {num_output} Output Tokens'
        ax = axes[row, col]
        if feature == 'Prefill Latency':
            io_ratio = [results[eval_len]['prefill']['t_io'][i] / results[eval_len]['prefill']['t_total'][i] for i in range(len(batch_sizes))]
            compute_ratio = [results[eval_len]['prefill']['t_compute'][i] / results[eval_len]['prefill']['t_total'][i] for i in range(len(batch_sizes))]
            mem_ratio = [results[eval_len]['prefill']['t_mem'][i] / results[eval_len]['prefill']['t_total'][i] for i in range(len(batch_sizes))]
            # 100% stacked bar chart
            ax.bar(batch_sizes_str, io_ratio, label='I/O', color='tab:blue', edgecolor='k')
            ax.bar(batch_sizes_str, compute_ratio, bottom=io_ratio, label='Compute', color='tab:orange', edgecolor='k')
            ax.bar(batch_sizes_str, mem_ratio, bottom=[io + compute for io, compute in zip(io_ratio, compute_ratio)], label='Memory', color='tab:green', edgecolor='k')
            ax.text(0.5, 1.18, task_name, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=13, fontweight='bold')
            ax.set_title('Prefill Latency Breakdown (%)', pad=2)
        elif feature == 'Generation Latency':
            io_ratio = [results[eval_len]['generate']['t_io'][i] / results[eval_len]['generate']['t_total'][i] for i in range(len(batch_sizes))]
            compute_ratio = [results[eval_len]['generate']['t_compute'][i] / results[eval_len]['generate']['t_total'][i] for i in range(len(batch_sizes))]
            mem_ratio = [results[eval_len]['generate']['t_mem'][i] / results[eval_len]['generate']['t_total'][i] for i in range(len(batch_sizes))]
            ax.bar(batch_sizes_str, io_ratio, label='I/O', color='tab:blue', edgecolor='k')
            ax.bar(batch_sizes_str, compute_ratio, bottom=io_ratio, label='Compute', color='tab:orange', edgecolor='k')
            ax.bar(batch_sizes_str, mem_ratio, bottom=[io + compute for io, compute in zip(io_ratio, compute_ratio)], label='Memory', color='tab:green', edgecolor='k')
            ax.set_title('Generation Latency Breakdown (%)', pad=2)
        elif feature == 'Prefill vs. Generation':
            prefill_ratio = []
            generation_ratio = []
            for i in range(len(batch_sizes)):
                prefill_total = results[eval_len]['prefill']['t_total'][i]
                generation_total = results[eval_len]['generate']['t_total'][i] * int(eval_len.split(',')[1])
                prefill_ratio.append(prefill_total / (prefill_total + generation_total))
                generation_ratio.append(generation_total / (prefill_total + generation_total))
            ax.bar(batch_sizes_str, prefill_ratio, label='Prefill', color='tab:purple', edgecolor='k')
            ax.bar(batch_sizes_str, generation_ratio, bottom=prefill_ratio, label='Generation', color='tab:red', edgecolor='k')
            ax.set_title('Prefill vs. Generation (%)', pad=2)
            
        ax.legend(loc = 'best')
        ax.set_ylim(0, 1)
        # set y-axis to %
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
        if row == 2:
            ax.set_xlabel('Batch Size', fontsize=14)

fig.savefig(f'tpu_perf.pdf', format='pdf', bbox_inches='tight')

# %%
# TPUv5p TCO
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)

ax1, ax2 = axes
# TCO per token, normalized
generate_tco_per_token_norm = []
prefill_tco_per_token_norm = []
norm = results['256, 64']['prefill']['tco_per_Mtoken'][0]
for i in range(len(batch_sizes)):
    generate_tco_per_token_norm.append(results['256, 64']['generate']['tco_per_Mtoken'][i] / norm)
    prefill_tco_per_token_norm.append(results['256, 64']['prefill']['tco_per_Mtoken'][i] / norm)
    print(generate_tco_per_token_norm[i], prefill_tco_per_token_norm[i])
ax1.plot(batch_sizes_str, generate_tco_per_token_norm, label='Generation', color='tab:red', marker='o')
ax1.plot(batch_sizes_str, prefill_tco_per_token_norm, label='Prefill', color='tab:purple', marker='x')
ax1.set_title('Normalized TCO per Token', pad=2)
ax1.legend(loc = 'best')
ax1.set_yscale('log')
ax1.set_xlabel('Batch Size', fontsize=12)

# TCO Breakdown 100% stacked bar chart
capex_ratio = []
opex_ratio = []
for i in range(len(batch_sizes)):
    capex_ratio.append(results['256, 64']['generate']['capex'][i] / (results['256, 64']['generate']['capex'][i] + results['256, 64']['generate']['opex'][i]))
    opex_ratio.append(results['256, 64']['generate']['opex'][i] / (results['256, 64']['generate']['capex'][i] + results['256, 64']['generate']['opex'][i]))
ax2.bar(batch_sizes_str, capex_ratio, label='CapEx', color='tab:blue', edgecolor='k')
ax2.bar(batch_sizes_str, opex_ratio, bottom=capex_ratio, label='OpEx', color='tab:orange', edgecolor='k')
ax2.set_title('TCO Breakdown (%)', pad=2)
ax2.legend(loc = 'best')
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_ylim(0, 1)
ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()], fontsize=8)

fig.savefig(f'tpu_tco.pdf', format='pdf', bbox_inches='tight')
# %%
# HBM vs SRAM
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
batch_sizes_str = [str(batch) for batch in batch_sizes]
with open('outputs/hbm_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    hbm_sys = split_sys(sys, 'server.package.hbm.config')

all_sys = {**hbm_sys}

for sram in ['sram_7nm', 'sram_5nm', 'sram_3nm']:
    with open('outputs/'+sram+'/gpt3.pkl', 'rb') as f:
        sys = pickle.load(f)
        label = sram.split('_')[1] + ' SRAM'
        all_sys[label] = sys

results = dict()
for mem in all_sys:
    results[mem] = {'latency': [], 'tco': []}
    sys = all_sys[mem][0]
    for batch in batch_sizes:
        perf = sys.batch_opt_generate_tco[batch]
        results[mem]['latency'].append(perf.generate_latency)
        results[mem]['tco'].append(perf.generate_tco_per_token)
# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
# Latency
for mem in all_sys:
    if 'TB/s' in mem:
        continue
    if 'GB' in mem:
        continue
    ax1.plot(batch_sizes_str, results[mem]['latency'], 
             label=mem,
             marker='o' if 'HBM' in mem else 'x')
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.text(0.5, 0.94, 'Per Token Latency', horizontalalignment='center', verticalalignment='center', 
         transform=ax1.transAxes, fontsize=12)
ax1.set_yscale('log')
ax1.legend(loc='lower center', bbox_to_anchor=(1.06, 1.0), ncol=6,
          handletextpad=0.1, columnspacing=0.6)
# Norm TCO
norm = results['HBM2E']['tco'][0]
for mem in all_sys:
    if 'TB/s' in mem:
        continue
    if 'GB' in mem:
        continue
    ax2.plot(batch_sizes_str, [tco / norm for tco in results[mem]['tco']], 
             label=mem,
             marker='o' if 'HBM' in mem else 'x')
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.text(0.5, 0.94, 'Normalized TCO per Token', horizontalalignment='center', verticalalignment='center', 
         transform=ax2.transAxes, fontsize=12)
ax2.set_yscale('log')

fig.savefig('hbm_vs_sram.pdf', format='pdf', bbox_inches='tight')
# %%
# Latency Breakdown
results = dict()
for mem in ['HBM3E', '5nm SRAM']:
    sys = all_sys[mem][0]
    results[mem] = {'I/O': [], 'Compute': [], 'Memory': [], 'Total': []}
    for batch in batch_sizes:
        perf = sys.batch_opt_generate_tco[batch]
        t_io, t_compute, t_mem = get_latency_breakdown(perf, 'generate')
        results[mem]['I/O'].append(t_io)
        results[mem]['Compute'].append(t_compute)
        results[mem]['Memory'].append(t_mem)
        results[mem]['Total'].append(t_io + t_compute + t_mem)

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
# 100% stacked bar chart
for i, mem in enumerate(['HBM3E', '5nm SRAM']):
    ax = axes[i]
    io_ratio = [results[mem]['I/O'][i] / results[mem]['Total'][i] for i in range(len(batch_sizes))]
    compute_ratio = [results[mem]['Compute'][i] / results[mem]['Total'][i] for i in range(len(batch_sizes))]
    mem_ratio = [results[mem]['Memory'][i] / results[mem]['Total'][i] for i in range(len(batch_sizes))]
    ax.bar(batch_sizes_str, io_ratio, label='I/O', color='tab:blue', edgecolor='k')
    ax.bar(batch_sizes_str, compute_ratio, bottom=io_ratio, label='Compute', color='tab:orange', edgecolor='k')
    ax.bar(batch_sizes_str, mem_ratio, bottom=[io + compute for io, compute in zip(io_ratio, compute_ratio)], label='Memory', color='tab:green', edgecolor='k') 
    ax.set_title(mem + ' Latency Breakdown', pad=2)
    ax.legend(loc = 'best')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()], fontsize=8)
fig.savefig('hbm_vs_sram_latency.pdf', format='pdf', bbox_inches='tight')

# %%
# HBM Exploration
batch_sizes = [2, 8, 32, 128]
with open('outputs/hbm_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    hbm_sys = split_sys(sys, 'server.package.hbm.config')

norm_latency = dict()
norm_tco = dict()
sys = hbm_sys['HBM3E'][0]
for batch in batch_sizes:
    perf = sys.batch_opt_generate_tco[batch]
    norm_latency[batch] = perf.generate_latency
    norm_tco[batch] = perf.generate_tco_per_token

# all_norm_latency = {'Capacity (GB)': dict(), 'Bandwidth (TB/s)': dict()}
# all_norm_tco = {'Capacity (GB)': dict(), 'Bandwidth (TB/s)': dict()}
all_norm_latency = dict()
all_norm_tco = dict()
bandwidth = ['1.1',]

for hbm in hbm_sys:
    sys = hbm_sys[hbm][0]
    if 'TB/s' in hbm:
        bandwidth.append(hbm.split(' ')[0])
    for batch in batch_sizes:
        perf = sys.batch_opt_generate_tco[batch]
        latency = perf.generate_latency
        tco = perf.generate_tco_per_token
        if hbm == 'HBM3E':
            all_norm_latency[batch] = [latency / norm_latency[batch]]
            all_norm_tco[batch] = [tco / norm_tco[batch]]
        if 'TB/s' in hbm:
            all_norm_latency[batch].append(latency / norm_latency[batch])
            all_norm_tco[batch].append(tco / norm_tco[batch])

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
# Latency
for i, batch in enumerate(batch_sizes):
    ax1.plot(bandwidth, all_norm_latency[batch], marker=markers[i], label=f'Batch {batch}')
ax1.set_xlabel('Bandwidth (TB/s)', fontsize=12)
ax1.set_ylim(0.02, 1.1)
ax1.text(0.5, 1.05, 'Normalized Token Latency', horizontalalignment='center', verticalalignment='center', 
         transform=ax1.transAxes, fontsize=12)
ax1.legend(loc='lower left', handletextpad=0.1, columnspacing=0.6, 
           fontsize=8, bbox_to_anchor=(0.0, 0.0))
# TCO
for i, batch in enumerate(batch_sizes):
    ax2.plot(bandwidth, all_norm_tco[batch], marker=markers[i], label=f'Batch {batch}')
ax2.set_xlabel('Bandwidth (TB/s)', fontsize=12)
ax2.text(0.5, 1.05, 'Normalized TCO per Token', horizontalalignment='center', verticalalignment='center', 
         transform=ax2.transAxes, fontsize=12)
ax2.legend(loc='upper left', handletextpad=0.1, columnspacing=0.6, 
           fontsize=8, bbox_to_anchor=(0.0, 1.0))

fig.savefig('hbm_explore.pdf', format='pdf', bbox_inches='tight')

# %%
# breakdown
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
batch_sizes_str = [str(batch) for batch in batch_sizes]
sys = hbm_sys['9.2 TB/s HBM'][0]
# Latency and TCO Breakdown
results = dict()
latency = {'I/O': [], 'Compute': [], 'Memory': []}
tco = {'CapEx': [], 'OpEx': []}
for batch in batch_sizes:
    perf = sys.batch_opt_generate_tco[batch]
    t_io, t_compute, t_mem = get_latency_breakdown(perf, 'generate')
    total = t_io + t_compute + t_mem
    latency['I/O'].append(t_io / total)
    latency['Compute'].append(t_compute / total)
    latency['Memory'].append(t_mem / total)

    tco_per_Mtoken, capex, opex = get_tco_breakdown(perf, 'generate')
    total = capex + opex
    tco['CapEx'].append(capex / total)
    tco['OpEx'].append(opex / total)

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
# Latency 100% stacked bar chart
ax1.bar(batch_sizes_str, latency['I/O'], label='I/O', color='tab:blue', edgecolor='k')
ax1.bar(batch_sizes_str, latency['Compute'], bottom=latency['I/O'], label='Compute', color='tab:orange', edgecolor='k')
ax1.bar(batch_sizes_str, latency['Memory'], bottom=[io + compute for io, compute in zip(latency['I/O'], latency['Compute'])], label='Memory', color='tab:green', edgecolor='k')
ax1.set_title('Latency Breakdown (%)', pad=2)
ax1.legend(loc = 'best')
ax1.set_ylim(0, 1)
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.set_yticklabels(['{:,.0%}'.format(x) for x in ax1.get_yticks()], fontsize=8)
# TCO 100% stacked bar chart
ax2.bar(batch_sizes_str, tco['CapEx'], label='CapEx', color='tab:blue', edgecolor='k')
ax2.bar(batch_sizes_str, tco['OpEx'], bottom=tco['CapEx'], label='OpEx', color='tab:orange', edgecolor='k')
ax2.set_title('TCO Breakdown (%)', pad=2)
ax2.legend(loc = 'best')
ax2.set_ylim(0, 1)
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()], fontsize=8)
fig.savefig('best_hbm.pdf', format='pdf', bbox_inches='tight')

# %%
# 3D DRAM Exploration
batch_sizes = [2, 8, 32, 128]
with open('outputs/dram_3d_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    mem_sys = split_sys(sys, 'server.package.mem_3d.config')

norm_latency = dict()
norm_tco = dict()
sys = mem_sys['9.2 Gbps, 3 pJ/bit'][0]
for batch in batch_sizes:
    perf = sys.batch_opt_generate_tco[batch]
    norm_latency[batch] = perf.generate_latency
    norm_tco[batch] = perf.generate_tco_per_token

all_norm_latency = dict()
all_norm_tco = dict()
mems = []

for mem in mem_sys:
    sys = mem_sys[mem][0]
    mems.append(mem.split(' ')[0] + '\n' + mem.split(' ')[2])
    for batch in batch_sizes:
        perf = sys.batch_opt_generate_tco[batch]
        latency = perf.generate_latency
        tco = perf.generate_tco_per_token
        if batch not in all_norm_latency:
            all_norm_latency[batch] = [norm_latency[batch]/ latency]
            all_norm_tco[batch] = [norm_tco[batch] / tco]
        else:
            all_norm_latency[batch].append(norm_latency[batch] / latency)
            all_norm_tco[batch].append(norm_tco[batch] / tco)

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
# Latency
for i, batch in enumerate(batch_sizes):
    ax1.plot(mems, all_norm_latency[batch], marker=markers[i], label=f'Batch {batch}')
ax1.text(0.5, 1.05, 'Latency Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax1.transAxes, fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)
ax1.set_xlabel('3D DRAM', fontsize=12)
ax1.text(-0.00, -0.1, 'Gb/s:', horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, fontsize=10)
ax1.text(-0.00, -0.205, 'pJ/b:', horizontalalignment='right', verticalalignment='center', transform=ax1.transAxes, fontsize=10)
ax1.grid(True, which='both', axis='y')
# TCO
for i, batch in enumerate(batch_sizes):
    ax2.plot(mems, all_norm_tco[batch], marker=markers[i], label=f'Batch {batch}')
ax2.set_xlabel('3D DRAM', fontsize=12)
ax2.text(-0.0, -0.1, 'Gb/s:', horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, fontsize=10)
ax2.text(-0.0, -0.205, 'pJ/b:', horizontalalignment='right', verticalalignment='center', transform=ax2.transAxes, fontsize=10)
ax2.text(0.5, 1.05, 'TCO/Token Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax2.transAxes, fontsize=12)
ax2.legend(loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)
ax2.grid(True, which='both', axis='y')

fig.savefig('3d_dram_explore.pdf', format='pdf', bbox_inches='tight')

# %%
# 3D DRAM TSV Exploration
batch_sizes = [2, 8, 32, 128]

with open('outputs/hbm_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    hbm_sys = split_sys(sys, 'server.package.hbm.config')

# for sram in ['sram_5nm']:
#     with open('outputs/'+sram+'/gpt3.pkl', 'rb') as f:
#         sys = pickle.load(f)
#         label = sram.split('_')[1] + ' SRAM'
#         all_sys[label] = sys

norm_latency = dict()
norm_tco = dict()
sys = hbm_sys['HBM3E'][0]
for batch in batch_sizes:
    perf = sys.batch_opt_generate_tco[batch]
    norm_latency[batch] = perf.generate_latency
    norm_tco[batch] = perf.generate_tco_per_token

mem_sys = dict()
for dram_3d in ['64tsvs', '96tsvs', '128tsvs', '160tsvs', '192tsvs', '224tsvs']:
    with open('outputs/dram_3d_'+dram_3d+'/gpt3.pkl', 'rb') as f:
        sys = pickle.load(f)
        label = dram_3d.split('tsvs')[0] + ' TSVs'
        mem_sys[label] = sys

all_norm_latency = dict()
all_norm_tco = dict()
mems = []

for mem in mem_sys:
    sys = mem_sys[mem][0]
    mems.append(mem.split(' ')[0])
    for batch in batch_sizes:
        perf = sys.batch_opt_generate_tco[batch]
        latency = perf.generate_latency
        tco = perf.generate_tco_per_token
        if batch not in all_norm_latency:
            all_norm_latency[batch] = [norm_latency[batch]/ latency]
            all_norm_tco[batch] = [norm_tco[batch] / tco]
        else:
            all_norm_latency[batch].append(norm_latency[batch] / latency)
            all_norm_tco[batch].append(norm_tco[batch] / tco)
# Add batch average
all_norm_latency['Average'] = [sum([all_norm_latency[batch][i] for batch in batch_sizes]) / len(batch_sizes) for i in range(len(mems))]
all_norm_tco['Average'] = [sum([all_norm_tco[batch][i] for batch in batch_sizes]) / len(batch_sizes) for i in range(len(mems))]

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
# Latency
for i, batch in enumerate(batch_sizes):
    ax1.plot(mems, all_norm_latency[batch], marker=markers[i], label=f'Batch {batch}')
ax1.plot(mems, all_norm_latency['Average'], marker='P', label='Average', color='tab:gray', linestyle='--')
ax1.text(0.5, 1.05, 'Latency Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax1.transAxes, fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)
ax1.set_xlabel('Data TSVs per Vault', fontsize=12)
ax1.grid(True, which='both', axis='y')
# TCO
for i, batch in enumerate(batch_sizes):
    ax2.plot(mems, all_norm_tco[batch], marker=markers[i], label=f'Batch {batch}')
ax2.plot(mems, all_norm_tco['Average'], marker='P', label='Average', color='tab:gray', linestyle='--')
ax2.set_xlabel('Data TSVs per Vault', fontsize=12)
ax2.text(0.5, 1.05, 'TCO/Token Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax2.transAxes, fontsize=12)
# ax2.legend(loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)
ax2.grid(True, which='both', axis='y')

fig.savefig('3d_dram_tradeoff.pdf', format='pdf', bbox_inches='tight')

# %%
# Prefill IO
batch = 8
with open('outputs/prefill_io_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    io_sys = split_sys(sys, 'num_servers')

lat_breakdown = {'I/O': [], 'Compute': [], 'Memory': []}
tput_per_node = []
tco_per_Mtoken = []
num_nodes = []
for io in io_sys:
    num_nodes.append(io)
    sys = io_sys[io][0]
    perf = sys.batch_opt_prefill_tco[4]
    t_io, t_compute, t_mem = get_latency_breakdown(perf, 'prefill')
    lat_breakdown['I/O'].append(t_io)
    lat_breakdown['Compute'].append(t_compute)
    lat_breakdown['Memory'].append(t_mem)
    tput_per_node.append(perf.prefill_throughput / sys.num_chips)
    tco_per_Mtoken.append(perf.prefill_tco_per_token * 1e6)

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
str_num_nodes = [str(node) for node in num_nodes]
# latency breakdown stacked bar
norm = lat_breakdown['I/O'][0]
for k in lat_breakdown:
    lat_breakdown[k] = [lat / norm for lat in lat_breakdown[k]]

ax1.bar(str_num_nodes, lat_breakdown['I/O'], label='I/O', color='tab:blue', edgecolor='k')
ax1.bar(str_num_nodes, lat_breakdown['Compute'], bottom=lat_breakdown['I/O'],
        label='Compute', color='tab:orange', edgecolor='k')
ax1.bar(str_num_nodes, lat_breakdown['Memory'], bottom=[io + compute for io, compute in zip(lat_breakdown['I/O'], lat_breakdown['Compute'])],
        label='Memory', color='tab:green', edgecolor='k') 
ax1.set_xlabel('Number of Servers', fontsize=12)
ax1.text(0.5, 1.05, 'Normalized Prefill Latency', horizontalalignment='center', verticalalignment='center', 
         transform=ax1.transAxes, fontsize=12)
ax1.set_ylim(0, 3.6)
ax1.legend(loc = 'best', ncol=1, handletextpad=0.2, columnspacing=0.6, fontsize=10)

# norm tco per token
norm = tco_per_Mtoken[0]
tco_per_Mtoken = [tco / norm for tco in tco_per_Mtoken]
ax2.plot(str_num_nodes, tco_per_Mtoken, marker='o') 
ax2.set_xlabel('Number of Servers', fontsize=12)
ax2.text(0.5, 1.05, 'Normalized Prefill TCO per Token', horizontalalignment='center', verticalalignment='center',
            transform=ax2.transAxes, fontsize=12)
ax2.set_yscale('log')

fig.savefig('prefill_io_explore.pdf', format='pdf', bbox_inches='tight')

# %%
# allreduce algorithms comparison
def get_allreduce_time(p, a, B, N, algo='Ring'):
    if algo == 'Ring':
        return 2 * (p-1) * a + 2 * (p-1) / p * N / B
    elif algo == '2D Ring':
        # return 4 * (math.sqrt(p) -1) * a + 4 * (math.sqrt(p)-1) / math.sqrt(p) * N / B
        return 4 * (math.sqrt(p) -1) * a + 2 * (math.sqrt(p)-1) / math.sqrt(p) * N / B
    elif algo == 'Two Tree':
        return 4 * math.log2(p) * a + 4 * N / B + 8 * math.sqrt(math.log2(p) * a * N / B)
    elif algo == 'Linear Pipeline':
        # return 2 * p * a + 2 * N / B + 2 * math.sqrt(math.log2(p) * a * N / B)
        return 2 * p * a + 2 * N / B + 2 * math.sqrt(math.log2(p) * a * N / B)
    elif algo == 'Rabenseifner':
        return 2 * math.log2(p) * a + 2 * (p-1) / p * N * math.log2(p) / B
    elif algo == '4D Hypercube':
        # return 8 * (p**0.25 -1) * a + 8 * (p**0.25-1) / p**0.25 * N / B
        return 8 * (p**0.25 -1) * a + 2 * (p**0.25-1) / p**0.25 * N / B
    else:
        raise ValueError('Invalid algorithm')

algos = ['Ring', '2D Ring', 'Linear Pipeline', '4D Hypercube']
a = 1e-8 # 1e-8 s = 10 ns
B = 300e9 

message_kb = []
nodes = 16
bw1 = {algo: [] for algo in algos}
for i in range(1, 11):
    message_kb.append(2**i)
    byte = 2**i * 1000
    for algo in algos:
        bw1[algo].append(byte / get_allreduce_time(nodes, a, B, byte, algo) / 1e9)

num_nodes = [8, 16, 32, 64, 128, 256, 512]
byte = 128 * 1000
bw2 = {algo: [] for algo in algos}
for nodes in num_nodes:
    for algo in algos:
        bw2[algo].append(byte / get_allreduce_time(nodes, a, B, byte, algo) / 1e9)

# plot
fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.23, hspace=0.3)
ax1, ax2 = axes
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
for i, algo in enumerate(algos):
    ax1.plot(message_kb, bw1[algo], marker=markers[i], label=algo)
    ax2.plot(num_nodes, bw2[algo], marker=markers[i], label=algo)

ax1.set_xlabel('Message Size (KB)', fontsize=12)
ax1.set_xscale('log')
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax2.set_xlabel('Number of Nodes', fontsize=12)
ax2.set_xscale('log')
ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)

# ax1.legend(loc='best')
ax1.legend(loc = 'lower center', 
           bbox_to_anchor=(1.1, 0.97),
           ncol=6, 
           handletextpad=0.2, columnspacing=0.6, fontsize=10)

fig.savefig('allreduce_algo.pdf', format='pdf', bbox_inches='tight')

# %%
# comparing with TPUv5p and B200 GPUs
from structs.TCO import TCO
batch_sizes = [2, 8, 32, 128]
# batch_sizes = [1, 4, 16, 64]
# batch_sizes = [4, 16, 64, 256]
hardwares = ['tpuv5p_eval', 'b200_eval', 'cc_3d']
models = ['llama2', 'gpt3', 'mtnlg','palm']
model_labels = {'llama2': 'Llama-2 70B', 
                'gpt3': 'GPT-3 175B', 
                'mtnlg': 'MT-NLG 530B',
                'palm': 'PaLM 540B'}
model_srvs = {'llama2': 2, 'gpt3': 4, 'mtnlg': 8, 'palm': 8}
# eval_lens = ['256, 1024', '1024, 256']
eval_lens = ['256, 64', '64, 256']

tco_opt_latency = dict()
tco_opt_tco = dict()
lat_opt_latency = dict()
lat_opt_tco = dict()

for hw in hardwares:
    for model in models:
        with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
            all_sys = pickle.load(f)
            sys_srvs = split_sys(all_sys, 'num_servers')
            num_srvs = model_srvs[model]
            if hw == 'tpuv5p_eval':
                num_srvs *= 2
            sys_eval_len = split_sys(sys_srvs[num_srvs], 'eval_len')
            for eval_len in eval_lens:
                key = f'{hw}_{model}_{eval_len}'
                sys = sys_eval_len[eval_len][0]
                if key not in tco_opt_latency:
                    tco_opt_latency[key] = []
                    tco_opt_tco[key] = []
                    lat_opt_latency[key] = []
                    lat_opt_tco[key] = []
                for batch in batch_sizes:
                    perf = sys.batch_opt_generate_tco[batch]
                    tco_opt_latency[key].append(perf.prefill_latency + perf.generate_latency)
                    tco_opt_tco[key].append(perf.prefill_tco_per_token)
                    perf = sys.batch_opt_generate_lat[batch]
                    lat_opt_latency[key].append(perf.prefill_latency + perf.generate_latency)
                    lat_opt_tco[key].append(perf.prefill_tco_per_token)

# plot
import numpy as np
# fig, axes = plt.subplots(4, 6, figsize=(16, 7))
fig, axes = plt.subplots(2, 8, figsize=(16, 3.6))
fig.subplots_adjust(wspace=0.16, hspace=0.16)
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
colors = ['tab:blue', 'tab:orange', 'tab:green']
x = np.arange(len(batch_sizes) + 1)
width = 0.4
offset = [-0.2, 0.2]
batch_sizes_str = [str(batch) for batch in batch_sizes]
x_labels = batch_sizes_str + ['Mean']
# Row 0: TCO OPT Latency
for i, model in enumerate(models):
    for j, eval_len in enumerate(eval_lens):
        ax = axes[0, i*2+j]
        norm_key = f'{hardwares[-1]}_{model}_{eval_len}'
        for k, hw in enumerate(hardwares[:-1]):
            key = f'{hw}_{model}_{eval_len}'
            all_lat =  []
            for l, batch in enumerate(batch_sizes):
                all_lat.append(tco_opt_latency[key][l] / tco_opt_latency[norm_key][l])
            # add mean
            all_lat.append(np.mean(all_lat))
            if 'tpu' in hw:
                label = 'Over TPUv5p'
            elif 'b200' in hw:
                label = 'Over NVIDIA B200 GPU'
            ax.bar(x + offset[k], all_lat, width, facecolor=colors[k], 
                   edgecolor='k', label=label)
            for l in range(len(batch_sizes) + 1):
                ax.text(x[l] + offset[k], all_lat[l] + 0.0, 
                        f'{all_lat[l]:.1f}', 
                        fontsize=6, ha='center', va='bottom')
            if k == 0:
                ax.set_ylim(0, max(all_lat) + 0.5)

        title = f'{model_labels[model]}\n{eval_len.split(",")[0]} input, {eval_len.split(",")[1][1:]} output'
        ax.text(0.5, 1.15, model_labels[model],
                ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.text(0.5, 1.05, f'{eval_len.split(",")[0]} input, {eval_len.split(",")[1][1:]} output',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

        if i == 0 and j == 0:
            ax.set_ylabel('Latency\nSpeedup', fontsize=12)
            ax.legend(loc='lower center', 
                      bbox_to_anchor=(4.56, 1.17), 
                      ncol=2,
                      fontsize=12)
# Row 1: TCO OPT Inference TCO
for i, model in enumerate(models):
    for j, eval_len in enumerate(eval_lens):
        ax = axes[1, i*2+j]
        norm_key = f'{hardwares[-1]}_{model}_{eval_len}'
        for k, hw in enumerate(hardwares[:-1]):
            all_tco = []
            for l, batch in enumerate(batch_sizes):
                all_tco.append(tco_opt_tco[f'{hw}_{model}_{eval_len}'][l] / tco_opt_tco[norm_key][l])
            # add mean
            all_tco.append(np.mean(all_tco))
            ax.bar(x + offset[k], all_tco, width, facecolor=colors[k],
                   edgecolor='k')
            for l in range(len(batch_sizes) + 1):
                ax.text(x[l] + offset[k], all_tco[l] + 0.0, 
                        f'{all_tco[l]:.1f}', 
                        fontsize=6, ha='center', va='bottom')
            if k == 0:
                ax.set_ylim(0, max(all_tco) + 0.5)
        if i == 0 and j == 0:
            ax.set_ylabel('TCO/Request\nImprovement', fontsize=12)
# # Row 2: TCO OPT Latency
# for i, model in enumerate(models):
#     for j, eval_len in enumerate(eval_lens):
#         ax = axes[2, i*2+j]
#         norm_key = f'{hardwares[-1]}_{model}_{eval_len}'
#         for k, hw in enumerate(hardwares[:-1]):
#             key = f'{hw}_{model}_{eval_len}'
#             all_lat =  []
#             for l, batch in enumerate(batch_sizes):
#                 all_lat.append(lat_opt_latency[key][l] / lat_opt_latency[norm_key][l])
#             # add mean
#             all_lat.append(np.mean(all_lat))
#             ax.bar(x + offset[k], all_lat, width, facecolor=colors[k], 
#                    edgecolor='k', label=label)
#             for l in range(len(batch_sizes) + 1):
#                 ax.text(x[l] + offset[k], all_lat[l] + 0.0, 
#                         f'{all_lat[l]:.1f}', 
#                         fontsize=6, ha='center', va='bottom')
#             if k == 0:
#                 ax.set_ylim(0, max(all_lat) + 0.5)

#         if i == 0 and j == 0:
#             ax.set_ylabel('Latency\nSpeedup', fontsize=12)
# # Row 3: lat OPT Inference TCO
# for i, model in enumerate(models):
#     for j, eval_len in enumerate(eval_lens):
#         ax = axes[3, i*2+j]
#         norm_key = f'{hardwares[-1]}_{model}_{eval_len}'
#         for k, hw in enumerate(hardwares[:-1]):
#             all_tco = []
#             for l, batch in enumerate(batch_sizes):
#                 all_tco.append(lat_opt_tco[f'{hw}_{model}_{eval_len}'][l] / lat_opt_tco[norm_key][l])
#             # add mean
#             all_tco.append(np.mean(all_tco))
#             ax.bar(x + offset[k], all_tco, width, facecolor=colors[k],
#                    edgecolor='k')
#             for l in range(len(batch_sizes) + 1):
#                 ax.text(x[l] + offset[k], all_tco[l] + 0.0, 
#                         f'{all_tco[l]:.1f}', 
#                         fontsize=6, ha='center', va='bottom')
#             if k == 0:
#                 ax.set_ylim(0, max(all_tco) + 0.5)
#             ax.set_xlabel('Batch Size')
#         if i == 0 and j == 0:
#             ax.set_ylabel('TCO/Request\nImprovement', fontsize=12)
for ax in axes.flatten():
    ax.set_xticks(x, x_labels, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
for ax in axes[-1]:
    ax.set_xlabel('Batch Size')

fig.savefig('hw_comparison.pdf', format='pdf', bbox_inches='tight')

# %%
latency = dict()
tco = dict()
model = 'llama2'
eval_len = '256, 64'

for hw in hardwares:
    with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
        all_sys = pickle.load(f)
        sys_srvs = split_sys(all_sys, 'num_servers')
        num_srvs = model_srvs[model]
        if hw == 'tpuv5p_eval':
            num_srvs *= 2
        sys_eval_len = split_sys(sys_srvs[num_srvs], 'eval_len')
        eval_len = '256, 64'
        key = f'{hw}_{model}_{eval_len}'
        sys = sys_eval_len[eval_len][0]
        print(key, 'num srvs:', sys.num_servers)
        perf = sys.batch_opt_prefill_lat[2]
        # perf = sys.batch_opt_prefill_tco[128]
        print(1 / (perf.prefill_latency + perf.generate_latency))
        print(perf.srv_tco.fix_part * sys.num_servers,
              perf.srv_tco.power_part * sys.num_servers,)
        # t_io, t_compute, t_mem = get_latency_breakdown(perf, 'prefill')
        # print(t_io, t_compute, t_mem, )
        # print(perf.tco_per_token * 1e6)
        # print(perf.mapping)
# %%
hw = 'cc_3d'
model = 'palm'
from dataclasses import replace
# for hw in hardwares:
for hw in ['cc_3d']:
    with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
        all_sys = pickle.load(f)
        sys_srvs = split_sys(all_sys, 'num_servers')
        num_srvs = model_srvs[model]
        if hw == 'tpuv5p_eval':
            num_srvs *= 2
        sys_eval_len = split_sys(sys_srvs[num_srvs], 'eval_len')
        eval_len = '64, 256'
        key = f'{hw}_{model}_{eval_len}'
        sys = sys_eval_len[eval_len][0]
        perf = sys.batch_opt_generate_tco[128]
        # print(tco_opt_latency[key][3])
        print(key)
        print(perf.tco_per_token * 1e6)
        print(perf.mapping)
        print(perf.prefill_latency, perf.generate_latency, 
              perf.prefill_latency + perf.generate_latency)

        t_io, t_compute, t_mem = get_latency_breakdown(perf, 'generate')
        lat = perf._get_micro_batch_latency('generate')
        t_io = lat.communication
        print(t_io, t_compute, t_mem)
        print(lat.atten_communication1,
              lat.atten_communication2,
              lat.fc_communication,)

        mapping = Mapping(t=128, p=4, micro_batch=32,
                          prefill_micro_batch=32, 
                         hybrid=True, prefill_batch=32, prefill_t=128, prefill_p=1) 
        perf = replace(perf, mapping=mapping)
        perf.update()
        print('new map')
        print(perf.tco_per_token * 1e6)
        print(perf.prefill_latency, perf.generate_latency, 
              perf.prefill_latency + perf.generate_latency)

        t_io, t_compute, t_mem = get_latency_breakdown(perf, 'generate')
        lat = perf._get_micro_batch_latency('generate')
        t_io = lat.communication
        print(t_io, t_compute, t_mem)
        print(lat.atten_communication1,
              lat.atten_communication2,
              lat.fc_communication,) 
# %%
hw = 'cc_3d'
model = 'palm'
with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
    all_sys = pickle.load(f)
    sys_srvs = split_sys(all_sys, 'num_servers')
    num_srvs = model_srvs[model]
    sys_eval_len = split_sys(sys_srvs[num_srvs], 'eval_len')
    eval_len = '64, 256'
    key = f'{hw}_{model}_{eval_len}'
    sys = sys_eval_len[eval_len][0]
    all_mappings = sys.gen_mappings(batch=128, min_ctx_len=256+64)
    for mapping in all_mappings:
        print(mapping)