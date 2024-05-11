# %%
import pickle, sys, math, functools
import numpy as np
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

def get_tco_breakdown(perf):
    srv_tco, tco_per_input_token, tco_per_output_token = perf._get_tco()

    tco_per_Mtoken = tco_per_output_token * 1e6
    capex = srv_tco.fix_part * perf.system.num_servers
    opex = srv_tco.power_part * perf.system.num_servers

    return tco_per_Mtoken, capex, opex

def get_tco_detail_breakdown(perf):
    srv_tco, _, _ = perf._get_tco()
    capex_ratio = srv_tco.fix_part / srv_tco.total
    capex_chip_ratio = capex_ratio * (sys.server.cost_all_package / sys.server.cost)
    capex_sys_ratio = capex_ratio * (sys.server.cost - sys.server.cost_all_package) / sys.server.cost

    opex_ratio = srv_tco.power_part / srv_tco.total
    srv_power = srv_tco.srv_power
    core_power_ratio = 1 - (sys.other_tdp / sys.num_servers / srv_power)

    prefill_core_energy = perf._get_core_energy('prefill')
    generate_core_energy = perf._get_core_energy('generate')
    prefill_ratio = perf.prefill_latency / (perf.prefill_latency + perf.generate_latency)
    generate_ratio = perf.generate_latency / (perf.prefill_latency + perf.generate_latency)

    core_compute_energy_ratio = prefill_ratio * (prefill_core_energy.fma / prefill_core_energy.total) + \
                                generate_ratio * (generate_core_energy.fma / generate_core_energy.total)
    core_mem_energy_ratio = prefill_ratio * (prefill_core_energy.mem / prefill_core_energy.total) + \
                            generate_ratio * (generate_core_energy.mem / generate_core_energy.total)
    core_io_energy_ratio = prefill_ratio * (prefill_core_energy.comm / prefill_core_energy.total) + \
                            generate_ratio * (generate_core_energy.comm / generate_core_energy.total)

    opex_compute_ratio = opex_ratio * core_power_ratio * core_compute_energy_ratio
    opex_mem_ratio = opex_ratio * core_power_ratio * core_mem_energy_ratio
    opex_io_ratio = opex_ratio * core_power_ratio * core_io_energy_ratio
    opex_other_ratio = opex_ratio * (1 - core_power_ratio)

    return capex_chip_ratio, capex_sys_ratio, opex_compute_ratio, opex_mem_ratio, opex_io_ratio, opex_other_ratio

colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
hatches = ['//', '\\\\', 'x', '-', '+', 'x', 'o', 'O', '.', '*']
# %%
colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17']
hatches = ['//', '\\\\', 'x', '-', '+', 'o', 'O', '.', '*']
# TPUv5p
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# batch_sizes = [1, 4, 16, 64, 256, 1024]
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

            tco_per_Mtoken, capex, opex = get_tco_breakdown(perf)
            results[eval_len][stage]['tco_per_Mtoken'].append(tco_per_Mtoken)
            results[eval_len][stage]['capex'].append(capex)
            results[eval_len][stage]['opex'].append(opex)

fig, axes = plt.subplots(3, 2, figsize=(10, 6.5))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
width = 0.6

features =  ['Prefill Latency', 'Decoding Latency', 'Prefill vs. Decoding']
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
            ax.bar(batch_sizes_str, io_ratio, label='I/O', 
                   color=colors[0], hatch=hatches[0], edgecolor='k', width=width)
            ax.bar(batch_sizes_str, compute_ratio, bottom=io_ratio, label='Compute', 
                   color=colors[1], hatch=hatches[1], edgecolor='k', width=width)
            ax.bar(batch_sizes_str, mem_ratio, bottom=[io + compute for io, compute in zip(io_ratio, compute_ratio)], label='Memory', 
                   color=colors[2], hatch=hatches[2], edgecolor='k', width=width)
            ax.text(0.5, 1.18, task_name, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=13, fontweight='bold')
            ax.set_title('Prefill Latency Breakdown (%)', pad=2)
        elif feature == 'Decoding Latency':
            io_ratio = [results[eval_len]['generate']['t_io'][i] / results[eval_len]['generate']['t_total'][i] for i in range(len(batch_sizes))]
            compute_ratio = [results[eval_len]['generate']['t_compute'][i] / results[eval_len]['generate']['t_total'][i] for i in range(len(batch_sizes))]
            mem_ratio = [results[eval_len]['generate']['t_mem'][i] / results[eval_len]['generate']['t_total'][i] for i in range(len(batch_sizes))]
            ax.bar(batch_sizes_str, io_ratio, label='I/O', 
                   color=colors[0], hatch=hatches[0], edgecolor='k', width=width)
            ax.bar(batch_sizes_str, compute_ratio, bottom=io_ratio, label='Compute', 
                   color=colors[1], hatch=hatches[1], edgecolor='k', width=width)
            ax.bar(batch_sizes_str, mem_ratio, bottom=[io + compute for io, compute in zip(io_ratio, compute_ratio)], label='Memory', 
                   color=colors[2], hatch=hatches[2], edgecolor='k', width=width)
            ax.set_title('Decoding Latency Breakdown (%)', pad=2)
        elif feature == 'Prefill vs. Decoding':
            prefill_ratio = []
            generation_ratio = []
            for i in range(len(batch_sizes)):
                prefill_total = results[eval_len]['prefill']['t_total'][i]
                generation_total = results[eval_len]['generate']['t_total'][i]
                prefill_ratio.append(prefill_total / (prefill_total + generation_total))
                generation_ratio.append(generation_total / (prefill_total + generation_total))
            ax.bar(batch_sizes_str, prefill_ratio, label='Prefill', 
                   color=colors[4], edgecolor='k', hatch=hatches[0], width=width)
            ax.bar(batch_sizes_str, generation_ratio, bottom=prefill_ratio, label='Decoding', 
                   color=colors[6], edgecolor='k', hatch=hatches[1], width=width)
            ax.set_title('Prefill vs. Decoding (%)', pad=2)
            
        # reverse legend order
        handles, labels = ax.get_legend_handles_labels()
        if row == 2:
            ax.legend(reversed(handles), reversed(labels), loc='upper left', framealpha=1)
        else:
            ax.legend(reversed(handles), reversed(labels), loc='best', framealpha=1)
        ax.set_ylim(0, 1)
        # set y-axis to %
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
        if row == 2:
            ax.set_xlabel('Batch Size', fontsize=14)

fig.savefig(f'tpu_perf.pdf', format='pdf', bbox_inches='tight')

    
# %%
# TPUv5p TCO
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
batch_sizes_str = [str(batch) for batch in batch_sizes]
with open('outputs/tpuv5p/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    eval_len_sys = split_sys(sys, 'eval_len')

results = dict()
for eval_len in eval_len_sys:
    results[eval_len] = dict()
    print('input, output:', eval_len)
    sys = eval_len_sys[eval_len][0]
    results[eval_len] = {'CapEx: Chip': [], 'CapEx: Sys': [],
                         'Opex: Cmpt': [], 'Opex: Mem': [], 
                         'Opex: I/O': [], 'OpEx: Other': []}
    for batch in batch_sizes:
        perf = sys.batch_opt_generate_tco[batch]
        capex_chip_ratio, capex_sys_ratio, opex_compute_ratio, opex_mem_ratio, opex_io_ratio, opex_other_ratio = get_tco_detail_breakdown(perf)

        results[eval_len]['CapEx: Chip'].append(capex_chip_ratio)
        results[eval_len]['CapEx: Sys'].append(capex_sys_ratio)
        results[eval_len]['Opex: Cmpt'].append(opex_compute_ratio)
        results[eval_len]['Opex: Mem'].append(opex_mem_ratio)
        results[eval_len]['Opex: I/O'].append(opex_io_ratio)
        results[eval_len]['OpEx: Other'].append(opex_other_ratio)

fig, axes = plt.subplots(1, 2, figsize=(8, 2.4))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
width = 0.6

for col, eval_len in enumerate(eval_lens):
    ax = axes[col]
    # 100% stacked bar chart of TCO breakdown
    bottom = [0] * len(batch_sizes)
    for i, key in enumerate(['CapEx: Chip', 'CapEx: Sys', 'Opex: Cmpt', 'Opex: Mem', 'Opex: I/O', 'OpEx: Other']):
        ax.bar(batch_sizes_str, results[eval_len][key], bottom=bottom,
                label=key, color=colors[i], edgecolor='k', hatch=hatches[i], width=width)
        bottom = [b + r for b, r in zip(bottom, results[eval_len][key])]
    input_tokens, output_tokens = eval_len.split(', ')
    ax.text(0.5, 1.0, f'{input_tokens} Input Tokens, {output_tokens} Output Tokens', 
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes, fontsize=12)
    ax.set_xlabel('Batch Size', fontsize=12)
    # ax.set_ylim(0, 1)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()], fontsize=8)
    if col == 0:
        ax.set_ylabel('TCO Breakdown (%)', fontsize=12)
    if col == 1:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, 
            loc='lower right', bbox_to_anchor=(1.1, 1.06), ncol=6,
            handletextpad=0.1, columnspacing=0.6)

fig.savefig(f'tpu_tco.pdf', format='pdf', bbox_inches='tight')
# %%
# HBM vs SRAM
colors = ['#7fc97f','#beaed4','#fdc086','#386cb0','#f0027f','#bf5b17']
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
# batch_sizes = [1, 2, 4, 8, 16, 32]
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
        results[mem]['latency'].append(perf.generate_latency + perf.prefill_latency)
        results[mem]['tco'].append(perf.generate_tco_per_token)

fig, axes = plt.subplots(1, 2, figsize=(8, 2.2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
# Latency
i = 0
lat_norm = results['3nm SRAM']['latency'][0]
for mem in all_sys:
    if 'TB/s' in mem:
        continue
    if 'GB' in mem:
        continue
    for j in range(len(batch_sizes)):
        results[mem]['latency'][j] = results[mem]['latency'][j] / lat_norm
    ax1.plot(batch_sizes_str, results[mem]['latency'], 
             label=mem, color=colors[i],
             marker='o' if 'HBM' in mem else 'x')
    i += 1
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.text(0.5, 1.0, 'Normalized Latency', 
        horizontalalignment='center', verticalalignment='bottom', 
         transform=ax1.transAxes, fontsize=12)
ax1.set_yscale('log')
# Norm TCO
norm = results['HBM2E']['tco'][0]
i= 0 
for mem in all_sys:
    if 'TB/s' in mem:
        continue
    if 'GB' in mem:
        continue
    ax2.plot(batch_sizes_str, [tco / norm for tco in results[mem]['tco']], 
             label=mem, color = colors[i],
             marker='o' if 'HBM' in mem else 'x')
    i += 1

# Zoom region
x1, x2 = 1.8, 2.2
y1, y2 = 0.02, 0.07
axins = ax2.inset_axes([0.05, 0.05, 0.2, 0.5],
                       xlim=(x1, x2), ylim=(y1, y2), 
                       xticklabels=[], yticklabels=[])
i = 0
for mem in all_sys:
    axins.plot(batch_sizes_str, [tco / norm for tco in results[mem]['tco']], 
             label=mem, color = colors[i],
             marker='o' if 'HBM' in mem else 'x')
    i += 1
axins.set_yticks([], minor=True)
ax2.indicate_inset_zoom(axins, edgecolor='black')

x1, x2 = 2.8, 3.2
y1, y2 = 0.005, 0.021
axins2 = ax2.inset_axes([0.75, 0.45, 0.2, 0.5],
                       xlim=(x1, x2), ylim=(y1, y2), 
                       xticklabels=[], yticklabels=[])
i = 0
for mem in all_sys:
    axins2.plot(batch_sizes_str, [tco / norm for tco in results[mem]['tco']], 
             label=mem, color = colors[i],
             marker='o' if 'HBM' in mem else 'x')
    i += 1
ax2.indicate_inset_zoom(axins2, edgecolor='black')


ax2.set_xlabel('Batch Size', fontsize=12)
ax2.text(0.5, 1.0, 'Normalized TCO per Request', 
horizontalalignment='center', verticalalignment='bottom', 
         transform=ax2.transAxes, fontsize=12)
ax2.set_yscale('log')
ax2.legend(loc='lower right', bbox_to_anchor=(.97, 1.08), ncol=6,
          handletextpad=0.1, columnspacing=0.6)

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

fig, axes = plt.subplots(1, 2, figsize=(8, 1.8))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
# 100% stacked bar chart
for i, mem in enumerate(['HBM3E', '5nm SRAM']):
    ax = axes[i]
    io_ratio = [results[mem]['I/O'][i] / results[mem]['Total'][i] for i in range(len(batch_sizes))]
    compute_ratio = [results[mem]['Compute'][i] / results[mem]['Total'][i] for i in range(len(batch_sizes))]
    mem_ratio = [results[mem]['Memory'][i] / results[mem]['Total'][i] for i in range(len(batch_sizes))]
    ax.bar(batch_sizes_str, io_ratio, label='I/O', 
           color=colors[0], edgecolor='k', hatch = hatches[0])
    ax.bar(batch_sizes_str, compute_ratio, bottom=io_ratio, label='Compute', 
           color=colors[1], edgecolor='k', hatch = hatches[1])
    ax.bar(batch_sizes_str, mem_ratio, bottom=[io + compute for io, compute in zip(io_ratio, compute_ratio)], label='Memory', 
           color=colors[2], edgecolor='k', hatch = hatches[2])
    ax.set_title(mem + ' Latency Breakdown', pad=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()], fontsize=8)
    # reverse legend order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), loc='best')

fig.savefig('hbm_vs_sram_latency.pdf', format='pdf', bbox_inches='tight')

# %%
# HBM Exploration
# batch_sizes = [2, 4, 8, 16]
# batch_sizes = [2, 8, 32]
batch_sizes = [1, 4, 16, 64]
hbm_server_num = {'24 GB HBM': 16,
                  '36 GB HBM': 11,
                  '48 GB HBM': 8,
                  '72 GB HBM': 6,
                  '96 GB HBM': 4,
                  '144 GB HBM': 3}
hbm_bw = []
bw_tco = {batch: [] for batch in batch_sizes}
with open('outputs/hbm_bw_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    hbm_sys = split_sys(sys, 'server.package.hbm.config')
    for batch in batch_sizes:
        norm = None
        for hbm in hbm_sys:
            if '13.8' in hbm:
                continue
            bw = hbm.split(' ')[0]
            if bw not in hbm_bw:
                hbm_bw.append(bw)
            sys = hbm_sys[hbm][0]
            perf = sys.batch_opt_generate_tco[batch]
            if norm is None:
                norm = perf.generate_tco_per_token
            bw_tco[batch].append(norm / perf.generate_tco_per_token)

hbm_cap = []
cap_tco = {batch: [] for batch in batch_sizes}
with open('outputs/hbm_cap_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    hbm_sys = split_sys(sys, 'server.package.hbm.config')
    for batch in batch_sizes:
        norm = None
        for hbm in hbm_sys:
            cap = hbm.split(' ')[0]
            if cap not in hbm_cap:
                hbm_cap.append(cap)
            # sys_srvs = split_sys(hbm_sys[hbm], 'num_servers')
            # sys = sys_srvs[hbm_server_num[hbm]][0]
            sys = hbm_sys[hbm][0]
            perf = sys.batch_opt_generate_tco[batch]
            if norm is None:
                norm = perf.generate_tco_per_token
            cap_tco[batch].append(norm / perf.generate_tco_per_token)

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
ax1, ax2 = axes
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
# Bandwidth Exploration
for i, batch in enumerate(batch_sizes):
    ax1.plot(hbm_bw, bw_tco[batch], color=colors[i],
             marker=markers[i], label=f'Batch {batch}')
ax1.set_xlabel('HBM Stack Bandwidth (TB/s)', fontsize=12)
ax1.text(0.5, 1.05, 'TCO/Request Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax1.transAxes, fontsize=12)
ax1.legend(loc='best', handletextpad=0.1, columnspacing=0.6, 
           fontsize=10,)
# Capacity Exploration
for i, batch in enumerate(batch_sizes):
    ax2.plot(hbm_cap, cap_tco[batch], color=colors[i],
             marker=markers[i], label=f'Batch {batch}')
ax2.set_xlabel('HBM Stack Capacity (GB)', fontsize=12)
ax2.text(0.5, 1.05, 'TCO/Request Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax2.transAxes, fontsize=12)
ax2.legend(loc='best', handletextpad=0.1, columnspacing=0.6, 
           fontsize=10,)

ax1.set_ylim(.9, 3)
ax2.set_ylim(.9, 3)
fig.savefig('hbm_explore.pdf', format='pdf', bbox_inches='tight')

# %%
# breakdown
with open('outputs/hbm_bw_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    hbm_sys = split_sys(sys, 'server.package.hbm.config')

# batch_sizes = [2, 8, 32, 128]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
batch_sizes_str = [str(batch) for batch in batch_sizes]
sys = hbm_sys['9.2 TB/s HBM'][0]
# Latency and TCO Breakdown
results = dict()
latency = {'I/O': [], 'Compute': [], 'Memory': []}
tco = {'CapEx: Chip': [], 'CapEx: System': [], 
       'OpEx: Compute': [], 'OpEx: Memory': [], 'OpEx: I/O': [], 'OpEx: Other': []}
for batch in batch_sizes:
    perf = sys.batch_opt_generate_tco[batch]
    t_io, t_compute, t_mem = get_latency_breakdown(perf, 'generate')
    total = t_io + t_compute + t_mem
    latency['I/O'].append(t_io / total)
    latency['Compute'].append(t_compute / total)
    latency['Memory'].append(t_mem / total)


    capex_chip_ratio, capex_sys_ratio, opex_compute_ratio, opex_mem_ratio, opex_io_ratio, opex_other_ratio = get_tco_detail_breakdown(perf)
    tco['CapEx: Chip'].append(capex_chip_ratio)
    tco['CapEx: System'].append(capex_sys_ratio)
    tco['OpEx: Compute'].append(opex_compute_ratio)
    tco['OpEx: Memory'].append(opex_mem_ratio)
    tco['OpEx: I/O'].append(opex_io_ratio)
    tco['OpEx: Other'].append(opex_other_ratio)

# fig, axes = plt.subplots(1, 2, figsize=(8, 2))
# fig.subplots_adjust(wspace=0.6, hspace=0.3)
# ax1, ax2 = axes
fig, ax2 = plt.subplots(1, 1, figsize=(6, 3))
width = 0.6
# Latency 100% stacked bar chart
# bottom = [0] * len(batch_sizes)
# for i, key in enumerate(['I/O', 'Compute', 'Memory']):
#     ax1.bar(batch_sizes_str, latency[key], bottom=bottom, label=key, 
#             color=colors[i], edgecolor='k', hatch=hatches[i], width=width)
#     bottom = [b + r for b, r in zip(bottom, latency[key])]
# ax1.set_title('Latency Breakdown (%)', pad=2)
# ax1.set_xlabel('Batch Size', fontsize=12)
# ax1.set_yticklabels(['{:,.0%}'.format(x) for x in ax1.get_yticks()], fontsize=8)
# ax1.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0), ncol=1,
#           handletextpad=0.1, columnspacing=0.6)
# TCO 100% stacked bar chart
bottom = [0] * len(batch_sizes)
for i, key in enumerate(['CapEx: Chip', 'CapEx: System', 'OpEx: Compute', 'OpEx: Memory', 'OpEx: I/O', 'OpEx: Other']):
    ax2.bar(batch_sizes_str, tco[key], bottom=bottom, label=key, 
            color=colors[i], edgecolor='k', hatch=hatches[i], width=width)
    bottom = [b + r for b, r in zip(bottom, tco[key])]
ax2.set_title('9.2 TB/s per Stack HBM', pad=2)
ax2.set_ylabel('TCO Breakdown (%)',fontsize=12)
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()], fontsize=8)
# reverse legend order
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(reversed(handles), reversed(labels),
          loc='lower left', bbox_to_anchor=(1.0, 0.0), ncol=1,
          handletextpad=0.1, columnspacing=0.6)
ax2.set_ylim(0, 1)
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
colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
batch_sizes = [2, 8, 32]
# batch_sizes = [4, 16, 64]
target_eval_len = '64, 256'

mem_sys = dict()
# for dram_3d in ['64tsvs', '96tsvs', '128tsvs', '160tsvs', '192tsvs', '224tsvs']:
#     with open('outputs/dram_3d_'+dram_3d+'/gpt3.pkl', 'rb') as f:
#         sys = pickle.load(f)
#         label = dram_3d.split('tsvs')[0] + ' TSVs'
#         mem_sys[label] = sys
with open('outputs/dram_3d_tsv_explore/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    eval_len_sys = split_sys(sys, 'eval_len')
    for eval_len in eval_len_sys:
        if eval_len == target_eval_len:
            mem_sys = split_sys(eval_len_sys[eval_len], 'server.package.mem_3d.config')
            for dram_3d in mem_sys:
                mem_sys[dram_3d] = mem_sys[dram_3d]

all_norm_latency = dict()
all_norm_tco = dict()
mems = []

for mem in mem_sys:
    sys = mem_sys[mem][0]
    mems.append(mem.split(' ')[0])
    for batch in batch_sizes:
        perf = sys.batch_opt_generate_tco[batch]
        latency = (perf.generate_latency + perf.prefill_latency)
        tco = perf.generate_tco_per_token
        if mem == '64 TSVs':
            norm_latency[batch] = latency
            norm_tco[batch] = tco
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
    ax1.plot(mems, all_norm_latency[batch], 
    marker=markers[i], color=colors[i+4],
    label=f'Batch {batch}')
# ax1.plot(mems, all_norm_latency['Average'], marker='P', label='Average', color='tab:gray', linestyle='--')
ax1.text(0.5, 1.05, 'Latency Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax1.transAxes, fontsize=12)
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)
ax1.set_xlabel('Data TSVs per Vault', fontsize=12)
ax1.grid(True, which='both', axis='y')
# TCO
for i, batch in enumerate(batch_sizes):
    ax2.plot(mems, all_norm_tco[batch], 
             marker=markers[i], color=colors[i+4],
             label=f'Batch {batch}')
# ax2.plot(mems, all_norm_tco['Average'], marker='P', label='Average', color='tab:gray', linestyle='--')
ax2.set_xlabel('Data TSVs per Vault', fontsize=12)
ax2.text(0.5, 1.05, 'TCO/Request Improvement', horizontalalignment='center', verticalalignment='center', 
         transform=ax2.transAxes, fontsize=12)

ax2.legend(loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)
# ax2.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0), ncol=1,
        #    handletextpad=0.1, columnspacing=0.6, fontsize=9)
ax2.grid(True, which='both', axis='y')

fig.savefig('3d_dram_tradeoff.pdf', format='pdf', bbox_inches='tight')

# %%
# Prefill IO
batch = 2
model = 'gpt3'
with open(f'outputs/prefill_io_explore/{model}.pkl', 'rb') as f:
    sys = pickle.load(f)
    io_sys = split_sys(sys, 'num_servers')

lat_breakdown = {'I/O': [], 'Compute': [], 'Memory': []}
tput_per_node = []
tco_per_Mtoken = []
num_nodes = []
for io in io_sys:
    if io > 16:
        continue
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
fig.subplots_adjust(wspace=0.47, hspace=0.3)
width = 0.6
ax1, ax2 = axes
str_num_nodes = [str(node) for node in num_nodes]
# latency breakdown stacked bar
norm = lat_breakdown['I/O'][0]
for k in lat_breakdown:
    lat_breakdown[k] = [lat / norm for lat in lat_breakdown[k]]

norm = tco_per_Mtoken[0]
tco_per_Mtoken = [tco / norm for tco in tco_per_Mtoken]

# ax1.yaxis.tick_right()
ax1.bar(str_num_nodes, lat_breakdown['I/O'], label='I/O', 
        color=colors[0], edgecolor='k', hatch=hatches[0], width=width)
ax1.bar(str_num_nodes, lat_breakdown['Compute'], bottom=lat_breakdown['I/O'], label='Compute', 
        color=colors[1], edgecolor='k', hatch=hatches[1], width=width)
ax1.bar(str_num_nodes, lat_breakdown['Memory'], bottom=[io + compute for io, compute in zip(lat_breakdown['I/O'], lat_breakdown['Compute'])], label='Memory', 
      color=colors[2], edgecolor='k', hatch=hatches[2], width=width)
ax1.set_ylabel('Normalized Latency', fontsize=12)
ax1.set_ylim(0, 6)

ax1_sub = ax1.twinx()
# ax1_sub.yaxis.tick_left()
ax1_sub.plot(str_num_nodes, tco_per_Mtoken, marker='o', color=colors[4], label='TCO/Request')
ax1_sub.set_ylabel('Norm. TCO/Request', fontsize=12)
ax1_sub.set_ylim(0.9, 1.8)

ax1.set_xlabel('Number of Servers', fontsize=12)
# ax1.text(0.5, 1.05, 'Normalized Prefill Latency', horizontalalignment='center', verticalalignment='center', 
        #  transform=ax1.transAxes, fontsize=12)

# combine two legends
handles, labels = ax1.get_legend_handles_labels()
handles_r, labels_r = ax1_sub.get_legend_handles_labels()
handles += handles_r
ax1.legend(handles, labels + labels_r, ncol=2,
           loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)

# norm tco per token
# norm = tco_per_Mtoken[0]
# tco_per_Mtoken = [tco / norm for tco in tco_per_Mtoken]
# ax2.plot(str_num_nodes, tco_per_Mtoken, marker='o') 
# ax2.set_xlabel('Number of Servers', fontsize=12)
# ax2.text(0.5, 1.05, 'Normalized Prefill TCO per Token', horizontalalignment='center', verticalalignment='center',
#             transform=ax2.transAxes, fontsize=12)
# ax2.set_yscale('log')

# Max context length
if model == 'gpt3':
    model_size_byte = 175 * 1024 * 1024 * 1024 * 2 
    kv_size_byte_per_token = 2 * 96 * 2 * 12288
elif model == 'palm':
    model_size_byte = 175 * 1024 * 1024 * 1024
    kv_size_byte_per_token = 2 * 96 * 2 * 12288
server_mem_byte = sys.server.dram
max_ctx_len = {4: [], 64: []}
for num_node in num_nodes:
    total_byte = server_mem_byte * num_node
    mem_for_kv = total_byte - model_size_byte
    for batch in max_ctx_len:
        max_ctx_len[batch].append(mem_for_kv / kv_size_byte_per_token / batch)

markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
for i, batch in enumerate(max_ctx_len):
    ax2.plot(str_num_nodes, max_ctx_len[batch], 
             marker=markers[i], color=colors[i+4],
             label=f'Batch {batch}')
    for x, y in zip(str_num_nodes, max_ctx_len[batch]): 
        if i == 1:
            ax2.text(x, y, f'{y/1000:.1f}K', fontsize=8, ha='center', va='bottom')
        else:
            ax2.text(x, y/1.6, f'{y/1000:.1f}K', fontsize=8, ha='center', va='top')
ax2.set_xlabel('Number of Servers', fontsize=12)
# ax2.text(0.5, 1.05, 'Max Context Length', 
        #  horizontalalignment='center', verticalalignment='center',
            # transform=ax2.transAxes, fontsize=12)
ax2.set_ylabel('Max Context Length', fontsize=12)
ax2.set_ylim(0.5e2, 1000e3)
ax2.legend(loc='best', handletextpad=0.1, columnspacing=0.6, fontsize=9)
ax2.set_yscale('log')
fig.savefig('prefill_io_explore.pdf', format='pdf', bbox_inches='tight')

# %%
# Dynamic Parallelism
all_sys = dict()
with open('outputs/dp_mp_overhead/gpt3.pkl', 'rb') as f:
    sys = pickle.load(f)
    eval_sys = split_sys(sys, 'eval_len')
    for eval_len in eval_sys:
        all_sys[eval_len] = dict()
        dp_mp_sys = split_sys(eval_sys[eval_len], 'num_servers')
        for num_srvs in dp_mp_sys:
            all_sys[eval_len][num_srvs] = dp_mp_sys[num_srvs]

baseline_lat = dict()
opt_lat = dict()
all_lat = dict()
num_srv = 8
opt_prefill_num_srv = 1
# eval_lens = ['256, 64', '64, 256',  '512, 1024']
eval_lens = ['256, 64', '512, 128']
alpha = 1e-8
beta = 1/300e9
batch_sizes = [16, 32, 64, 128]

for eval_len in eval_lens:
    baseline_lat[eval_len] = {'Prefill': [], 'Decoding': []}
    opt_lat[eval_len] = {'Prefill': [], 'Decoding': [], 'Alltoall': [], 'Allgather': []}
    for batch in batch_sizes:
        # baseline
        sys = all_sys[eval_len][num_srv][0]
        perf = sys.batch_opt_generate_tco[batch]
        # perf = sys.batch_opt_generate_lat[batch]
        # norm = perf.prefill_latency + perf.generate_latency
        norm = perf.prefill_latency
        baseline_lat[eval_len]['Prefill'].append(perf.prefill_latency / norm)
        baseline_lat[eval_len]['Decoding'].append(perf.generate_latency / norm)
        # optimized
        if eval_len == '1024, 256' and batch == 128:
            opt_prefill_num_srv = 4
        opt_sys = all_sys[eval_len][opt_prefill_num_srv][0]
        sub_batch = batch // (num_srv // opt_prefill_num_srv)
        opt_perf = opt_sys.batch_opt_generate_tco[sub_batch]
        # opt_perf = opt_sys.batch_opt_generate_lat[batch]
        opt_lat[eval_len]['Prefill'].append(opt_perf.prefill_latency / norm)
        opt_lat[eval_len]['Decoding'].append(perf.generate_latency / norm)

        p = num_srv // opt_prefill_num_srv
        input_len = int(eval_len.split(',')[0])
        N = input_len * 96 * 2 * 12288
        alltoall_lat = (p - 1) * (alpha + 0.5 * beta * N )
        N = 2 * 96 * 12 * 12288 * 12288 / p
        allgather_lat = (p - 1) * (alpha + beta * N / p)

        # if input_len == '64' or input_len == '256':
        #     if batch <= 32:
        #         allgather_lat = 0
        #         alltoall_lat = 0
        # if input_len == '1024':
        #     if batch <= 8:
        #         allgather_lat = 0
        #         alltoall_lat = 0
        opt_lat[eval_len]['Alltoall'].append(alltoall_lat / norm)
        opt_lat[eval_len]['Allgather'].append(allgather_lat / norm)

fig, axes = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.15, hspace=0.3)
width = 0.4
import numpy as np
x = np.arange(len(batch_sizes))
offset = [-0.5*width, 0.5*width]
for i, eval_len in enumerate(eval_lens):
    ax = axes[i]
    # baseline
    bottom = [0] * len(batch_sizes)
    # for j, key in enumerate(['Prefill', 'Decoding']):
    for j, key in enumerate(['Prefill']):
        ax.bar(x + offset[0], baseline_lat[eval_len][key],
                bottom=bottom, label='Base '+key, color=colors[j], 
                edgecolor='k', hatch=hatches[j], width=width)
        bottom = [b + r for b, r in zip(bottom, baseline_lat[eval_len][key])]
    # optimized
    # for j, key in enumerate(['Opt. Prefill', 'Overhead']):
    bottom = [0] * len(batch_sizes)
    # for j, key in enumerate(['Prefill', 'Decoding', 'Alltoall', 'Allgather']):
    for j, key in enumerate(['Prefill', 'Alltoall', 'Allgather']):
        if key == 'Allgather':
            alpha = 0.4
        else:
            alpha = 1
        ax.bar(x + offset[1], opt_lat[eval_len][key],
                bottom=bottom, label='DP '+ key, alpha=alpha,
                color=colors[j+1], edgecolor='k', hatch=hatches[j+1], width=width)
        bottom = [b + r for b, r in zip(bottom, opt_lat[eval_len][key])]

    for j in range(len(batch_sizes)):
        # opt_lat_total = opt_lat[eval_len]['Prefill'][j] + opt_lat[eval_len]['Decoding'][j] + opt_lat[eval_len]['Alltoall'][j] + opt_lat[eval_len]['Allgather'][j]
        opt_lat_total = opt_lat[eval_len]['Prefill'][j] + opt_lat[eval_len]['Alltoall'][j] + opt_lat[eval_len]['Allgather'][j]
        ax.text(j + offset[1]+0.07, opt_lat_total, f'{opt_lat_total:.2f}x', 
                fontsize=9, ha='center', va='bottom')
    
    input_len = eval_len.split(',')[0]
    output_len = eval_len.split(',')[1][1:]
    # ax.set_title(f'{input_len} input, {output_len} output', pad=2)
    ax.set_title(f'{input_len} Input Tokens ', pad=2)
    ax.set_xlabel('Batch Size', fontsize=12)
    if i == 0:
        ax.set_ylabel('Normalized Latency', fontsize=12)
    if i == 1:
        # reverse legend order
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, 
                #   loc='lower left', bbox_to_anchor=(-0.1, 1.065), ncol=6,
                  loc='lower left', bbox_to_anchor=(0.99, 0.065), ncol=1,
                  handletextpad=0.1, columnspacing=0.6, fontsize=10)
    ax.set_xticks(x, batch_sizes)
    ax.set_yticklabels(['{:,.1f}'.format(x) for x in ax.get_yticks()], fontsize=8)

fig.savefig('dp_mp_overhead.pdf', format='pdf', bbox_inches='tight')

# %%
# allreduce algorithms comparison
def get_allreduce_time(p, a, B, N, algo='Ring'):
    if algo == 'Ring':
        return 2 * (p-1) * a + 2 * (p-1) / p * N / B
    elif algo == '2D Ring':
        # return 4 * (math.sqrt(p) -1) * a + 4 * (math.sqrt(p)-1) / math.sqrt(p) * N / B
        return 4 * (math.sqrt(p) -1) * a + 2 * (math.sqrt(p)-1) / math.sqrt(p) * N / B
    elif algo == 'Two Tree':
        return 4 * math.log2(p) * a + 2 * N / B + 4 * math.sqrt(2 * math.log2(p) * a * N / B)
    elif algo == 'Linear Pipeline':
        # return 2 * p * a + 2 * N / B + 2 * math.sqrt(math.log2(p) * a * N / B)
        return 2 * p * a + 2 * N / B + 2 * math.sqrt(math.log2(p) * a * N / B)
    elif algo == 'Rabenseifner':
        return 2 * math.log2(p) * a + 2 * (p-1) / p * N * math.log2(p) / B
    elif algo == 'Hypercube Ring':
        # return 8 * (p**0.25 -1) * a + 8 * (p**0.25-1) / p**0.25 * N / B
        return 8 * (p**0.25 -1) * a + 2 * (p**0.25-1) / p**0.25 * N / B
    else:
        raise ValueError('Invalid algorithm')

algos = ['Ring', '2D Ring', 'Two Tree', 'Hypercube Ring']
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
ax1.legend(loc = 'lower center', 
            bbox_to_anchor=(1.1, 0.97),
            ncol=6, 
           handletextpad=0.2, columnspacing=0.6, fontsize=10)

fig.savefig('allreduce_algo.pdf', format='pdf', bbox_inches='tight')

# %%
# comparing with TPUv5p and B200 GPUs
from structs.TCO import TCO
hatches = ['/', '\\']
# batch_sizes = [8, 32, 128]
batch_sizes = [4, 16, 64]
# batch_sizes = [4, 16, 64, 256]
hardwares = ['tpuv5p_eval', 'b200_eval', 'cc_3d']
models = ['llama2', 'gpt3', 'mtnlg','palm']
model_labels = {'llama2': 'Llama-2 70B', 
                'gpt3': 'GPT-3 175B', 
                'mtnlg': 'MT-NLG 530B',
                'palm': 'PaLM 540B'}
# model_srvs = {'llama2': 2, 'gpt3': 4, 'mtnlg': 8, 'palm': 8}
model_srvs = {'llama2': 2, 'gpt3': 4, 'mtnlg': 4, 'palm': 4}
# eval_lens = ['256, 1024', '1024, 256']
eval_lens = {
    'llama2': ['256, 64', '64, 256'],
    'gpt3': ['1024, 256', '256, 1024'],
    # 'gpt3': ['256, 64', '256, 64'],
    # 'mtnlg': ['256, 64', '64, 256'],
    'mtnlg': ['1024, 256', '256, 1024'],
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
                if hw == 'cc_3d':
                    base_num_srvs = num_srvs
                    base_sys = sys_srvs[num_srvs][0]
                    if model == 'llama2' or model == 'gpt3':
                        sub_num_srvs = 1
                    else:
                        sub_num_srvs = 2
                    sub_sys = sys_srvs[sub_num_srvs][0]
                    if key not in tco_opt_latency:
                        tco_opt_latency[key] = []
                        tco_opt_prefill_lat[key] = []
                        tco_opt_generate_lat[key] = []
                        tco_opt_tco[key] = []
                        lat_opt_latency[key] = []
                        lat_opt_tco[key] = []
                    for batch in batch_sizes:
                        sub_batch = batch // (base_num_srvs // sub_num_srvs)
                        perf = base_sys.batch_opt_generate_tco[batch]
                        sub_perf = sub_sys.batch_opt_prefill_lat[sub_batch]
                        # Prefill
                        prefill_latency = sub_perf.prefill_latency
                        # Decoding
                        generate_latency = perf.generate_latency

                        speedup = (perf.prefill_latency + generate_latency) / (prefill_latency + generate_latency)
                        tco = perf.generate_tco_per_token / speedup

                        tco_opt_latency[key].append(prefill_latency + generate_latency)
                        tco_opt_prefill_lat[key].append(prefill_latency)
                        tco_opt_generate_lat[key].append(generate_latency)
                        tco_opt_tco[key].append(tco)
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
# fig, axes = plt.subplots(3, 8, figsize=(16, 4.6))
fig, axes = plt.subplots(2, 8, figsize=(16, 3.6))
fig.subplots_adjust(wspace=0.16, hspace=0.16)
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
                all_lat.append(tco_opt_latency[key][l] / tco_opt_latency[norm_key][l])
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
                ax.text(x[l] + offset[k], all_lat[l] + 0.0, 
                        f'{all_lat[l]:.1f}', 
                        fontsize=6, ha='center', va='bottom')
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
                      bbox_to_anchor=(4.56, 1.17), 
                      ncol=2,
                      fontsize=12)
# # Row 1: Generation Latency
# for i, model in enumerate(models):
#     for j, eval_len in enumerate(eval_lens[model]):
#         ax = axes[1, i*2+j]
#         norm_key = f'{hardwares[-1]}_{model}_{eval_len}'
#         for k, hw in enumerate(hardwares[:-1]):
#             key = f'{hw}_{model}_{eval_len}'
#             all_lat =  []
#             for l, batch in enumerate(batch_sizes):
#                 all_lat.append(tco_opt_generate_lat[key][l] / tco_opt_generate_lat[norm_key][l])
#             # add mean
#             all_lat.append(np.mean(all_lat))
#             ax.bar(x + offset[k], all_lat, width, facecolor=colors[k], 
#                    hatch=hatches[k],
#                    edgecolor='k', label=label)
#             for l in range(len(batch_sizes) + 1):
#                 ax.text(x[l] + offset[k], all_lat[l] + 0.0, 
#                         f'{all_lat[l]:.1f}', 
#                         fontsize=6, ha='center', va='bottom')
#             if k == 0:
#                 ax.set_ylim(0, max(all_lat) + 0.5)

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
                ax.text(x[l] + offset[k], all_tco[l] + 0.0, 
                        f'{all_tco[l]:.1f}', 
                        fontsize=6, ha='center', va='bottom')
            if k == 0:
                ax.set_ylim(0, max(all_tco) + 0.5)
        if i == 0 and j == 0:
            ax.set_ylabel('TCO/Request\nImprovement', fontsize=12)

for ax in axes.flatten():
    ax.set_xticks(x, x_labels, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
for ax in axes[-1]:
    ax.set_xlabel('Batch Size')

fig.savefig('hw_comparison.pdf', format='pdf', bbox_inches='tight')

# %%
# batch = 8
# hardwares = ['tpuv5p_eval', 'b200_eval', 'cc_3d']
# model = 'mtnlg'
# # eval_len = '256, 64'
# eval_len = '64, 256'
# num_srvs = 4
# for hw in hardwares:
#     with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
#         all_sys = pickle.load(f)
#         sys_srvs = split_sys(all_sys, 'num_servers')
#         if hw == 'tpuv5p_eval':
#             num_srvs *= 2
#         sys_eval_len = split_sys(sys_srvs[num_srvs], 'eval_len')
#         sys = sys_eval_len[eval_len][0]
#         perf = sys.batch_opt_generate_tco[batch]
#         print(f'{hw}: prefill: {perf.prefill_latency:.4f}, decoding: {perf.generate_latency:.4f}')
#         print(sys.server.dram / 1024 / 1024/ 1024)
#         print(sys.server.package.tdp)
# %%
# Hardware Comparison: Latency and TCO Breakdown
# batch_sizes = [2, 8, 32, 128]
batch_sizes = [4, 16, 64, 128]
hardwares = ['tpuv5p_eval', 'b200_eval', 'cc_3d']
model = 'gpt3'
model_srvs = {'llama2': 2, 'gpt3': 4, 'mtnlg': 4, 'palm': 4}
# eval_len = '256, 64'
eval_len = '1024, 256'
# eval_len = '256, 1024'

latency_breakdown = dict()
tco_breakdown = dict()

for hw in hardwares:
    with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
        key = f'{hw}_{model}_{eval_len}'
        all_sys = pickle.load(f)
        sys_eval_len = split_sys(all_sys, 'eval_len')
        sys_srvs = split_sys(sys_eval_len[eval_len], 'num_servers')
        num_srvs = model_srvs[model]
        if hw == 'tpuv5p_eval':
            num_srvs *= 2
        sys = sys_srvs[num_srvs][0]
        if key not in latency_breakdown:
            latency_breakdown[key] = {'I/O': [], 'Compute': [], 'Memory': []}
            tco_breakdown[key] = {'CapEx': [], 'OpEx': []}
        for batch in batch_sizes:
            perf = sys.batch_opt_generate_tco[batch]
            if hw == 'cc_3d' and batch >= num_srvs:
                if model == 'llama2' or model == 'gpt3':
                    sub_num_srvs = 1
                else:
                    sub_num_srvs = 2
                sub_sys = sys_srvs[sub_num_srvs][0]
                sub_batch = batch // (base_num_srvs // sub_num_srvs)
                sub_perf = sub_sys.batch_opt_prefill_lat[sub_batch]
                pre_io, pre_compute, pre_mem = get_latency_breakdown(sub_perf, 'prefill')
            else:
                pre_io, pre_compute, pre_mem = get_latency_breakdown(perf, 'prefill')
            gen_io, gen_compute, gen_mem = get_latency_breakdown(perf, 'generate')
            latency_breakdown[key]['I/O'].append(pre_io + gen_io)
            latency_breakdown[key]['Compute'].append(pre_compute + gen_compute)
            latency_breakdown[key]['Memory'].append(pre_mem + gen_mem)

            tco_per_Mtoken, capex, opex = get_tco_breakdown(perf)
            tco_breakdown[key]['CapEx'].append(capex)
            tco_breakdown[key]['OpEx'].append(opex)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
fig.subplots_adjust(wspace=0.2, hspace=0.16)
markers = ['o', 'x', '^', 's', 'D', 'v', 'P', 'X', 'H']
hatches = ['//', '.', '\\\\']
colors = [colors[0], colors[1], colors[2]]
x = np.arange(len(batch_sizes))
width = 0.3
offset = [-width, 0, width]
batch_sizes_str = [str(batch) for batch in batch_sizes]
x_labels = batch_sizes_str
# ax1: breakdown
for i, batch in enumerate(batch_sizes):
    norm = latency_breakdown[f'tpuv5p_eval_{model}_{eval_len}']['I/O'][i] + \
           latency_breakdown[f'tpuv5p_eval_{model}_{eval_len}']['Compute'][i] + \
           latency_breakdown[f'tpuv5p_eval_{model}_{eval_len}']['Memory'][i]
    for k, key in enumerate(latency_breakdown):
        latency_breakdown[key]['I/O'][i] /= norm
        latency_breakdown[key]['Compute'][i] /= norm
        latency_breakdown[key]['Memory'][i] /= norm

for i, key in enumerate(latency_breakdown):
    ax1.bar(x + offset[i], latency_breakdown[key]['I/O'], 
            width=width, facecolor=colors[i], hatch=hatches[0], 
            edgecolor='k')
    ax1.bar(x + offset[i], latency_breakdown[key]['Compute'],
            bottom=latency_breakdown[key]['I/O'],
            width=width, facecolor=colors[i], hatch=hatches[1],
            edgecolor='k')
    ax1.bar(x + offset[i], latency_breakdown[key]['Memory'],
            bottom=[io + compute for io, compute in zip(latency_breakdown[key]['I/O'], latency_breakdown[key]['Compute'])],
            width=width, facecolor=colors[i], hatch=hatches[2],
            edgecolor='k')
ax1.bar(0,0,0, label='Memory', facecolor='white', edgecolor='k', hatch=hatches[2])
ax1.bar(0,0,0, label='Compute', facecolor='white', edgecolor='k', hatch=hatches[1])
ax1.bar(0,0,0, label='I/O', facecolor='white', edgecolor='k', hatch=hatches[0])
ax1_r = ax1.twinx()
hw_labels = {'tpuv5p_eval': 'TPUv5p', 
             'b200_eval': 'NVIDIA B200 GPU', 
             'cc_3d': 'This work'}
for i, hw in enumerate(hardwares):
    ax1_r.bar(0,0,0, label=hw_labels[hw], facecolor=colors[i], edgecolor='k')
ax1_r.axis('off')
ax1.legend(loc='best', fontsize=8, ncol=3)
ax1_r.legend(loc='lower center', bbox_to_anchor=(1.1, 0.97), ncol=3, 
             fontsize=10)
ax1.set_xticks(x, x_labels)
ax1.set_xlabel('Batch Size', fontsize=12)
ax1.set_ylabel('Latency Breakdown', fontsize=12)
ax1.set_ylim(0, 1.2)

# ax2: TCO breakdown
for i, batch in enumerate(batch_sizes):
    norm = tco_breakdown[f'tpuv5p_eval_{model}_{eval_len}']['CapEx'][i]
    for k, key in enumerate(tco_breakdown):
        tco_breakdown[key]['CapEx'][i] /= norm
        tco_breakdown[key]['OpEx'][i] /= norm

for i, key in enumerate(tco_breakdown):
    ax2.bar(x + offset[i], tco_breakdown[key]['CapEx'], 
            width=width, facecolor=colors[i], hatch=hatches[0],
            edgecolor='k')
    ax2.bar(x + offset[i], tco_breakdown[key]['OpEx'],
            bottom=tco_breakdown[key]['CapEx'],
            width=width, facecolor=colors[i], hatch=hatches[1],
            edgecolor='k')
ax2.bar(0,0,0, label='OpEx', facecolor='white', edgecolor='k', hatch=hatches[1])
ax2.bar(0,0,0, label='CapEx', facecolor='white', edgecolor='k', hatch=hatches[0])
ax2.legend(loc='best', fontsize=8, ncol=2)
ax2.set_xticks(x, x_labels)
ax2.set_xlabel('Batch Size', fontsize=12)
ax2.set_ylabel('TCO Breakdown', fontsize=12)
ax1.set_yticklabels(ax1.get_yticks(), fontsize=8)
ax2.set_yticklabels(ax2.get_yticks(), fontsize=8)

fig.savefig('hw_comparison_breakdown.pdf', format='pdf', bbox_inches='tight')

# %%
# Optimization breakdown
# batch_sizes = [1, 2, 4, 8, 16]
# batch_sizes = [2, 8, 32, 128]
hardwares = ['cc_3d_sys1', 'cc_3d_sys15' ,'cc_3d_sys2', 'cc_3d_sys3', 'cc_3d']
models = ['llama2', 'gpt3', 'mtnlg','palm']
batch = 32
batch_sizes = [4, 8, 16, 32, 64]
# eval_len = '1024, 256'
# eval_len = '256, 64'
eval_lens = ['64, 256', '256, 64']
# eval_lens = ['1024, 256', '256, 1024']
num_srvs = 4

latency = dict()
prefill_lat = dict()
generate_lat = dict()
tco = dict()

for hw in hardwares:
    model = 'gpt3'
    # for model in models:
    for eval_len in eval_lens:
        key = f'{hw}_{eval_len}'
        for batch in batch_sizes:
            with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
                all_sys = pickle.load(f)
                sys_eval_len = split_sys(all_sys, 'eval_len')
                sys_srvs = split_sys(sys_eval_len[eval_len], 'num_servers')
                sys = sys_srvs[num_srvs][0]
                if key not in latency:
                    latency[key] = []
                    tco[key] = []
                    prefill_lat[key] = []
                    generate_lat[key] = []
                perf = sys.batch_opt_generate_tco[batch]
                if (hw == 'cc_3d_sys3' or hw == 'cc_3d') and batch >= num_srvs:
                    if model == 'llama2' or model == 'gpt3':
                        sub_num_srvs = 1
                    else:
                        sub_num_srvs = 2
                    sub_sys = sys_srvs[sub_num_srvs][0]
                    sub_batch = batch // (num_srvs // sub_num_srvs)
                    sub_perf = sub_sys.batch_opt_prefill_lat[sub_batch]
                    latency[key].append(sub_perf.prefill_latency + perf.generate_latency)
                    prefill_lat[key].append(sub_perf.prefill_latency)
                    speedup = (perf.prefill_latency + perf.generate_latency) / (sub_perf.prefill_latency + perf.generate_latency)
                else:
                    latency[key].append(perf.prefill_latency + perf.generate_latency)
                    prefill_lat[key].append(perf.prefill_latency)
                    speedup = 1
                generate_lat[key].append(perf.generate_latency)
                tco[key].append(perf.tco_per_token / speedup)

        # add average
        latency[key].append(np.mean(latency[key]))
        tco[key].append(np.mean(tco[key]))
        prefill_lat[key].append(np.mean(prefill_lat[key]))
        generate_lat[key].append(np.mean(generate_lat[key]))

# fig, ax1 = plt.subplots(1, 1, figsize=(7, 2.2))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 4))
fig.subplots_adjust(wspace=0.18, hspace=0.29)
colors = ['tab:grey','#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
hatch = ['//', '\\', '+', 'o', 'x']
width = 0.175
offset = [-2, -1, 0, 1, 2]
offset = [off * width for off in offset]
models = ['Llama-2', 'GPT-3', 'MT-NLG', 'PaLM', 'Mean']
batch_sizes_str = [str(batch) for batch in batch_sizes]
batch_sizes_str.append('Mean')
x = np.arange(len(batch_sizes_str))
# ax1: latency
axes = [ax1, ax2]
for i, eval_len in enumerate(eval_lens):
    ax = axes[i]

    for j, batch in enumerate(batch_sizes_str):
        norm = latency[f'cc_3d_sys1_'+eval_len][j]
        for k, hw in enumerate(hardwares):
            key = f'{hw}_{eval_len}'
            latency[key][j] = norm / latency[key][j]

    for j, hw in enumerate(hardwares):
        key = f'{hw}_{eval_len}'
        ax.bar(x + offset[j], latency[key], 
                width=width, facecolor=colors[j], 
                hatch=hatch[j],
                edgecolor='k')
        if j > 0:
            # for l in range(len(batch_sizes_str)):
            l = -1
            ax.text(x[l] + offset[j], latency[key][l] + 0.0, 
                    f'{latency[key][l]:.1f}', 
                    fontsize=8, ha='center', va='bottom')
    ax.set_xticks(x, batch_sizes_str)
    ax.set_ylabel('Latency Speedup')
    ax.set_title(f'{eval_len.split(",")[0]} Input Tokens, {eval_len.split(",")[1][1:]} Output Tokens', pad=1)
ax1.set_ylim(0, 5)
ax2.set_ylim(0, 4)
ax2.set_xlabel('Batch Size', fontsize=13)

# ax2: TCO
# tco = generate_lat
# for i, model in enumerate(models):
#     norm = tco['cc_3d_sys1'][i]
#     for k, key in enumerate(tco):
#         tco[key][i] = norm / tco[key][i]
# 
# for i, key in enumerate(tco):
#     ax2.bar(x + offset[i], tco[key], 
#             width=width, facecolor=colors[i], 
#             hatch=hatch[i],
#             edgecolor='k')
#     if i > 0:
#         for l in range(len(models)):
#             ax2.text(x[l] + offset[i], tco[key][l] + 0.0, 
#                     f'{tco[key][l]:.1f}', 
#                     fontsize=8, ha='center', va='bottom')
# ax2.set_xticks(x, models)
# ax2.set_ylabel('TCO/Request Imporv.')
# ax1.set_yticklabels(ax1.get_yticks(), fontsize=10)
# ax2.set_yticklabels(ax2.get_yticks(), fontsize=10)
# ax2.set_ylim(0, 4.3)

ax1_r = ax1.twinx()
hw_labels = {'cc_3d_sys1':  'Chiplet Baseline', 
             'cc_3d_sys15': '+Naive 3D Mem',
             'cc_3d_sys2': '+Opt. 3D Mem',
             'cc_3d_sys3': '+Opt. 3D Mem\n+Dynamic Para.',
             'cc_3d': '+Opt. 3D MEM\n+Dynamic Para.\n+Hypercube '}

for i, hw in enumerate(hardwares):
    ax1_r.bar(0,0,0, label=hw_labels[hw], 
              hatch=hatch[i],
              facecolor=colors[i], edgecolor='k')

ax1_r.legend(loc='center left', bbox_to_anchor=(1., -0.1), ncol=1, 
             fontsize=10, handletextpad=0.4, labelspacing=2)

ax1_r.axis('off')

fig.savefig('opt_breakdown.pdf', format='pdf', bbox_inches='tight')

# %%
hardwares = ['cc_3d_sys3']
models = ['llama2', 'gpt3', 'mtnlg','palm']
batch = 32
# batch_sizes = [4, 8, 16, 32, 64]
batch_sizes = [16, 32, 64]
eval_lens = ['64, 256', '256, 64']
num_srvs = 16

for hw in hardwares:
    model = 'gpt3'
    # for model in models:
    for eval_len in eval_lens:
        print('eval len: ', eval_len)
        key = f'{hw}_{eval_len}'
        for batch in batch_sizes:
            print('batch: ', batch)
            with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
                all_sys = pickle.load(f)
                sys_eval_len = split_sys(all_sys, 'eval_len')
                sys_srvs = split_sys(sys_eval_len[eval_len], 'num_servers')
                for num_srvs in [1, 2 ,4, 8, 16]:
                    sys = sys_srvs[num_srvs][0]
                    sub_batch = batch // (16 // num_srvs)
                    if num_srvs == 16:
                        perf = sys.batch_opt_generate_tco[sub_batch]
                        # perf = sys.batch_opt_prefill_lat[sub_batch]
                    else:
                        perf = sys.batch_opt_prefill_lat[sub_batch]
                    print(f'Sub batch={sub_batch}, {num_srvs} servers: prefill: {perf.prefill_latency:.4f}, decoding: {perf.generate_latency:.4f}')