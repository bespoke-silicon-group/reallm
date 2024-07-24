# %%
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

import pickle, sys, math, functools
sys.path.append('micro_arch_sim/')
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
    
# %%
hatches = ['\\', '*', '/' , '|', '-', '+', 'x', 'o', 'O', '.']
colors = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
model_labels = {'llama2': 'Llama-2 70B', 
                'gpt3': 'GPT-3 175B', 
                'mtnlg': 'MT-NLG 530B',
                'palm': 'PaLM 540B'}
model_srvs = {'llama2': 1, 'gpt3': 2, 'mtnlg': 4, 'palm': 2}
eval_lens = ['1024, 256', '256, 1024']

def phase_breakdown(hardwares: list, models: list, 
                    batch_sizes: list,
                    eval_lens = eval_lens,
                    hatches = hatches, colors = colors,
                    model_srvs = model_srvs,
                    ):      
    # choose the mapping optimized for latency
    prefill_lat = dict()
    generate_lat = dict()
    prefill_energy = dict()
    generate_energy = dict()
    max_ctx_len = dict()

    for hw in hardwares:
        for model in models:
            num_srvs = model_srvs[model]
            with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
                all_sys = pickle.load(f)
                sys_eval_len = split_sys(all_sys, 'eval_len')
                for eval_len in eval_lens:
                    sys_srvs = split_sys(sys_eval_len[eval_len], 'num_servers')
                    key = f'{hw}_{model}_{eval_len}'
                    sys = sys_srvs[num_srvs][0]
                    if key not in prefill_lat:
                        prefill_lat[key] = []
                        generate_lat[key] = []
                        prefill_energy[key] = []
                        generate_energy[key] = []
                        max_ctx_len[key] = []
                    for batch in batch_sizes:
                        if batch not in sys.batch_opt_prefill_lat:
                            prefill_lat[key].append(math.inf)
                            generate_lat[key].append(math.inf)
                            prefill_energy[key].append(math.inf)
                            generate_energy[key].append(math.inf)
                            max_ctx_len[key].append(0.0)
                        else:
                            perf = sys.batch_opt_generate_lat[batch]
                            prefill_lat[key].append(perf.prefill_latency)
                            generate_lat[key].append(perf.generate_latency)
                            prefill_energy[key].append(perf.prefill_core_energy.total)
                            generate_energy[key].append(perf.generate_core_energy.total)
                            kv_cache_mem = sys.total_mem - sys.model.model_size_byte
                            max_ctx_len[key].append(kv_cache_mem / sys.model.kv_cache_size_per_token_byte / batch)
    # plot
    fig, axes = plt.subplots(3, 4, figsize=(10, 5.4))
    fig.subplots_adjust(wspace=0.2, hspace=0.18)
    x = np.arange(len(batch_sizes))
    width = 0.4
    offset = [-0.2, 0.2]
    batch_sizes_str = [str(batch) for batch in batch_sizes]
    x_labels = batch_sizes_str
    fig.suptitle('Overall Comparison', fontsize=16, y=1.08)
    # Row 0: Latency
    for i, model in enumerate(models):
        for j, eval_len in enumerate(eval_lens):
            ax = axes[0, i*2+j]
            all_prefill = dict()
            all_generate = dict()
            for hw in hardwares:
                norm_key = f'{hardwares[0]}_{model}_{eval_len}'
                key = f'{hw}_{model}_{eval_len}'
                all_prefill[hw] = []
                all_generate[hw] = []
                for l, batch in enumerate(batch_sizes):
                    norm = prefill_lat[norm_key][l] + generate_lat[norm_key][l]
                    all_prefill[hw].append(prefill_lat[key][l] / norm)
                    all_generate[hw].append(generate_lat[key][l] / norm)
            for k, hw in enumerate(hardwares):
                # prefill and generate stacked bar
                bottom = [0] * len(batch_sizes)
                for s, stage in enumerate(['prefill', 'generate']):
                    if stage == 'prefill':
                        all_stage = all_prefill
                    else:
                        all_stage = all_generate
                    ax.bar(x + offset[k], all_stage[hw], width, facecolor=colors[k],
                           hatch=hatches[s], edgecolor='k', bottom=bottom,
                           label=hw+' '+stage)
                    for l in range(len(batch_sizes)):
                        bottom[l] += all_stage[hw][l]

            ax.text(0.5, 1.15, model_labels[model],
                ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
            ax.text(0.5, 1.05, f'{eval_len.split(",")[0]} in, {eval_len.split(",")[1][1:]} out',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=10)

            if i == 0 and j == 0:
                ax.set_ylabel('Latency', fontsize=12)
                ax.legend(loc='lower center', 
                          bbox_to_anchor=(2.2, 1.17), 
                          ncol=2,
                          fontsize=12)
    # Row 1 : Lat OPT Inference Energy
    for i, model in enumerate(models):
        for j, eval_len in enumerate(eval_lens):
            ax = axes[1, i*2+j]
            all_prefill = dict()
            all_generate = dict()
            for hw in hardwares:
                norm_key = f'{hardwares[0]}_{model}_{eval_len}'
                key = f'{hw}_{model}_{eval_len}'
                all_prefill[hw] = []
                all_generate[hw] = []
                for l, batch in enumerate(batch_sizes):
                    norm = prefill_energy[norm_key][l] + generate_energy[norm_key][l]
                    all_prefill[hw].append(prefill_energy[key][l] / norm)
                    all_generate[hw].append(generate_energy[key][l] / norm)
            for k, hw in enumerate(hardwares):
                # prefill and generate stacked bar
                bottom = [0] * len(batch_sizes)
                for s, stage in enumerate(['prefill', 'generate']):
                    if stage == 'prefill':
                        all_stage = all_prefill
                    else:
                        all_stage = all_generate
                    ax.bar(x + offset[k], all_stage[hw], width, facecolor=colors[k],
                           hatch=hatches[s], edgecolor='k', bottom=bottom,
                           label=hw+' '+stage)
                    for l in range(len(batch_sizes)):
                        bottom[l] += all_stage[hw][l]

            if i == 0 and j == 0:
                ax.set_ylabel('Energy', fontsize=12)
    # Row 2: Max Context Length
    for i, model in enumerate(models):
        for j, eval_len in enumerate(eval_lens):
            ax = axes[2, i*2+j]
            for hw in hardwares:
                key = f'{hw}_{model}_{eval_len}'
                ax.plot(x, max_ctx_len[key], marker='o', 
                        color=colors[hardwares.index(hw)], label=hw)
            ax.set_yscale('log')
            if i == 0 and j == 0:
                ax.set_ylabel('Max Cxt Len', fontsize=12)

    for ax in axes.flatten():
        ax.set_xticks(x, x_labels, fontsize=9)
        ax.tick_params(axis='y', labelsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Batch Size', fontsize=12)

    fig.savefig('phase_breakdown.pdf', format='pdf', bbox_inches='tight')

def detail_breakdown(phase: str,
                     hardwares: list, models: list, 
                     batch_sizes: list,
                     eval_lens = eval_lens,
                     hatches = hatches, colors = colors,
                     model_srvs = model_srvs,
                     ):      
    # choose the mapping optimized for latency
    compt_lat = dict()
    mem_lat = dict()
    comm_lat = dict()
    compt_energy = dict()
    mem_energy = dict()
    comm_energy = dict()
    util = dict()

    for hw in hardwares:
        for model in models:
            num_srvs = model_srvs[model]
            with open(f'outputs/{hw}/{model}.pkl', 'rb') as f:
                all_sys = pickle.load(f)
                sys_eval_len = split_sys(all_sys, 'eval_len')
                for eval_len in eval_lens:
                    sys_srvs = split_sys(sys_eval_len[eval_len], 'num_servers')
                    key = f'{hw}_{model}_{eval_len}'
                    sys = sys_srvs[num_srvs][0]
                    if key not in compt_lat:
                        compt_lat[key] = []
                        mem_lat[key] = []
                        comm_lat[key] = []
                        compt_energy[key] = []
                        mem_energy[key] = []
                        comm_energy[key] = []
                        util[key] = []
                    for batch in batch_sizes:
                        if batch not in sys.batch_opt_prefill_lat:
                            compt_lat[key].append(math.inf)
                            mem_lat[key].append(math.inf)
                            comm_lat[key].append(math.inf)
                            compt_energy[key].append(math.inf)
                            mem_energy[key].append(math.inf)
                            comm_energy[key].append(math.inf)
                            util[key].append(0.0)
                        else:
                            perf = sys.batch_opt_generate_lat[batch]
                            t_io, t_compute, t_mem = get_latency_breakdown(perf, phase)
                            compt_lat[key].append(t_compute)
                            mem_lat[key].append(t_mem)
                            comm_lat[key].append(t_io)
                            if phase == 'prefill':
                                energy = perf.prefill_core_energy
                                util[key].append(perf.prefill_utilization)
                            else:
                                energy = perf.generate_core_energy
                                util[key].append(perf.generate_utilization)
                            compt_energy[key].append(energy.fma)
                            mem_energy[key].append(energy.mem)
                            comm_energy[key].append(energy.comm)

    # plot
    fig, axes = plt.subplots(3, 4, figsize=(10, 5.4))
    fig.subplots_adjust(wspace=0.2, hspace=0.18)
    x = np.arange(len(batch_sizes))
    width = 0.4
    offset = [-0.2, 0.2]
    batch_sizes_str = [str(batch) for batch in batch_sizes]
    x_labels = batch_sizes_str
    if phase == 'prefill':
        fig.suptitle('Prefill', fontsize=16, y=1.12)
    else:
        fig.suptitle('Decode', fontsize=16, y=1.12)
    # Row 0: Latency
    for i, model in enumerate(models):
        for j, eval_len in enumerate(eval_lens):
            ax = axes[0, i*2+j]
            all_compt = dict()
            all_mem = dict()
            all_comm = dict()
            for hw in hardwares:
                norm_key = f'{hardwares[0]}_{model}_{eval_len}'
                key = f'{hw}_{model}_{eval_len}'
                all_compt[hw] = []
                all_mem[hw] = []
                all_comm[hw] = []
                for l, batch in enumerate(batch_sizes):
                    norm = compt_lat[norm_key][l] + mem_lat[norm_key][l] + comm_lat[norm_key][l]
                    all_compt[hw].append(compt_lat[key][l] / norm)
                    all_mem[hw].append(mem_lat[key][l] / norm)
                    all_comm[hw].append(comm_lat[key][l] / norm)
            for k, hw in enumerate(hardwares):
                # prefill and generate stacked bar
                bottom = [0] * len(batch_sizes)
                for s, stage in enumerate(['compute', 'mem', 'comm']):
                    if stage == 'compute':
                        all_stage = all_compt
                    elif stage == 'mem':
                        all_stage = all_mem
                    else:
                        all_stage = all_comm
                    ax.bar(x + offset[k], all_stage[hw], width, facecolor=colors[k],
                           hatch=hatches[s], edgecolor='k', bottom=bottom,
                           label=hw+' '+stage)
                    for l in range(len(batch_sizes)):
                        bottom[l] += all_stage[hw][l]

            ax.text(0.5, 1.15, model_labels[model],
                ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
            ax.text(0.5, 1.05, f'{eval_len.split(",")[0]} in, {eval_len.split(",")[1][1:]} out',
                ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
            ax.tick_params(axis='y', labelsize=9)

            if i == 0 and j == 0:
                ax.set_ylabel('Latency', fontsize=12)
                ax.legend(loc='lower center', 
                          bbox_to_anchor=(2.2, 1.17), 
                          ncol=2,
                          fontsize=12)
    # Row 1 : Lat OPT Inference Energy
    for i, model in enumerate(models):
        for j, eval_len in enumerate(eval_lens):
            ax = axes[1, i*2+j]
            all_compt = dict()
            all_mem = dict()
            all_comm = dict()
            for hw in hardwares:
                norm_key = f'{hardwares[0]}_{model}_{eval_len}'
                key = f'{hw}_{model}_{eval_len}'
                all_compt[hw] = []
                all_mem[hw] = []
                all_comm[hw] = []
                for l, batch in enumerate(batch_sizes):
                    norm = compt_energy[norm_key][l] + mem_energy[norm_key][l] + comm_energy[norm_key][l]
                    all_compt[hw].append(compt_energy[key][l] / norm)
                    all_mem[hw].append(mem_energy[key][l] / norm)
                    all_comm[hw].append(comm_energy[key][l] / norm)
            for k, hw in enumerate(hardwares):
                # prefill and generate stacked bar
                bottom = [0] * len(batch_sizes)
                for s, stage in enumerate(['compute', 'mem', 'comm']):
                    if stage == 'compute':
                        all_stage = all_compt
                    elif stage == 'mem':
                        all_stage = all_mem
                    else:
                        all_stage = all_comm
                    ax.bar(x + offset[k], all_stage[hw], width, facecolor=colors[k],
                           hatch=hatches[s], edgecolor='k', bottom=bottom,
                           label=hw+' '+stage)
                    for l in range(len(batch_sizes)):
                        bottom[l] += all_stage[hw][l]

            ax.tick_params(axis='y', labelsize=9)
            if i == 0 and j == 0:
                ax.set_ylabel('Energy', fontsize=12)
    # Row 2: Utilization
    for i, model in enumerate(models):
        for j, eval_len in enumerate(eval_lens):
            ax = axes[2, i*2+j]
            for hw in hardwares:
                key = f'{hw}_{model}_{eval_len}'
                ax.plot(x, util[key], marker='o', 
                        color=colors[hardwares.index(hw)], label=hw)
            # ax.set_yscale('log')
            ax.set_ylim(0.0, 1)
            # set y-axis to percentage
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            if i == 0 and j == 0:
                ax.set_ylabel('Utilization', fontsize=12)
            ax.tick_params(axis='y', labelsize=6)

    for ax in axes.flatten():
        ax.set_xticks(x, x_labels, fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Batch Size', fontsize=12)

    fig.savefig(f'{phase}_breakdown', format='pdf', bbox_inches='tight')
      

# %%
hardwares = ['b200', 'cc_3d_baseline']
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
models = ['gpt3', 'mtnlg']
# models = ['llama2', 'gpt3']
model_srvs = {'llama2': 1, 'gpt3': 2, 'mtnlg': 4, 'palm': 2}
# eval_lens = ['1024, 256', '256, 1024']
eval_lens = ['4096, 16384', '16384, 4096']
phase_breakdown(hardwares, models, batch_sizes, eval_lens=eval_lens, model_srvs=model_srvs)
# %%
detail_breakdown('prefill', hardwares, models, batch_sizes, eval_lens=eval_lens, model_srvs=model_srvs)
# %%
detail_breakdown('generate', hardwares, models, batch_sizes, eval_lens=eval_lens, model_srvs=model_srvs)
# %%
