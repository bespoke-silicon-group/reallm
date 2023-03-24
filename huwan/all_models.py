# %%
# read all models
real_models = ['gpt2', 'tnlg', 'gpt3', 'mtnlg']
scale_models = ['gpt3', 'gpt_340B', 'gpt_700B', 'gpt_1360B']
ctx_models = ['gpt3', 'gpt3_ctx_8K', 'gpt3_ctx_32K', 'gpt3_ctx_128K']

models_label = {'gpt2':  'GPT2-1.4B',
                'tnlg':  'Turing NLG-17B',
                'gpt3':  'GPT3-175B',
                'mtnlg': 'MT-NLG-540B',
                'gpt3_ctx_8K': 'GPT3-8K-CTX',
                'gpt3_ctx_32K': 'GPT3-32K-CTX',
                'gpt3_ctx_128K': 'GPT3-128K-CTX',
                'gpt_340B':  'GPT-340B',
                'gpt_700B':  'GPT-700B',
                'gpt_1360B': 'GPT-1360B',
                }

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

def pareto_front_filter_2d(df: pd.DataFrame, x_label: str, y_label: str, x_lower_is_better: bool=True, y_lower_is_better: bool=True) -> pd.DataFrame:
    # 2D pareto front is quite simple. We first sort by X, and break ties of X
    # by then sorting on Y. The first row is always on the pareto front when
    # sorted this way. Then we traverse the rows, thus X is getting worse and
    # worse. We compare Y to the best Y we've seen and only add the row to the
    # pareto front if Y is better.
    result = pd.DataFrame(columns=df.columns)
    best_y_seen = None
    for idx, row in df.sort_values(by=[x_label, y_label], ascending=[x_lower_is_better, y_lower_is_better]).iterrows():
        if (best_y_seen is None) or (y_lower_is_better and row[y_label] < best_y_seen) or (not y_lower_is_better and row[y_label] > best_y_seen):
            best_y_seen = row[y_label]
            result.loc[idx] = row
    return result

hw_csv = "../asic_cloud_sim_v2/exploration.csv"
hw = pd.read_csv(hw_csv)

dfs = {}
for model in real_models:
    all_csv = "../app_perf_sim_v2/"+model+"_all.csv"
    pd.set_option("display.max.columns", None)
    df = pd.read_csv(all_csv)
    df = pd.concat([hw.loc[map(lambda x: x-1, df["srv_id"].to_numpy().tolist())].reset_index(drop=True), df], axis=1)
    dfs[model] = df

# %%
# add some data
# for model in dfs:
#     df = dfs[model]
#     df['W/tput'] = df.apply(lambda row: (row['num_srvs']*row['chips_per_srv']*row['power_per_chip']*row['utilization']) / row['tput'], axis=1)
#     # add $/tput
#     df['$/tput'] = df.apply(lambda row: row['all_srv_cost'] / row['tput'], axis=1)
#     # add all_tco/sec
#     life_time_years = 1.5
#     total_sec = life_time_years * 365 * 24 * 3600
#     df['all_tco/sec'] = df.apply(lambda row: row['all_tco'] / total_sec, axis=1)
#     # add tco/token
#     df['tco/token'] = df.apply(lambda row: row['all_tco/sec'] / row['tput'], axis=1)
#     df['tco/1ktoken'] = df.apply(lambda row: row['all_tco/sec'] / row['tput'] * 1000, axis=1)
#     # add real_w
#     df['real_w'] = df.apply(lambda row: (row['num_srvs']*row['chips_per_srv']*row['power_per_chip']*row['utilization']), axis=1)
#     # add tops/mb
#     df['tops/mb'] = df.apply(lambda row: row['tops_per_chip'] / row['sram_per_chip'], axis=1)
#     # add all_area
#     df['all_area'] = df.apply(lambda row: row['num_srvs']*row['chips_per_srv']*row[' [5]die_area'], axis=1)
#     # add latency in ms
#     df['latency_ms'] = df.apply(lambda row: row['latency']/1e3, axis=1)
#     # add 1/tput
#     df['1/tput'] = df.apply(lambda row: 1/row['tput'], axis=1)

# %%
# plot all data points

for model in real_models:
    df = dfs[model]
    print('========================', model, '======================')
    print("       Exploration Space:   ", df.shape[0])
    print(" ")
    print("Number of Server Designs:   ", len(pd.unique(df["srv_id"])))
    print("        Chips per Server:   ", f'{df["chips_per_srv"].min()}', '~', f'{df["chips_per_srv"].max()}')
    print("         Chip Size (mm2):   ", f'{df[" [5]die_area"].min() :.2f}', '~', f'{df[" [5]die_area"].max():.3f}')
    print("          Chip SRAM (MB):   ", f'{df["sram_per_chip"].min():.3f}', '~', f'{df["sram_per_chip"].max():.3f}')
    print("               Chip TOPS:   ", f'{df["tops_per_chip"].min():.3f}', '~', f'{df["tops_per_chip"].max():.3f}')
    print(" ")
    print("         Pipeline Stages:   ", f'{df["p"].min()}', '~', f'{df["p"].max()}')
    print(" Tensor Parallelism Size:   ", f'{df["t"].min()}', '~', f'{df["t"].max()}')
    print(" ")
    # print("           Hardware Cost:   ", f'{df["all_srv_cost"].min():,.0f}', '~', f'{df["all_srv_cost"].max():,.0f} $')
    print("         Total Power (W):   ", f'{df["real_w"].min():,.2f}', '~', f'{df["real_w"].max():,.0f}')
    print("                 TCO ($):   ", f'{df["all_tco"].min():,.0f}', '~', f'{df["all_tco"].max():,.0f}')
    print(" ")

# Optimal designs
for model in real_models:
    df = dfs[model]
    tco_optimal = df.loc[pd.to_numeric(df['tco/token']).idxmin()]
    # tput_optimal = df.loc[pd.to_numeric(df['tput']).idxmax()]
    energy_optimal = df.loc[pd.to_numeric(df['W/tput']).idxmin()] 
    latency_optimal = df.loc[pd.to_numeric(df['latency']).idxmin()]
    features = {'chip_id': 'Chip ID', ' [5]die_area': 'Die Size (mm2)', 
                'sram_per_chip': "SRAM per Chip (MB)", 'tops_per_chip': 'TOPS per Chip', 
                'srv_id': 'Srv ID', 'chips_per_srv': 'Chips per Srv', 
                'num_srvs': 'Num of Srvs', 'all_srv_cost': 'All CapEx ($)',
                't': 'Tensor Para Size', 'p': 'Pipeline Para Size', 
                'batch': 'Batch Size', 'micro_batch': 'Micro-Batch Size',
                'latency_ms': 'Latency (ms)', 'W/tput': 'Joules/1K Tokens', 'tco/1ktoken': 'Cents/1K Tokens', 
    }
    print(models_label[model], ',   Latency Optimal,     Energy Optimal,      TCO Optimal')
    for f in features:
        if f == 'W/tput' or f == 'tco/1ktoken':
            print(f'{features[f]}, {latency_optimal[f]*1e3}, {energy_optimal[f]*1e3}, {tco_optimal[f]*1e3}')
        else:
            print(f'{features[f]}, {latency_optimal[f]}, {energy_optimal[f]}, {tco_optimal[f]}')

    print(" ")
# %%
# Latency vs TCO

markers = ['<', 'p', 'D', 'D', 'o', 's', '^']
fig, axes = plt.subplots(1, 1, figsize=(8,5), dpi=200)
ax = axes
for model in real_models:
    df = dfs[model]
    pareto_n1 = pareto_front_filter_2d(df, "all_tco", "latency_ms", )
    ax.plot(pareto_n1["all_tco"], pareto_n1["latency_ms"], linestyle='--', 
            markeredgecolor='black', markeredgewidth=0.5,
            marker=markers.pop(), markersize=10, label=models_label[model])

ax.set_xlabel("TCO Budget ($)")
ax.set_ylabel("Latency (ms)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(which='both', c='whitesmoke')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=12)

# %%
# Throughput vs TCO

markers = ['<', 'p', 'D', 'D', 'o', 's', '^']
fig, axes = plt.subplots(1, 1, figsize=(8,5), dpi=200)
ax = axes

# scale_out_label = False
# for i in range(20):
#     x_start_power = (i/2)-1
#     x_vals = []
#     y_vals = []
#     y_start_power = 2
#     for i in [1, 10, 100, 1000, 10000]:
#         x_vals.append(10**x_start_power*i)
#         y_vals.append(10**y_start_power*i)
#     if not scale_out_label:
#         ax.plot(x_vals, y_vals, c='red', linestyle='--', linewidth=1, alpha=0.3, label='Scale-Out',)
#         scale_out_label = True
#     else:
#         ax.plot(x_vals, y_vals, c='red', linestyle='--', linewidth=1, alpha=0.3)

for model in ctx_models:
    df = dfs[model]
    pareto_n1 = pareto_front_filter_2d(df, "all_tco", "1/tput", )
    ax.plot(pareto_n1["all_tco"], pareto_n1["tput"], linestyle='None', 
            markeredgecolor='black', markeredgewidth=0.5,
            marker=markers.pop(), markersize=10, label=models_label[model])

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(6e4, 5e7)
ax.set_ylim(200, 1e5)

ax.set_xlabel("TCO Budget ($)")
ax.set_ylabel("Throughput (tokens/sec)")
ax.grid(which='both', c='whitesmoke')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right", fontsize=12)
# %%
# all pareto 
markers = ['<', 'p', 'D', 'D', 'o', 's', '^']
fig, axes = plt.subplots(1, 1, figsize=(8,6), dpi=200)
ax = axes
latency = 10000
top_point = None
for model in ctx_models:
    df = dfs[model]
    pareto_all = pareto_front_filter_2d(df, "tco/1ktoken", "latency_ms", )
    ax.plot(pareto_all["tco/1ktoken"], pareto_all["latency_ms"], 
            linestyle='None',
            markeredgecolor='black', markeredgewidth=0.5,
            marker=markers.pop(), markersize=10, label=models_label[model])

# color='salmon'
# for model in ctx_models:
#     df = dfs[model]
#     pareto_all = pareto_front_filter_2d(df, "tco/1ktoken", "latency_ms", )
#     latency_optimal = pareto_all.loc[pd.to_numeric(pareto_all['latency_ms']).idxmin()]

#     if top_point == None:
#         top_point = (latency_optimal["tco/1ktoken"], latency_optimal["latency_ms"]) 
#         last_y = latency_optimal["latency_ms"] 
#     else:
#         ax.plot([latency_optimal["tco/1ktoken"], top_point[0]], 
#                 [latency_optimal["latency_ms"], latency_optimal["latency_ms"]],
#                 linestyle=':', color=color)

#         top = (top_point[0], last_y)
#         bot = (top_point[0], latency_optimal["latency_ms"])
#         ax.annotate("", xy=bot, xytext=top,
#                     arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
#         ax.text(bot[0], math.sqrt(top[1]*bot[1]),
#                 f'{(top[1]/bot[1]):.1f}x',
#                 fontsize='large',
#                 ha='center', va="center",
#                 bbox=dict(boxstyle="round",
#                            edgecolor='black',
#                            facecolor=color)
#                 )
#         last_y = latency_optimal["latency_ms"] 

# color='deepskyblue'
# right_point = None
# for model in ctx_models[::-1]:
#     df = dfs[model]
#     pareto_all = pareto_front_filter_2d(df, "tco/1ktoken", "latency_ms", )
#     tco_optimal = pareto_all.loc[pd.to_numeric(pareto_all['tco/1ktoken']).idxmin()]
#     if right_point == None:
#         right_point = (tco_optimal["tco/1ktoken"], tco_optimal["latency_ms"]) 
#         last_x = tco_optimal["tco/1ktoken"]
#     else:
#         ax.plot([tco_optimal["tco/1ktoken"], tco_optimal["tco/1ktoken"]], 
#                 [tco_optimal["latency_ms"], right_point[1]],
#                 linestyle=':', color=color)

#         left  = (last_x, right_point[1])
#         right = (tco_optimal["tco/1ktoken"], right_point[1])
#         ax.annotate("", xy=left, xytext=right,
#                     arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
#         ax.text(math.sqrt(left[0]*right[0]), left[1], 
#                 f'{(left[0]/right[0]):.1f}x',
#                 fontsize='large',
#                 ha='center', va="center",
#                 bbox=dict(boxstyle="round",
#                            edgecolor='black',
#                            facecolor=color)
#                 )
#         last_x = tco_optimal["tco/1ktoken"]


ax.set_xlabel("TCO per 1K tokens ($)", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(which='both', c='whitesmoke')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="upper right", fontsize=12)
# %%
df = dfs['gpt3']
tput_optimal = df.loc[pd.to_numeric(df['tput']).idxmax()]
token_price_optimal = df.loc[pd.to_numeric(df['tco/token']).idxmin()]
print(tput_optimal)
print(token_price_optimal)
# %%
# Throughput vs Energy

fig, axes = plt.subplots(1, 1, figsize=(8,5), dpi=200)
ax = axes
markers = ['<', 'p', 'D', 'D', 'o', 's', '^']

# scale_out_label = False
# for i in range(20):
#     x_start_power = (i/2)-1
#     x_vals = []
#     y_vals = []
#     y_start_power = 2
#     for i in [1, 10, 100, 1000, 10000]:
#         x_vals.append(10**x_start_power*i)
#         y_vals.append(10**y_start_power*i)
#     if not scale_out_label:
#         ax.plot(x_vals, y_vals, c='red', linestyle='--', linewidth=1, alpha=0.3, label='Scale-Out',)
#         scale_out_label = True
#     else:
#         ax.plot(x_vals, y_vals, c='red', linestyle='--', linewidth=1, alpha=0.3)

for model in scale_models:
    df = dfs[model]
    pareto_n1 = pareto_front_filter_2d(df, "1/tput", "real_w")
    ax.plot(pareto_n1["real_w"], pareto_n1["tput"], linestyle='None', 
            markeredgecolor='black', markeredgewidth=0.5,
            marker=markers.pop(), markersize=10, label=models_label[model])
    # tco_optimal = pareto_n1.loc[pd.to_numeric(pareto_n1['all_tco']).idxmin()]

ax.set_xscale("log")
ax.set_yscale("log")
# ax.set_xlim(1e3, 2.5e6)
# ax.set_ylim(1e2, 7e5)
# print(x_left)
# x_left_power = math.log10(x_left)
# x_right_power = math.log10(x_right)
# y_bot_power = math.log10(y_bot)
# y_top_power = math.log10(y_top)

ax.set_xlabel("Power Budget (W)")
ax.set_ylabel("Throughput (tokens/sec)")
ax.grid(which='both', c='whitesmoke')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc="lower right", fontsize=12)
