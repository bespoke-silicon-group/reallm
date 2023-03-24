# %%
model = "gpt3"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

hw_csv = "../asic_cloud_sim_v2/exploration.csv"
all_csv = "../app_perf_sim_v2/"+model+"_all.csv"

pd.set_option("display.max.columns", None)

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

df = pd.read_csv(all_csv)
print(df.columns)

#%%
# concat hw infor into the main df
hw = pd.read_csv(hw_csv)
df = pd.concat([hw.loc[map(lambda x: x-1, df["srv_id"].to_numpy().tolist())].reset_index(drop=True), df], axis=1)
#df.head()
# %%
# # add some data
# # add W/tput
# df['W/tput'] = df.apply(lambda row: (row['num_srvs']*row['chips_per_srv']*row['power_per_chip']*row['utilization']) / row['tput'], axis=1)
# # add $/tput
# df['$/tput'] = df.apply(lambda row: row['all_srv_cost'] / row['tput'], axis=1)
# # add all_tco/sec
# life_time_years = 1.5
# total_sec = life_time_years * 365 * 24 * 3600
# df['all_tco/sec'] = df.apply(lambda row: row['all_tco'] / total_sec, axis=1)
# # add tco/token
# df['tco/token'] = df.apply(lambda row: row['all_tco/sec'] / row['tput'], axis=1)
# df['tco/1ktoken'] = df.apply(lambda row: row['all_tco/sec'] / row['tput'] * 1000, axis=1)
# # add real_w
# df['real_w'] = df.apply(lambda row: (row['num_srvs']*row['chips_per_srv']*row['power_per_chip']*row['utilization']), axis=1)
# # add tops/mb
# df['tops/mb'] = df.apply(lambda row: row['tops_per_chip'] / row['sram_per_chip'], axis=1)
# # add all_area
# df['all_area'] = df.apply(lambda row: row['num_srvs']*row['chips_per_srv']*row[' [5]die_area'], axis=1)
# # add latency in ms
# df['latency_ms'] = df.apply(lambda row: row['latency']/1e3, axis=1)
# # add 1/tput
# df['1/tput'] = df.apply(lambda row: 1/row['tput'], axis=1)

# print(df.columns)
#     >  ['# [1]tech_node', ' [2]sram_per_asic', ' [3]tops_per_asic',
#     >   ' [4]watts_per_asic', ' [5]die_area', ' [6]die_cost',
#     >   ' [7]asics_per_server', ' [8]sram_per_server', ' [9]server_power',
#     >   ' [10]tops_per_server', ' [11]server_cost', ' [12]life_time_tco',
#     >   ' [13]DCAmortization', ' [14]DCInterest', ' [15]DCOpex',
#     >   ' [16]SrvAmortization', ' [17]SrvInterest', ' [18]SrvOpex',
#     >   ' [19]SrvPower', ' [20]PUEOverhead', ' [21]cost_per_tops',
#     >   ' [22]watts_per_tops', ' [23]tco_per_tops',
#     >   ' [24]max_die_power_per_server', ' [25]die_yield', ' ', 'srv_id',
#     >   'chip_tops', 'chip_sram', 'power_per_chip', 'chips_per_srv', 'srv_tco',
#     >   'num_srvs', 't', 'p', 'batch', 'latency', 'compute_latency',
#     >   'communicate_latency', 'tput', 'latency_best', 'tput_best', 'all_tco',
#     >   'all_srv_cost', 'utilization', 'all_tco/tops', 'all_tco/tput', 'W/tput',
#     >   '$/tput']

# df.head()

# %%
# plot all data points

plt.figure(figsize=(10,8), dpi=300)
plt.yscale("log")
plt.xscale("log")
plt.xlabel("TCO/Tokens/sec")
plt.ylabel("Latency (us)")
for batch_size in pd.unique(df["batch"]):
    series = df[df["batch"] == batch_size]
    plt.scatter(x=series["all_tco/tput"], y=series["latency"], s=0.04, marker='.', label=batch_size)

plt.legend(title="Batch Size", loc="upper right", markerscale=20)

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

# %%
tco_optimal = df.loc[pd.to_numeric(df['tco/token']).idxmin()]
tput_optimal = df.loc[pd.to_numeric(df['tput']).idxmax()]
latency_optimal = df.loc[pd.to_numeric(df['latency']).idxmin()]
print(latency_optimal)
print(tput_optimal)
print(tco_optimal)

# %%
# Plot GPT3 Pareto Curve

feature_x = "tco/token"
feature_y = "latency"

pareto_n1    = pareto_front_filter_2d( df[df["batch"] ==    1], feature_x, feature_y, )
pareto_n2    = pareto_front_filter_2d( df[df["batch"] ==    2], feature_x, feature_y, )
pareto_n4    = pareto_front_filter_2d( df[df["batch"] ==    4], feature_x, feature_y, )
pareto_n8    = pareto_front_filter_2d( df[df["batch"] ==    8], feature_x, feature_y, )
pareto_n16   = pareto_front_filter_2d( df[df["batch"] ==   16], feature_x, feature_y, )
pareto_n32   = pareto_front_filter_2d( df[df["batch"] ==   32], feature_x, feature_y, )
pareto_n64   = pareto_front_filter_2d( df[df["batch"] ==   64], feature_x, feature_y, )
pareto_n128  = pareto_front_filter_2d( df[df["batch"] ==  128], feature_x, feature_y, )
pareto_n256  = pareto_front_filter_2d( df[df["batch"] ==  256], feature_x, feature_y, )
pareto_n512  = pareto_front_filter_2d( df[df["batch"] ==  512], feature_x, feature_y, )
pareto_n1024 = pareto_front_filter_2d( df[df["batch"] == 1024], feature_x, feature_y, )

all_batch = {1: pareto_n1, 2: pareto_n2, 4: pareto_n4, 8: pareto_n8, 16: pareto_n16, 
             32: pareto_n32, 64: pareto_n64, 128: pareto_n128, 256: pareto_n256,
             512: pareto_n512, 1024: pareto_n1024}

fig, axes = plt.subplots(1, 1, figsize=(8,6), dpi=200)
ax = axes

# feature_x = "all_tco/tput"
# feature_y = "tput"
for batch in all_batch:
    tmp_df = all_batch[batch].sort_values(feature_x, ascending=True)
    ax.plot(tmp_df[feature_x], tmp_df[feature_y], 
            linestyle='--', marker='o', markersize=5, label=batch)

    for index, data in tmp_df.iterrows():
        u = float(data["utilization"])
    ax.annotate(f'{u:.1%}', (data[feature_x], data[feature_y]))

ax.set_xscale("log")
#ax.set_xlim(right=56)
#ax.set_xlabel("TCO/Tokens/sec")
ax.set_xlabel(feature_x)

ax.set_yscale("log")
# ax.set_ylabel("Latency (us)")
ax.set_ylabel(feature_y)

ax.set_title(model+" Pareto Curve")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, title="Batch Size", ncol=3, loc="upper left")


# %%
all_batch[256].sort_values("tco/token", ascending=True)
# %%
all_batch[1024].sort_values("tco/token", ascending=True)

# %%
# Generate $/tput vs W/tput

n1  = df[df["batch"] ==  1]
n8  = df[df["batch"] ==  8]
n64 = df[df["batch"] == 64]

pareto_n1  = pareto_front_filter_2d( n1, "W/tput", "$/tput" )
pareto_n8  = pareto_front_filter_2d( n8, "W/tput", "$/tput" )
pareto_n64 = pareto_front_filter_2d( n64, "W/tput", "$/tput" )

tco_optimal_n1 = pareto_n1.loc[pd.to_numeric(pareto_n1['all_tco/tput']).idxmin()]
tco_optimal_n8 = pareto_n8.loc[pd.to_numeric(pareto_n8['all_tco/tput']).idxmin()]
tco_optimal_n64 = pareto_n64.loc[pd.to_numeric(pareto_n64['all_tco/tput']).idxmin()]

fig, axes = plt.subplots(1, 1, figsize=(8,6), dpi=200)
ax = axes

#ax.scatter(n1["W/tput"], n1["$/tput"], s=1.0, c="blue")
#ax.scatter(n8["W/tput"], n8["$/tput"], s=1.0, c="red")
#ax.scatter(n64["W/tput"], n64["$/tput"], s=1.0, c="green")
ax.plot(pareto_n1["W/tput"], pareto_n1["$/tput"], linestyle='--', marker='o', label=1, color="blue")
ax.plot(pareto_n8["W/tput"], pareto_n8["$/tput"], linestyle='--', marker='o', label=8, color="red")
ax.plot(pareto_n64["W/tput"], pareto_n64["$/tput"], linestyle='--', marker='o', label=64, color="green")

ax.plot(tco_optimal_n1["W/tput"], tco_optimal_n1["$/tput"], marker='*', color="blue", markersize=15)
ax.plot(tco_optimal_n8["W/tput"], tco_optimal_n8["$/tput"], marker='*', color="red", markersize=15)
ax.plot(tco_optimal_n64["W/tput"], tco_optimal_n64["$/tput"], marker='*', color="green", markersize=15)

ax.set_xscale("log")
ax.set_xlabel("W/tput")
#ax.set_xlim(0.5, 1.5)

ax.set_yscale("log")
ax.set_ylabel("$/tput")
#ax.set_ylim(2, 16)

ax.set_title(model)
ax.legend(title="Batch Size")

pareto_n64.sort_values("$/tput", ascending=True)

# %%
# all pareto 
pareto_all = pareto_front_filter_2d(df, "tco/token", "latency", )
pareto_all.sort_values("latency", ascending=True)
latency_optimal = pareto_all.loc[pd.to_numeric(pareto_all['latency']).idxmax()]

tpu_price = 3.22 # per chip-hour, from google cloud
tpu_tco = 0.12 # per chip-hour, from our TCO model. Did not consider HBM, not accurate.
# from figure 1, decoding latency 
# tpu_540B={"latency":[32, 45, 180], "chip_ms_per_token":[64,16*1.41, 8*1.41], "num_chips":[128, 64, 64], "batch":[64, 128, 1024]}
# from figure 1 and table 2
tpu_540B={
          "num_chips":         [        64,        64,       64,        64], 
          "batch":             [        64,       128,      512,      1024],
          "latency":           [   1820/64,        45,  6000/64,       180], 
          "tput":              [64*64/1.82, 128/0.045, 64*512/6, 1024/0.18], 
          # "chip_ms_per_token": [64,  16*1.41, 8*1.41], 
          }
tpu={"tco/token":[], "latency":[]}
for i in range(3):
    tpu["latency"].append(tpu_540B["latency"][i] /1.7) # GPT3 estimates
    tpu["tco/token"].append(tpu_price*tpu_540B["num_chips"][i]/3600/tpu_540B["tput"][i]*1000 / 2.2) # GPT3 estimates

gpu_price = 3.05 # per GPU per hour, from google cloud
gpu_tco = 1.67 # per GPU per hour, from our TCO Model
gpu_tput = [18]
gpu_latency = [620]
gpu_tflops = [70]
gpu_num = [16]
gpu={"tco/token":[], "latency":[]}
for i in range(len(gpu_tput)):
    gpu["latency"].append(gpu_latency[i])
    gpu["tco/token"].append(gpu_tco/3600/gpu_tput[i] * 1000)

fig, axes = plt.subplots(1, 1, figsize=(8,6), dpi=200)
ax = axes
ax.plot(pareto_all["tco/token"]*1000, pareto_all["latency"]/1e3, 
        linestyle='--', marker='^', 
        markeredgecolor='black', markerfacecolor='tab:blue', c='tab:blue',
        markeredgewidth=0.5, markersize=10,
        label="Chiplet Cloud")
ax.plot(tpu["tco/token"], tpu["latency"],
        linestyle='--', marker='s', 
        markeredgecolor='black', markerfacecolor='tab:green', c='tab:green',
        markeredgewidth=0.5, markersize=10,
        label="TPU")
ax.plot(gpu["tco/token"], gpu["latency"],
        linestyle='--', marker='o', 
        markeredgecolor='black', markerfacecolor='tab:red', c='tab:red',
        markeredgewidth=0.5, markersize=10,
        label="GPU")

cc_opt  = (latency_optimal["tco/token"]*1000, latency_optimal["latency"]/1000)
tpu_opt = (tpu["tco/token"][2], tpu["latency"][2])
gpu_opt = (gpu["tco/token"][0], gpu["latency"][0])

# TCO improvement
color = 'salmon'
ax.axvline(x=cc_opt[0], linestyle=':', c=color, linewidth=1.5)
for right in [tpu_opt, gpu_opt]:
    left = (cc_opt[0], right[1])
    ax.annotate("", xy=left, xytext=right,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
    ax.text(math.sqrt(right[0]*left[0]), right[1], 
            f'{(right[0]/left[0]):.1f}x',
            fontsize='large',
            ha='center', va="center",
            bbox=dict(boxstyle="round",
                       edgecolor='black',
                       facecolor=color
                       )
            )

# Latency improvement
color = 'deepskyblue'
ax.axhline(y=cc_opt[1], linestyle=':', c=color, linewidth=1.5)
for top in [tpu_opt, gpu_opt]:
    bot = (top[0], cc_opt[1])
    ax.annotate("", xy=bot, xytext=top,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
    ax.text(top[0], math.sqrt(top[1]*bot[1]),
            f'{(top[1]/bot[1]):.1f}x',
            fontsize='large',
            ha='center', va="center",
            bbox=dict(boxstyle="round",
                       edgecolor='black',
                       facecolor=color
                       )
            )

ax.set_xlabel("TCO per 1K tokens ($)", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(which='both', c='whitesmoke')

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="lower right", fontsize=12)


# %%
# generate all pareto 
# pareto_all = pareto_front_filter_2d(df[(df["batch"]==8) & (df["p"]==144)], "tco/token", "latency", )
pareto_all = pareto_front_filter_2d(df[(df["batch"]==128)], "tco/token", "latency", )
pareto_all.sort_values("latency", ascending=True)
pareto_all.head()

# %%
# How different factors affect the pareto points?

fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9, 7), dpi=300)
plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)

markers = ['1', '+', '2' ,'x', '3', '|', '4']
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive','tab:cyan']

num_bins=5
marker_size = 30

ax = axes[0][0]
feature = "all_tco"
_, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
for i in range(num_bins):
    left = bin_edges[i]
    right = bin_edges[i+1]
    series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
    ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
               s=marker_size,
               marker=markers[i%len(markers)],
               label=f"[{left/1000:.0f}K, {right/1000:.0f}K)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title="TCO Budget($)", loc="upper right", markerscale=1.3, framealpha=1)

# ax = axes[0][1]
# feature = "real_w"
# _, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
# for i in range(num_bins):
#     left = bin_edges[i]
#     right = bin_edges[i+1]
#     series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
#     if model == 'gpt3':
#         ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                    s=marker_size,
#                    marker=markers[i%len(markers)],
#                    label=f"[{left:.0f}, {right:.0f})")
#     else:
#         ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                    s=marker_size,
#                    marker=markers[i%len(markers)],
#                    label=f"[{left/1000:.0f}K, {right/1000:.0f}K)")
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], title="Power Budget(W)", loc="upper right", markerscale=1.3, framealpha=1)

ax = axes[0][1]
i = 0
p_df = pareto_all.sort_values("p", ascending=True)
for p in pd.unique(p_df["p"]):
    series = p_df[p_df["p"] == p]
    if not series.empty:
        ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
                   s=marker_size,
                   marker=markers[i%len(markers)],
                   label=p)
    i += 1
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], ncol=2, title="Pipeline Stages", loc="upper right", markerscale=1.3, framealpha=1)

# ax = axes[1][1]
# i = 0
# p_df = pareto_all.sort_values("t", ascending=True)
# for batch in pd.unique(p_df["t"]):
#     series = p_df[p_df["t"] == batch]
#     if not series.empty:
#         ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                    s=marker_size,
#                    marker=markers[i%len(markers)],
#                    label=batch)
#     i += 1
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], ncol=4, title="T", loc="upper right", markerscale=1.3, framealpha=1)

# ax = axes[2][0]
# feature = "chips_per_srv"
# i = 0
# p_df = pareto_all.sort_values(feature, ascending=True)
# for p in pd.unique(p_df[feature]):
#     series = p_df[p_df[feature] == p]
#     if not series.empty:
#         ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                    s=marker_size,
#                    # facecolors='none', 
#                    # edgecolor=colors[i%len(colors)], 
#                    marker=markers[i%len(markers)],
#                    label=p)
#     i += 1
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], ncol=2, title="Chips per Server", loc="upper right", markerscale=1.3, framealpha=1)

ax = axes[1][0]
feature = "peak_tops"
_, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
for i in range(num_bins):
    left = bin_edges[i]
    right = bin_edges[i+1]
    series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
    ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
               s=marker_size,
               marker=markers[i%len(markers)],
               label=f"[{left:.0f}, {right:.0f})")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title="Total TOPS", loc="upper right", markerscale=1.3, framealpha=1)

ax = axes[1][1]
feature = "all_area"
_, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
for i in range(num_bins):
    left = bin_edges[i]
    right = bin_edges[i+1]
    if i == num_bins-1:
        series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] <= right)]
    else:
        series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
    ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
               s=marker_size,
               marker=markers[i%len(markers)],
               label=f"[{left/1000:.0f}K, {right/1000:.0f}K)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], ncol=1, title="Total Area (mm2)", loc="upper right", markerscale=1.3, framealpha=1)

ax = axes[2][0]
feature = "num_srvs"
i = 0
p_df = pareto_all.sort_values(feature, ascending=True)
for p in pd.unique(p_df[feature]):
    series = p_df[p_df[feature] == p]
    if not series.empty:
        ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
                   s=marker_size,
                   marker=markers[i%len(markers)],
                   label=p)
    i += 1
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], ncol=3, title="Num of Servers", loc="upper right", markerscale=1.3, framealpha=1)


ax = axes[2][1]
feature = "utilization"
_, bin_edges = np.histogram(pareto_all[feature], bins=5)
for i in range(5):
    left = bin_edges[i]
    right = bin_edges[i+1]
    series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
    ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
               s=marker_size,
               marker=markers[i%len(markers)],
               label=f"[{left:.3f}, {right:.3f})")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], ncol=1, title="Utilization", loc="upper right", markerscale=1.3, framealpha=1)


# ax = axes[1][0]
# feature = " [5]die_area"
# _, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
# for i in range(num_bins):
#     left = bin_edges[i]
#     right = bin_edges[i+1]
#     series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
#     ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                s=marker_size,
#                marker=markers[i%len(markers)],
#                label=f"[{left:.0f}, {right:.0f})")
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], title="Chip Area (mm2)", loc="upper right", markerscale=1.3, framealpha=1)

# ax = axes[1][1]
# feature = "tops/mb"
# _, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
# for i in range(num_bins):
#     left = bin_edges[i]
#     right = bin_edges[i+1]
#     series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
#     ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                s=marker_size,
#                marker=markers[i%len(markers)],
#                label=f"[{left:.3f}, {right:.3f})")
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], title="TOPS/MB", loc="upper right", markerscale=1.3, framealpha=1)

# ax = axes[3][0]
# feature = " [10]tops_per_server"
# _, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
# for i in range(num_bins):
#     left = bin_edges[i]
#     right = bin_edges[i+1]
#     series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
#     ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                s=marker_size,
#                marker=markers[i%len(markers)],
#                label=f"[{left:.0f}, {right:.0f})")
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], title="TOPS per Server", loc="upper right", markerscale=1.3, framealpha=1)

# ax = axes[3][0]
# feature = "t"
# i = 0
# p_df = pareto_all.sort_values(feature, ascending=True)
# for p in pd.unique(p_df[feature]):
#     series = p_df[p_df[feature] == p]
#     if not series.empty:
#         ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                    s=marker_size,
#                    # facecolors='none', 
#                    # edgecolor=colors[i%len(colors)], 
#                    marker=markers[i%len(markers)],
#                    label=p)
#     i += 1
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], ncol=4, title="t", loc="upper right", markerscale=1.3, framealpha=1)

# ax = axes[3][1]
# feature = " [8]sram_per_server"
# _, bin_edges = np.histogram(pareto_all[feature], bins=num_bins)
# for i in range(num_bins):
#     left = bin_edges[i]
#     right = bin_edges[i+1]
#     series = pareto_all[(pareto_all[feature] >= left) & (pareto_all[feature] < right)]
#     ax.scatter(series["tco/token"]*1000, series["latency"]/1e3, 
#                s=marker_size,
#                marker=markers[i%len(markers)],
#                label=f"[{left/1000:.0f}, {right/1000:.0f})")
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], title="GB per Server", loc="upper right", markerscale=1.3, framealpha=1)


for ax_row in axes:
    for ax in ax_row:
        ax.grid(True)

if model == 'gpt2':
    fig.text(0.5, 0.01, "TCO per 1K tokens ($)", ha='center')
    fig.text(0.02, 0.5, "Latency (ms)", va='center', rotation='vertical')
    fig.text(0.5, 0.96, "GPT2 Pareto Curve", ha='center')
elif model == 'gpt3':
    fig.text(0.5, 0.03, "TCO per 1K tokens ($)", ha='center')
    fig.text(0.02, 0.5, "Latency (ms)", va='center', rotation='vertical')
    fig.text(0.5, 0.96, "GPT3, Batch Size = 128, Pareto Curve", ha='center')
elif model == 'mtnlg':
    fig.text(0.5, 0.025, "TCO per 1K tokens ($)", ha='center')
    fig.text(0.03, 0.5, "Latency (ms)", va='center', rotation='vertical')
    fig.text(0.5, 0.96, "MT-NLG Pareto Curve", ha='center')



# for index, data_point in pareto_all.iterrows():
#     batch = data_point["batch"]
#     x = data_point["tco/token"]*1000
#     y = data_point["latency"]/1e3
#     label = f"{batch}"
#     plt.annotate(label, (x,y), textcoords='offset points', xytext=(0,5), ha='center', fontsize=8)
# 
# plt.legend(title="All TCO ($)"


# %%
fig, axes = plt.subplots(1, 1, figsize=(10,6), dpi=300)
ax = axes
# plt.yscale("log")
plt.xscale("log")
ax.set_xlabel("Cost per 1K tokens ($)")
ax.set_ylabel("Latency (ms)")

for batch_size in pd.unique(df["batch"]):
    series = pareto_all[pareto_all["batch"] == batch_size]
    if not series.empty:
        ax.plot(series["tco/token"]*1000, series["latency"]/1e3, linestyle='--', marker='o', markersize=3, label=batch_size)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title="Batch Size", loc="upper right")

for index, data_point in pareto_all.iterrows():
    batch = int(data_point["all_tco"])
    label = f"{batch}"
    x = data_point["tco/token"]*1000
    y = data_point["latency"]/1e3
    ax.annotate(label, (x,y), textcoords='offset points', xytext=(0,5), ha='center', fontsize=4)

plt.show()

# %%
# RUN EVERYTHING BY HITTING 'Run Above'