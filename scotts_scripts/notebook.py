# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List

hw_csv = "../asic_cloud_sim_v2/exploration.csv"
all_csv = "../app_perf_sim_v2/all.csv"

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

hw = pd.read_csv(hw_csv)

# concat hw infor into the main df
df = pd.concat([hw.loc[map(lambda x: x-1, df["srv_id"].to_numpy().tolist())].reset_index(drop=True), df], axis=1)
df.head()
# %%

# add W/tput
df['W/tput'] = df.apply(lambda row: (row['num_srvs']*row['chips_per_srv']*row['power_per_chip']*row['utilization']) / row['tput'], axis=1)
#df['W/tput'] = df.apply(lambda row: (row['num_srvs']*row['chips_per_srv']*row['power_per_chip']) / row['tput'], axis=1)

# add $/tput
df['$/tput'] = df.apply(lambda row: row['all_srv_cost'] / row['tput'], axis=1)

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

df.head()

#%%

plt.figure(figsize=(8,6), dpi=200)
plt.yscale("log")
plt.xscale("log")
for batch_size in pd.unique(df["batch"]):
    series = df[df["batch"] == batch_size]
    plt.scatter(x=series["all_tco/tput"], y=series["latency"], s=0.1)

# %%
# Plot GPT3 Pareto Curve

pareto_n1    = pareto_front_filter_2d( df[df["batch"] ==    1], "all_tco/tput", "latency", )
pareto_n2    = pareto_front_filter_2d( df[df["batch"] ==    2], "all_tco/tput", "latency", )
pareto_n4    = pareto_front_filter_2d( df[df["batch"] ==    4], "all_tco/tput", "latency", )
pareto_n8    = pareto_front_filter_2d( df[df["batch"] ==    8], "all_tco/tput", "latency", )
pareto_n16   = pareto_front_filter_2d( df[df["batch"] ==   16], "all_tco/tput", "latency", )
pareto_n32   = pareto_front_filter_2d( df[df["batch"] ==   32], "all_tco/tput", "latency", )
pareto_n64   = pareto_front_filter_2d( df[df["batch"] ==   64], "all_tco/tput", "latency", )
pareto_n128  = pareto_front_filter_2d( df[df["batch"] ==  128], "all_tco/tput", "latency", )
pareto_n256  = pareto_front_filter_2d( df[df["batch"] ==  256], "all_tco/tput", "latency", )
pareto_n512  = pareto_front_filter_2d( df[df["batch"] ==  512], "all_tco/tput", "latency", )
pareto_n1024 = pareto_front_filter_2d( df[df["batch"] == 1024], "all_tco/tput", "latency", )

fig, axes = plt.subplots(1, 1, figsize=(8,6), dpi=200)

ax = axes

ax.plot(pareto_n1["all_tco/tput"],   pareto_n1["latency"]  , linestyle='--', marker='o', label=1)
ax.plot(pareto_n2["all_tco/tput"],   pareto_n2["latency"]  , linestyle='--', marker='o', label=2)
ax.plot(pareto_n4["all_tco/tput"],   pareto_n4["latency"]  , linestyle='--', marker='o', label=4)
ax.plot(pareto_n8["all_tco/tput"],   pareto_n8["latency"]  , linestyle='--', marker='o', label=8)
ax.plot(pareto_n16["all_tco/tput"],  pareto_n16["latency"] , linestyle='--', marker='o', label=16)
ax.plot(pareto_n32["all_tco/tput"],  pareto_n32["latency"] , linestyle='--', marker='o', label=32)
ax.plot(pareto_n64["all_tco/tput"],  pareto_n64["latency"] , linestyle='--', marker='o', label=64)
ax.plot(pareto_n128["all_tco/tput"], pareto_n128["latency"], linestyle='--', marker='o', label=128)
ax.plot(pareto_n256["all_tco/tput"], pareto_n256["latency"], linestyle='--', marker='o', label=256)
ax.plot(pareto_n512["all_tco/tput"], pareto_n512["latency"], linestyle='--', marker='o', label=512)
ax.plot(pareto_n1024["all_tco/tput"], pareto_n1024["latency"], linestyle='--', marker='o', label=1024)

ax.set_xscale("log")
#ax.set_xlim(right=56)
ax.set_xlabel("TCO/Tokens/sec")

ax.set_yscale("log")
ax.set_ylabel("Latency (us)")

ax.set_title("GPT3 Pareto Curve")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], title="Batch Size", loc="upper right")

pareto_n128.sort_values("latency", ascending=True)
pareto_n1.sort_values("latency", ascending=True)

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

ax.set_title("GPT3")
ax.legend(title="Batch Size")

pareto_n64.sort_values("$/tput", ascending=True)

# %%
# RUN EVERYTHING BY HITTING 'Run Above'