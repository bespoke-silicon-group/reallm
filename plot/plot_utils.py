import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

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

# Print model exploration info
def print_exploration_info(models_df, models_name):
    for model in models_name:
        df = models_df[model]
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

def print_optimal_designs(models_df, models_label, ctx_len=2048):
    print(f'Context Length = {ctx_len}')
    for model in models_label:
        df = models_df[model]
        df = df[df['real_ctx_len']==ctx_len]
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
            if f == 'W/tput':
                print(f'{features[f]}, {latency_optimal[f]*1e3}, {energy_optimal[f]*1e3}, {tco_optimal[f]*1e3}')
            elif f == 'tco/1ktoken':
                print(f'{features[f]}, {latency_optimal[f]*100}, {energy_optimal[f]*100}, {tco_optimal[f]*100}')
            else:
                print(f'{features[f]}, {latency_optimal[f]}, {energy_optimal[f]}, {tco_optimal[f]}')

        print(" ")

def compare_gpu_tpu(models_df, plot_dir='results', ctx_len=2048):

    df = models_df['gpt3']
    df = df[df['real_ctx_len']==ctx_len]
    pareto_gpt3 = pareto_front_filter_2d(df, "tco/1ktoken", "latency_ms", )
    df = models_df['palm']
    df = df[df['real_ctx_len']==ctx_len]
    pareto_palm = pareto_front_filter_2d(df, "tco/1ktoken", "latency_ms", )

    tpu_price = 1.45 # 3Y reservation, per chip-hour, from google cloud: https://cloud.google.com/tpu/pricing#v4-pricing
    tpu_tco = 0.31 # per chip-hour, from our TCO model, not used
    # from TPU paper figure 1 and table 2
    tpu_540B_perf={
        "num_chips":         [        64,        64,       64,        64], 
        "batch":             [        64,       128,      512,      1024],
        "latency_ms":        [   1820/64,        45,  6000/64,       180], 
        "tput":              [64*64/1.82, 128/0.045, 64*512/6, 1024/0.18], 
        }
    tpu_540B={"tco/1ktoken":[], "latency_ms":[]}
    for i in range(3):
        tpu_540B["latency_ms"].append(tpu_540B_perf["latency_ms"][i])
        tpu_540B["tco/1ktoken"].append(tpu_price*tpu_540B_perf["num_chips"][i]/3600/tpu_540B_perf["tput"][i]*1000)

    gpu_price = 1.10 # per GPU per hour, from Lambda
    gpu_tco = 1.67 # per GPU per hour, from our TCO Model, not used

    # From Deep Speed Fig 5
    gpu_gpt2 = {"latency_ms": [100], 
                "tco/1ktoken": [gpu_price/(160*3600/1000)]
                }
    gpu_gpt3 = {"latency_ms": [620], 
                "tco/1ktoken": [gpu_price/(18*3600/1000)]
                }

    markers = ['<', 'p', 'D', 'D', 'o', 's', '^']
    fig, axes = plt.subplots(1, 2, figsize=(10,4), dpi=200)
    plt.tight_layout(pad=3.0, w_pad=0.2, h_pad=0.5)

    # GPT-3
    ax = axes[0]
    ax.plot(pareto_gpt3["tco/1ktoken"], pareto_gpt3["latency_ms"], linestyle='--', 
            markeredgecolor='black', markeredgewidth=0.5, markerfacecolor='tab:blue', c='tab:blue', 
            marker=markers[0], markersize=15, label='Ours')
    ax.plot(gpu_gpt3["tco/1ktoken"], gpu_gpt3["latency_ms"],
        linestyle='--', marker=markers[1], 
        markeredgecolor='black', markerfacecolor='tab:red', c='tab:red',
        markeredgewidth=0.5, markersize=15,
        label="A100 GPU")
    tco_optimal = pareto_gpt3.loc[pd.to_numeric(pareto_gpt3['tco/1ktoken']).idxmin()]
    cc_opt = (tco_optimal["tco/1ktoken"], tco_optimal["latency_ms"])
    gpu_opt = (gpu_gpt3["tco/1ktoken"][0], gpu_gpt3["latency_ms"][0])
    # TCO improvement
    color = 'salmon'
    ax.axvline(x=cc_opt[0], linestyle=':', c=color, linewidth=1.5)
    left = (cc_opt[0], gpu_opt[1])
    ax.annotate("", xy=left, xytext=gpu_opt,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
    ax.text(math.sqrt(gpu_opt[0]*left[0]), gpu_opt[1], 
            f'{(gpu_opt[0]/left[0]):.1f}x',
            fontsize=14,
            ha='center', va="center",
            bbox=dict(boxstyle="round",
                    edgecolor='black',
                    facecolor=color
                    )
            )
    # Latency improvement
    color = 'skyblue'
    ax.axhline(y=cc_opt[1], linestyle=':', c=color, linewidth=1.5)
    bot = (gpu_opt[0], cc_opt[1])
    ax.annotate("", xy=bot, xytext=gpu_opt,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
    ax.text(gpu_opt[0], math.sqrt(gpu_opt[1]*bot[1]),
            f'{(gpu_opt[1]/bot[1]):.1f}x',
            fontsize=12,
            ha='center', va="center",
            bbox=dict(boxstyle="round",
                    edgecolor='black',
                    facecolor=color
                    )
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(right=3e-2)
    ax.set_ylim(top=1e3)
    ax.grid(which='both', c='whitesmoke')
    ax.text(2.4e-4, 2.1e2, 'GPT-3', fontsize=17, weight='bold')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(0.05, 0.3), 
              fontsize=14)

    # PaLM 540B
    ax = axes[1]
    ax.plot(pareto_palm["tco/1ktoken"], pareto_palm["latency_ms"], linestyle='--', 
            markeredgecolor='black', markeredgewidth=0.5, markerfacecolor='tab:blue', c='tab:blue', 
            marker=markers[0], markersize=15, label='Ours')
    ax.plot(tpu_540B["tco/1ktoken"], tpu_540B["latency_ms"],
        linestyle='--', marker=markers[2], 
        markeredgecolor='black', markerfacecolor='tab:green', c='tab:green',
        markeredgewidth=0.5, markersize=12,
        label="TPUv4")
    tco_optimal = pareto_palm.loc[pd.to_numeric(pareto_palm['tco/1ktoken']).idxmin()]
    cc_opt = (tco_optimal["tco/1ktoken"], tco_optimal["latency_ms"])
    tpu_opt = (tpu_540B["tco/1ktoken"][2], tpu_540B["latency_ms"][2])
    # TCO improvement
    color = 'salmon'
    ax.axvline(x=cc_opt[0], linestyle=':', c=color, linewidth=1.5)
    left = (cc_opt[0], tpu_opt[1])
    ax.annotate("", xy=left, xytext=tpu_opt,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
    ax.text(math.sqrt(tpu_opt[0]*left[0]), tpu_opt[1], 
            f'{(tpu_opt[0]/left[0]):.1f}x',
            fontsize=14,
            ha='center', va="center",
            bbox=dict(boxstyle="round",
                    edgecolor='black',
                    facecolor=color
                    )
            )
    # Latency improvement
    color = 'skyblue'
    ax.axhline(y=cc_opt[1], linestyle=':', c=color, linewidth=1.5)
    bot = (tpu_opt[0], cc_opt[1])
    ax.annotate("", xy=bot, xytext=tpu_opt,
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=1))
    ax.text(tpu_opt[0], math.sqrt(tpu_opt[1]*bot[1]),
            f'{(tpu_opt[1]/bot[1]):.1f}x',
            fontsize=12,
            ha='center', va="center",
            bbox=dict(boxstyle="round",
                    edgecolor='black',
                    facecolor=color
                    )
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=1.2e2)
    ax.grid(which='both', c='whitesmoke')

    ax.text(4e-4, 5e1, 'PaLM 540B', fontsize=17, weight='bold')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower left', bbox_to_anchor=(0.05, 0.3), 
              fontsize=14)

    fig.text(0.5, 0.04, "TCO per 1K Tokens ($)", ha='center', fontsize=16)
    fig.text(0.015, 0.5, "Token Generation Latency (ms)", va='center', rotation='vertical', fontsize=14)

    plt.savefig(plot_dir+'/gpu_compare.pdf', format='pdf', bbox_inches='tight')

def asic_profit(gpu_price_per_hour=1.1, gpu_tput=18, 
                chiplet_NRE=35e6, cc_sys_tput=33791, cents_per_1k_tokens=0.018,
                plot_dir='results'):
    # when should go Chiplet Could
    # x: average tokens/sec
    # y: Cents/1K Token, for GPU and Chiplet Clouds
    bing_tokens_per_sec = 5.4e6
    google_tokens_per_sec = 49e6
    chatgpt_daily_user = 13e6
    chatgpt_request_per_user_per_day = 5
    chatgpt_tokens_per_request = 1024
    chatgpt_tokens_per_day = chatgpt_daily_user*chatgpt_request_per_user_per_day*chatgpt_tokens_per_request
    chatgpt_tokens_per_sec = chatgpt_tokens_per_day/24/3600
    x = []
    gpu_1ktoken = []
    cc_1ktoken = []
    gpu_cost_per_1k_tokens = gpu_price_per_hour/3600/(gpu_tput/1000)
    NRE_per_sec = chiplet_NRE/1.5/365/24/3600
    TCO_per_sec = cents_per_1k_tokens * (cc_sys_tput / 1000) / 100

    index = 0
    for i in range(3, 9):
        base = 10**i
        step = 10**(i-1)
        for tokens_per_sec in range(base, base*10, step):
            x.append(tokens_per_sec)
            gpu_1ktoken.append(gpu_cost_per_1k_tokens)
            num_cc_sys = math.ceil(tokens_per_sec/cc_sys_tput)
            cost_per_sec = NRE_per_sec + TCO_per_sec * num_cc_sys
            cc_cost_per_1k_tokens = cost_per_sec/(tokens_per_sec/1000) 
            cc_1ktoken.append(cc_cost_per_1k_tokens)

            if abs(cc_cost_per_1k_tokens-gpu_cost_per_1k_tokens) < 0.05*gpu_cost_per_1k_tokens:
                break_even = tokens_per_sec 
                break_even_index = index
            index += 1

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=200)
    ax.plot(x, gpu_1ktoken, label='GPU (TCO)', c='tab:red',
            lw=2)
    ax.plot(x, cc_1ktoken, label='ASIC (NRE+TCO)', linestyle='dashdot',c='tab:green',
            lw=2)

    ax.fill_between(x[break_even_index:], gpu_1ktoken[break_even_index:], cc_1ktoken[break_even_index:],
                    fc='lightblue', alpha=0.5,
                    label='Improvement'
                    # ec='grey', hatch = '*',
                    )

    print('Break-even point is:', break_even)
    ax.axvline(x=break_even, linestyle=':', c='k', linewidth=0.8)
    ax.text(break_even*0.96, 0.9, 'Break-even', fontsize='large', 
            ha='right', va='top')

    ax.axvline(x=bing_tokens_per_sec, linestyle=':', c='k', linewidth=0.8)
    ax.text(bing_tokens_per_sec*0.96, 0.9, 'Bing', fontsize='large',
            ha='right', va='top')

    ax.axvline(x=google_tokens_per_sec, linestyle=':', c='k', linewidth=0.8)
    ax.text(google_tokens_per_sec*0.96, 0.9, 'Google\nSearch', fontsize='large',
            ha='right', va='top')

    ax.axvline(x=chatgpt_tokens_per_sec, linestyle=':', c='k', linewidth=0.8)
    ax.text(chatgpt_tokens_per_sec*0.96, 0.9, 'ChatGPT', fontsize='large',
            ha='right', va='top')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(left=1000, right=1e8)
    ax.set_xlabel("GPT-3 Tokens Generated per Second", fontsize='large')
    ax.set_ylabel("Cost per 1K Tokens ($)", fontsize='large')

    ax.legend(loc='lower left', fontsize='large', framealpha=1.0)
    ax.set_axisbelow(True)
    ax.grid(which='both', c='whitesmoke')

    plt.savefig(plot_dir+'/asic_profit.pdf', format='pdf', bbox_inches='tight')

def design_space_exploration(models_df, models_label, plot_dir='results'):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 5), dpi=200)
    plt.tight_layout(pad=3.0, w_pad=0.5, h_pad=0.5)
    axes = [ax1, ax2, ax3, ax4]

    colors = ['darkblue', 'royalblue','lightblue', 
              'lightgreen', 'yellowgreen', 'yellow', 
              'gold',
              'orange', 'orangered', 'darkred', 
              'plum']

    for model in models_label:
        df = models_df[model]
        df = df[df['real_ctx_len']==2048]
        ax = axes.pop(0)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_axisbelow(True)
        ax.grid(which='major', axis='both', c='whitesmoke')

        i = 0
        for batch in pd.unique(df['batch']):
            series = df[df['batch']==batch]
            s = ax.scatter(x=series["tco/1ktoken"], y=series["latency_ms"], 
                           label=batch,
                           c=colors[i],
                           s=0.0001)
            s.set_rasterized(True)
            i += 1

        ax.set_title(models_label[model], y=1.0, pad=-13)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='center left',
               bbox_to_anchor=(0.87,0.5),
               markerscale=1000,
               framealpha=1,
               title='Batch Size')

    fig.text(0.5, 0.039, "TCO/1K Tokens ($)", ha='center', fontsize=15)
    fig.text(0.025, 0.5, "Latency (ms)", va='center', rotation='vertical', fontsize='large')

    plt.savefig(plot_dir+'/design_space.pdf', format='pdf', bbox_inches='tight')

def chip_size(models_df, model='gpt3', ctx_len=2048, plot_dir='results'):
    # Chip Size: TCO and Latency
    # x = area, y_left = latency, y_right = tco/1k tokens
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3), dpi=200)
    plt.tight_layout(pad=3.0, w_pad=2.5, h_pad=0.5)

    df = models_df[model]
    df = df[df['real_ctx_len']==ctx_len]
    df = df[(df['batch']==64) & (df['tput']>20000)]

    feature = " [5]die_area"
    areas = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    areas_tco_token = []
    areas_latency = []
    areas_capex_ratio = []
    for i in range(len(areas)-1):
        left = areas[i]
        right = areas[i+1]
        series = df[(df[feature] > left) & (df[feature] <= right)]
        if series.shape[0] > 0:
            tco_optimal = series.loc[pd.to_numeric(series['all_tco']).idxmin()]
            areas_tco_token.append(tco_optimal['all_tco'])
            areas_capex_ratio.append(tco_optimal['all_srv_cost']/tco_optimal['all_tco'])
        else:
            areas_tco_token.append(0.0)
            areas_capex_ratio.append(0.0)

    areas_tco_token_breakdown = {'CapEx': [], 'OpEx': []}
    for i in range(len(areas_capex_ratio)):
        areas_tco_token_breakdown['CapEx'].append(areas_tco_token[i] * areas_capex_ratio[i])
        areas_tco_token_breakdown['OpEx'].append(areas_tco_token[i] * (1-areas_capex_ratio[i]))
    

    x = [50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 650.0, 750.0]
    bottom = np.zeros(len(x))
    for label, val in areas_tco_token_breakdown.items():
        ax1.bar(x, val, 60, edgecolor='black', linewidth=1.5, label=label, bottom=bottom)
        bottom += val

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc="upper left", framealpha=1)

    ax1.set_xlabel("Chip Area (mm$^2$)", fontsize=14)
    ax1.set_ylabel("TCO ($)", fontsize=14)
    ax1.set_title('Batch=64, Tput>20,000 tokens/s', x=0.53, y=.98, fontsize=13)

    ax1.set_axisbelow(True)
    ax1.grid(which='major', axis='y', c='grey')
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    df = models_df[model]
    df = df[(df['batch']==16) & (df['all_tco']<2.5e5)]
    df = df[df['real_ctx_len']==ctx_len]

    areas_latency_breakdown = {'Compute': [], 'Communicate': []}
    areas_latency = []
    pt = []
    for i in range(len(areas)-1):
        left = areas[i]
        right = areas[i+1]
        series = df[(df[feature] > left) & (df[feature] <= right)]
        if series.shape[0] > 0:
            latency_optimal = series.loc[pd.to_numeric(series['latency_ms']).idxmin()]
            areas_latency_breakdown['Compute'].append(latency_optimal['compute_latency']/1000)
            areas_latency_breakdown['Communicate'].append(latency_optimal['communicate_latency']/1000)
            areas_latency.append(latency_optimal['latency_ms'])
            pt.append(f"({latency_optimal['p']}, {latency_optimal['t']})")
        else:
            areas_latency_breakdown['Compute'].append(0.0)
            areas_latency_breakdown['Communicate'].append(0.0)
            areas_latency.append(0.0)
            pt.append('(0, 0)')


    ax2.plot(x, areas_latency, marker='^', markersize=10, linewidth=2)
    ax2.set_ylim(bottom=0, top=1.25)

    ax2.set_xlabel("Chip Area (mm$^2$)", fontsize=14)
    ax2.set_ylabel("Per-Token Latency (ms)", fontsize=14)
    ax2.set_title('Batch=16, TCO<$250,000', fontsize=13, y=0.98)

    ax2.set_axisbelow(True)
    ax2.grid(which='major', axis='y', c='grey')

    plt.savefig(plot_dir+'/chip_size.pdf', format='pdf', bbox_inches='tight')

def batch_size(models_df, models_label, plot_dir='results'):
    # Batch size
    optimal_points = None

    fig, ((ax1, ax2,), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 4.7), dpi=200)
    plt.tight_layout(pad=3.0, w_pad=.4, h_pad=0.1)
    markers = ['<', 'p', 'D',  'o', 's', '^']
    all_ax = [ax1, ax2, ax3, ax4]

    i = 0
    for model in models_label:
        ax = all_ax[i]
        i += 1
        j = 0
        df = models_df[model]
        for ctx in pd.unique(df['real_ctx_len']):
            df = models_df[model]
            df = df[df['real_ctx_len']==ctx]
            tco_1ktoken = []
            batch = []
            for b in pd.unique(df['batch']):
                batch.append(str(b))
                series = df[df['batch']==b]
                series = series.sort_values('tco/1ktoken', ascending=True)
                tco_optimal = series.iloc[0]
                if optimal_points is None:
                    optimal_points = series.head(1)
                else:
                    optimal_points = pd.concat([optimal_points, series.head(1)], axis=0)
                tco_1ktoken.append(tco_optimal['tco/1ktoken'])

            p1, = ax.plot(batch, tco_1ktoken, marker=markers[j], 
                lw=2, markersize=6, label=ctx)
            optimal_val = min(tco_1ktoken)
            optimal_index = tco_1ktoken.index(optimal_val)
            ax.plot(batch[optimal_index], optimal_val, marker='*', c=p1.get_color(), 
                    markersize=20)
        
            j += 1

        ax.set_yscale('log')
        ax.grid(which='both', c='whitesmoke')

        if ax == ax1:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], loc='best', ncol=1, 
                    title='Context')

        if 'palm' in model:
            ax.set_title(models_label[model]+'\nMulti-Query', y=1.0, pad=-35, fontsize=14)
        else:
            ax.set_title(models_label[model]+'\nMulti-Head', y=1.0, pad=-35, fontsize=14)


    fig.text(0.5, 0.05, "Batch Size", ha='center', fontsize=14)
    fig.text(0.015, 0.5, "TCO/1K Tokens ($)", va='center', rotation='vertical', fontsize=14)

    plt.savefig(plot_dir+'/batch_size_all_models.pdf', format='pdf', bbox_inches='tight')

def design_choice(models_df, HBM_models_df, models_label, ctx_len=2048, plot_dir='results'):
    # vs. HBM and large chip
    model_performace = {'large': [], 'HBM':[], 'SRAM': []}
    model_names = []
    for model in models_label:
        model_names.append(models_label[model])
        df_cc = models_df[model]
        df_cc = df_cc[df_cc['real_ctx_len']==ctx_len]
        df_large = df_cc[df_cc[" [5]die_area"]>700.0]
        df_HBM_chiplet = HBM_models_df[model]
        df_HBM_chiplet = df_HBM_chiplet[df_HBM_chiplet['real_ctx_len']==ctx_len]
        # df_HBM_large = dfs_HBM_large[model]
        # df_HBM_large = df_HBM_large[df_HBM_large['real_ctx_len']==2048]

        systems = {'large': df_large,'HBM': df_HBM_chiplet, 'SRAM': df_cc}

        for sys in systems:
            df = systems[sys]
            series = df.sort_values('tco/1ktoken', ascending=True)
            tco_optimal = series.iloc[0]
            model_performace[sys].append(tco_optimal['tco/1ktoken'])

    for i in range(4):
        model_performace['large'][i] = (model_performace['large'][i]/model_performace['SRAM'][i])
        model_performace['HBM'][i] = (model_performace['HBM'][i]/model_performace['SRAM'][i])
        model_performace['SRAM'][i] = 1

    model_performace['large'].append(np.exp(np.log(model_performace['large']).mean()))
    model_performace['HBM'].append(np.exp(np.log(model_performace['HBM']).mean()))
    model_performace['SRAM'].append(1)
        
    fig, ax = plt.subplots(figsize=(7, 2.7), dpi=200)
    x = np.arange(1, 8, 1.5)
    w = 0.35
    ax.bar(x-0.4, model_performace['SRAM'], width=w, label='Chiplet Cloud')
    ax.bar(x, model_performace['large'], width=w, label='Large Chip')
    ax.bar(x+0.4, model_performace['HBM'], width=w, label='HBM')

    for i in range(5):
        improvement = (model_performace['large'][i]/model_performace['SRAM'][i])
        ax.text(x[i]-0.05, model_performace['large'][i]*1.01, f'{improvement:.2f}x', ha='center', fontsize='large')

        improvement = (model_performace['HBM'][i]/model_performace['SRAM'][i])
        ax.text(x[i]+0.45, model_performace['HBM'][i]*1.01, f'{improvement:.2f}x', ha='center', fontsize='large')

    model_names.append('GeoMean')
    ax.set_xticks(x, model_names)
    ax.set_ylabel('Normalized TCO/Token', fontsize='large')
    ax.grid(which='both', axis='y', c='grey')
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', ncol=3, bbox_to_anchor=(1.0, 1.0), handletextpad=0.1,
            fontsize=11, framealpha=1.0)
    ax.set_ylim(top=3.5)

    plt.savefig(plot_dir+'/design_choice.pdf', format='pdf', bbox_inches='tight')

def p_sweep(models_df, models_label, ctx_len=2048, plot_dir='results'):
    import matplotlib.ticker as mtick

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 4.6), dpi=200)
    plt.tight_layout(pad=3.0, w_pad=2., h_pad=0.85)

    axes = [ax1, ax2, ax3, ax4]

    # for model in real_models:
    for model in ['gpt2', 'palm']:
        for batch in [16, 128]:
            ax = axes.pop(0)
            ax.set_title(models_label[model]+f', Batch={batch}', x=0.5, y=.97)
            srv_id = 645
            df = models_df[model]
            df = df[df['batch']==batch]
            df = df[df['real_ctx_len']==ctx_len]
            df = df[df['srv_id']==srv_id]

            x = []
            y1 = []
            y3 = []

            for p in pd.unique(df['p']):
                series = df[(df['p']==p)]
                x.append(str(p))
                series = series.sort_values('tco/1ktoken', ascending=True)
                tco_optimal = series.iloc[0]
                y1.append(tco_optimal['tco/1ktoken'])
                y3.append(tco_optimal['utilization'])

            ax_right = ax.twinx()

            l1 = ax.plot(x, y1, marker='^', label='TCO/Token', c='tab:blue')
            l2 = ax_right.plot(x, y3,marker='o',  label='Utilization', c='tab:orange')
            lns = l1+l2
            labs = [l.get_label() for l in lns]
            ax.set_yscale('log')

            ax.tick_params(axis='y', colors=l1[0].get_color())
            ax_right.tick_params(axis='y', colors=l2[0].get_color())
            ax_right.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax.grid(which='both', c='whitesmoke')

    ax3.legend(lns, labs, loc='lower left', bbox_to_anchor=(-0.02, 0.25), fontsize=10)

    fig.text(0.5, 0.055, "Pipeline Stages", ha='center', fontsize='14')
    fig.text(0.01, 0.46, "TCO/1K Tokens ($)", va='center', rotation='vertical', fontsize='14', c='tab:blue')
    fig.text(0.98, 0.5, "Utilization", va='center', rotation=270, fontsize='14', c='tab:orange')

    plt.savefig(plot_dir+'/p_sweep.pdf', format='pdf', bbox_inches='tight')

def server_flexibility(models_df, models_label, ctx_len=2048, plot_dir='results'):
    # Multiple Tasks
    model_best = {}
    for model in models_label:
        df = models_df[model]
        df = df[df['real_ctx_len']==ctx_len]
        model_best[model] = df["tco/1ktoken"].min()
    srvs_perf = [None]
    print('Different Models:')
    for srv_id in tqdm(range(1, 1073)):
        srv_perf = {}
        for model in models_label:
            df = models_df[model]
            perf = df[(df['srv_id']==srv_id)&(df['real_ctx_len']==ctx_len)]['tco/1ktoken'].min()
            srv_perf[model] = (model_best[model])/perf
        srvs_perf.append(srv_perf)

    srvs_perf_sum = {}
    for srv_id in range(1, 1073):
        sum_perf = 0
        for t in srvs_perf[srv_id]:
            sum_perf += srvs_perf[srv_id][t]
        srvs_perf_sum[srv_id] = sum_perf
    sort_srvs_perf = {k: v for k, v in sorted(srvs_perf_sum.items(), key=lambda item: item[1], reverse=True)}

    # Multiple contex
    ctx_best = {}
    for ctx in [256, 2048, 8196]:
        df = models_df['gpt3']
        df = df[df['real_ctx_len']==ctx]
        ctx_best[ctx] = df["tco/1ktoken"].min()

    ctx_srvs_perf = [None]
    print('Different Context Length:')
    for srv_id in tqdm(range(1, 1073)):
        ctx_srv_perf = {}
        for ctx in [256, 2048, 8196]:
            df = models_df['gpt3']
            perf = df[(df['srv_id']==srv_id)&(df['real_ctx_len']==ctx)]['tco/1ktoken'].min()
            ctx_srv_perf[ctx] = (ctx_best[ctx])/perf
        ctx_srvs_perf.append(ctx_srv_perf)

    ctx_srvs_perf_sum = {}
    for srv_id in range(1, 1073):
        sum_perf = 0
        for t in ctx_srvs_perf[srv_id]:
            sum_perf += ctx_srvs_perf[srv_id][t]
        ctx_srvs_perf_sum[srv_id] = sum_perf
    sort_ctx_srvs_perf = {k: v for k, v in sorted(ctx_srvs_perf_sum.items(), key=lambda item: item[1], reverse=True)}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9,3.2), dpi=200)
    plt.tight_layout(pad=1.5, w_pad=0.1,)

    num_srvs = 10
    model_perfs_breakdown = {'gpt2':[], 'tnlg':[], 'gpt3':[], 'palm':[]}
    srvs = []
    plot_x = []
    i = 1
    for srv_id in sort_srvs_perf:
        if i > num_srvs:
            break
        srvs.append(str(srv_id))
        for model in models_label:
            model_perfs_breakdown[model].append(srvs_perf[srv_id][model])
        plot_x.append(i)
        i += 1

    bottom = np.zeros(num_srvs)
    for label, val in model_perfs_breakdown.items():
        ax.bar(plot_x, val, 0.4, edgecolor='black', linewidth=1.2, label=models_label[label], bottom=bottom)
        bottom += val

    ax.set_xticks(plot_x, srvs, fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y', c='grey')
    ax.set_ylim(top=4.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc="upper center", fontsize=12, 
            bbox_to_anchor=(0.5,1.12), labelspacing=0.1
    )

    ax = ax2
    ctx_perfs_breakdown = {256:[], 2048:[], 8196:[]}
    srvs = []
    plot_x = []
    i = 1
    for srv_id in sort_ctx_srvs_perf:
        if i > num_srvs:
            break
        srvs.append(str(srv_id))
        for ctx in ctx_perfs_breakdown:
            ctx_perfs_breakdown[ctx].append(ctx_srvs_perf[srv_id][ctx])
        plot_x.append(i)
        i += 1

    bottom = np.zeros(num_srvs)
    for label, val in ctx_perfs_breakdown.items():
        ax.bar(plot_x, val, 0.4, edgecolor='black', linewidth=1.2, label=f'CTX-{label}', bottom=bottom)
        bottom += val

    ax.set_xticks(plot_x, srvs, fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(which='both', axis='y', c='grey')
    ax.set_ylim(top=3.0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=2, loc="upper center", fontsize=12,
            bbox_to_anchor=(0.5,1.12), labelspacing=0.1)

    fig.text(0.5, 0.002, "Server ID", ha='center', fontsize=17)
    fig.text(0.001, 0.5, "Aggregated Performance", va='center', rotation='vertical', fontsize='15',
            )

    plt.savefig(plot_dir+'/multi_models.pdf', format='pdf', bbox_inches='tight')

def compare_memory(plot_dir='results'):

    ddr =  {'size':    16, 'area': 469.8, 'bw': 25.6, 'read_pj': 20}
    hbm =  {'size':    24, 'area': 768,   'bw':  307, 'read_pj': 4}
    sram = {'size': 0.001, 'area': 0.3,   'bw':    8, 'read_pj': 0.2}

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.5), dpi=200)

    total_size = 48 # GB
    data = {'DDR4': ddr, 'HBM2e': hbm, 'SRAM\n(7nm)':sram}
    
    for name in data:
        d = data[name]
        num_blocks = total_size / d['size']
        total_area = num_blocks * d['area']
        total_bw = num_blocks * d['bw']
        # x: read_pj_per_total_bw
        x = d['read_pj'] / total_bw
        # y: total_area_per_total_bw
        y = total_area / total_bw
        ax.plot([x], [y], marker="o", markersize=80, alpha=0.4)
        ax.text(x, y, name,horizontalalignment='center', verticalalignment='center',)

        
    arrowprops=dict(arrowstyle='->', color='tab:red', linewidth=3, mutation_scale=40)
    ax.annotate("",  xytext=(0.3, 2), xy=(3e-6, 0.02), size = 50, arrowprops=arrowprops)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim(bottom=0.01)
    ax.set_xlim(left=1e-7)

    row_labels=['DDR4','HBM2e','SRAM']
    col_labels=['Size (GB)','Area ($mm^2$)','Bandwidth\n(GB/s)','Read Energy\n(pJ/bit)']
    table_vals=[]
    for name in data:
        d = data[name]
        table_vals.append([d['size'], d['area'], d['bw'], d['read_pj']])

    ax.text(2e-6, 13.5, 'Typical Memory Blocks', c='black',size=12) 
    ax.text(5e-4, 0.06, 'Better TCO/Token', c='tab:red', size=23)

    ax.set_xlabel(r'Read Energy per Total Bandwidth ($\frac{pJ/bit}{GB/s}$)', size =15)
    ax.set_ylabel(r'Area per Total Bandwidth ($\frac{mm^2}{GB/s}$)', size=15)

    ax.set_axisbelow(True)
    ax.grid(which='both', c='whitesmoke')

    the_table = ax.table(
        cellText=table_vals,
        colWidths = [0.14, 0.165, 0.163, 0.178],
        rowLabels=row_labels,
        colLabels=col_labels,
        bbox = [0.1, 0.5, 0.52, 0.4],
        # facecolor='white'
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.set_zorder(100)

    plt.savefig(plot_dir+'/mem_compare.pdf', bbox_inches='tight', pad_inches=0.1)