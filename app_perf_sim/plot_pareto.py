import utils
import dfplot
import matplotlib.pyplot as plt 

# Plot lim
# Name       thru_opt             delay_opt            tco_per_thru       tco_delay
#            x           y        x           y        x         y        x          y
# BERT                   (0,6)                (0,6)    (47,52)   (50,200) (0.08,0.5) (0,2)
# GPT2       (150k,300k) (0,0.4)  (150k,300k) (0,0.4)  (4.5,4.8) (7,15)   (0.04,0.2) (0,0.5)
# T-NLG      (150k,250k) (0,0.5)  (150k,250k) (0,0.5)  (5.3,5.7) (8,20)   (0.1,0.7)  (0,4)
# MT-NLG-A   (180k,240k) (0,0.02) (180k,240k) (0,0.02) (5.1,5.7) (9,18)   (0.9,0.02) (0.02,0.03)

csvfile = 'MT-NLG-Atten_7nm_routing_opt_results.csv'
df = utils.csv2df(csvfile)

valid_df = df[df['asic_hot']==0.0]
valid_df = valid_df[valid_df['server_hot']==0.0]
df_asic_hot = df[df['asic_hot']>0.0]
df_server_hot = df[df['server_hot']>0.0]

annotate = True

draw_hot = False
key = 'asics_per_server'
markersize = 3

# Thruput optimal
ax = dfplot.line(valid_df, x='opt_thru', y='opt_thru_delay', key=key, sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=markersize)

if draw_hot:
  for dataframe in [df_asic_hot, df_server_hot]:
    if not dataframe.empty:
      ax = dfplot.line(dataframe, x='opt_thru', y='opt_thru_delay', key=key, sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=markersize, ax=ax, color='red')
ax.get_legend().remove()
#  ax.set_xlim(100e3, 250e3)
#  ax.set_ylim(0, 0.5)

fig = ax.get_figure()
fig.savefig('thru_opt.pdf')
plt.close(fig)

# Delay optimal
ax = dfplot.line(valid_df, x='opt_delay_thru', y='opt_delay', key=key, sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=markersize)

if draw_hot:
  for dataframe in [df_asic_hot, df_server_hot]:
    if not dataframe.empty:
      ax = dfplot.line(dataframe, x='opt_delay_thru', y='opt_delay', key=key, sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=markersize, ax=ax, color='red')
ax.get_legend().remove()
#  ax.set_xlim(180e3, 240e3)
#  ax.set_ylim(0.01, 0.02)

if annotate:
  for idx, row in df.iterrows():
   ax.annotate("(%d, %d)" % (int(row.sram_per_asic), int(row.tops_per_asic)),xy=(row.opt_delay_thru, row.opt_delay), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('delay_opt.pdf')
plt.close(fig)

# tco_per_opt_thru
ax = dfplot.line(valid_df, x='watts_per_opt_thru', y='cost_per_opt_thru', key=key, sort_key=True, keyloc='upper right', xlabel='Watts/Throughput', ylabel='Cost/Throughput', linestyle='',markersize=markersize)

if draw_hot:
  for dataframe in [df_asic_hot, df_server_hot]:
    if not dataframe.empty:
      ax = dfplot.line(dataframe, x='watts_per_opt_thru', y='cost_per_opt_thru', key=key, sort_key=True, keyloc='upper right', xlabel='Power/Throughput', ylabel='Cost/Throughput', linestyle='',markersize=markersize, ax=ax, color='red')

#  ax.set_xlim(5.1, 5.7)
#  ax.set_ylim(9, 18)
ax.get_legend().remove()

if annotate:
  for idx, row in df.iterrows():
   ax.annotate("(%d, %d)" % (int(row.sram_per_asic), int(row.tops_per_asic)),xy=(row.watts_per_opt_thru, row.cost_per_opt_thru+0.05), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('thru_pareto.pdf')
plt.close(fig)

# tco_opt_delay
ax = dfplot.line(valid_df, x='watts_opt_delay', y='cost_opt_delay', key=key, sort_key=True, keyloc='upper right', xlabel='Watts Latency Product (W*s)', ylabel='Cost Latency Product ($*s)', linestyle='',markersize=markersize)
if draw_hot:
  for dataframe in [df_asic_hot, df_server_hot]:
    if not dataframe.empty:
      ax = dfplot.line(dataframe, x='watts_opt_delay', y='cost_opt_delay', key=key, sort_key=True, keyloc='upper right', xlabel='Watts Latency Product (W*s)', ylabel='Cost Latency Product ($*s)', linestyle='',markersize=markersize, ax=ax, color='red')

#  ax.set_xlim(0.005, 0.02)
#  ax.set_ylim(0.02, 0.03)
ax.get_legend().remove()

if annotate:
  for idx, row in df.iterrows():
   ax.annotate("(%d, %d)" % (int(row.sram_per_asic), int(row.tops_per_asic)),xy=(row.watts_opt_delay, row.cost_opt_delay), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('delay_pareto.pdf')
plt.close(fig)

