import utils
import dfplot
import matplotlib.pyplot as plt 

csvfile = 'routing_opt_results.csv'
df = utils.csv2df(csvfile)
df_GPU = utils.csv2df('GPU.csv')

df_asic_hot = df[df['asic_hot']>0.0]
df_server_hot = df[df['server_hot']>0.0]

annotate = False

key = None
markersize = 3

# Thruput and Latency
ax = dfplot.line(df, x='thru', y='delay', key=None, sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=markersize)
for dataframe in [df_asic_hot, df_server_hot]:
  if not dataframe.empty:
    ax = dfplot.line(dataframe, x='thru', y='delay', key=None, sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=markersize, ax=ax, color='red')
#  ax = dfplot.line(df_GPU, x='thru', y='delay', key=None, sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=10, ax=ax, color='black', marker='d')
ax.get_legend().remove()

fig = ax.get_figure()
fig.savefig('thru_delay.pdf')
plt.close(fig)


ax = dfplot.line(df, x='watts_per_thru', y='cost_per_thru', key=None, sort_key=True, keyloc='upper right', xlabel='Watts/Throughput', ylabel='Cost/Throughput', linestyle='',markersize=markersize)
for dataframe in [df_asic_hot, df_server_hot]:
  if not dataframe.empty:
    ax = dfplot.line(df_server_hot, x='watts_per_thru', y='cost_per_thru', key=None, sort_key=True, keyloc='upper right', xlabel='Power/Throughput', ylabel='Cost/Throughput', linestyle='',markersize=markersize, ax=ax, color='red')
#  ax = dfplot.line(df_GPU, x='watts_per_thru', y='cost_per_thru', key=None, sort_key=True, keyloc='upper right', xlabel='Power/Throughput', ylabel='Cost/Throughput', linestyle='',markersize=10, ax=ax, marker='d')

#  ax.set_xlim(5.5, 7)
#  ax.set_ylim(10, 40)
ax.get_legend().remove()

if annotate:
  for idx, row in df.iterrows():
   ax.annotate("(%d, %d)" % (int(row.sram_per_asic), int(row.tops_per_asic)),xy=(row.watts_per_thru, row.cost_per_thru+0.05), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('thru_pareto.pdf')
plt.close(fig)

ax = dfplot.line(df, x='watts_delay', y='cost_delay', key=None, sort_key=True, keyloc='upper right', xlabel='Watts Latency Product (W*s)', ylabel='Cost Latency Product ($*s)', linestyle='',markersize=markersize)
for dataframe in [df_asic_hot, df_server_hot]:
  if not dataframe.empty:
    ax = dfplot.line(df_server_hot, x='watts_delay', y='cost_delay', key=None, sort_key=True, keyloc='upper right', xlabel='Watts Latency Product (W*s)', ylabel='Cost Latency Product ($*s)', linestyle='',markersize=markersize, ax=ax, color='red')
#  ax = dfplot.line(df_GPU, x='watts_delay', y='cost_delay', key=None, sort_key=True, keyloc='upper right', xlabel='Watts Latency Product (W*s)', ylabel='Cost Latency Product ($*s)', linestyle='',markersize=10, ax=ax, marker='d')

#  ax.set_xlim(2.4, 3)
#  ax.set_ylim(4, 24)
ax.get_legend().remove()

if annotate:
  for idx, row in df.iterrows():
   ax.annotate("(%d, %d)" % (int(row.sram_per_asic), int(row.tops_per_asic)),xy=(row.watts_delay, row.cost_delay+0.05), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('delay_pareto.pdf')
plt.close(fig)

