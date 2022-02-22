import utils
import dfplot
import matplotlib.pyplot as plt 

csvfile = 'routing_opt_results.csv'
df = utils.csv2df(csvfile)

#  ax = dfplot.line(df, x='thru', y='lat', key='sram_per_asic', sort_key=True, keyloc='upper right', xlabel='Throughput', ylabel='Latency (ms)', linestyle='',markersize=8)

ax = dfplot.line(df, x='watts_per_thru', y='cost_per_thru', key='sram_per_asic', sort_key=True, keyloc='upper right', xlabel='Watts per thru', ylabel='Cost per thru', linestyle='',markersize=8)

ax.set_xlim(5.5, 7)
ax.set_ylim(10, 40)
ax.get_legend().remove()

for idx, row in df.iterrows():
 ax.annotate("(%d, %d)" % (int(row.sram_per_asic), int(row.tops_per_asic)),xy=(row.watts_per_thru, row.cost_per_thru+0.05), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('thru_pareto.pdf')
plt.close(fig)

ax = dfplot.line(df, x='watts_delay', y='cost_delay', key='sram_per_asic', sort_key=True, keyloc='upper right', xlabel='Watts_delay (W*s)', ylabel='Cost_delay($*s)', linestyle='',markersize=8)

ax.set_xlim(2.4, 3)
ax.set_ylim(4, 24)
ax.get_legend().remove()

for idx, row in df.iterrows():
 ax.annotate("(%d, %d)" % (int(row.sram_per_asic), int(row.tops_per_asic)),xy=(row.watts_delay, row.cost_delay+0.05), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('delay_pareto.pdf')
plt.close(fig)

