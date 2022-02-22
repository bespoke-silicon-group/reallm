import utils
import dfplot
import matplotlib.pyplot as plt 


csvfile = 'results.csv'
df = utils.csv2df(csvfile)

#  df2 = df[['tech_node', 'sram_per_asic']]

#  group_keys=['sram_per_asic', 'tops_per_asic']
group_keys='sram_per_asic'

ax = dfplot.line(df, x='watts_per_tops', y='cost_per_tops', key=None, sort_key=True, keyloc='upper right', xlabel='W/TOps/s', ylabel='\$/TOps/s', linestyle='',markersize=8)

ax.set_xlim(1.5, 1.7)
ax.set_ylim(2.8, 7.4)
ax.get_legend().remove()

for idx, row in df.iterrows():
  ax.annotate("(%d, %.1f)" % (int(row.sram_per_asic), float(row.tops_per_asic)),xy=(row.watts_per_tops, row.cost_per_tops+0.05), xycoords='data', size=5, va='center', ha='center')

fig = ax.get_figure()
fig.savefig('test.pdf')
plt.close(fig)


