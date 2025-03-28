# %%
import matplotlib.pyplot as plt
import numpy as np

# Original simulation time (in minutes)
# original_time = 1e4
original_time = (43779/60.00/1600)*10000

# Our method
# 1000 -> 29527s
# 1500 -> 31348s
# 1600 -> 43779s
library_build_time = 43779/60.0  # Convert hours to minutes
new_simulation_time = 27.85

speedup = original_time / new_simulation_time

total_new_time = library_build_time + new_simulation_time

# Data for plotting
categories = ['Baseline', 'ReaLLM Kernel\n Build Time', 'ReaLLM Trace\n Simulation Time']
times = [original_time, library_build_time, new_simulation_time]  # Scale original time

# Create bar chart
plt.figure(figsize=(6, 3))
plt.bar(categories, times, color=['tab:orange', 'grey', 'green'], alpha=0.7, edgecolor='black')
plt.ylabel('Time (minutes)')
plt.title('Simulation Time Speedup with ReaLLM')
plt.yscale('log')

# Dashed line for kernel build time
plt.bar(1, library_build_time, color='grey', hatch='//', alpha=0.5)

# Arrow to indicate speedup
plt.annotate('', xy=(2, new_simulation_time*1.6), xytext=(0, times[0]),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
plt.text(1.9, times[0] / 25, f'{speedup:.1f}x Speedup', color='red', fontsize=12, ha='center')

# Annotate values
# for i, time in enumerate(times):
plt.text(0, times[0] * 1.0, f'${original_time:.1f}$ min (est.)', ha='center', fontsize=12)
plt.text(1, times[1] * 1.08, f'{library_build_time:.1f} min', ha='center', fontsize=12)
plt.text(2, times[2] * 1.08, f'{new_simulation_time:.1f} min', ha='center', fontsize=12)
#set y-limit
plt.ylim(1e1, 2e4)

plt.grid(axis='y', linestyle='--', alpha=0.7)

# save to pdf
plt.savefig('speedup.pdf', bbox_inches='tight')

# %%
