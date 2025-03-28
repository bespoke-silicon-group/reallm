# %%
from kernel_generation import llama, deepseek, llama70b, llama405b, deepseekv2, deepseekv3, gen_parallelism, gen_moe_parallelism

# %%
# llama3_70b_num_nodes = [8, 32, ]
# llama3_405b_num_nodes = [32, 128, ]
# deepseek_v2_num_nodes = [8, 32]
# deepseek_v3_num_nodes = [32, 128]

# all_kernel_sizes = dict()

# prefill_blocks = [64, 128, 256, 512, 1024, 2048, 4096]
# num_decode_blocks = [128]
# decode_ctxs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
#                 16384, 32768, 65536, 131072]

# model = llama70b
# all_parallelism = []
# for nodes in llama3_70b_num_nodes:
#     all_parallelism += gen_parallelism(nodes)

# for prefill_len in prefill_blocks:
#     for num_decode in num_decode_blocks:
#         if num_decode <= len(decode_ctxs):
#                 decode_lens = decode_ctxs[:num_decode]
#         else:
#             decode_lens = decode_ctxs + [decode_ctxs[-1]] * (num_decode - len(decode_ctxs))
#         for parallelism in all_parallelism:
#             kernel_sizes = model.get_kernel_sizes(prefill_len, decode_lens, parallelism)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load CSV file (Replace with actual file path)
df = pd.read_csv("H100_matmul_lat.csv")  # Ensure CSV has columns: B, M, K, N, latency

# 🚀 Fix: Strip spaces from column names
df.columns = df.columns.str.strip()

# Define fixed and varying dimensions
fixed_dims = ['B', 'M', 'K']  # Fixed dimensions
varying_dim = 'N'  # The dimension to interpolate over

# Function for linear interpolation
def interpolate_latency(group):
    group = group.sort_values(by=varying_dim)
    unique_values = group[varying_dim].values
    latencies = group['latency'].values

    if len(unique_values) < 2:
        return pd.DataFrame()  # Not enough points to interpolate

    # Linear interpolation
    linear_interp = interp1d(unique_values, latencies, kind='linear', fill_value="extrapolate")

    # Generate interpolated values for all integer N in range
    N_min, N_max = unique_values.min(), unique_values.max()
    N_new = np.arange(N_min, N_max + 1)  # All integers between min and max

    interpolated_linear = linear_interp(N_new)

    # Create DataFrame with interpolated values
    interp_df = pd.DataFrame({varying_dim: N_new, "linear_latency": interpolated_linear})

    # Add fixed dimensions
    for dim in fixed_dims:
        interp_df[dim] = group[dim].iloc[0]

    return interp_df

# Apply linear interpolation for each (B, M, K) group
interpolated_results = df.groupby(fixed_dims, group_keys=False).apply(interpolate_latency).reset_index(drop=True)

# Merge back with original dataset
df_combined = pd.concat([df, interpolated_results], ignore_index=True).sort_values(by=fixed_dims + [varying_dim])

# Save to a new CSV file
df_combined.to_csv("interpolated_matmul_results_linear.csv", index=False)

# Debugging: Print sample data
print(df_combined.head())

# %%
