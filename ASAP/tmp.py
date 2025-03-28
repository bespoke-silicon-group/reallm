# %%
import numpy as np
from collections import Counter
from typing import List, Tuple
import math
import os, sys
sys.path.append('../')
sys.path.append('../LLMCompass')
from LLMCompass.design_space_exploration.dse import template_to_system, read_architecture_template
from LLMCompass.software_model.utils import Tensor, data_type_dict
from LLMCompass.software_model.matmul import *
# %%
# %%
# read all kernel sizes
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
df = pd.read_csv("H100_matmul_lat.csv")

# Load the dataset
df.columns = df.columns.str.strip()  # Clean column names

df = df.drop_duplicates(subset=['B', 'M', 'K', 'N'], keep='last')  # Drop duplicates

# Define keys (B, M, K, N)
fixed_keys = ['B', 'M', 'K', 'N']

# Function to compute Hamming distance (number of differing dimensions)
def hamming_distance(shape1, shape2):
    """Computes Hamming distance (number of differing dimensions)."""
    return sum(a != b for a, b in zip(shape1, shape2))

# Function to compute sum of absolute differences
def sum_absolute_difference(shape1, shape2):
    """Computes the sum of absolute differences across all dimensions."""
    return sum(abs(a - b) for a, b in zip(shape1, shape2))

# Function to find the best two shapes for interpolation
def find_best_shapes(query_shape):
    """
    Finds the best two closest shapes:
    - First determines which dimensions are fixed by **minimum Hamming distance**.
    - Filters the dataset to only include rows where those dimensions match the query.
    - Within this subset, finds the two closest shapes using **sum of absolute differences**.
    """
    existing_shapes = df[fixed_keys].values  # Extract all existing (B, M, K, N) shapes

    # Compute Hamming distances
    distances = [(hamming_distance(query_shape, tuple(shape)), tuple(shape)) for shape in existing_shapes]

    # Find the **minimum Hamming distance** (fewest differing dimensions)
    min_hamming_dist = min(d[0] for d in distances)
    
    # Identify the first shape with this minimal Hamming distance
    best_match_shape = next(d[1] for d in distances if d[0] == min_hamming_dist)

    # Determine which dimensions are fixed
    fixed_dimensions = [idx for idx, (q, b) in enumerate(zip(query_shape, best_match_shape)) if q == b]

    # Filter dataset to only include rows that match the query in fixed dimensions
    subset_df = df.copy()
    for idx, col in enumerate(fixed_keys):
        if idx in fixed_dimensions:  # Only keep rows where fixed dimensions match
            subset_df = subset_df[subset_df[col] == query_shape[idx]]

    # Extract subset shapes
    subset_shapes = subset_df[fixed_keys].values

    if len(subset_shapes) < 2:
        return None  # Not enough points to interpolate

    # Compute sum of absolute differences within the subset
    abs_diffs = [(sum_absolute_difference(query_shape, shape), shape) for shape in subset_shapes]
    
    # Sort by absolute difference
    abs_diffs.sort(key=lambda x: x[0])

    # Select the closest two shapes
    best_shapes = [shape for _, shape in abs_diffs[:2]]  # Take at most 2

    return best_shapes if len(best_shapes) == 2 else None  # Return only if we have two valid points


# Function for batch interpolation
def batch_interpolate_latency(queries):
    """
    Processes multiple queries and interpolates latency values for them.
    If an exact match is found, return it directly.
    """
    results = []

    for B, M, K, N in queries:
        query_shape = (B, M, K, N)

        # Check if exact match exists
        exact_match = df[(df['B'] == B) & (df['M'] == M) & 
                         (df['K'] == K) & (df['N'] == N)]
        if not exact_match.empty:
            latency_exact = exact_match['latency'].values[0]
            print(f"Exact match found for {query_shape}: {latency_exact:.2f} ms")
            results.append((B, M, K, N, latency_exact))
            continue

        # Find the best two shapes
        best_shapes = find_best_shapes(query_shape)

        # If no valid shapes found, return None
        if not best_shapes:
            print(f"Not enough valid data for interpolation: {query_shape}")
            results.append((B, M, K, N, None))
            continue

        varying_idx = next(i for i, (q, s1, s2) in enumerate(zip(query_shape, best_shapes[0], best_shapes[1])) if s1 != s2)

        # Extract latencies
        latencies = []
        shape_values = []
        for shape in best_shapes:
            latency = df[(df['B'] == shape[0]) & (df['M'] == shape[1]) &
                         (df['K'] == shape[2]) & (df['N'] == shape[3])]['latency'].values
            if len(latency) > 0:
                latencies.append(latency[0])
                shape_values.append(shape[varying_idx]) # Use the varying dimension for interpolation

        # If only one valid point, return its latency
        if len(latencies) == 1:
            print(f"Only one close match found for {query_shape}: {latencies[0]:.2f} ms")
            results.append((B, M, K, N, latencies[0]))
            continue

        # Perform interpolation on the detected varying dimension
        interp_func = interp1d(shape_values, latencies, kind='linear', fill_value="extrapolate")
        query_value = query_shape[varying_idx]
        latency_interpolated = interp_func(query_value)

        print(f"Interpolated latency for {query_shape}: {latency_interpolated:.2f} ms (Using {best_shapes})")
        results.append((B, M, K, N, latency_interpolated))

    return results

# Example Queries
queries = [
    (1, 8, 128, 1000),
    (1, 8, 512, 5000),
    (2, 16, 256, 700),
    (3, 32, 512, 15000)  # Adjust queries based on your dataset
]

# Run batch interpolation
interpolated_results = batch_interpolate_latency(queries)
print(interpolated_results)

# # Convert results to DataFrame for saving or analysis
# df_results = pd.DataFrame(interpolated_results, columns=['B', 'M', 'K', 'N', 'Interpolated_Latency'])

# # Save results to a CSV file
# df_results.to_csv("interpolated_results.csv", index=False)

# # Display a preview
# print(df_results)

# %%
hamming_distance((1, 8, 128, 1), (1, 8, 128, 1000))