import logging
from scheduler import SimKernel, LLMKernel
from hardware import HardwareNode, Hardware
from model import KernelSizes, llama, deepseek
import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d, LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import KDTree

class Performance:
    def __init__(self, 
                 kernel: SimKernel,
                 hardware: Hardware,
                 method: str,
                 prefill_attn_mem_latency: float = 0.0,
                 prefill_attn_compute_latency: float = 0.0,
                 prefill_fc_mem_latency: float = 0.0,
                 prefill_fc_compute_latency: float = 0.0,
                 prefill_io_latency: float = 0.0,
                 decode_attn_mem_latency: float = 0.0,
                 decode_attn_compute_latency: float = 0.0,
                 decode_fc_mem_latency: float = 0.0,
                 decode_fc_compute_latency: float = 0.0,
                 decode_io_latency: float = 0.0):
        self.kernel = kernel
        self.hardware = hardware
        self.method = method
        self.prefill_attn_mem_latency = prefill_attn_mem_latency
        self.prefill_fc_mem_latency = prefill_fc_mem_latency
        self.prefill_attn_compute_latency = prefill_attn_compute_latency
        self.prefill_fc_compute_latency = prefill_fc_compute_latency
        self.decode_attn_mem_latency = decode_attn_mem_latency
        self.decode_fc_mem_latency = decode_fc_mem_latency
        self.decode_attn_compute_latency = decode_attn_compute_latency
        self.decode_fc_compute_latency = decode_fc_compute_latency

        self.prefill_attn_latency = prefill_attn_mem_latency + prefill_attn_compute_latency
        if prefill_attn_mem_latency > prefill_attn_compute_latency:
            self.prefill_attn_bottleneck = 'mem'
        else:
            self.prefill_attn_bottleneck = 'compute'
        self.prefill_fc_latency = max(prefill_fc_mem_latency, prefill_fc_compute_latency)
        if prefill_fc_mem_latency > prefill_fc_compute_latency:
            self.prefill_fc_bottleneck = 'mem'
        else:
            self.prefill_fc_bottleneck = 'compute'
        self.prefill_latency = self.prefill_attn_latency + self.prefill_fc_latency + prefill_io_latency
        self.prefill_io_latency = prefill_io_latency
        if prefill_io_latency > max(self.prefill_attn_latency, self.prefill_fc_latency):
            self.prefill_bottleneck = 'io'
        elif self.prefill_attn_latency > self.prefill_fc_latency:
            self.prefill_bottleneck = self.prefill_attn_bottleneck
        else:
            self.prefill_bottleneck = self.prefill_fc_bottleneck

        self.decode_attn_latency = max(decode_attn_mem_latency, decode_attn_compute_latency)
        if decode_attn_mem_latency > decode_attn_compute_latency:
            self.decode_attn_bottleneck = 'mem'
        else:
            self.decode_attn_bottleneck = 'compute'
        self.decode_fc_latency = max(decode_fc_mem_latency, decode_fc_compute_latency)
        if decode_fc_mem_latency > decode_fc_compute_latency:
            self.decode_fc_bottleneck = 'mem'
        else:
            self.decode_fc_bottleneck = 'compute'
        self.decode_io_latency = decode_io_latency
        self.decode_latency = self.decode_attn_latency + self.decode_fc_latency + decode_io_latency
        if decode_io_latency > max(self.decode_attn_latency, self.decode_fc_latency):
            self.decode_bottleneck = 'io'
        elif self.decode_attn_latency > self.decode_fc_latency:
            self.decode_bottleneck = self.decode_attn_bottleneck
        else:
            self.decode_bottleneck = self.decode_fc_bottleneck
        
        self.latency = self.prefill_latency + self.decode_latency

class HardwareSim:
    def __init__(self, hardware, method = 'roofline', scheduler_algo = 'baseline', max_ctx_len: int = 1):
        self.hardware = hardware
        self.method = method
        self.scheduler_algo = scheduler_algo
        self.max_ctx_len = max_ctx_len

        self.kernels_perf = []
        self.accept_new_request = True
    
    def run(self, sim_kernel: SimKernel) -> tuple:
        hw_flops = self.hardware.flops
        hw_mem_bw = self.hardware.mem_bw
        hw_mem_size = self.hardware.mem_size

        # check if the kernel fits in the memory
        # model_byte = sim_kernel.model.model_size_byte
        if sim_kernel.prefill_kernel is not None:
            num_tokens = sim_kernel.prefill_kernel.n
            num_reqs = 1
        else:
            num_tokens = 0
            num_reqs = 0
        if sim_kernel.decode_kernel is not None:
            for ctx_len in sim_kernel.decode_kernel.ctx:
                num_tokens += ctx_len 
                num_reqs += 1
        # max_kv_cache_byte = sim_kernel.model.kv_cache_size_per_token_byte * self.max_ctx_len * num_reqs
        # if model_byte + max_kv_cache_byte >= hw_mem_size:
        #     if self.accept_new_request:
        #         logging.debug(f"Model size {model_byte / 1e9:.2f}GB + kv cache size {max_kv_cache_byte / 1e9:.2f}GB > HBM size {hw_mem_size / 1e9:.2f}GB")
        #         logging.debug(f"decode {sim_kernel.decode_kernel.n} tasks")
        #         logging.debug('Rejecting new request')
        #     self.accept_new_request = False
        # else:
        #     if self.accept_new_request == False:
        #         logging.debug(f"Model size {model_byte / 1e9:.2f}GB + kv cache size {max_kv_cache_byte / 1e9:.2f}GB < HBM size {hw_mem_size / 1e9:.2f}GB")
        #         logging.debug(f"decode {sim_kernel.decode_kernel.n} tasks")
        #         logging.debug('Accepting new request')
        #     self.accept_new_request = True

        # kv_cache_byte = sim_kernel.model.kv_cache_size_per_token_byte * num_tokens
        # if model_byte + kv_cache_byte > hw_mem_size:
        #     logging.info(f"Model size {model_byte / 1e9:.2f}GB + kv cache size {kv_cache_byte / 1e9:.2f}GB > HBM size {hw_mem_size / 1e9:.2f}GB")
        #     logging.info(f"Prefill n {sim_kernel.prefill_kernel.n}, decode n {sim_kernel.decode_kernel.n}, num_tokens {num_tokens}")
        #     raise ValueError(f"Model size {model_byte / 1e9:.2f}GB + kv cache size {kv_cache_byte / 1e9:.2f}GB > HBM size {hw_mem_size / 1e9:.2f}GB")

        if self.method == 'llmcompass':
            model = sim_kernel.model
            hw_name = self.hardware.node.name
            num_nodes = self.hardware.num_nodes
            parallelism = self.hardware.parallelism

            if sim_kernel.prefill_kernel == None and sim_kernel.decode_kernel == None:
                raise ValueError(f"Both prefill and decode kernels are None")
            elif sim_kernel.prefill_kernel == None:
                prefill_len = 0
                decode_lens = sim_kernel.decode_kernel.ctx
            elif sim_kernel.decode_kernel == None:
                prefill_len = sim_kernel.prefill_kernel.n
                decode_lens = []
            else:
                prefill_len = sim_kernel.prefill_kernel.n
                decode_lens = sim_kernel.decode_kernel.ctx
            
            if prefill_len > 0 and len(decode_lens) > 0 and self.scheduler_algo == 'continuous':
                raise ValueError(f"Prefill len {prefill_len} > 0 and decode lens {decode_lens} > 0 in continuous scheduler")
            

            if self.scheduler_algo == 'baseline':
                # prefill_kernel_sizes = model.get_kernel_sizes(prefill_len, [], parallelism)
                # decode_kernel_sizes = model.get_kernel_sizes(0, decode_lens, parallelism)
                # latency = find_kernel_latency(hw_name, prefill_kernel_sizes) + find_kernel_latency(hw_name, decode_kernel_sizes)
                kernel_sizes = model.get_kernel_sizes(prefill_len, decode_lens, parallelism)
                latency = find_kernel_latency(hw_name, kernel_sizes)
            elif 'continuous' in self.scheduler_algo or 'mixed-sarathi' in self.scheduler_algo:
                kernel_sizes = model.get_kernel_sizes(prefill_len, decode_lens, parallelism)
                latency = find_kernel_latency(hw_name, kernel_sizes)
            else:
                raise ValueError(f"Unknown scheduler algorithm {self.scheduler_algo}")
            
            # Add IO latency
            E, T, P, C = parallelism
            num_layers = model.num_layers
            # Allreduce latency
            if 'llama' in model.name:
                num_bytes = 2 * (prefill_len + len(decode_lens)) * model.d_model
                allreduce_latency = self.hardware.get_allreduce_latency(num_bytes, T)
            else:
                if T != 1 or C != 1:
                    raise ValueError(f"Unsupported parallelism {parallelism} for model {model.name}")
                allreduce_latency = 0.0

            io_latency = (2 * allreduce_latency * math.ceil(num_layers / P))

            latency += io_latency

            # print(f"prefill_len {prefill_len}, decode_lens {decode_lens}, io_latency {io_latency}, latency {latency}")
                 
        else:
            raise ValueError(f"Unknown method {self.method}")
        

        # logging.debug(f"HardwareSim:")
        # if sim_kernel.prefill_kernel is not None:
        #     logging.debug(f"             Prefill layer {sim_kernel.prefill_kernel.l_start} to {sim_kernel.prefill_kernel.l_end}")
        #     logging.debug(f"             Prefill n {sim_kernel.prefill_kernel.n}")
        # if sim_kernel.decode_kernel is not None:
        #     logging.debug(f"             Decode layer {sim_kernel.decode_kernel.l_start} to {sim_kernel.decode_kernel.l_end}")
        #     logging.debug(f"             Decode n {sim_kernel.decode_kernel.n}")
        # logging.debug(f"             latency {latency}")
        return latency, self.accept_new_request

def find_kernel_latency(hw_name, kernel_sizes):
    total_latency = 0
    for kernel_type in kernel_sizes.keys():
        csv_file = f'{hw_name}_{kernel_type}_lat.csv'
        lat = batch_interpolate_latency(csv_file, kernel_sizes[kernel_type].kernel_sizes)
        # print(f"{kernel_type} latency: {lat:.3e} s")
        total_latency += lat
    return  total_latency



# Function to compute Hamming distance (number of differing dimensions)
def hamming_distance(shape1, shape2):
    """Computes Hamming distance (number of differing dimensions)."""
    return sum(a != b for a, b in zip(shape1, shape2))

# Function to compute sum of absolute differences
def sum_absolute_difference(shape1, shape2):
    """Computes the sum of absolute differences across all dimensions."""
    return sum(abs(a - b) for a, b in zip(shape1, shape2))

# Function to find the best two shapes for interpolation
def find_best_shapes(df, query_shape, fixed_keys):
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
    # best_shapes = [shape for _, shape in abs_diffs[:2]]  # Take at most 2
    if len(abs_diffs) >= 5:
        best_shapes = [shape for _, shape in abs_diffs[:5]]
    else:
        best_shapes = [shape for _, shape in abs_diffs[:]]

    return best_shapes if len(best_shapes) >= 2 else None  # Return only if we have two valid points


def find_nearest_neighbor(query_shape, existing_shapes, values):
    tree = KDTree(existing_shapes)  # Create KD-tree for fast nearest neighbor search
    _, nearest_idx = tree.query(query_shape)  # Find nearest point
    return values[nearest_idx]  # Return nearest latency

# Function to detect varying dimensions
def detect_varying_dims(query_shape, best_shapes):
    """
    Identifies which dimensions are changing across the best_shapes.
    Returns a list of indices of varying dimensions.
    """
    return [idx for idx in range(len(query_shape)) if len(set(shape[idx] for shape in best_shapes)) > 1]

# Perform either 1D or 2D interpolation based on available points
def interpolate_latency(query_shape, best_shapes, latencies, dimension_keys):
    """
    Dynamically selects 1D or 2D interpolation based on the number of varying dimensions.
    - If one dimension varies → Use `interp1d()` (1D interpolation).
    - If two dimensions vary and we have at least 3 points → Use `LinearNDInterpolator()` (2D interpolation).
    - If more than two dimensions vary, return the closest available latency.
    """
    varying_dims = detect_varying_dims(query_shape, best_shapes)

    if len(varying_dims) == 1:
        # 1D Interpolation (Single Dimension Varies)
        varying_idx = varying_dims[0]
        shape_values = [shape[varying_idx] for shape in best_shapes]
        query_value = query_shape[varying_idx]

        # Ensure at least two unique values
        if len(set(shape_values)) < 2:
            return latencies[0]  # Return the closest known value

        # 1D Interpolation
        interp_func = interp1d(shape_values, latencies, kind='linear', fill_value="extrapolate")
        return interp_func(query_value)

    elif len(varying_dims) == 2 and len(best_shapes) >= 3:
        # 2D Interpolation (Two Dimensions Vary, at least 3 points needed)
        dim1, dim2 = varying_dims
        points = np.array([[shape[dim1], shape[dim2]] for shape in best_shapes])  # (x, y) coordinates
        values = np.array(latencies)  # Corresponding latencies

        # 2D Interpolation using LinearNDInterpolator
        interp_func = LinearNDInterpolator(points, values)
        query_point = np.array([query_shape[dim1], query_shape[dim2]])
        latency_interpolated =  interp_func(query_point)
        if np.isnan(latency_interpolated):
            # print(f"⚠ Warning: {query_point} is outside convex hull. Using nearest neighbor instead.")
            nearest_interp = NearestNDInterpolator(points, values)
            latency_interpolated = nearest_interp(query_point)
        return np.squeeze(latency_interpolated)

    else:
        print(f"⚠ Warning: More than 2 dimensions vary or not enough points. Returning closest match.")
        return latencies[0]  # Return the closest known value


# Function for batch interpolation
def batch_interpolate_latency(csv_name, kernel_sizes, verbose=False):
    """
    Processes multiple queries and interpolates latency values for them.
    If an exact match is found, return it directly.
    """
    df = pd.read_csv(csv_name)  # Load the dataset

    # Load the dataset
    df.columns = df.columns.str.strip()  # Clean column names

    dimension_keys = list(df.columns[:-1])  # Extract dimension keys
    df = df.drop_duplicates(subset=dimension_keys, keep='last')  # Drop duplicates

    total_latency = 0

    if 'matmul' in csv_name:
        overhead = 1.0e-5
        # print(f"Overhead for matmul: {overhead}")
    else:
        overhead = 0.3e-5
        # print(f"Overhead for others: {overhead}")
    # elif 'softmax' in csv_name or 'mul' in csv_name or 'silu' in csv_name:
    #     overhead = 1.2e-5
    #     # print(f"Overhead for softmax: {overhead}")
    # elif 'layernorm' in csv_name:
    #     overhead = 4.5e-5
        # print(f"Overhead for others: {overhead}")

    for shape, freq in kernel_sizes.items():
        if len(dimension_keys) == 4: # Matmul
            if len(shape) != 4:
                B = 1
                M, K, N = shape
            else:
                B, M, K, N = shape
            query_shape = (B, M, K, N)
            exact_match = df[(df['B'] == B) & (df['M'] == M) & 
                             (df['K'] == K) & (df['N'] == N)]
        elif len(dimension_keys) == 2: # others
            if len(shape) == 2:
                M, N = shape
            elif len(shape) == 3:
                M = shape[0] * shape[1]
                N = shape[2]
            else:
                raise ValueError(f"Unsupported shape: {shape}")
            query_shape = (M, N)
            exact_match = df[(df['M'] == M) & (df['N'] == N)]
        else:
            raise ValueError(f"Unsupported dimension keys: {dimension_keys}")

        if not exact_match.empty:
            latency_exact = exact_match['latency'].values[0]
            total_latency += latency_exact * freq
            if verbose:
                print(f"Exact match found for {query_shape}: {latency_exact:.3e} s")
            # results.append((B, M, K, N, latency_exact))
            continue

        # Find the best two shapes
        best_shapes = find_best_shapes(df, query_shape, dimension_keys)

        # If no valid shapes found, return None
        if not best_shapes:
            # results.append((B, M, K, N, None))
            # raise ValueError(f"Not enough valid data for interpolation: {query_shape}")
            nearest_latency = find_nearest_neighbor(query_shape, df[dimension_keys].values, df['latency'].values)
            total_latency += (nearest_latency + overhead) * freq
            # print(f"Not enough valid data for interpolation: {query_shape}. Using nearest neighbor: {nearest_latency:.3e} s")
            continue

        # Extract latencies
        latencies = []
        for shape in best_shapes:
            if len(dimension_keys) == 4:
                latency = df[(df['B'] == shape[0]) & (df['M'] == shape[1]) &
                             (df['K'] == shape[2]) & (df['N'] == shape[3])]['latency'].values
            elif len(dimension_keys) == 3:
                latency = df[(df['B'] == shape[0]) & (df['M'] == shape[1]) &
                             (df['N'] == shape[2])]['latency'].values
            elif len(dimension_keys) == 2:
                latency = df[(df['M'] == shape[0]) & (df['N'] == shape[1])]['latency'].values

            if len(latency) > 0:
                latencies.append(latency[0])

        # If only one valid point, return its latency
        if len(latencies) == 1:
            print(f"Only one close match found for {query_shape}: {latencies[0]:.3e} s")
            # results.append((B, M, K, N, latencies[0]))
            total_latency += (latencies[0] + overhead) * freq
            continue

        latency_interpolated = interpolate_latency(query_shape, best_shapes,latencies, dimension_keys)

        # Perform interpolation on the detected varying dimension
        # interp_func = interp1d(shape_values, latencies, kind='linear', fill_value="extrapolate")
        # query_value = query_shape[varying_idx]
        # latency_interpolated = interp_func(query_value)

        if verbose:
            print(f"Interpolated latency for {query_shape}: {latency_interpolated:.3e} s (Using {best_shapes} with latency {latencies})")
            
        # results.append((B, M, K, N, latency_interpolated))
        total_latency += (latency_interpolated + overhead) * freq
    
    # return the sum of latencies
    return total_latency
