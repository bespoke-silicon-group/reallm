# %%
import sys
from time import time
from line_profiler import profile
import cProfile, pstats
sys.path.append("./LLMCompass")
# import LLMCompass.software_model.matmul as matmul
# import LLMCompass.software_model.utils as utils
from LLMCompass.software_model.utils import Tensor, data_type_dict
from LLMCompass.software_model.matmul import *
from LLMCompass.hardware_model.device import device_dict
from LLMCompass.software_model.fast_matmul import Matmul as FastMatmul
from math import ceil

# from LLMCompass.software_model.matmul import MatMul
# matmul_instance = MatMul()
# result = matmul_instance.multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
# print(result)
# %%

pcb = device_dict["A100_80GB_fp16"]
compile_mode="heuristic-GPU"
data_type=data_type_dict["fp16"]

K = 12288
N = K
M = 2**6

for M in range(5, 6):
    M = 2**M
    mm = Matmul(data_type)
    input1 = Tensor([M, K], data_type)
    input2 = Tensor([K, N], data_type)
    _ = mm(input1, input2)
    start = time.perf_counter()
    uarch_lat = mm.compile_and_simulate(pcb, compile_mode)
    sim_time = time.perf_counter() - start

    mm = FastMatmul(data_type)
    _ = mm(input1, input2)
    start = time.perf_counter()
    fast_uarch_lat = mm.compile_and_simulate(pcb, compile_mode)
    fast_sim_time = time.perf_counter() - start

    print(f"Matrix size {M}x{K}x{N} finished")
    print(f"uarch latency: {uarch_lat:.3e}, sim time: {sim_time:.3f}s")
    print(f"fast uarch latency: {fast_uarch_lat:.3e}, sim time: {fast_sim_time:.3f}s")
    error_rate = (fast_uarch_lat - uarch_lat) / uarch_lat
    print(f"Error rate: {error_rate:.3f}, speedup: {sim_time / fast_sim_time:.3f}")
# %%
table_name = "-918226037868254367"
l2_compute_cycle_count_table = pd.read_csv(
    f"./l2_sim_table/{table_name}.csv",
    header=None,
    names=[
        "M",
        "N",
        "K",
        "l1_tile_M",
        "l1_tile_N",
        "l1_tile_K",
        "l1_loop_order",
        "l0_M_tiling_factor",
        "l0_N_tiling_factor",
        "l0_K_tiling_factor",
        "dataflow",
        "cycle_count",
    ],
)
l2_compute_cycle_count_table.set_index(
    [
        "M",
        "N",
        "K",
        "l1_tile_M",
        "l1_tile_N",
        "l1_tile_K",
        "l1_loop_order",
        "l0_M_tiling_factor",
        "l0_N_tiling_factor",
        "l0_K_tiling_factor",
        "dataflow",
    ],
    inplace=True,
)

# Debug: Print the DataFrame's index.
print("DataFrame index names:", l2_compute_cycle_count_table.index.names)
# print("DataFrame index values:", l2_compute_cycle_count_table.index.tolist())

# Construct your key.
key = ('32', '32', '8192', '32', '32', '512', 'knm', '4', '1', '1', 'os')
# to str
print("Lookup key:", key)
print("Key type:", type(key), "Length:", len(key))

# Check if the key exists.
if key in l2_compute_cycle_count_table.index:
    cycle_count = l2_compute_cycle_count_table.loc[key, "cycle_count"]
    print("Cycle count from cache:", cycle_count)
else:
    print("Key not found in index. Running simulation...")



# %%
print(pcb.compute_module.core_count)
print(pcb.compute_module.l2_bandwidth_per_cycle)
# %%
from collections import defaultdict

def precompute_dependency(M, N, K, l1_tile_M, l1_tile_N, l1_tile_K, loop_order, core_count):
    M_l1_t = M // l1_tile_M
    N_l1_t = N // l1_tile_N
    K_l1_t = K // l1_tile_K
    M_remain = M % l1_tile_M
    N_remain = N % l1_tile_N
    K_remain = K % l1_tile_K

    ceil_M_l1_t = ceil(M / l1_tile_M)
    ceil_N_l1_t = ceil(N / l1_tile_N)
    ceil_K_l1_t = ceil(K / l1_tile_K)

    M_K_tile_size = np.zeros(
        [ceil_M_l1_t, ceil_K_l1_t], dtype=int
    )
    M_K_tile_size[:M_l1_t, :K_l1_t] = l1_tile_M * l1_tile_K
    if M_remain > 0:
        M_K_tile_size[-1, :K_l1_t] = M_remain * l1_tile_K
    if K_remain > 0:
        M_K_tile_size[:M_l1_t, -1] = l1_tile_M * K_remain
    if M_remain > 0 and K_remain > 0:
        M_K_tile_size[-1, -1] = M_remain * K_remain

    K_N_tile_size = np.zeros(
        [ceil_K_l1_t, ceil_N_l1_t], dtype=int
    )
    K_N_tile_size[:K_l1_t, :N_l1_t] = l1_tile_K * l1_tile_N
    if K_remain > 0:
        K_N_tile_size[-1, :N_l1_t] = K_remain * l1_tile_N
    if N_remain > 0:
        K_N_tile_size[:K_l1_t, -1] = l1_tile_K * N_remain
    if K_remain > 0 and N_remain > 0:
        K_N_tile_size[-1, -1] = K_remain * N_remain

    M_N_tile_size = np.zeros(
        [ceil_M_l1_t, ceil_N_l1_t], dtype=int
    )
    M_N_tile_size[:M_l1_t, :N_l1_t] = l1_tile_M * l1_tile_N
    if M_remain > 0:
        M_N_tile_size[-1, :N_l1_t] = M_remain * l1_tile_N
    if N_remain > 0:
        M_N_tile_size[:M_l1_t, -1] = l1_tile_M * N_remain
    if M_remain > 0 and N_remain > 0:
        M_N_tile_size[-1, -1] = M_remain * N_remain

    total_cycle_count = 0
    previous_batch_Read_M_K = np.zeros(
        [ceil_M_l1_t, ceil_K_l1_t], dtype=bool
    )
    previous_batch_Read_K_N = np.zeros(
        [ceil_K_l1_t, ceil_N_l1_t], dtype=bool
    )
    previous_batch_Read_M_N = np.zeros(
        [ceil_M_l1_t, ceil_N_l1_t], dtype=bool
    )
    previous_batch_Write_M_N = np.zeros(
        [ceil_M_l1_t, ceil_N_l1_t], dtype=bool
    )
    previous_batch_compute_cycle_count = 0


    active_l1_tile_list = []
    for m, n, k in Matmul.generate_tile_loops(ceil_M_l1_t, ceil_N_l1_t, ceil_K_l1_t, loop_order):

        active_l1_tile_list.append((m, n, k))
        if (
            m == ceil_M_l1_t - 1
            and n == ceil_N_l1_t - 1
            and k == ceil_K_l1_t - 1
        ):
            pass
        elif (
            len(active_l1_tile_list) < core_count
        ):
            continue

        assert (
            len(active_l1_tile_list) <= core_count
        )
        current_batch_Read_M_K = np.zeros(
            [ceil(M / l1_tile_M), ceil(K / l1_tile_K)], dtype=bool
        )
        current_batch_Read_K_N = np.zeros(
            [ceil(K / l1_tile_K), ceil(N / l1_tile_N)], dtype=bool
        )
        current_batch_Read_M_N = np.zeros(
            [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
        )
        current_batch_Write_M_N = np.zeros(
            [ceil(M / l1_tile_M), ceil(N / l1_tile_N)], dtype=bool
        )

        current_batch_compute_cycle_count = 0
        for i in range(len(active_l1_tile_list)):
            temp_m, temp_n, temp_k = active_l1_tile_list[i]
            current_batch_Read_M_K[temp_m, temp_k] = 1
            current_batch_Read_K_N[temp_k, temp_n] = 1
            current_batch_Read_M_N[temp_m, temp_n] = temp_k > 0
            current_batch_Write_M_N[temp_m, temp_n] = 1

        # if one output tile in this batch shares input/output with another output tile in the previous batch, assign them to the same core to avoid data movement
        # note that of the three input matrix mk, kn, mn, at most one of them can be the same if we change m,n,k
        current_batch_M_K_read_count = np.sum(
            (current_batch_Read_M_K * (~previous_batch_Read_M_K))
            * M_K_tile_size
        )
        current_batch_K_N_read_count = np.sum(
            (current_batch_Read_K_N * (~previous_batch_Read_K_N))
            * K_N_tile_size
        )
        current_batch_M_N_read_count = np.sum(
            (
                current_batch_Read_M_N
                * (~(previous_batch_Read_M_N + previous_batch_Write_M_N))
            )
            * M_N_tile_size
        )
        previous_batch_M_N_write_count = np.sum(
            (previous_batch_Write_M_N * (~current_batch_Read_M_N))
            * M_N_tile_size
        )

        # read current batch while compute and write previous batch
        current_batch_read_count = (
            current_batch_M_K_read_count
            + current_batch_K_N_read_count
            + current_batch_M_N_read_count
        )
        # current_batch_read_cycle_count = ceil(
        #     current_batch_read_count
        #     * chiplet_module.compute_module.core.systolic_array.input_word_size
        #     / chiplet_module.compute_module.l2_bandwidth_per_cycle
        # )
        # prvious_batch_write_cycle_count = ceil(
        #     previous_batch_M_N_write_count
        #     * chiplet_module.compute_module.core.systolic_array.output_word_size
        #     / chiplet_module.compute_module.l2_bandwidth_per_cycle
        # )

        total_cycle_count += (
            max(
                current_batch_read_cycle_count,
                previous_batch_compute_cycle_count,
            )
            + prvious_batch_write_cycle_count
        )

        previous_batch_compute_cycle_count = current_batch_compute_cycle_count
        previous_batch_Read_M_K = copy.deepcopy(current_batch_Read_M_K)
        previous_batch_Read_K_N = copy.deepcopy(current_batch_Read_K_N)
        previous_batch_Read_M_N = copy.deepcopy(current_batch_Read_M_N)
        previous_batch_Write_M_N = copy.deepcopy(current_batch_Write_M_N)

        active_l1_tile_list = []

    
    transition_counts = defaultdict(int)
    for i in range(len(ordered_keys) - 1):
        transition = (ordered_keys[i], ordered_keys[i+1])
        transition_counts[transition] += 1
    
    # For each transition, compute an approximate cost.
    # This cost function might be based on a fixed reuse factor.
    reuse_factor = 0.5  # For example, assume 50% data reuse.
    transition_costs = {}
    for (key1, key2), count in transition_counts.items():
        # Assume a function that computes the full cost for a given tile.
        full_cost = compute_full_tile_read_write_cost(key1)
        if key1 == key2:
            cost = 0
        else:
            cost = (1 - reuse_factor) * full_cost
        transition_costs[(key1, key2)] = (cost, count)
    
    return {
        "ordered_keys": ordered_keys,
        "transition_costs": transition_costs,
    }
# %%
K = 12288
N = K
sim_times = 1

roofline_lats = dict()
uarch_lats = dict()
uarch_runtimes = dict()

for M in range(10, 12):
    M = 2**M
    mm = Matmul(data_type)
    input1 = Tensor([M, K], data_type)
    input2 = Tensor([K, N], data_type)
    _ = mm(input1, input2)

    roofline_lat = mm.roofline_model(pcb)

    run_time = []
    for i in range(sim_times):
        start = time.perf_counter()
        uarch_lat = mm.compile_and_simulate(pcb, compile_mode)
        end = time.perf_counter()
        run_time.append(end - start)

    key = f"{M}x{K}x{N}"
    avg_run_time = sum(run_time) / len(run_time)

    roofline_lats[key] = roofline_lat
    uarch_lats[key] = uarch_lat
    uarch_runtimes[key] = avg_run_time
    print(f"Matrix size {key} finished")
    print(f"Roofline latency: {roofline_lat:.2e}")
    print(f"uarch latency: {uarch_lat:.2e}, avg runtime: {avg_run_time:.2f}s")



# %%
cProfile.run("mm.compile_and_simulate(pcb, compile_mode)", "profile.out")
p = pstats.Stats("profile.out")
p.sort_stats("cumtime").print_stats(20)
# %%



# Example parameters (replace with your actual values)
# These should be based on your L1 tiling sizes.
K = 12288
N = K
M = 2**10

l1_tile_M = 32
l1_tile_N = 32
l1_tile_K = 16

# Calculate the number of tiles along each dimension.
num_tiles_M = ceil(M / l1_tile_M)
num_tiles_N = ceil(N / l1_tile_N)
num_tiles_K = ceil(K / l1_tile_K)

# Choose a loop order, for example 'mnk'
loop_order = 'mnk'

# Get the indices using your generate_tile_loops function
indices = list(Matmul.generate_tile_loops(num_tiles_M, num_tiles_N, num_tiles_K, loop_order))

print("Total number of (m, n, k) indices:", len(indices))
print("First 10 indices:", indices[:10])
# find unique indices
unique_indices = set(indices)
print("Total number of unique (m, n, k) indices:", len(unique_indices))

# %%
