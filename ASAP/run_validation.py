# %%
import sys
sys.path.append('../')
sys.path.append('llm_serving_sim/')
from llm_serving_sim.simulator import Simulator
from llm_serving_sim.hardware_sim import HardwareSim
from llm_serving_sim.hardware import Hardware, A100, H100
from llm_serving_sim.model import llama70b, llama405b, deepseekv2, deepseekv3
from llm_serving_sim.scheduler import Scheduler
import logging
sys.path.append('../LLMCompass')
from LLMCompass.design_space_exploration.dse import template_to_system, read_architecture_template
from LLMCompass.software_model.utils import Tensor, data_type_dict
from LLMCompass.software_model.matmul import *
from LLMCompass.software_model.layernorm import *
from LLMCompass.software_model.softmax import *

# %%
import json

test_hw = 'h100'

if test_hw == 'h100':
    json_name = 'h100-nvl-94gb_kernel_profile'
elif test_hw == 'a100':
    json_name = 'a100-sxm-80gb_kernel_profile'
file_name = f'real_kernel_time/{json_name}.json'

with open(file_name, 'r') as f:
    data = json.load(f)

matmul1_kernel_sizes = []
real_matmul1_times = dict()
sim_matmul1_times = dict()

matmul2_kernel_sizes = []
real_matmul2_times = dict()
sim_matmul2_times = dict()

layernorm_kernel_sizes = []
real_layernorm_times = dict()
sim_layernorm_times = dict()

softmax_kernel_sizes = []
real_softmax_times = dict()
sim_softmax_times = dict()

for key in data.keys():
    print(key)
    tests = data[key]
    for test in tests:
        # print(test)
        if 'matmul1' in key:
            M = test['m']
            K = test['k']
            N = test['n']
            matmul1_kernel_sizes.append((M, K, N))
            num_ops = 2 * M * K * N
            avg_time = test['avg_time']
            achived_tflops = num_ops / avg_time / 1e12
            print(f'MatMul1 - M: {M}, K: {K}, N: {N}, achieved TFLOPS: {achived_tflops}, avg_time: {avg_time}')
            real_matmul1_times[(M, K, N)] = avg_time
        elif 'matmul2' in key:
            M = test['m']
            K = test['k']
            N = test['n']
            matmul2_kernel_sizes.append((M, K, N))
            num_ops = 2 * M * K * N
            avg_time = test['avg_time']
            achived_tflops = num_ops / avg_time / 1e12
            print(f'MatMul2 - M: {M}, K: {K}, N: {N}, achieved TFLOPS: {achived_tflops}, avg_time: {avg_time}')
            real_matmul2_times[(M, K, N)] = avg_time
        elif 'layernorm' in key:
            M = test['m']
            N = test['n']
            if M < 1024:
                continue
            layernorm_kernel_sizes.append((M, N))
            avg_time = test['avg_time']
            print(f'LayerNorm - M: {M}, N: {N}, avg_time: {avg_time}')
            real_layernorm_times[(M, N)] = avg_time
        elif 'softmax' in key:
            M = test['m']
            N = test['n']
            if M != 1024:
                continue
            if N < 1024:
                continue
            softmax_kernel_sizes.append((M, N))
            avg_time = test['avg_time']
            print(f'Softmax - M: {M}, N: {N}, avg_time: {avg_time}')
            real_softmax_times[(M, N)] = avg_time
# %%
if test_hw == 'h100':
    out_file = 'valid_h100_matmul_lat.csv'
    hw_name = 'GH100'
elif test_hw == 'a100':
    out_file = 'valid_a100_matmul_lat.csv'
    hw_name = 'GA100'

hw_specs = read_architecture_template(f'../LLMCompass/configs/{hw_name}.json')
lc_system = template_to_system(hw_specs)
device = lc_system.device
compile_mode = 'heuristic-GPU'
data_type = data_type_dict["fp16"]

exist_matmul_sizes = set()

if os.path.exists(out_file):
    # read all existing matmul sizes
    f = open(out_file, 'r')
    lines = f.readlines()
    for line in lines[1:]:
        B, M, K, N, lat = line.strip().split(',')
        if B == '1':
            exist_matmul_sizes.add((int(M), int(K), int(N)))
            sizes = (int(M), int(K), int(N))
            if sizes in real_matmul1_times:
                sim_matmul1_times[sizes] = float(lat)
            else:
                sim_matmul2_times[sizes] = float(lat)
            # sim_matmul_times[(int(M), int(K), int(N))] = float(lat)
        else:
            exist_matmul_sizes.add((int(B), int(M), int(K), int(N)))
    f.close()
else:
    f = open(out_file, 'w')
    f.write('B, M, K, N, latency\n')
    f.close()

print(f'LLMCompass Decode Kernel Performance Simulation Begin for all kernels on {hw_name}')
# for matmul_size in all_kernel_sizes['matmul']:
matmul_kernel_sizes = matmul1_kernel_sizes + matmul2_kernel_sizes
for i, matmul_size in enumerate(matmul_kernel_sizes):
    if matmul_size in exist_matmul_sizes:
        print(f'Skipping Matmul Size {i+1}/{len(matmul_kernel_sizes)}:', matmul_size)
        continue
    print(f'Evaluate Matmul Size {i+1}/{len(matmul_kernel_sizes)}:', matmul_size)
    f = open(out_file, 'a')
    if len(matmul_size) == 3:
        M = matmul_size[0]
        K = matmul_size[1]
        N = matmul_size[2]
        mm = Matmul(data_type)
        input1 = Tensor([M, K], data_type)
        input2 = Tensor([K, N], data_type)
        _ = mm(input1, input2)
        lat = mm.compile_and_simulate(device, compile_mode)
        f.write(f'1, {M}, {K}, {N}, {lat}\n')
    elif len(matmul_size) == 4:
        B = matmul_size[0]
        M = matmul_size[1]
        K = matmul_size[2]
        N = matmul_size[3]
        mm = BatchedMatmul(data_type)
        input1 = Tensor([B, M, K], data_type)
        input2 = Tensor([B, K, N], data_type)
        _ = mm(input1, input2)
        lat = mm.compile_and_simulate(device, compile_mode)
        f.write(f'{B}, {M}, {K}, {N}, {lat}\n')
    # sim_matmul_times[matmul_size] = lat
    print(f'{lat * 1e6} us')
    f.close()

# %%
data_type = data_type_dict["fp16"]
for layernorm_size in layernorm_kernel_sizes:
    M = layernorm_size[0]
    N = layernorm_size[1]
    print(f'Evaluate LayerNorm Size:', layernorm_size)
    ln = LayerNorm(data_type)
    input1 = Tensor([M, N], data_type)
    _ = ln(input1)
    lat = ln.compile_and_simulate(device, compile_mode)
    print(f'{lat * 1e6} us')
    sim_layernorm_times[layernorm_size] = lat
for softmax_size in softmax_kernel_sizes:
    M = softmax_size[0]
    N = softmax_size[1]
    print(f'Evaluate Softmax Size:', softmax_size)
    sm = Softmax(data_type)
    input1 = Tensor([M, N], data_type)
    _ = sm(input1)
    lat = sm.compile_and_simulate(device, compile_mode)
    print(f'{lat * 1e6} us')
    sim_softmax_times[softmax_size] = lat
# %%
# compare real and sim matmul latencies
import matplotlib.pyplot as plt

# overhead = 1.0e-5
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.24, hspace=0.3)
for ax in axes:
    if ax == axes[0]:
        real_lat = list(real_matmul1_times.values())
        sim_lat = list(sim_matmul1_times.values())
        x = []
        for M, K, N in real_matmul1_times.keys():
            x.append(N)
        ax.set_title('MatMul: M=131072, K=512')
        ax.set_xlabel('N', fontsize=13)
    else:
        real_lat = list(real_matmul2_times.values())
        sim_lat = list(sim_matmul2_times.values())
        x = []
        for M, K, N in real_matmul2_times.keys():
            x.append(M)
        ax.set_title('MatMul: N=16384, K=7168')
        ax.set_xlabel('M', fontsize=13)
    # sim_lat = [lat + overhead for lat in sim_lat]
    # avg_error = sum([abs(real - sim) for real, sim in zip(real_lat, sim_lat)]) / len(real_lat)
    avg_error_ratio = sum([abs(real - sim) / real for real, sim in zip(real_lat, sim_lat)]) / len(real_lat)
    print(f'Average Error: {avg_error_ratio * 100:.2f}%')
    ax.plot(x, real_lat, 'o', label='Real Latency', linestyle='-')
    ax.plot(x, sim_lat, 'x', label='Simulated Latency', linestyle='--')
    ax.set_ylabel('Latency (s)', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

fig.savefig(f'valid_matmul.pdf', bbox_inches='tight')
# %%
# compare real and sim layernorm and softmax latencies
fig, axes = plt.subplots(1, 2, figsize=(9, 3))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.24, hspace=0.3)

for ax in axes:
    if ax == axes[0]:
        real_lat = list(real_layernorm_times.values())
        sim_lat = list(sim_layernorm_times.values())
        x = []
        for M, N in real_layernorm_times.keys():
            x.append(M)
        ax.set_title('LayerNorm: N=8192')
        ax.set_xlabel('M', fontsize=13)
        overhead = 30e-5
    else:
        real_lat = list(real_softmax_times.values())
        sim_lat = list(sim_softmax_times.values())
        x = []
        for M, N in real_softmax_times.keys():
            x.append(N)
        ax.set_title('Softmax: M=64')
        ax.set_xlabel('N', fontsize=13)
        overhead = 1.2e-5
    # sim_lat = [lat + overhead for lat in sim_lat]
    print(sim_lat)
    avg_error_ratio = sum([abs(real - sim) / real for real, sim in zip(real_lat, sim_lat)]) / len(real_lat)
    error_ratios = [abs(real - sim) / real for real, sim in zip(real_lat, sim_lat)]
    print(f'Average Error: {avg_error_ratio * 100:.2f}%')
    print(error_ratios)
    ax.plot(x, real_lat, 'o', label='Real Latency', linestyle='-')
    ax.plot(x, sim_lat, 'x', label='Simulated Latency', linestyle='--')
    ax.set_ylabel('Latency (s)', fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # change
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

fig.savefig(f'valid_layernorm_softmax.pdf', bbox_inches='tight')
# %%
# compare real and sim softmax latencies
fig, axes = plt.subplots(1, 1, figsize=(5.5, 3))
real_lat = list(real_softmax_times.values())
sim_lat = list(sim_softmax_times.values())
overhead = 1.1e-5
sim_lat = [lat + overhead for lat in sim_lat]
x = []
for M, N in real_softmax_times.keys():
    x.append(N)
ax = axes
ax.set_title('Softmax: (M=64, N)')
ax.set_xlabel('N', fontsize=13)
avg_error_ratio = sum([abs(real - sim) / real for real, sim in zip(real_lat, sim_lat)]) / len(real_lat)
print(f'Average Error: {avg_error_ratio * 100:.2f}%')
ax.plot(x, real_lat, 'o', label='Real Latency', linestyle='-')
ax.plot(x, sim_lat, 'x', label='Simulated Latency', linestyle='--')
ax.set_ylabel('Latency (s)', fontsize=13)
ax.set_xscale('log')
# ax.set_yscale('log')
ax.legend()
ax.grid(True, linestyle="--", alpha=0.6)

fig.savefig(f'valid_softmax.pdf', bbox_inches='tight')

# %%
for layernorm_size in layernorm_kernel_sizes:
    M = layernorm_size[0]
    N = layernorm_size[1]
    print(f'Evaluate LayerNorm Size:', layernorm_size)
    ln = LayerNorm(data_type)
    input1 = Tensor([M, N], data_type)
    _ = ln(input1)
    lat = ln.compile_and_simulate(device, compile_mode)
    print(f'{lat * 1e6} us')
    sim_layernorm_times[layernorm_size] = lat
print(sim_layernorm_times)
# %%
# json_name = 'h100x4-nvl-94gb-llama-3.1-70b-instruct-sharegpt-s1.0-m100'
# file_name = f'real_trace_time/{json_name}.json'
# file_to_write = f'traces/{file_name.split("/")[-1][:-5]}.csv'
# # logging.basicConfig(level=logging.DEBUG)

# scheduler_algo = 'continuous'
# max_ctx_len = 8192
# io_algo = 'multishot'
# num_nodes = 4
# model = llama70b
# parallelism = (1, 4, 1, 1) # EP, TP, PP, CP
# eval_hardware = Hardware(node=H100, 
#                             num_nodes=num_nodes,
#                             parallelism=parallelism,
#                             io_algo=io_algo,
# )

# hardware_sim = HardwareSim(
#     hardware=eval_hardware,
#     method='llmcompass',
#     scheduler_algo=scheduler_algo,
#     max_ctx_len = max_ctx_len,
# )

# scheduler = Scheduler(
#     algo=scheduler_algo,
# )

# eval_hardware.node.name = 'H100'
# hw_name = f"4H100-validation"

# sim = Simulator(
#     model = llama70b,
#     trace=file_to_write,
#     scheduler=scheduler,
#     hardware_sim=hardware_sim,
#     end_time=200,
#     start_reqs=0,
#     end_reqs=100,
# )
# sim.run()

# %%
