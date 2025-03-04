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
# read all kernel sizes
eval_kernel_types = ['matmul', 'softmax', 'mul', 'layernorm', 'silu']
all_kernel_sizes = {}
for kernel_type in eval_kernel_types:
    all_kernel_sizes[kernel_type] = []
    # with open(f'{kernel_type}_sizes.txt', 'r') as f:
    with open(f'parts_{kernel_type}_sizes.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            size = line.strip().split(',')
            all_kernel_sizes[kernel_type].append(tuple(map(int, size)))
# %%
out_file = 'matmul_lat.csv'
hw_name = 'GH100'
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
        B, M, K, N, _ = line.strip().split(',')
        if B == '1':
            exist_matmul_sizes.add((int(M), int(K), int(N)))
        else:
            exist_matmul_sizes.add((int(B), int(M), int(K), int(N)))
    f.close()
else:
    f = open(out_file, 'w')
    f.write('B, M, K, N, latency\n')
    f.close()

print(f'LLMCompass Decode Kernel Performance Simulation Begin for all kernels on {hw_name}')
# for matmul_size in all_kernel_sizes['matmul']:
for i, matmul_size in enumerate(all_kernel_sizes['matmul']):
    if matmul_size in exist_matmul_sizes:
        print(f'Skipping Matmul Size {i+1}/{len(all_kernel_sizes["matmul"])}:', matmul_size)
        continue
    print(f'Evaluate Matmul Size {i+1}/{len(all_kernel_sizes["matmul"])}:', matmul_size)
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
    print(f'{lat * 1e6} us')
    f.close()
# %%
