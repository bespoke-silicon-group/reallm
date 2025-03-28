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
from LLMCompass.software_model.softmax import *
from LLMCompass.software_model.mul import *
from LLMCompass.software_model.layernorm import *
from LLMCompass.software_model.silu import *
# %%
# read all kernel sizes
eval_kernel_types = ['softmax', 'mul', 'layernorm', 'silu']
eval_kernel_dims = {'softmax': 2, 'mul': 2, 'layernorm': 2, 'silu': 2}

for kernel_type in eval_kernel_types:
    all_kernel_sizes = []
    # with open(f'{kernel_type}_sizes.txt', 'r') as f:
    with open(f'{kernel_type}_sizes.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            size = line.strip().split(',')
            if kernel_type == 'softmax':
                # multiply the first two dimensions
                size[0] = str(int(size[0]) * int(size[1]))
                size.pop(1)
            all_kernel_sizes.append(tuple(map(int, size)))
    
    # hw_name = 'GH100'
    for hw_name in ['GA100', 'GH100_fast_main', 'GH100_more_compute1', 'GH100_more_compute2', 'GH100_more_l2']:
        out_file = f'{hw_name[1:]}_{kernel_type}_lat.csv'
        hw_specs = read_architecture_template(f'../LLMCompass/configs/{hw_name}.json')
        lc_system = template_to_system(hw_specs)
        device = lc_system.device
        data_type = data_type_dict["fp16"]

        exist_matmul_sizes = set()

        f = open(out_file, 'w')
        num_dims = eval_kernel_dims[kernel_type]
        if num_dims == 2:
            f.write('M, N, latency\n')
        elif num_dims == 3:
            f.write('B, M, N, latency\n')
        else:
            raise ValueError(f'Unsupported kernel type: {kernel_type}')

        print(f'LLMCompass Kernel Performance Simulation Begin for all {kernel_type} on {hw_name}')
        for i, size in enumerate(all_kernel_sizes):
            size = list(size)
            if kernel_type == 'softmax':
                sm = Softmax(data_type)
                input1 = Tensor(size, data_type)
                _ = sm(input1)
                lat = sm.compile_and_simulate(device)
                f.write(f'{size[0]}, {size[1]}, {lat}\n')
            elif kernel_type == 'layernorm':
                ln = LayerNorm(data_type)
                input1 = Tensor(size, data_type)
                _ = ln(input1)
                lat = ln.compile_and_simulate(device, compile_mode='heuristic-GPU')
                f.write(f'{size[0]}, {size[1]}, {lat}\n')
            elif kernel_type == 'silu':
                sl = SiLU(data_type)
                input1 = Tensor(size, data_type)
                _ = sl(input1)
                lat = sl.compile_and_simulate(device, compile_mode='heuristic-GPU')
                f.write(f'{size[0]}, {size[1]}, {lat}\n')
            elif kernel_type == 'mul':
                mm = Mul(data_type)
                input1 = Tensor(size, data_type)
                input2 = Tensor(size, data_type)
                _ = mm(input1, input2)
                lat = mm.compile_and_simulate(device, compile_mode='heuristic-GPU')
                f.write(f'{size[0]}, {size[1]}, {lat}\n')
        f.close()
# %%
