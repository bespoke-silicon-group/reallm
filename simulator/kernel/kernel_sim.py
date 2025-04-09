import os, sys
from pathlib import Path

# Add LLMCompass to the Python path
llmcompass_path = Path(__file__).resolve().parents[2] / "LLMCompass"
if str(llmcompass_path) not in sys.path:
    sys.path.append(str(llmcompass_path))

from LLMCompass.design_space_exploration.dse import template_to_system, read_architecture_template
from LLMCompass.software_model.utils import Tensor, data_type_dict
from LLMCompass.software_model.matmul import *
from LLMCompass.software_model.softmax import *
from LLMCompass.software_model.mul import *
from LLMCompass.software_model.layernorm import *
from LLMCompass.software_model.silu import *

def kernel_perf_sim(hardware_llmcompass_json, kernel_lib_dir, 
                   eval_kernel_types = ['matmul', 'softmax', 'layernorm', 'mul', 'silu'],
                   compile_mode = 'heuristic-GPU'):

    hw_specs = read_architecture_template(hardware_llmcompass_json)
    lc_system = template_to_system(hw_specs)
    device = lc_system.device
    data_type = data_type_dict["fp16"]

    # read all kernel sizes
    for kernel_type in eval_kernel_types:
        all_kernel_sizes = []
        path = os.path.join(kernel_lib_dir, f'{kernel_type}_sizes.txt')
        with open(f'{path}', 'r') as f:
            lines = f.readlines()
            for line in lines:
                size = line.strip().split(',')
                if kernel_type == 'softmax':
                    # multiply the first two dimensions
                    size[0] = str(int(size[0]) * int(size[1]))
                    size.pop(1)
                all_kernel_sizes.append(tuple(map(int, size)))

        exist_matmul_sizes = set()

        hardware_name = hardware_llmcompass_json.split('/')[-1].split('.')[0]
        out_file = os.path.join(kernel_lib_dir, f'{hardware_name}_{kernel_type}_latency.csv')
        
        if kernel_type == 'matmul':
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

            print(f'LLMCompass Matmul Simulation Begin for all kernels on {hardware_llmcompass_json}')
            for i, matmul_size in enumerate(all_kernel_sizes):
                if matmul_size in exist_matmul_sizes:
                    print(f'Skipping Matmul Size {i+1}/{len(all_kernel_sizes)}:', matmul_size)
                    continue
                print(f'Evaluate Matmul Size {i+1}/{len(all_kernel_sizes)}:', matmul_size)
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
        else:
            eval_kernel_dims = {'softmax': 2, 'mul': 2, 'layernorm': 2, 'silu': 2}

            f = open(out_file, 'w')
            num_dims = eval_kernel_dims[kernel_type]
            if num_dims == 2:
                f.write('M, N, latency\n')
            elif num_dims == 3:
                f.write('B, M, N, latency\n')
            else:
                raise ValueError(f'Unsupported kernel type: {kernel_type}')

            print(f'LLMCompass Kernel Performance Simulation Begin for all {kernel_type} on {hardware_llmcompass_json}')
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



