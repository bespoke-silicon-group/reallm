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

def kernel_perf_roofline(hardware_llmcompass_json, kernel_type, shape):

    hw_specs = read_architecture_template(hardware_llmcompass_json)
    lc_system = template_to_system(hw_specs)
    device = lc_system.device
    data_type = data_type_dict["fp16"]

    if kernel_type == 'matmul':
        if len(shape) == 3:
            M = shape[0]
            K = shape[1]
            N = shape[2]
            mm = Matmul(data_type)
            input1 = Tensor([M, K], data_type)
            input2 = Tensor([K, N], data_type)
            _ = mm(input1, input2)
            lat = mm.roofline_model(device)
        elif len(shape) == 4:
            B = shape[0]
            M = shape[1]
            K = shape[2]
            N = shape[3]
            mm = BatchedMatmul(data_type)
            input1 = Tensor([B, M, K], data_type)
            input2 = Tensor([B, K, N], data_type)
            _ = mm(input1, input2)
            lat = mm.roofline_model(device)
    else:
        shape = list(shape)
        if kernel_type == 'softmax':
            sm = Softmax(data_type)
            input1 = Tensor(shape, data_type)
            _ = sm(input1)
            lat = sm.roofline_model(device)
        elif kernel_type == 'layernorm':
            ln = LayerNorm(data_type)
            input1 = Tensor(shape, data_type)
            _ = ln(input1)
            lat = ln.roofline_model(device)
        elif kernel_type == 'silu':
            sl = SiLU(data_type)
            input1 = Tensor(shape, data_type)
            _ = sl(input1)
            lat = sl.roofline_model(device)
        elif kernel_type == 'mul':
            mm = Mul(data_type)
            input1 = Tensor(shape, data_type)
            input2 = Tensor(shape, data_type)
            _ = mm(input1, input2)
            lat = mm.roofline_model(device)
        else:
            raise ValueError(f"Unknown kernel type {kernel_type} for roofline model")
    return lat



