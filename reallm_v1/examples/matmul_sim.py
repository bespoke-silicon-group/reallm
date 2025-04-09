# %%
import sys
# sys.path.append("..")
sys.path.append("/mnt/users/ssd3/homes/huwan/workspace/reallm_v2")
sys.path.append("/mnt/users/ssd3/homes/huwan/workspace/reallm_v2/sw")
sys.path.append("/mnt/users/ssd3/homes/huwan/workspace/reallm_v2/examples/LLMCompass")
from phases.hardware_exploration import hardware_exploration
from structs.System import System
from structs.Model import Model
from scripts.gen_llmcompass_config import gen_llmcompass_config
import yaml, pickle

import os
import onnx
from onnx_utils import *
from framework.Network import Network
from framework.backend.ShapeInference import ShapeInference
from framework.backend.BaselineOptimizer import BaselineOptimizer
from framework.backend.PerformanceSim import PerformanceSim
import numpy as np
# %%
# load the original model and convert it to the latest opset version
# model_id = "meta-llama/Llama-3.2-1B"
# model_name = model_id.split('/')[1]
# model_path = f"../../onnx_model/{model_name}/model.onnx"
model_path = "simple_llm.onnx"

original_model = onnx.load(model_path, load_external_data=False)
network = Network.from_onnx(model_path, load_external_data=True)
# %%
T = 4
P = 1
C = 1
p_network = Network.from_onnx_partition(model_path, T, P, C)
# %%
inputs_dict = get_model_input_shapes(original_model)
# list to tuple
for input_name, input_shape in inputs_dict.items():
    inputs_dict[input_name] = tuple(input_shape)
print(inputs_dict)
# %%
# dynamic_dims = {'batch_size': 2,
#                 'sequence_length': 1,
#                 'past_sequence_length': 64}
# dynamic_dims["past_sequence_length + 1"] = dynamic_dims["past_sequence_length"] + 1
# 
# # Get the input shapes
# inputs_dict = get_model_input_shapes(original_model)
# for input_name, input_shape in inputs_dict.items():
#     for dim in input_shape:
#         if isinstance(dim, str):
#             if dim not in dynamic_dims:
#                 dynamic_dims[dim] = 1
#                 print(f"Adding dynamic dim {dim} to dynamic_dims, set to 1")
#     inputs_dict[input_name] = list([dynamic_dims[dim] if isinstance(dim, str) else dim for dim in input_shape])
# %%
# Shape Inference
for nn in [network, p_network]:
    baseline_optimizer = BaselineOptimizer(nn)
    baseline_optimizer.run()
    shape_infer = ShapeInference(nn)
    shape_infer.run(inputs_dict)
    preset_in_dict = {}
    shape_infer.run_symbolic(preset_in_dict)

# # %%
# # generate onnx model with dynamic shape inference
# dynamic_model_path = f"model_dynamic.onnx"
# # copy the original model to the dynamic model
# os.system(f"cp {model_path} {dynamic_model_path}")
# dynamic_model = onnx.load(dynamic_model_path, load_external_data=False)
# opset_version = get_model_opset(dynamic_model)
# # %%
# # add tensor value info
# from onnx import TensorProto
# for tensor in network.tensors:
#     T = network.tensors[tensor]
#     if T.id in network.inputs:
#         continue
#     print(f'{T.id}: {T.shape}')
#     value_info = onnx.helper.make_tensor_value_info(T.id, TensorProto.FLOAT, T.shape)
#     dynamic_model.graph.value_info.append(value_info)
# onnx.save(dynamic_model, dynamic_model_path)
# %%
# Get all Gemm operators
for E in network.iter():
    if E.type == 'Gemm' or E.type == 'MatMul':
        A_shape = network.lookup_tensor(E.A).shape
        B_shape = network.lookup_tensor(E.B).shape
        print(f"Gemm {E.id}: A: {A_shape}, B: {B_shape}")

print("=========================================")
for E in p_network.iter():
    if E.type == 'Gemm' or E.type == 'MatMul':
        A_shape = p_network.lookup_tensor(E.A).shape
        B_shape = p_network.lookup_tensor(E.B).shape
        print(f"Gemm {E.id}: A: {A_shape}, B: {B_shape}")

# %%
# Set the hardware
hw_config_path = "inputs/test_hw.yaml"
constants_path = "inputs/7nm_default.yaml"
output_dir = "./outputs"
hw_name = hw_config_path.split('/')[-1].split('.')[0]

hw_config = yaml.safe_load(open(hw_config_path, 'r'))
constants = yaml.safe_load(open(constants_path, 'r'))
hardware_exploration(hw_config, constants, output_dir, hw_name, True)
gen_llmcompass_config(hw_config_path, '../../LLMCompass/configs/template.json')

hw_pickle = f'{output_dir}/{hw_name}/{hw_name}.pkl'
with open(hw_pickle, 'rb') as f:
    servers = pickle.load(f)
    if len(servers) > 1:
        print(f"Warning: Multiple servers found in {hw_pickle}, using the first one")
    srv = servers[0]
    hw_node = srv.package
    print("=========================================")
    print(f"Hardware Config:\nPerf: {hw_node.tops} TFLOPS\nDRAM BW: {hw_node.dram_bw_TB_per_sec} TB/s")
# Set system config
sys_config_path = "inputs/sys_default.yaml"
sys_config = yaml.safe_load(open(sys_config_path, 'r'))
sys_config['server'] = srv
sys_config['model'] = Model(name='simple_llm', num_layers=3, d=128, num_heads=1)
if 'workload' in sys_config:
    sys_config.pop('workload')
# hardware = System(**sys_config, update_on_init=False)
hw_system = System(**sys_config, sw_update_on_init=False)
# %%
T = hw_system.default_mapping['t']
P = hw_system.default_mapping['p']
C = hw_system.default_mapping['c']
num_nodes = hw_system.num_packages
print("=========================================")
print(f"System Config:\nNum Chips: {num_nodes}\nTensor Parallelism: {T}\nPipeline Parallelism: {P}\nContext Parallelism: {C}")

# %%
# generate new operators based on T, P, and C


# %%
# Roofline Simulation
llmcompass_json = f"outputs/{hw_name}/llmcompass.json"
ops_to_sim = ['Gemm']
perf_sim= PerformanceSim(network, hw_system, llmcompass_json, ops_to_sim)
perf_sim.run(method = 'roofline', debug = True)

# %%
perf_sim= PerformanceSim(p_network, hw_system, llmcompass_json, ops_to_sim)
perf_sim.run(method = 'roofline', debug = True)
# %%
perf_sim.run(method = 'uarch', debug = True)
# %%
for E in network.iter():
    print(f"{E.id}: {E.type}")