# %%
import sys
sys.path.append("..")

import os
import onnx
from onnx_utils import *
from framework.Network import Network
from framework.backend.ShapeInference import ShapeInference
from framework.backend.BaselineOptimizer import BaselineOptimizer
import numpy as np
# %%
# Convert Huggingface model to ONNX
# from optimum.onnxruntime import ORTModelForCausalLM
# import torch
# model_id = "meta-llama/Llama-3.2-1B"
# model_id = "HuggingFaceTB/SmolLM-135M"
# ort_model = ORTModelForCausalLM.from_pretrained(model_id, export=True)
# ort_model.save_pretrained(f"../onnx_model/{model_id.split('/')[1]}")
# %%
# load the original model and convert it to the latest opset version
model_id = "meta-llama/Llama-3.2-1B"
model_name = model_id.split('/')[1]
opset_version = 22

model_path = f"../../onnx_model/{model_name}/model.onnx"
original_model = onnx.load(model_path, load_external_data=False)
converted_model = onnx.version_converter.convert_version(original_model, opset_version)
converted_model_path = f"../../onnx_model/{model_name}/model_opset_{opset_version}.onnx"
onnx.save(converted_model, converted_model_path)
# %%
network = Network.from_onnx(model_path, load_external_data=False)
# print(network.__str__())
# %%
# Set the dynamic dimensions
dynamic_dims = {'batch_size': 2,
                'sequence_length': 1,
                'past_sequence_length': 64}
dynamic_dims["past_sequence_length + 1"] = dynamic_dims["past_sequence_length"] + 1

# Get the input shapes
inputs_dict = get_model_input_shapes(original_model)
for input_name, input_shape in inputs_dict.items():
    for dim in input_shape:
        if isinstance(dim, str):
            if dim not in dynamic_dims:
                dynamic_dims[dim] = 1
                print(f"Adding dynamic dim {dim} to dynamic_dims, set to 1")
    inputs_dict[input_name] = list([dynamic_dims[dim] if isinstance(dim, str) else dim for dim in input_shape])
    print(f"{input_name}: {inputs_dict[input_name]}")
# %%
# Shape Inference
baseline_optimizer = BaselineOptimizer(network)
baseline_optimizer.run()
shape_infer = ShapeInference(network)
shape_infer.run(inputs_dict)
# %%
def get_num_unk_shape_tensors(network):
    unk_shape_tensors = []
    for tensor in network.tensors:
        T = network.tensors[tensor]
        shape = T.shape
        if not all([isinstance(dim, int) or isinstance(dim, np.int64) for dim in shape]):
            unk_shape_tensors.append(T)
    return unk_shape_tensors

print(f"Before symbolic inference: {len(get_num_unk_shape_tensors(network))} unknown shape tensors")
# %%
# attention_mask: [batch_size, past_sequence_length + 1]
# set causal attention mask, i.e., mask all elements in the past sequence
attention_mask = np.zeros((dynamic_dims["batch_size"], dynamic_dims["past_sequence_length"] + 1), dtype=np.float32)
attention_mask[:, -1] = 1.0

preset_in_dict = {'attention_mask': attention_mask}
shape_infer.run_symbolic(preset_in_dict)
print(f"After symbolic inference: {len(get_num_unk_shape_tensors(network))} unknown shape tensors")
# %%
# generate onnx model with dynamic shape inference
dynamic_model_path = f"../../onnx_model/{model_name}/model_dynamic.onnx"
# copy the original model to the dynamic model
os.system(f"cp {model_path} {dynamic_model_path}")
dynamic_model = onnx.load(dynamic_model_path, load_external_data=False)
opset_version = get_model_opset(dynamic_model)

for value_info in dynamic_model.graph.value_info:
    print(value_info.name)
    print(f'\t orignal shape: {get_valueinfo_shape(value_info)}')
    if value_info.name not in network.tensors:
        print(f"Tensor {value_info.name} not found in network tensors")
    else:
        inferred_shape = network.lookup_tensor(value_info.name).shape
        print(f'\t inferred shape: {inferred_shape}')
        # set the shape to the inferred shape
        for i, dim in enumerate(value_info.type.tensor_type.shape.dim):
            if (dim.dim_value > 0) and (dim.dim_value != inferred_shape[i]):
                print(f"Error: {value_info.name} dim {i} {dim.dim_value} != {inferred_shape[i]}")
            # print(i, dim)
            dim.dim_value = inferred_shape[i]

onnx.save(dynamic_model, dynamic_model_path)
# %%
