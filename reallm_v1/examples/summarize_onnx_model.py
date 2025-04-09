# %%
import sys

sys.path.append("..")

# %%
import onnx
import onnx_utils

# SET ME TO THE PATH OF AN ONNX MODEL
ONNX_MODEL_PATH = "./mnist-cnn.onnx"

print("Loading model...", end="")
model = onnx.load(ONNX_MODEL_PATH)
print(" Done!")

# %%
print("Model Summary")
print("=============")

print(f"Opset Version = {onnx_utils.get_model_opset(model)}")
print(f"Inputs: {onnx_utils.get_model_input_shapes(model)}")
print(f"Outputs: {onnx_utils.get_model_input_shapes(model)}")

print(f"Parameters:")
parameters = onnx_utils.get_model_initializer_np_data(model)
total_bytes = 0
for name, data in parameters.items():
    total_bytes += data.nbytes
    print(f"\t{name}: {data.nbytes}")
print(f"\tTotal bytes: {total_bytes}")

print(f"Operators:")
operators = onnx_utils.get_model_node_optypes(model)
for name, op in operators:
    print(f"\t{name}: {op}")
print()

# %%
print("Model Program")
print("=============")

for node in model.graph.node:
    (ins, outs, attrs) = onnx_utils.get_node_args(node)

    in_list = list(map(lambda x: f"{x.name}:{x.value}", ins))
    out_list = list(map(lambda x: f"{x.name}:{x.value}", outs))
    attr_list = list(map(lambda x: f"{x.name}:{x.value}", attrs))
    print(
        ", ".join(out_list)
        + f" = {node.op_type}("
        + ", ".join(in_list + attr_list)
        + ");"
    )
