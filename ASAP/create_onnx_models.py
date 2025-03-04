# %%
import onnx
from onnx import helper, TensorProto
# %%
# Llama3 70B
# vocab_size = 10000
embedding_dim = 8192
ffn_dim = 28672
num_heads = 64
num_layers = 2
batch_size = 1

seq_len = 1024
# Create the input tensor (token IDs)
# input_tensor = helper.make_tensor_value_info('input_tokens',
#                                              TensorProto.INT32,
#                                              [None, seq_len])
# Create the output tensor (logits)
# output_tensor = helper.make_tensor_value_info('output_logits',
#                                               TensorProto.FLOAT,
#                                               [None, seq_len, vocab_size])
input_tensor = helper.make_tensor_value_info('inputs',
                                             TensorProto.FLOAT16,
                                             [batch_size, seq_len, embedding_dim])

# Embedding layer: map token IDs to embeddings
# embedding_weights = helper.make_tensor(
#     name='embedding_weights',
#     data_type=TensorProto.FLOAT,
#     dims=[vocab_size, embedding_dim],
#     vals=[0.1] * vocab_size * embedding_dim # dummy values
# )
# embedding_node = helper.make_node(
#     'Gather',
#     inputs=['embedding_weights', 'input_tokens'],
#     outputs=['token_embeddings'],
#     axis=0 # the first dimension of the embedding_weights tensor
# )
# Transformer Layer
graph_nodes = []
graph_initializers = []
for layer_idx in range(num_layers):
    if layer_idx == 0:
        layer_input = 'inputs'
    else:
        layer_input = f'ffn2_o_{layer_idx - 1}'
    # Attention layer
    qkv_weights = helper.make_tensor(
        name=f'qkv_weights_{layer_idx}',
        data_type=TensorProto.FLOAT,
        dims=[embedding_dim, embedding_dim * 3],
        vals=[0.1] * embedding_dim * embedding_dim * 3 # dummy values
    )
    qkv_node = helper.make_node(
        'MatMul',
        inputs=[layer_input, f'qkv_weights_{layer_idx}'],
        outputs=[f'qkv_{layer_idx}'],
        name=f'qkv_proj_{layer_idx}',
    )
    split_tensor = helper.make_tensor(
        name=f'split_{layer_idx}',
        data_type=TensorProto.INT64,
        dims=[3],
        vals=[embedding_dim] * 3
    )
    split_qkv_node = helper.make_node(
        "Split",
        inputs=[f'qkv_{layer_idx}', f'split_{layer_idx}'],
        outputs=[f'q_{layer_idx}', f'k_{layer_idx}', f'v_{layer_idx}'],
        axis=-1,
        # split=[embedding_dim] * 3
        name=f'split_qkv_{layer_idx}'
    )
    # q_weights = helper.make_tensor(
    #     name=f'q_weights_{layer_idx}',
    #     data_type=TensorProto.FLOAT,
    #     dims=[embedding_dim, embedding_dim],
    #     vals=[0.1] * embedding_dim * embedding_dim # dummy values
    # )
    # q_node = helper.make_node(
    #     "MatMul",
    #     inputs=[layer_input, f'q_weights_{layer_idx}'],
    #     outputs=[f'q_{layer_idx}']
    # )
    # k_weights = helper.make_tensor(
    #     name=f'k_weights_{layer_idx}',
    #     data_type=TensorProto.FLOAT,
    #     dims=[embedding_dim, embedding_dim],
    #     vals=[0.1] * embedding_dim * embedding_dim # dummy values
    # )
    # k_node = helper.make_node(
    #     "MatMul",
    #     inputs=[layer_input, f'k_weights_{layer_idx}'],
    #     outputs=[f'k_{layer_idx}']
    # )
    # v_weights = helper.make_tensor(
    #     name=f'v_weights_{layer_idx}',
    #     data_type=TensorProto.FLOAT,
    #     dims=[embedding_dim, embedding_dim],
    #     vals=[0.1] * embedding_dim * embedding_dim # dummy values
    # )
    k_transpose_node = helper.make_node(
        "Transpose",
        inputs=[f'k_{layer_idx}'],
        outputs=[f'k_transpose_{layer_idx}'],
        perm=[0, 2, 1],
        name=f'k_transpose_{layer_idx}'
    )
    matmul_qk_node = helper.make_node(
        "MatMul",
        inputs=[f'q_{layer_idx}', f'k_transpose_{layer_idx}'],
        outputs=[f's_{layer_idx}'],
        name=f'matmul_qk_{layer_idx}'
    )
    scale_tensor = helper.make_tensor(
        name=f'scale_{layer_idx}',
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[1.0 / (embedding_dim ** 0.5)], # scale factor
    )
    scale_node = helper.make_node(
        "Mul", # element-wise multiplication
        inputs=[f's_{layer_idx}', f'scale_{layer_idx}'],
        outputs=[f'scaled_s_{layer_idx}'],
        name=f'scale_{layer_idx}',
    )
    softmax_node = helper.make_node(
        "Softmax",
        inputs=[f'scaled_s_{layer_idx}'],
        outputs=[f'attn_p_{layer_idx}'],
        axis=-1, # apply softmax along the last dimension
        name=f'softmax_{layer_idx}',
    )
    matmul_pv_node = helper.make_node(
        'MatMul',
        inputs=[f'attn_p_{layer_idx}', f'v_{layer_idx}'],
        outputs=[f'pv_{layer_idx}'],
        name=f'matmul_pv_{layer_idx}',
    )
    attn_output_weights = helper.make_tensor(
        name=f'attn_output_weights_{layer_idx}',
        data_type=TensorProto.FLOAT,
        dims=[embedding_dim, embedding_dim],
        vals=[0.1] * embedding_dim * embedding_dim # dummy values
    )
    attn_output_node = helper.make_node(
        'MatMul',
        inputs=[f'pv_{layer_idx}', f'attn_output_weights_{layer_idx}'],
        outputs=[f'attn_o_{layer_idx}'],
        name=f'attn_output_{layer_idx}',
    )
    # Feed-forward layer
    ffn1_weights = helper.make_tensor(
        name=f'ffn1_weights_{layer_idx}',
        data_type=TensorProto.FLOAT,
        dims=[embedding_dim, ffn_dim],
        vals=[0.1] * embedding_dim * ffn_dim, # dummy values
    )
    ffn1_node = helper.make_node(
        "MatMul",
        inputs=[f'attn_o_{layer_idx}', f'ffn1_weights_{layer_idx}'],
        outputs=[f'ffn1_o_{layer_idx}'],
        name=f'ffn1_{layer_idx}',
    )
    ffn_relu_node = helper.make_node(
        "Relu",
        inputs=[f'ffn1_o_{layer_idx}'],
        outputs=[f'ffn1_relu_{layer_idx}'],
        name=f'ffn_relu_{layer_idx}',
    )
    ffn2_weights = helper.make_tensor(
        name=f'ffn2_weights_{layer_idx}',
        data_type=TensorProto.FLOAT,
        dims=[ffn_dim, embedding_dim],
        vals=[0.1] * ffn_dim * embedding_dim # dummy values
    )
    ffn2_node = helper.make_node(
        "MatMul",
        inputs=[f'ffn1_relu_{layer_idx}', f'ffn2_weights_{layer_idx}'],
        outputs=[f'ffn2_o_{layer_idx}'],
        name=f'ffn2_{layer_idx}',
    )

    graph_nodes.extend([qkv_node, split_qkv_node, k_transpose_node, matmul_qk_node, 
                        scale_node, softmax_node, matmul_pv_node, attn_output_node,
                        ffn1_node, ffn_relu_node, ffn2_node])
    graph_initializers.extend([qkv_weights, split_tensor, 
                               scale_tensor, attn_output_weights,
                               ffn1_weights, ffn2_weights])
# %%
# Build the graph
output_tensor = helper.make_tensor_value_info(f'ffn2_o_{num_layers - 1}',
                                              TensorProto.FLOAT16,
                                              [batch_size, seq_len, embedding_dim])
graph = helper.make_graph(
    nodes=graph_nodes,
    name='Simple-LLM',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=graph_initializers
)

model = helper.make_model(graph, producer_name='Simple-LLM')
onnx.checker.check_model(model)
onnx.save(model, 'simple_llm.onnx')


# %%
