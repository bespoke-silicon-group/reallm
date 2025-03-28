# %%
import sys
sys.path.append('llm_serving_sim/')
from llm_serving_sim.model import KernelSizes, llama, deepseek, llama70b, llama405b, deepseekv2, deepseekv3
# %%
# Batch Size Generation
# Given traces and SLO, generate prefill size and decode ctx sizes
prefill_blocks = [64, 128, 256, 512, 1024, 2048, 4096]
# num_decode_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
num_decode_blocks = [128]
decode_ctxs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
                16384, 32768, 65536, 131072]
# %%
# Parallelism Generation

def gen_parallelism(num_nodes: int):
    all_parallelism = [] # List of tuples (tensor_parallelism, pipeline_parallelism, context_parallelism)
    # generate tensor, pipeline and context parallelism
    for tensor_parallelism in range(1, num_nodes+1):
        for pipeline_parallelism in range(1, num_nodes+1):
            for context_parallelism in range(1, num_nodes+1):
                if tensor_parallelism * pipeline_parallelism * context_parallelism == num_nodes:
                    all_parallelism.append((tensor_parallelism, pipeline_parallelism, context_parallelism))
    return all_parallelism

def gen_moe_parallelism(num_nodes: int):
    all_parallelism = [] # List of tuples (expert_parallelism, tensor_parallelism, pipeline_parallelism, context_parallelism)
    # generate tensor, pipeline and context parallelism
    for expert_parallelism in range(8, num_nodes+1):
        for tensor_parallelism in range(1, num_nodes+1):
            for pipeline_parallelism in range(1, num_nodes+1):
                for context_parallelism in range(1, num_nodes+1):
                    if expert_parallelism * tensor_parallelism * pipeline_parallelism * context_parallelism == num_nodes:
                        all_parallelism.append((expert_parallelism, tensor_parallelism, pipeline_parallelism, context_parallelism))
    return all_parallelism

    
llama3_70b_num_nodes = [8, 32, ]
llama3_405b_num_nodes = [32, 128, ]
deepseek_v2_num_nodes = [8, 32]
deepseek_v3_num_nodes = [32, 128]

all_kernel_sizes = dict()

# for model in [llama70b, llama405b, deepseekv2, deepseekv3]:
# for model in [llama70b, deepseekv3]:
#     all_parallelism = []
#     if model == llama70b:
#         for nodes in llama3_70b_num_nodes:
#             all_parallelism += gen_parallelism(nodes)
#         print('llama70b')
#     elif model == llama405b:
#         for nodes in llama3_405b_num_nodes:
#             all_parallelism += gen_parallelism(nodes)
#         print('llama405b')
#     elif model == deepseekv2:
#         for nodes in deepseek_v2_num_nodes:
#             all_parallelism += gen_moe_parallelism(nodes)
#         print('deepseekv2')
#     elif model == deepseekv3:
#         for nodes in deepseek_v3_num_nodes:
#             all_parallelism += gen_moe_parallelism(nodes)
#         print('deepseekv3')
#     for prefill_len in prefill_blocks:
#         for num_decode in num_decode_blocks:
#             if num_decode <= len(decode_ctxs):
#                 decode_lens = decode_ctxs[:num_decode]
#             else:
#                 decode_lens = decode_ctxs + [decode_ctxs[-1]] * (num_decode - len(decode_ctxs))
#             for parallelism in all_parallelism:
#                 kernel_sizes = model.get_kernel_sizes(prefill_len, decode_lens, parallelism)
#                 for kernel_type in eval_kernel_types:
#                     if kernel_type not in all_kernel_sizes:
#                         all_kernel_sizes[kernel_type] = kernel_sizes[kernel_type].get_all_kernel_sizes()
#                     else:
#                         for kernel_size in kernel_sizes[kernel_type].get_all_kernel_sizes():
#                             if kernel_size not in all_kernel_sizes[kernel_type]:
#                                 all_kernel_sizes[kernel_type].append(kernel_size)

# %%
# Save all kernel sizes to file
# for kernel_type in eval_kernel_types:
#     print(f"{kernel_type}: {len(all_kernel_sizes[kernel_type])}")
#     # with open(f'{kernel_type}_sizes.txt', 'w') as f:
#     with open(f'parts_{kernel_type}_sizes.txt', 'w') as f:
#         for kernel_size in all_kernel_sizes[kernel_type]:
#             f.write(','.join(map(str, kernel_size)) + '\n')
# print(all_kernel_sizes['matmul'])
# %%