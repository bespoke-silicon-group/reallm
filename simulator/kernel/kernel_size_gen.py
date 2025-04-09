import os

# Parallelism Generation
def gen_parallelism(num_nodes: int):
    all_parallelism = [] # List of tuples (tensor_parallelism, pipeline_parallelism, context_parallelism)
    # generate tensor, pipeline and context parallelism
    for tensor_parallelism in range(1, num_nodes+1):
        for pipeline_parallelism in range(1, num_nodes+1):
            for context_parallelism in range(1, num_nodes+1):
                if tensor_parallelism * pipeline_parallelism * context_parallelism == num_nodes:
                    all_parallelism.append((1, tensor_parallelism, pipeline_parallelism, context_parallelism))
    return all_parallelism

def gen_moe_parallelism(num_nodes: int):
    all_parallelism = [] # List of tuples (expert_parallelism, tensor_parallelism, pipeline_parallelism, context_parallelism)
    # generate tensor, pipeline and context parallelism
    for expert_parallelism in range(1, num_nodes+1):
        for tensor_parallelism in range(1, num_nodes+1):
            for pipeline_parallelism in range(1, num_nodes+1):
                for context_parallelism in range(1, num_nodes+1):
                    if expert_parallelism * tensor_parallelism * pipeline_parallelism * context_parallelism == num_nodes:
                        all_parallelism.append((expert_parallelism, tensor_parallelism, pipeline_parallelism, context_parallelism))
    return all_parallelism

    
def gen_kernel_sizes(model, num_nodes, output_dir = 'workspace/kernel_lib',
                     prefill_blocks = [64, 128, 256, 512, 1024, 2048, 4096],
                     num_decode_blocks = [128],
                     decode_ctxs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                     eval_kernel_types = ['matmul', 'softmax', 'layernorm', 'mul', 'silu'],
                     ):
    if model.moe:
        all_parallelism = gen_moe_parallelism(num_nodes)
    else:
        all_parallelism = gen_parallelism(num_nodes)

    all_kernel_sizes = dict()
    for prefill_len in prefill_blocks:
        for num_decode in num_decode_blocks:
            if num_decode <= len(decode_ctxs):
                decode_lens = decode_ctxs[:num_decode]
            else:
                decode_lens = decode_ctxs + [decode_ctxs[-1]] * (num_decode - len(decode_ctxs))
            for parallelism in all_parallelism:
                kernel_sizes = model.get_kernel_sizes(prefill_len, decode_lens, parallelism)
                for kernel_type in eval_kernel_types:
                    if kernel_type not in all_kernel_sizes:
                        all_kernel_sizes[kernel_type] = kernel_sizes[kernel_type].get_all_kernel_sizes()
                    else:
                        for kernel_size in kernel_sizes[kernel_type].get_all_kernel_sizes():
                            if kernel_size not in all_kernel_sizes[kernel_type]:
                                all_kernel_sizes[kernel_type].append(kernel_size)

    # Save kernel sizes to files
    os.makedirs(output_dir, exist_ok=True)
    for kernel_type in eval_kernel_types:
        file_path = os.path.join(output_dir, f'{kernel_type}_sizes.txt')
        print(f"{kernel_type}: {len(all_kernel_sizes[kernel_type])} kernel sizes, saved to {file_path}")
        with open(f'{file_path}', 'w') as f:
            for kernel_size in all_kernel_sizes[kernel_type]:
                f.write(','.join(map(str, kernel_size)) + '\n')
