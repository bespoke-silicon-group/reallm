# %%
import numpy as np
import os, sys
from simulator.base.model import llama70b, llama405b, deepseekv2, deepseekv3
from simulator.kernel.kernel_size_gen import gen_kernel_sizes
from simulator.kernel.kernel_sim import kernel_perf_sim
from simulator.system.trace_gen import generate_code_traces, generate_conv_traces, download_azure_llm_traces
from simulator.system.system_sim import run_system_sim

# %%
# Configurations
workspace_dir = "workspace/"
model = llama70b
num_nodes = 8

# %%
# Generate Kernel Sizes
prefill_blocks = [1024]
decode_ctxs = [128]
kernel_lib_dir = os.path.join(workspace_dir, "kernel_lib")
gen_kernel_sizes(model, num_nodes, output_dir=kernel_lib_dir,
                 prefill_blocks=prefill_blocks,
                 decode_ctxs=decode_ctxs)

# %%
# Kernel Simulation
hardware_llmcompass_json = 'LLMCompass/configs/GA100.json'
kernel_perf_sim(hardware_llmcompass_json, kernel_lib_dir,
                eval_kernel_types=['matmul', 'softmax', 'layernorm', 'mul', 'silu'],
                compile_mode='heuristic-GPU')

# %%
# Download Azure LLM traces if not exists
trace_data_dir = os.path.join(workspace_dir, "traces")
download_azure_llm_traces(trace_data_dir)

# %%
# Trace generation
random_seed = 0
np.random.seed(random_seed)

requests_rates = [1, 3, 5, 7]

overwrite = False

# generate traces
code_distributions_file = os.path.join(trace_data_dir, "code_distributions.csv")
code_trace_file_temp = trace_data_dir + '/rr_code_{}.csv'
conv_distributions_file = os.path.join(trace_data_dir, "conv_distributions.csv")
conv_trace_file_temp = trace_data_dir + '/rr_conv_{}.csv'
for req_rate in requests_rates:
    trace_file = os.path.join(trace_data_dir, f'rr_code_{req_rate}.csv')
    if not os.path.exists(trace_file) or overwrite:
        generate_code_traces(
            max_requests=50000,
            end_time=500,
            request_rates=[req_rate],
            code_distributions_file=code_distributions_file,
            trace_filename_template=code_trace_file_temp)
    trace_file = os.path.join(trace_data_dir, f'rr_conv_{req_rate}.csv')
    if not os.path.exists(trace_file) or overwrite:
        generate_conv_traces(
            max_requests=50000,
            end_time=500,
            request_rates=[req_rate],
            conv_distributions_file=conv_distributions_file,
            trace_filename_template=conv_trace_file_temp)

print("Generated traces at different request rates")

# %%
# Run System Simulation
trace_file = os.path.join(trace_data_dir, 'rr_code_1.csv')
run_system_sim(model, trace_file, 
               hw_node_name='GA100', num_nodes=num_nodes,
               parallelism=(1, 8, 1, 1), io_algo='multishot',
                scheduler_algo='mixed-sarathi', prefill_chunk=2048,
                workspace_dir=workspace_dir,
                end_reqs=2,
)
