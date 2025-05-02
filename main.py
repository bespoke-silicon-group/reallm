import yaml
import argparse
import os
import numpy as np
from simulator.base.model import llama70b, llama405b, deepseekv2, deepseekv3
from simulator.kernel.kernel_size_gen import gen_kernel_sizes
from simulator.kernel.kernel_sim import kernel_perf_sim
from simulator.system.trace_gen import generate_traces, download_azure_llm_traces
from simulator.system.system_sim import run_system_sim

def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def setup_workspace(workspace_dir):
    """Create workspace directories if they don't exist."""
    os.makedirs(os.path.join(workspace_dir, "kernel_lib"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "traces"), exist_ok=True)
    return workspace_dir

def run_kernel_generation(config, workspace_dir, model):
    """Run kernel generation and simulation."""
    
    kernel_config = config.get('kernel_sim', {})
    prefill_blocks = kernel_config.get('prefill_blocks', [1024])
    decode_ctxs = kernel_config.get('decode_ctxt', [128])
    print(f"Running kernel generation and simulation with prefill_blocks: {prefill_blocks} and decode_ctxs: {decode_ctxs} ...")
    
    kernel_lib_dir = os.path.join(workspace_dir, "kernel_lib")
    
    # Generate kernel sizes
    num_nodes = config.get('system').get('num_devices')
    gen_kernel_sizes(model, num_nodes, output_dir=kernel_lib_dir,
                     prefill_blocks=prefill_blocks,
                     decode_ctxs=decode_ctxs)
    
    # Run kernel simulation
    hardware_name = config.get('system', {}).get('device', 'GA100')
    hardware_json = os.path.join('configs', 'device', f'{hardware_name}.json')
    compile_mode = kernel_config.get('compile_mode', 'heuristic-GPU')
    kernel_types = kernel_config.get('kernel_types', ['softmax', 'layernorm', 'mul', 'silu', 'matmul'])
    
    kernel_perf_sim(hardware_json, kernel_lib_dir,
                    eval_kernel_types=kernel_types,
                    compile_mode=compile_mode)
    
    print("Kernel generation and simulation completed.")

def run_trace_generation(config, workspace_dir):
    """Download and generate traces."""
    print("Downloading and generating traces...")
    
    trace_data_dir = os.path.join(workspace_dir, "traces")
    download_azure_llm_traces(trace_data_dir)
    
    trace_config = config.get('trace_gen', {})
    random_seed = trace_config.get('random_seed', 0)
    np.random.seed(random_seed)
    
    requests_rates = trace_config.get('request_rates', [1])
    max_requests = trace_config.get('max_requests', 50000)
    end_time = trace_config.get('end_time', 500)
    tasks = trace_config.get('tasks', ['code', 'conv'])
    
    # Generate traces
    for task in tasks:
        print(f"Generating traces for {task} task...")
        distribution_file = os.path.join(trace_data_dir, f"{task}_distributions.csv")
        generate_traces(max_requests=max_requests,
                        end_time=end_time,
                        request_rates=requests_rates,
                        pt_distributions_file=distribution_file,
                        trace_filename_template=trace_data_dir + f'/rr_{task}_{{}}.csv')

                
    print("Trace generation completed.")

def run_system_simulation(config, workspace_dir, model, 
                          trace_override=None, task_override=None, rate_override=None,
                          ):
    """Run system simulation."""
    system_sim_config = config.get('system_sim', {})
    sim_method = system_sim_config.get('sim_method')

    if trace_override:
        traces = [trace_override]
    else:
        trace_dir = os.path.join(workspace_dir, "traces")
        if task_override and rate_override:
            traces = []
            for task in task_override:
                for rate in rate_override:
                    trace_filename = f'rr_{task}_{rate}.csv'
                    if os.path.exists(os.path.join(trace_dir, trace_filename)):
                        traces.append(os.path.join(trace_dir, trace_filename))
                    else:
                        print(f"Warning: Trace file {trace_filename} not found. Skipping.")
        elif 'tasks' in system_sim_config and 'request_rates' in system_sim_config:
            tasks = system_sim_config['tasks']
            rates = system_sim_config['request_rates']
            if isinstance(tasks, str):
                tasks = [tasks]
            if isinstance(rates, str):
                rates = [rates]
            traces = []
            for task in tasks:
                for rate in rates:
                    trace_filename = f'rr_{task}_{rate}.csv'
                    if os.path.exists(os.path.join(trace_dir, trace_filename)):
                        traces.append(os.path.join(trace_dir, trace_filename))
                    else:
                        print(f"Warning: Trace file {trace_filename} not found. Skipping.")
        else:
            traces = system_sim_config.get('traces', [])
            if isinstance(traces, str):
                traces = [traces]
    
    # Get system simulation parameters
    system_config = config.get('system', {})
    hw_node_name = system_config.get('device')
    num_nodes = system_config.get('num_devices', 1)
    attn_parallelism = system_config.get('attention_parallelism')
    ffn_parallelism = system_config.get('ffn_parallelism')

    # attn and ffn parallelism have to be the same for now
    if attn_parallelism != ffn_parallelism:
        raise ValueError("Attention and FFN parallelism must be the same.")
    
    io_algo = config.get('io_algo', 'multishot')
    scheduler_algo = config.get('batching_algo', 'mixed-sarathi')
    prefill_chunk = config.get('prefill_chunk', 2048)

    end_reqs = system_sim_config.get('end_reqs', 10)

    print(f"Running {sim_method} simulation on {num_nodes} {hw_node_name} devices...")
    print(f"system_config: {system_config}")
    for trace_file in traces:
        print(f"Simulating trace: {trace_file}...")
        run_system_sim(model=model, 
                       trace=trace_file,
                       hw_node_name=hw_node_name,
                       num_nodes=num_nodes,
                       parallelism=attn_parallelism, 
                       io_algo=io_algo,
                       scheduler_algo=scheduler_algo, 
                       prefill_chunk=prefill_chunk,
                       sim_method=sim_method,
                       end_reqs=end_reqs,
                       workspace_dir=workspace_dir,
        )
                  
    print("System simulation completed.")

def main(args):
    """Main function to run the simulation based on arguments."""
    # Load config
    config = load_config(args.config)
    
    # Setup workspace
    workspace_dir = setup_workspace(args.workspace_dir or "workspace/")
    
    # Get model from config
    model_name = config.get('model', 'llama70b')
    models = {
        'llama70b': llama70b,
        'llama405b': llama405b,
        'deepseekv2': deepseekv2, 
        'deepseekv3': deepseekv3
    }
    model = models.get(model_name, llama70b)
    
    # Run the requested mode
    if args.mode in ['all', 'kernel']:
        run_kernel_generation(config, workspace_dir, model)
    
    if args.mode in ['all', 'trace']:
        run_trace_generation(config, workspace_dir)
    
    if args.mode in ['all', 'sim']:
        run_system_simulation(config, workspace_dir, model, 
                              trace_override=args.trace,
                              task_override=args.task,
                              rate_override=args.rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run chiplet cloud simulation.')
    parser.add_argument('--config', type=str, default='configs/system/default_homo.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--workspace_dir', type=str, default='workspace/',
                        help='Path to workspace directory')
    parser.add_argument('--mode', type=str, choices=['all', 'kernel', 'trace', 'sim'], 
                        default='all', help='Mode to run')
    parser.add_argument('--trace', type=str, help='Specific trace file to use for simulation')
    parser.add_argument('--task', type=str, choices=['code', 'conv'], nargs='+', help='Task type(s) for trace selection (space-separated)')
    parser.add_argument('--rate', type=int, nargs='+', help='Request rate(s) for trace selection (space-separated)')
    
    args = parser.parse_args()
    main(args)