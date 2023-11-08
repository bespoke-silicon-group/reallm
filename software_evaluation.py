import argparse
import pickle
import math
import time
import multiprocessing
from typing import Optional
from structs.Model import Model
from structs.System import System
from structs.Server import Server
from structs.Performance import Performance
from tests.models import gpt2, megatron, gpt3, gopher, mtnlg, bloom, palm, llama2

def system_eval(server: Server, model: Model, num_servers: int, max_ctx_len_batch_1: int, max_batch: int, asplos_version: bool = False) -> Optional[Performance]:
    system = System(server=server, model=model, num_servers=num_servers, max_ctx_len_batch_1=max_ctx_len_batch_1, max_batch=max_batch, asplos_version=asplos_version)
    if system.valid:
        return system.generate_throughput_opt_perf
    else:
        return None

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--hw-pkl', type=str)
    parser.add_argument('--weight-sparsity', type=str, default='0')
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    model_name = args.model
    hw_pkl = args.hw_pkl
    results_dir = args.results_dir
    if args.weight_sparsity != '0':
        weight_sparsity = float(int(args.weight_sparsity) / 100)
    else:
        weight_sparsity = 0.0
    verbose = args.verbose
    
    print('Generated design points for:', model_name)
    
    all_models = [gpt2, megatron, gpt3, gopher, mtnlg, bloom, palm, llama2]
    model = [m for m in all_models if m.name == model_name][0]
    num_servers = 64
    max_ctx_len_batch_1 = 2048 * 128
    max_batch = 1024

    system_eval_args = []
    with open(hw_pkl, 'rb') as f:
        servers = pickle.load(f)
        for server in servers:
            system_eval_args.append((server, model, num_servers, max_ctx_len_batch_1, max_batch))

    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_perf = pool.starmap(system_eval, system_eval_args)
    elapsed_time = time.time() - start_time
    print(f'Finished evaluating {len(system_eval_args)} systems in {elapsed_time} seconds.')

    best_tco = math.inf
    for throughput_opt_perf in all_perf:
        if throughput_opt_perf == None:
            continue
        if throughput_opt_perf.tco_per_token < best_tco:
            best_tco = throughput_opt_perf.tco_per_token
            generate_opt_perf = throughput_opt_perf

    with open(results_dir+'/'+model.name+'.pkl', 'wb') as f:
      pickle.dump(generate_opt_perf, f)

    system = generate_opt_perf.system

    if verbose:
        print('=================================')
        print('num of srvs:', system.num_servers, ", mb/chip:", system.server.package.chip.sram_mb, ", tops/chip", system.server.package.chip.tops, ", chips/srv", system.server.num_chips)
        print('area', system.server.package.chip.area)
        print("batch:",generate_opt_perf.batch, generate_opt_perf.mapping)
        print('tks/sec/chip:', generate_opt_perf.generate_throughput_per_chip, ', tco/1M token:', generate_opt_perf.tco_per_token * 1e6)
        print('utilization:', generate_opt_perf.generate_utilization)
        print('tco', generate_opt_perf.srv_tco.total * num_servers[model_name])
        print(generate_opt_perf.srv_tco.fix_part * num_servers[model_name], generate_opt_perf.srv_tco.power_part * num_servers[model_name], generate_opt_perf.srv_tco.fix_part * num_servers[model_name])
        print('=================================')
        mb_latency = generate_opt_perf._get_micro_batch_latency('generate')
        print('micro_batch latency:', mb_latency.total_us)
        print('compute latency:', mb_latency.compute_us, 'commu latency:', mb_latency.communication_us)
        print('pipe stage:', mb_latency.pipeline_stage_us)
        print('atten:', mb_latency.atten_qkv_us, mb_latency.atten_matmul1_us, mb_latency.atten_communication1_us, mb_latency.atten_matmul2_us, mb_latency.atten_fc_us, mb_latency.atten_communication2_us)
        print('fc:', mb_latency.fc1_us, mb_latency.fc2_us, mb_latency.fc_communication_us)

        print(system.max_ctx_len_batch_1 / 2048)
        print(system.total_mem / 1e9)

             


