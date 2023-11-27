import argparse, pickle, math, time, os, yaml
import multiprocessing
from typing import Optional
from structs.Model import Model
from structs.System import System
from structs.Server import Server
from structs.Performance import Performance

def system_eval(server: Server, srv_id: int, model: Model, num_servers: Optional[int], max_ctx_len_batch_1: int, max_batch: int, asplos_version: bool = False) -> Optional[System]:
    system = System(server=server, server_id=srv_id, model=model, num_servers=num_servers, max_ctx_len_batch_1=max_ctx_len_batch_1, max_batch=max_batch, asplos_version=asplos_version)
    if system.valid:
        return system
    else:
        return None

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--hardware', type=str)
    parser.add_argument('--max-num-servers', type=int, default=128)
    parser.add_argument('--max-ctx-len-batch-1', type=int, default=512*1024)
    parser.add_argument('--max-batch', type=int, default=1024)
    parser.add_argument('--weight-sparsity', type=str, default='0')
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model_config = yaml.safe_load(f)
        model_name = model_config['Model']['name']
        model = Model(**model_config['Model'])
    print('Generated design points for:', model_name)

    system_eval_args = []
    with open(args.hardware, 'rb') as f:
        servers = pickle.load(f)
        srv_id = 0
        for server in servers:
            # set number of servers
            # num_servers = 1
            # while num_servers <= args.max_num_servers:
            #     system_eval_args.append((server, model, num_servers, args.max_ctx_len_batch_1, args.max_batch))
            #     num_servers *= 2

            # set max context length
            for max_ctx_len_batch_1 in [128*1024]:
                system_eval_args.append((server, srv_id, model, None, max_ctx_len_batch_1, args.max_batch))
            srv_id += 1

    start_time = time.time()
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_systems = pool.starmap(system_eval, system_eval_args)
    elapsed_time = time.time() - start_time
    with open(args.results_dir+'/'+model.name+'.pkl', 'wb') as f:
      pickle.dump(all_systems, f)
    print(f'Finished evaluating {len(system_eval_args)} systems in {elapsed_time} seconds.')

             


