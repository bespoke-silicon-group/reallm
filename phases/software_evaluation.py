import argparse, pickle, time, yaml, os
import multiprocessing
from typing import Optional
from structs.Model import Model
from structs.System import System
from structs.HardwareConfig import expand_dict
from utils.performance_dump import perf_to_csv

def system_eval(config: dict, verbose: bool = False) -> Optional[System]:
    system = System(**config)
    if system.valid:
        return system
    else:
        if verbose:
            print(f'Invalid system design: {system.invalid_reason}')
        return None

def software_evaluation(model_config: dict, sys_config: Optional[dict], hw_pickle: str, results_dir: str, verbose: bool = False):

    model_name = model_config['Model']['name']
    model = Model(**model_config['Model'])
    print('Generated design points for:', model_name)

    if sys_config is not None:
        if 'workload' in sys_config:
            if 'max_batch' in sys_config['workload']:
                sys_config['max_batch'] = sys_config['workload']['max_batch']
            if 'eval_len' in sys_config['workload']:
                sys_config['eval_len'] = sys_config['workload']['eval_len']
            sys_config.pop('workload')
    else:
        sys_config = dict()
        sys_config['num_servers'] = []
        max_num_servers = model.num_layers
        num_servers = 1
        while num_servers <= max_num_servers:
            if model.num_layers % num_servers == 0:
                # only consider the case where the number of layers is divisible by the number of servers
                sys_config['num_servers'].append(num_servers)
            num_servers += 1
    sys_config['model'] = model
    all_configs = expand_dict(sys_config)

    system_eval_args = []
    with open(hw_pickle, 'rb') as f:
        servers = pickle.load(f)
        for srv in servers:
            for cfg in all_configs:
                config = cfg.copy()
                config['server'] = srv
                system_eval_args.append((config, verbose))

    start_time = time.time()
    all_systems = []
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_systems = pool.starmap(system_eval, system_eval_args)
    
    # Remove None values
    all_systems = [sys for sys in all_systems if sys is not None]
    elapsed_time = time.time() - start_time

    hardware_name = hw_pickle.split('/')[-1].split('.')[0]
    result_pickle_path = f'{results_dir}/{hardware_name}/{model.name}.pkl'
    with open(result_pickle_path, 'wb') as f:
      pickle.dump(all_systems, f)
    # comment out the performance dump for now 
    # csv_path = f'{results_dir}/{hardware_name}/{model.name}.csv' 
    # perf_to_csv(result_pickle_path, csv_path)
    print(f'Finished evaluating {len(system_eval_args)} systems in {elapsed_time} seconds.')

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--hardware', type=str)
    parser.add_argument('--sys-config', type=str)
    parser.add_argument('--results-dir', type=str, default='outputs')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    model_config = yaml.safe_load(open(args.model, 'r'))
    if os.path.exists(args.sys_config) == False:
        args.sys_config = 'inputs/software/system/sys_default.yaml'
        print(f'Warning: System configuration file not found. Using default configuration: {args.sys_config}')
    sys_config = yaml.safe_load(open(args.sys_config, 'r'))
    software_evaluation(model_config, sys_config, args.hardware, args.results_dir, args.verbose)

    