import argparse, pickle, time, yaml
import multiprocessing
from typing import Optional
from structs.Model import Model
from structs.System import System
from structs.HardwareConfig import expand_dict

def system_eval(config: dict, verbose: bool = False) -> Optional[System]:
    system = System(**config)
    if system.valid:
        return system
    else:
        if verbose:
            print(f'Invalid system design: {system.invalid_reason}')
        return None

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--hardware', type=str)
    parser.add_argument('--hw-config', type=str)
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    with open(args.model, 'rb') as f:
        model_config = yaml.safe_load(f)
        model_name = model_config['Model']['name']
        model = Model(**model_config['Model'])
    print('Generated design points for:', model_name)

    hw_config = yaml.safe_load(open(args.hw_config, 'r'))
    if 'System' in hw_config:
        if model_name in hw_config['System']:
            sys_config = hw_config['System'][model_name][0]
        elif 'all' in hw_config['System']:
            sys_config = hw_config['System']['all'][0]
        else:
            raise ValueError(f'No system configuration found for model {model_name}')
    else:
        sys_config = dict()
        sys_config['num_servers'] = []
        # # For GPT-3 exploration, set number of servers
        # if 'hbm' in args.hardware:
        #     max_num_servers = 64 # reduce search space for HBM
        # elif 'dram_3d' in args.hardware:
        #     max_num_servers = 16 # reduce search space for 3D DRAM
        # else:
        #     max_num_servers = model.num_layers
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
    with open(args.hardware, 'rb') as f:
        servers = pickle.load(f)
        for srv in servers:
            for cfg in all_configs:
                config = cfg.copy()
                config['server'] = srv
                system_eval_args.append((config, args.verbose))

    start_time = time.time()
    all_systems = []
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
        all_systems = pool.starmap(system_eval, system_eval_args)
    
    # Remove None values
    all_systems = [sys for sys in all_systems if sys is not None]
    elapsed_time = time.time() - start_time

    with open(args.results_dir+'/'+model.name+'.pkl', 'wb') as f:
      pickle.dump(all_systems, f)
    print(f'Finished evaluating {len(system_eval_args)} systems in {elapsed_time} seconds.')
