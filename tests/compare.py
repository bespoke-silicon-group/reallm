import argparse, pickle, sys, functools

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def split_sys(systems, key):
    sys_dict = dict()
    for _sys in systems:
        val = rgetattr(_sys, key)
        if isinstance(val, list):
            old_val = val
            val = str(old_val[0])
            for v in old_val[1: ]:
                val += ', '
                val += str(v)
        if val in sys_dict:
            sys_dict[val].append(_sys)
        else:
            sys_dict[val] = [_sys]
    return sys_dict

def get_latency_breakdown(perf, stage):
    lat = perf._get_micro_batch_latency(stage)
    io_ratio = lat.communication / lat.total
    layer_compute = 0
    layer_mem = 0
    for k in ['atten_qkv', 'atten_matmul1', 'atten_matmul2', 'atten_fc', 'fc1', 'fc2']:
        mm = getattr(lat, k)
        if mm.block_ldst_time > mm.block_comp_time:
            layer_mem += mm.block_ldst_time
        else:
            layer_compute += mm.block_comp_time
    mem_ratio = (1 - io_ratio) * layer_mem / (layer_compute + layer_mem)
    compute_ratio = (1 - io_ratio) * layer_compute / (layer_compute + layer_mem)
    if stage == 'prefill':
        latency = perf.prefill_latency
    else:
        latency = perf.generate_latency
    t_io = io_ratio * latency
    t_compute = compute_ratio * latency
    t_mem = mem_ratio * latency
    return t_io, t_compute, t_mem

def get_tco_breakdown(perf):
    srv_tco, tco_per_input_token, tco_per_output_token = perf._get_tco()

    tco_per_Mtoken = tco_per_output_token * 1e6
    capex = srv_tco.fix_part * perf.system.num_servers
    opex = srv_tco.power_part * perf.system.num_servers

    return tco_per_Mtoken, capex, opex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hardware', type=str)
    parser.add_argument('--model', type=str)
    args = parser.parse_args()
    hw = args.hardware
    model = args.model

    mismatch = False
    for comparison in ['hw', 'sw']:
        if comparison == 'hw':
            target_csv = f'tests/{hw}/{hw}.csv'
            gen_csv = f'outputs/{hw}/{hw}.csv'
        else:
            target_csv = f'tests/{hw}/{model}.csv'
            gen_csv = f'outputs/{hw}/{model}.csv'
        with open(target_csv, 'r') as f1, open(gen_csv, 'r') as f2:
            target_lines = f1.readlines()
            gen_lines = f2.readlines()
            header = target_lines[0].split(',')
            for i in range(1, len(target_lines)):
                target_specs = target_lines[i].split(',')
                gen_specs = gen_lines[i].split(',')
                for j in range(len(target_specs)):
                    if isinstance(gen_specs[j], str):
                        continue
                    if float(gen_specs[j]) == 0.0:
                        error_rate = abs(float(gen_specs[j]) - float(target_specs[j])) 
                    else:
                        error_rate = abs(float(gen_specs[j]) - float(target_specs[j])) / float(gen_specs[j])
                    if error_rate > 0.0001:
                        mismatch = True
                        print(f'Mismatch on {header[j]}. Target: {target_specs[j]}, Gen: {gen_specs[j]}')
                        break

    if not mismatch:
        print(f'Validation passed for {hw} on {model}!')