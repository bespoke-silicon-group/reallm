import pickle, sys, math, functools

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

def get_batch_opt_sys(systems, max_batch=1024):
    batch_prefill_lat = dict()
    batch_prefill_tco = dict()
    batch_generate_lat = dict()
    batch_generate_tco = dict()
    batch_prefill_lat_sys = dict()
    batch_prefill_tco_sys = dict()
    batch_generate_lat_sys = dict()
    batch_generate_tco_sys = dict()
    batch = 1
    while batch <= max_batch:
        batch_prefill_lat[batch] = math.inf
        batch_prefill_tco[batch] = math.inf
        batch_generate_lat[batch] = math.inf
        batch_generate_tco[batch] = math.inf
        batch *= 2

    for _sys in systems:
        batch = 1
        while batch <= max_batch:
            if batch in _sys.batch_opt_prefill_lat:
                if _sys.batch_opt_prefill_lat[batch].prefill_latency < batch_prefill_lat[batch]:
                    batch_prefill_lat[batch] = _sys.batch_opt_prefill_lat[batch].prefill_latency
                    batch_prefill_lat_sys[batch] = _sys
            if batch in _sys.batch_opt_prefill_tco:
                if _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token < batch_prefill_tco[batch]:
                    batch_prefill_tco[batch] = _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token
                    batch_prefill_tco_sys[batch] = _sys
            if batch in _sys.batch_opt_generate_lat:
                if _sys.batch_opt_generate_lat[batch].generate_latency < batch_generate_lat[batch]:
                    batch_generate_lat[batch] = _sys.batch_opt_generate_lat[batch].generate_latency
                    batch_generate_lat_sys[batch] = _sys
            if batch in _sys.batch_opt_generate_tco:
                if _sys.batch_opt_generate_tco[batch].generate_tco_per_token < batch_generate_tco[batch]:
                    batch_generate_tco[batch] = _sys.batch_opt_generate_tco[batch].generate_tco_per_token
                    batch_generate_tco_sys[batch] = _sys
            batch *= 2
    return batch_prefill_lat_sys, batch_prefill_tco_sys, batch_generate_lat_sys, batch_generate_tco_sys

def get_opt_sys_batch(systems, target='generate_tco'):
    opt = math.inf
    for batch in systems:
        _sys = systems[batch]
        if target == 'generate_tco':
            if _sys.batch_opt_generate_tco[batch].generate_tco_per_token < opt:
                opt = _sys.batch_opt_generate_tco[batch].generate_tco_per_token
                opt_sys = _sys
                opt_batch = batch
        elif target == 'generate_lat':
            if _sys.batch_opt_generate_lat[batch].generate_latency < opt:
                opt = _sys.batch_opt_generate_lat[batch].generate_latency
                opt_sys = _sys
                opt_batch = batch
        elif target == 'prefill_tco':
            if _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token < opt:
                opt = _sys.batch_opt_prefill_tco[batch].prefill_tco_per_token
                opt_sys = _sys
                opt_batch = batch
        elif target == 'prefill_lat':
            if _sys.batch_opt_prefill_lat[batch].prefill_latency < opt:
                opt = _sys.batch_opt_prefill_lat[batch].prefill_latency
                opt_sys = _sys
                opt_batch = batch
    return opt_sys, opt_batch

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

# def get_tco_detail_breakdown(perf):
#     srv_tco, _, _ = perf._get_tco()
#     capex_ratio = srv_tco.fix_part / srv_tco.total
#     capex_chip_ratio = capex_ratio * (sys.server.cost_all_package / sys.server.cost)
#     capex_sys_ratio = capex_ratio * (sys.server.cost - sys.server.cost_all_package) / sys.server.cost

#     opex_ratio = srv_tco.power_part / srv_tco.total
#     srv_power = srv_tco.srv_power
#     core_power_ratio = 1 - (sys.other_tdp / sys.num_servers / srv_power)

#     prefill_core_energy = perf._get_core_energy('prefill')
#     generate_core_energy = perf._get_core_energy('generate')
#     prefill_ratio = perf.prefill_latency / (perf.prefill_latency + perf.generate_latency)
#     generate_ratio = perf.generate_latency / (perf.prefill_latency + perf.generate_latency)

#     core_compute_energy_ratio = prefill_ratio * (prefill_core_energy.fma / prefill_core_energy.total) + \
#                                 generate_ratio * (generate_core_energy.fma / generate_core_energy.total)
#     core_mem_energy_ratio = prefill_ratio * (prefill_core_energy.mem / prefill_core_energy.total) + \
#                             generate_ratio * (generate_core_energy.mem / generate_core_energy.total)
#     core_io_energy_ratio = prefill_ratio * (prefill_core_energy.comm / prefill_core_energy.total) + \
#                             generate_ratio * (generate_core_energy.comm / generate_core_energy.total)

#     opex_compute_ratio = opex_ratio * core_power_ratio * core_compute_energy_ratio
#     opex_mem_ratio = opex_ratio * core_power_ratio * core_mem_energy_ratio
#     opex_io_ratio = opex_ratio * core_power_ratio * core_io_energy_ratio
#     opex_other_ratio = opex_ratio * (1 - core_power_ratio)

#     return capex_chip_ratio, capex_sys_ratio, opex_compute_ratio, opex_mem_ratio, opex_io_ratio, opex_other_ratio


def perf_to_csv(result_pickle_path: str, csv_path: str):
    o_file = open(csv_path, 'w')
    o_file.write('opt_goal, input_len, output_len, batch, prefill_sec, decoding_sec, tco_per_Mtoken, prefill_io_sec, prefill_compute_sec, prefill_mem_sec, decoding_io_sec, decoding_compute_sec, decoding_mem_sec, capex, opex\n')

    with open(result_pickle_path, 'rb') as f:
        all_sys = pickle.load(f)
        eval_len_sys = split_sys(all_sys, 'eval_len')

    target_batch = 32
    for opt_goal in ['prefill_lat', 'decoding_lat', 'tco']:
        for eval_len in eval_len_sys:
            max_batch = eval_len_sys[eval_len][0].max_batch
            batch_pre_lat_sys, _, batch_gen_lat_sys, batch_tco_sys = get_batch_opt_sys(eval_len_sys[eval_len], max_batch)
            # find the hardware system best for the optimization goal and target batch size
            if opt_goal == 'prefill_lat':
                _sys = batch_pre_lat_sys[target_batch]
            elif opt_goal == 'decoding_lat':
                _sys = batch_gen_lat_sys[target_batch]
            else:
                _sys = batch_tco_sys[target_batch]

            batch = 1
            while batch <= max_batch:
                # find the mapping best for the optimization goal
                if opt_goal == 'prefill_lat':
                    if batch in _sys.batch_opt_prefill_lat:
                        perf = _sys.batch_opt_prefill_lat[batch]
                    else:
                        batch *= 2
                        continue
                elif opt_goal == 'decoding_lat':
                    if batch in _sys.batch_opt_generate_lat:
                        perf = _sys.batch_opt_generate_lat[batch]
                    else:
                        batch *= 2
                        continue
                else:
                    if batch in _sys.batch_opt_generate_tco:
                        perf = _sys.batch_opt_generate_tco[batch]
                    else:
                        batch *= 2
                        continue
                prefill_io, prefill_compute, prefill_mem = get_latency_breakdown(perf, 'prefill')
                generate_io, generate_compute, generate_mem = get_latency_breakdown(perf, 'generate')
                tco_per_Mtoken, capex, opex = get_tco_breakdown(perf)
                o_file.write(f'{opt_goal}, {eval_len}, {batch}, {perf.prefill_latency}, {perf.generate_latency}, {tco_per_Mtoken}, {prefill_io}, {prefill_compute}, {prefill_mem}, {generate_io}, {generate_compute}, {generate_mem}, {capex}, {opex}\n')
                batch *= 2
    
    o_file.close()
