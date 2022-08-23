import math

mapping_default = {
    't_srv': 1, # total used srvs per pipe stage
    't_pkg': 1, # total used pkgs per pipe stage
    't_chip': 1, # total used  chips per pipe stage
    'p': 1, # p * t_chips = total chips 
    'partition': {'FC1': 'col','FC2': 'row'},
}

def product_of_two(num):
    pairs = []
    for i in range(1, num+1):
        if num % i == 0:
            pairs.append([i, int(num/i)])
    return pairs

########################################
# Collective Opeartion Timing
########################################
def pipeline_collective(p, n, bw, T_start=0.01):
    T_byte = 1 / bw / 1000 # in us

    if p == 1:
        return 0
    elif p == 2:
        T = T_start+n*T_byte
    else:
        m = math.ceil(math.sqrt(n*(p-2)*T_byte/T_start))
        T = (T_start+n/m*T_byte)*(p+m-2)

    return T

def ring_allreduce(p, n, bw, T_start=0.01):
    T_byte = 1 / bw / 1000 # in us

    if p == 1:
        return 0
    else:
        T = 2*(p-1)*((n/p)*T_byte+T_start)
    return T

def reduce_scatter(p, n, bw, T_start=0.01):
    T_byte = 1 / bw / 1000 # in us

    if p == 1:
        return 0
    else:
        T = (p-1)*((n/p)*T_byte+T_start)
    return T

def allgather(p, n, bw, T_start=0.01):
    T_byte = 1 / bw / 1000 # in us

    if p == 1:
        return 0
    else:
        T = (p-1)*((n/p)*T_byte+T_start)
    return T

########################################

def generate_routings(sys_spec, algo_spec):

    num_srvs = sys_spec['num_srvs']
    pkgs_per_srv = sys_spec['pkgs_per_srv']
    chips_per_pkg = sys_spec['chips_per_pkg']
    num_layers = algo_spec['num_layers']
    num_trans_stages = num_layers * 3

    mappings = []

    # Server level mapping
    if num_srvs > 1:
        all_srv_t_p = product_of_two(num_srvs)
        for srv_t_p in all_srv_t_p:
            srv_t = srv_t_p[0]
            srv_p = srv_t_p[1]
            if srv_p > num_trans_stages:
                continue
            if num_trans_stages % srv_p != 0:
                continue
            trans_stage_per_pipe_stage = num_trans_stages / srv_p
            if trans_stage_per_pipe_stage > 1 and trans_stage_per_pipe_stage % 3 != 0:
                continue

            mapping = mapping_default.copy()
            mapping['t_srv'] = srv_t
            mapping['t_pkg'] = srv_t * sys_spec['pkgs_per_srv']
            mapping['t_chip'] = srv_t * sys_spec['pkgs_per_srv'] * sys_spec['chips_per_pkg']
            mapping['p'] = srv_p

            mappings.append(mapping)
#             print('srv', mapping)

    # Package level mapping
    all_pkg_t_p = product_of_two(pkgs_per_srv)
    for pkg_t_p in all_pkg_t_p:
        pkg_t = pkg_t_p[0]
        pkg_p = pkg_t_p[1]
        total_p = pkg_p * num_srvs
        if total_p > num_trans_stages:
            continue
        if num_trans_stages % total_p != 0:
            continue
        trans_stage_per_pipe_stage = num_trans_stages / total_p
        if trans_stage_per_pipe_stage > 1 and trans_stage_per_pipe_stage % 3 != 0:
            continue

        mapping = mapping_default.copy()
        mapping['p'] = total_p
        mapping['t_srv'] = num_srvs / total_p
        if mapping['t_srv'] > 1.0:
            continue
        mapping['t_pkg'] = pkg_t
        mapping['t_chip'] = pkg_t * sys_spec['chips_per_pkg']

        mappings.append(mapping)
#         print('pkg', mapping)

    # Chip level mapping
    if chips_per_pkg > 1:
        all_chip_t_p = product_of_two(chips_per_pkg)
        for chip_t_p in all_chip_t_p:
            chip_t = chip_t_p[0]
            chip_p = chip_t_p[1]
            total_p = chip_p * pkgs_per_srv * num_srvs
            if total_p > num_trans_stages:
                continue
            if num_trans_stages % total_p != 0:
                continue
            trans_stage_per_pipe_stage = num_trans_stages / total_p
            if trans_stage_per_pipe_stage > 1 and trans_stage_per_pipe_stage % 3 != 0:
                continue

            mapping = mapping_default.copy()
            mapping['p'] = total_p
            mapping['t_srv'] = num_srvs / total_p
            mapping['t_pkg'] = (num_srvs * pkgs_per_srv) / total_p
            if mapping['t_srv'] >= 1.0 or mapping['t_pkg'] >= 1.0:
                continue
            mapping['t_chip'] = chip_t

            mappings.append(mapping)
#             print('chip', mapping)

    return mappings

# Calculate latency
def get_latency(sys_spec, algo_spec, mapping, ts=None, verbose=False, batch=1):

    d = algo_spec['d']

    layers_per_pipe_stage = algo_spec['num_layers'] / mapping['p'] # 0.33 means one transformer stage

    chip_tops = sys_spec['chip_tops']
    c2c_bw = sys_spec['c2c_bw']
    p2p_bw = sys_spec['p2p_bw']
    s2s_bw = sys_spec['s2s_bw']
    if ts==None:
        T_start = sys_spec['T_start']
    else:
        T_start = ts
    hbm_bw = sys_spec['hbm_bw']

    srvs = mapping['t_srv']
    pkgs = mapping['t_pkg']
    chips = mapping['t_chip']

    if srvs > 1:
        # 1. Multi servers
        link_GBs = s2s_bw
        stage2stage_delay = batch*d*2/(s2s_bw*1e9) *1e6 + T_start # in us
    elif srvs == 1:
        # 2. One server
        link_GBs = p2p_bw
        stage2stage_delay = batch*d*2/(s2s_bw*1e9) *1e6 + T_start # in us
    else:
        if pkgs > 1:
            # 3. Multi pkgs
            link_GBs = p2p_bw
            stage2stage_delay = batch*d*2/(p2p_bw*1e9) *1e6 + T_start # in us
        elif pkgs == 1:
            # 4. One pkgs
            link_GBs = c2c_bw
            stage2stage_delay = batch*d*2/(p2p_bw*1e9) *1e6 + T_start # in us
        else:
            # 5. Multi chips
            link_GBs = c2c_bw
            stage2stage_delay = batch*d*2/(c2c_bw*1e9) *1e6 + T_start # in us

    t = chips

    if hbm_bw == None:
        compute_time = batch*batch*4*d*d*2/t/(chip_tops*1e12) * 1e6 # in us
    else:
        compute_time = 4*d*d*2/t/(hbm_bw*1e9) * 1e6 # in us

    if layers_per_pipe_stage >=1:
        # Atten: atten_compute + all_reduce
        atten_delays = [compute_time, ring_allreduce(t, batch*d*2, link_GBs, T_start)]

        fc1 = mapping['partition']['FC1']
        fc2 = mapping['partition']['FC2']
        if fc1 == 'row':
            if fc2 == 'row':
                # fc1: fc1_compute + reduce_scatter
                fc1_delays = [compute_time, reduce_scatter(t, batch*4*d*2, link_GBs, T_start)]
                # fc2: fc2_compute + all_reduce
                fc2_delays = [compute_time, ring_allreduce(t, batch*d*2, link_GBs, T_start)]
            elif fc2 == 'col': # all_reduce
                # fc1: fc1_compute + reduce_scatter
                fc1_delays = [compute_time, ring_allreduce(t, batch*4*d*2, link_GBs, T_start)]
                # fc2: fc2_compute + all_gather
                fc2_delays = [compute_time, allgather(t, batch*d*2, link_GBs, T_start)]
        elif fc1 == 'col':
            if fc2 == 'row':
                # fc1: fc1_compute
                fc1_delays = [compute_time]
                # fc2: fc2_compute + all_reduce
                fc2_delays = [compute_time, ring_allreduce(t, batch*d*2, link_GBs, T_start)]
            elif fc2 == 'col':
                # fc1: fc1_compute + all_gather
                fc1_delays = [compute_time, allgather(t, batch*4*d*2, link_GBs, T_start)]
                # fc2: fc2_compute + all_gather
                fc2_delays = [compute_time, allgather(t, batch*d*2, link_GBs, T_start)]

            pipe_stage_delay = (sum(atten_delays) + sum(fc1_delays) + sum(fc2_delays)) * layers_per_pipe_stage + stage2stage_delay
            total_delay = pipe_stage_delay * mapping['p']

            compute_delay = mapping['p'] * compute_time * 3 * layers_per_pipe_stage

            if verbose:
                print(mapping['p'])
                print('atten:', atten_delays, 'fc1:', fc1_delays, 'fc2:', fc2_delays, 'layers/pipe:', layers_per_pipe_stage)
                print(stage2stage_delay)
                print(pipe_stage_delay, total_delay)
    else:
        # Atten: atten_compute + reduce
        atten_delays = [compute_time, pipeline_collective(t, batch*d*2, link_GBs, T_start)]
        # fc1: bcast with fc1_compute
        fc1_delays = [compute_time]
        # fc2: fc2_compute + reduce
        fc2_delays = [compute_time, pipeline_collective(t, batch*d*2, link_GBs, T_start)]

        layer_delay = sum(atten_delays) + stage2stage_delay + sum(fc1_delays) + 4*stage2stage_delay + sum(fc2_delays) + stage2stage_delay
        total_delay = algo_spec['num_layers'] * layer_delay

        compute_delay = mapping['p'] * compute_time

        if verbose:
            print(mapping['p'])
            print(atten_delays, stage2stage_delay, fc1_delays, 4*stage2stage_delay, fc2_delays, stage2stage_delay)
            print(layer_delay, total_delay)

    communicate_delay = total_delay - compute_delay
    if verbose:
        print('compute delay:', compute_delay, 'communicate delay:', communicate_delay)

    return total_delay, [compute_delay, communicate_delay]

def opt_routing(sys, model, ts=None, verbose=False, opt_target='delay'):
    all_routings = generate_routings(sys, model)
    best_latency = 100000000000
    best_perf = 0.000000001
    
    all_results = []
    if opt_target == 'delay':
        for routing in all_routings:
            latency, detail_delay = get_latency(sys, model, routing, ts, verbose, 1)
            if latency < best_latency:
                best_latency = latency
                best_routing = routing
                [compute_delay, communicate_delay] = detail_delay
                best_thru = 1e6/(best_latency/best_routing['p'])
            if verbose:
                print(routing, latency)
                all_results.append([routing, latency, detail_delay])
    elif opt_target == 'perf':
        for routing in all_routings:
            for batch in [1, 2, 4, 8, 16, 32]:
                latency, detail_delay = get_latency(sys, model, routing, ts, verbose, batch)
                thru = 1e6/(latency/routing['p'])*batch
                perf = thru / latency
                if perf > best_perf:
                    best_perf = perf
                    best_routing = routing
                    best_latency = latency
                    best_thru = thru
                    [compute_delay, communicate_delay] = detail_delay
           
    # Throuput
    # if sys['hbm_bw'] is None:
    #     best_thru = 1e6/(best_latency/best_routing['p'])*batch
    # else: # for HBM systems
    #     total_tops = sys['chip_tops']*sys['chips_per_pkg']*sys['pkgs_per_srv']*sys['num_srvs']
    #     total_ops_per_token = model['num_layers']*24*model['d']*model['d']
    #     best_thru = total_tops*1e12/total_ops_per_token
    
    if verbose:
        return best_routing, best_latency, best_thru, [compute_delay, communicate_delay], all_results
    else:
        return best_routing, best_latency, best_thru, [compute_delay, communicate_delay]
