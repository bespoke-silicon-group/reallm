import math

mapping_default = {
    't_srv': 1, # total used srvs per pipe stage
    't_pkg': 1, # total used pkgs per pipe stage
    't_chip': 1, # total used chips per pipe stage
    'p': 1, # p * t_chips = total chips 
    'partition': {'FC1': 'col','FC2': 'row'},
    'all_srv': 1,
}

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

def generate_mappings(sys_spec, algo_spec):

    num_srvs = sys_spec['num_srvs']
    pkgs_per_srv = sys_spec['pkgs_per_srv']
    chips_per_pkg = sys_spec['chips_per_pkg']
    chip_sram = sys_spec['sram_per_chip']
    chip_kv_cache_max = chip_sram/2 
    chips_per_srv = pkgs_per_srv * chips_per_pkg
    srv_sram = chip_sram * chips_per_srv # in MB
    srv_kv_cache_max = srv_sram/2

    num_layers = algo_spec['num_layers']
    model_d = algo_spec['d']
    data_bytes = algo_spec['bytes_per_number']
    batch_size = algo_spec['batch_size']
    num_trans_stages = num_layers * 3
    model_param_size = num_trans_stages * (4 * model_d * model_d) * data_bytes # in bytes
    kv_cache_size =  num_layers * 2 * algo_spec['max_ctx_len'] * model_d * data_bytes # in bytes, per input
    if 'num_heads' in algo_spec:
       kv_cache_size /= algo_spec['num_heads']
    # if (batch_size * kv_cache_size) <= model_param_size:
    #   real_ctx_len = algo_spec['max_ctx_len']  
    # else:
    #   real_ctx_len = math.floor(model_param_size / (batch_size * num_layers * 2 * model_d * data_bytes))
    #   kv_cache_size =  num_layers * 2 * real_ctx_len * model_d * data_bytes # in bytes, per input
    total_mem_size = (model_param_size + batch_size * kv_cache_size) / 1e6 # in MB

    # generate pipeline parallelism size p, which is the number of pipeline stages
    all_p = []
    # for trans_stages_per_pipe_stage in range(1, num_trans_stages+1):
    #   p = math.ceil(num_trans_stages/trans_stages_per_pipe_stage)
    #   if p not in all_p:
    #     all_p.append(p)
    for i in range(1, num_layers+1):
       if num_layers % i == 0:
          all_p.append(i)

    mappings = []
    # generate tensor parallelism size t, which is the number of chips being used per pipeline stage
    for p in all_p:
      # this number could be a bit small for unbalanced pipelines
      mem_per_pipe_stage = total_mem_size / p
      pipe_stages_per_srv = srv_sram / mem_per_pipe_stage
      if pipe_stages_per_srv < 2:
        min_srvs = math.ceil(mem_per_pipe_stage / srv_sram)
        mapping = mapping_default.copy()
        mapping['p'] = p
        mapping['t_srv']   = min_srvs
        mapping['t_pkg']   = min_srvs * pkgs_per_srv
        mapping['t_chip']  = min_srvs * pkgs_per_srv * chips_per_pkg
        mapping['all_srv'] = min_srvs * p
        mappings.append(mapping)
      else:
        for srv_stages in range(1, math.floor(pipe_stages_per_srv)+1):
          # Todo: think about what if we use less pkgs per stage
          min_pkgs = math.ceil(1.0 / srv_stages * pkgs_per_srv)
          # 1 chip per pkg for now
          if min_pkgs * srv_stages <= pkgs_per_srv:
            mapping = mapping_default.copy()
            mapping['p'] = p
            if srv_stages == 1:
              mapping['t_srv'] = 1
            else:
              mapping['t_srv']  = 1.0 / srv_stages
            mapping['t_pkg'] = min_pkgs
            mapping['t_chip'] = mapping['t_pkg'] * chips_per_pkg
            mapping['all_srv'] = math.ceil(mapping['t_srv'] * p)
            mappings.append(mapping)

    return mappings

# Calculate latency
def get_latency(sys_spec, algo_spec, mapping, micro_batch, ts=None, prefill=None):

    d = algo_spec['d']
    batch = algo_spec['batch_size']
    data_bytes = algo_spec['bytes_per_number']
    if prefill:
      activation_row  = algo_spec['max_ctx_len'] * batch
    else:
      # activation_row  = batch
      activation_row  = micro_batch

    activation_col  = d
    activation_size = activation_row * activation_col

    layers_per_pipe_stage = algo_spec['num_layers'] / mapping['p'] # 0.33 means one transformer stage

    chip_tops = sys_spec['tops_per_chip']
    # assume 1 GHz
    chip_macs = int(chip_tops/2*1e3)
    chip_macs_row = round(math.sqrt(chip_macs))
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
        stage2stage_bw = s2s_bw
    elif srvs == 1:
        # 2. One server
        link_GBs = p2p_bw
        stage2stage_bw = s2s_bw
    else:
      # 3. < 1 server
        link_GBs = p2p_bw
        total_srvs = round(srvs * mapping['p']) # check this later
        num_stage2stage_links = mapping['p']-1
        num_srv_links = total_srvs-1 
        srvs_link_delay = num_srv_links/s2s_bw
        chip_link_delay = (num_stage2stage_links-num_srv_links)/p2p_bw
        stage2stage_bw = num_stage2stage_links/(srvs_link_delay+chip_link_delay)

        if mapping['p'] == 1:
          stage2stage_bw  = s2s_bw
        # if pkgs > 1:
        #     # 3. Multi pkgs
        #     link_GBs = p2p_bw
        #     stage2stage_bw = p2p_bw
        # elif pkgs == 1:
        #     # 4. One pkgs
        #     link_GBs = c2c_bw
        #     stage2stage_bw = p2p_bw
        # else:
        #     # 5. Multi chips
        #     link_GBs = c2c_bw
        #     stage2stage_bw = c2c_bw
    stage2stage_delay = activation_size*data_bytes/(stage2stage_bw*1e9) *1e6 + T_start # in us

    t = chips

    #######################################
    # Atten Delay (atten FC only)
    #######################################
    # Now we only consider this partition:
    # Q, K, V matrix
    # all chips get the whole activation of size (activation_row, d)
    # each chip has weight of size (d, 3d/t) --> d rows, 3d/t cols, 
    # so each chip: (activation_row, d) * (d, 3d/t)
    # the max macs per cycle is: 
    max_mac_parall = activation_row * math.ceil(3*d/t)
    atten_qkv_util = min(max_mac_parall/chip_macs_row, 1.0)
    atten_qkv_delay = activation_row*d*math.ceil(3*d/t)*2/(atten_qkv_util*chip_tops*1e12) * 1e6 # in us
    if hbm_bw != None:
      atten_qkv_delay = 3*d*d*2/t/(hbm_bw*1e9) * 1e6 # in us
    # atten last FC
    # each chip get the activation of size (activation_row, d/t)
    # each chip has weight of size (d/t, d) --> d/t rows, d cols, 
    # each chip: (activation_row, d/t) * (d/t, d)
    # so the max macs per cycle is: 
    max_mac_parall = activation_row * d
    max_mac_parall = math.ceil(d/t)
    atten_fc_util = min(max_mac_parall/chip_macs_row, 1.0)
    atten_fc_delay = activation_row*math.ceil(d/t)*d*2/(atten_fc_util*chip_tops*1e12) * 1e6 # in us
    if hbm_bw != None:
      atten_fc_delay = d*d*2/t/(hbm_bw*1e9) * 1e6 # in us

    #######################################
    # FC Delay
    #######################################
    # Now we only consider this partition:
    # FC 1
    # all chips get the whole activation of size (activation_row, d)
    # each chip has weight of size (d, 4d/t) --> d rows, 4d/t cols
    # so each chip: (activation_row, d) * (d, 4d/t)
    # the max macs per cycle is: 
    max_mac_parall = activation_row * math.ceil(4*d/t)
    fc1_util = min(max_mac_parall/chip_macs_row, 1.0)
    fc1_delay = activation_row*d*math.ceil(4*d/t)*2/(fc1_util*chip_tops*1e12) * 1e6 # in us
    if hbm_bw != None:
      fc1_delay = 4*d*d*2/t/(hbm_bw*1e9) * 1e6 # in us
    # FC 2
    # each chip get the activation of size (activation_row, 4d/t)
    # each chip has weight of size (4d/t, d) --> 4d/t rows, d cols, 
    # each chip: (activation_row, 4d/t) * (4d/t, d)
    # so the max macs per cycle is: 
    max_mac_parall = activation_row * d
    max_mac_parall = math.ceil(4*d/t)
    fc2_util = min(max_mac_parall/chip_macs_row, 1.0)
    fc2_delay = activation_row*math.ceil(4*d/t)*d*2/(fc2_util*chip_tops*1e12) * 1e6 # in us
    if hbm_bw != None:
      fc2_delay = 4*d*d*2/t/(hbm_bw*1e9) * 1e6 # in us

    if layers_per_pipe_stage >=1:
      # atten:
      atten_delays = [atten_qkv_delay, atten_fc_delay, ring_allreduce(t, activation_size*data_bytes, link_GBs, T_start)]
      # fc1: fc1_compute
      fc1_delays = [fc1_delay]
      # fc2: fc2_compute + all_reduce
      fc2_delays = [fc2_delay, ring_allreduce(t, activation_size*data_bytes, link_GBs, T_start)]

      pipe_stage_delay = (sum(atten_delays) + sum(fc1_delays) + sum(fc2_delays)) * layers_per_pipe_stage + stage2stage_delay

      total_delay = pipe_stage_delay * mapping['p']

      compute_delay = mapping['p'] * (atten_qkv_delay+atten_fc_delay+fc1_delay+fc2_delay) * layers_per_pipe_stage

      critical_pipe_stage_delay = pipe_stage_delay

    elif layers_per_pipe_stage > 0.4: 
      # 2 layers per 3 pipe stages,
      # pipeline stage could be: (atten, fc1), (fc2, atten), (fc1, fc2)
      # (atten, fc1)
      atten_delays_1 = [atten_qkv_delay, atten_fc_delay, ring_allreduce(t, activation_size*data_bytes, link_GBs, T_start)]
      fc1_delays_1 = [fc1_delay]
      pipe_stage_delay_1 = sum(atten_delays_1) + sum(fc1_delays_1) + stage2stage_delay*4
      # (fc2, atten)
      fc2_delays_2 = [fc2_delay, ring_allreduce(t, activation_size*data_bytes, link_GBs, T_start)]
      atten_delays_2 = [atten_qkv_delay, atten_fc_delay, pipeline_collective(t, activation_size*data_bytes, link_GBs, T_start)]
      pipe_stage_delay_2 = sum(fc2_delays_2) + sum(atten_delays_2) + stage2stage_delay
      # (fc1, fc2)
      fc1_delays_3 = [fc1_delay]
      fc2_delays_3 = [fc2_delay, pipeline_collective(t, activation_size*data_bytes, link_GBs, T_start)]
      pipe_stage_delay_3 = sum(fc1_delays_3) + sum(fc2_delays_3) + stage2stage_delay
      
      total_delay = (pipe_stage_delay_1 + pipe_stage_delay_2 + pipe_stage_delay_3) * mapping['p'] / 3

      compute_delay = mapping['p'] * (atten_qkv_delay+atten_fc_delay+fc1_delay+fc2_delay) * layers_per_pipe_stage

      critical_pipe_stage_delay = max(pipe_stage_delay_1, pipe_stage_delay_2, pipe_stage_delay_3)

    else: 
      # 1 layers per 3 pipe stages --> 1 transforer stage is 1 pipe stage
      # Atten: atten_compute + reduce
      atten_delays = [atten_qkv_delay, atten_fc_delay, pipeline_collective(t, activation_size*data_bytes, link_GBs, T_start)]
      # fc1: bcast with fc1_compute
      fc1_delays = [fc1_delay]
      # fc2: fc2_compute + reduce
      fc2_delays = [fc2_delay, pipeline_collective(t, activation_size*data_bytes, link_GBs, T_start)]

      layer_delay = sum(atten_delays) + stage2stage_delay + sum(fc1_delays) + 4*stage2stage_delay + sum(fc2_delays) + stage2stage_delay
      total_delay = algo_spec['num_layers'] * layer_delay

      compute_delay = mapping['p'] * (atten_qkv_delay+atten_fc_delay+fc1_delay+fc2_delay) * layers_per_pipe_stage

      critical_pipe_stage_delay = max(sum(atten_delays)+stage2stage_delay, sum(fc1_delays)+4*stage2stage_delay, sum(fc2_delays)+stage2stage_delay)

    communicate_delay = total_delay - compute_delay

    return total_delay, [compute_delay, communicate_delay, critical_pipe_stage_delay]

def opt_mapping(sys, model, ts=None):
  all_mappings = generate_mappings(sys, model)
  best_latency = 100000000000
  best_tput = 0.000000001
  
  all_results = []
  large_batch = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] 
  for context_len in [256, 2048, 8196]:
    model['max_ctx_len'] = context_len
    for batch in large_batch:
      model['batch_size'] = batch
      all_mappings = generate_mappings(sys, model)
      for mapping in all_mappings:
        for micro_batch in large_batch:
          if micro_batch <= batch:
            micro_batch_latency, detail_delay = get_latency(sys, model, mapping, micro_batch, ts, None)
            critical_pipe_stage_delay = detail_delay[2]
            if batch == 1:
              latency = micro_batch_latency
              tput = 1e6/latency
              multi_micro_batch_pipe_stage_delay = 0.0
            else:
              # latency = micro_batch_latency + critical_pipe_stage_delay*(batch/micro_batch)
              # the first token is generated and first pipeline stage is finished
              multi_micro_batch_pipe_stage_delay = critical_pipe_stage_delay*(batch/micro_batch)
              latency = max(micro_batch_latency, multi_micro_batch_pipe_stage_delay)
              tput = 1e6/latency * batch

            mapping_batch = {'t': mapping['t_chip'], 'p': mapping['p'], 'srvs': mapping['all_srv'], 
                             'batch': batch, 'micro_batch':micro_batch,
                             'micro_batch_latency': micro_batch_latency,
                             'batch_pipeline_latency': multi_micro_batch_pipe_stage_delay,
                             'real_ctx_len': context_len}

            if latency < best_latency:
              best_latency = latency
              best_latency_tput = tput

            if tput > best_tput:
              best_tput = tput
              best_tput_latency = latency

            all_results.append([mapping_batch, latency, detail_delay, tput])

  
  return best_latency, best_tput, all_results

