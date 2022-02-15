import math

def generate_routings(model, chiplet_size):
  # possible_routings = {'layer name': [[n_C, n_M], ]}
  possible_routings = {'Q':[], 'Atten_FC':[], 'FC1':[], 'FC2':[]}
  for layer in ['Q', 'Atten_FC', 'FC1', 'FC2']:
    layer_size = model[layer][0]*model[layer][1]*2 /1e6 # in MBytes
    num_chiplets = int(math.ceil(layer_size/chiplet_size))
    for i in range(1,num_chiplets+1):
      if num_chiplets % i == 0:
        possible_routings[layer].append([i, int(num_chiplets/i)])
      else:
        continue
  
  routings = []
  for Q_routing in possible_routings['Q']:
    for AFC_routing in possible_routings['Atten_FC']:
      for FC1_routing in possible_routings['FC1']:
        for FC2_routing in possible_routings['FC2']:
          new_routing = {'Q':Q_routing, 'K':Q_routing, 'V':Q_routing, 'Atten_FC':AFC_routing, 'FC1':FC1_routing, 'FC2':FC2_routing}
          routings.append(new_routing)
  
  # print(routings)
  return routings

def analysis(model, routing, link_GBs, chiplet_TOPS):
  # Now assume Q, K, V have the same routings
  if routing['K'] != routing['Q'] or routing['V'] != routing['Q']:
    print('QKV routing error!')
  
  all_in_traffics    = {}
  all_mid_traffics   = {}
  all_compute_stages = {}
  all_out_chiplets   = {}
  for layer in ['Q', 'Atten_FC', 'FC1', 'FC2']:
    n_C = routing[layer][0]
    n_M = routing[layer][1]
    in_traffics = model[layer][0]/n_C
    out_traffics = model[layer][1]/n_M
    out_chiplets = n_M
    compute_stages = n_C
    mid_traffics = out_traffics
  
    all_in_traffics[layer] = in_traffics
    all_mid_traffics[layer] = mid_traffics
    all_compute_stages[layer] = compute_stages
    all_out_chiplets[layer] = out_chiplets
  
  bottleneck = 0
  layer_delay = 0
  for layer in ['Q', 'Atten_FC', 'FC1', 'FC2']:
    num_chiplets = routing[layer][0] * routing[layer][1]
    if layer == 'Q':
      input_delay = all_in_traffics[layer]*2/(link_GBs*1e9) *1e6 # in us
    else:
      input_delay = all_in_traffics[layer]*num_chiplets*2/(pre_out_chiplets*link_GBs*1e9) *1e6 # in us
  
    mid_link_delay = all_mid_traffics[layer]*2/(link_GBs*1e9) *1e6 # in us
    mid_links_delay = (all_compute_stages[layer]-1)*mid_link_delay
  
    stage_delay = model[layer][0]*model[layer][1]*2/num_chiplets/(chiplet_TOPS*1e12) * 1000000 # in us
    compute_delay = all_compute_stages[layer]*stage_delay
  
    if input_delay > bottleneck:
      bottleneck = input_delay
    if mid_link_delay > bottleneck:
      bottleneck = mid_link_delay
    if stage_delay > bottleneck:
      bottleneck = stage_delay
    layer_delay += (input_delay+mid_links_delay+compute_delay)
  
    pre_out_chiplets = all_out_chiplets[layer]
  
    #  print(layer, 'Layers: ')
    #  print(input_delay,  mid_links_delay, compute_delay)
  
  #  print('Throughput is', 1e6/bottleneck, 'latency is ', 96*layer_delay)

  return 1e6/bottleneck, 96*layer_delay, bottleneck



D = 12288
# model = {'layer name': [C, M]}
model = {'Q': [D, D], 'K': [D, D], 'V': [D, D], 'Atten_FC': [D, D], 'FC1': [D, 4*D], 'FC2': [4*D, D]} 


chiplet_sizes = [40, 80, 160, 320] # MB
chiplet_TOPS = [40, 80, 160, 320] # TOPS

link_GBs = 100 
freq_GHz = 1 

for chiplet_size in chiplet_sizes:
  for TOPS in chiplet_TOPS:
    routings = generate_routings(model, chiplet_size)

    throughput = 0
    latency = 1e12
    th_best_routing = None
    bb = 0
    for routing in routings:
      new_th, new_delay, bottleneck = analysis(model, routing, link_GBs, TOPS)
      if new_th > throughput:
        throughput = new_th
        latency = new_delay
        th_best_routing = routing
        bb = bottleneck
      elif new_th == throughput:
        if new_delay < latency:
          latency = new_delay
          th_best_routing = routing
          bb = bottleneck
      #  if new_delay <= latency:
        #  latency = new_delay
        #  latency_best = routing
    
    #  print('========',chiplet_size,'MB  ', TOPS, 'TOPS  =========')
    #  print(int(throughput), int(latency), 'us')
    print(int(throughput), int(latency)/1000)
    #  print(th_best_routing, bb)

