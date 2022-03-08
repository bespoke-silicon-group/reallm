import math
import csv

app_layers_size = {
    'GPT3': {'Q': 320, 'Atten_FC': 320, 'FC1': 1280, 'FC2': 1280},
    'MT-NLG-Atten': {'Q': 850, 'Atten_FC': 850},
    'MT-NLG-FC': 3400
    }

INPUT_LEN = 50

def generate_routings(app, chiplet_size, chiplets_per_board):
  if app == 'BERT' or app == 'GPT2' or app == 'T-NLG': # multi-layer per chiplet
    # possible routing = total chiplets
    routings = [chiplets_per_board]
  elif app == 'GPT3':
    # possible_routings = {'layer name': [[n_C, n_M], ]}
    possible_routings = {'Q':[], 'Atten_FC':[], 'FC1':[], 'FC2':[]}
    for layer in ['Q', 'Atten_FC', 'FC1', 'FC2']:
      layer_size = app_layers_size[app][layer]
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
  elif app == 'MT-NLG-Atten':
    possible_routings = {'Q':[], 'Atten_FC':[]}
    for layer in ['Q', 'Atten_FC']:
      layer_size = app_layers_size[app][layer]
      num_chiplets = int(math.ceil(layer_size/chiplet_size))
      for i in range(1,num_chiplets+1):
        if num_chiplets % i == 0:
          possible_routings[layer].append([i, int(num_chiplets/i)])
        else:
          continue
    routings = []
    for Q_routing in possible_routings['Q']:
      for AFC_routing in possible_routings['Atten_FC']:
        new_routing = {'Q':Q_routing, 'K':Q_routing, 'V':Q_routing, 'Atten_FC':AFC_routing}
        routings.append(new_routing)
  elif app == 'MT-NLG-FC':
    routings = []
    num_chiplets = chiplets_per_board
    for i in range(1,num_chiplets+1):
      if num_chiplets % i == 0:
        routings.append([i, int(num_chiplets/i)])
      else:
        continue
  else:
    print('Error! Unknown applications!')
    routings = None
  
  return routings

def analysis(app, routing, link_GBs, chiplet_TOPS, verbose=False):

  if app == 'BERT':
    num_boards = 1
    chiplets_per_board = routing
    chiplet_time = (INPUT_LEN * 302e6 * 2) / (chiplets_per_board * chiplet_TOPS * 1e12) * 1e6 # in us
    link_time = (INPUT_LEN * 1024 * 2) / (link_GBs * 1e9) * 1e6 # in us
    board_delay = chiplets_per_board * (chiplet_time + link_time)
    if link_time > chiplet_time:
      bottleneck = link_time
      constraints = 'link'
    else:
      bottleneck = chiplet_time
      constraints = 'compute'
    if verbose:
      print(app, 'num of chiplets', chiplets_per_board, 'chiplet_time is ', chiplet_time, 'link time is', link_time)
  elif app == 'GPT2':
    num_boards = 1
    chiplets_per_board = routing
    chiplet_time = (1.48e9 * 2) / (chiplets_per_board * chiplet_TOPS * 1e12) * 1e6 # in us
    link_time = (12800*2) / (link_GBs * 1e9) * 1e6 # in us
    board_delay = chiplets_per_board * (chiplet_time + link_time)
    if link_time > chiplet_time:
      bottleneck = link_time
      constraints = 'link'
    else:
      bottleneck = chiplet_time
      constraints = 'compute'
    if verbose:
      print(app, 'num of chiplets', chiplets_per_board, 'chiplet_time is ', chiplet_time, 'link time is', link_time)
  elif app == 'T-NLG':
    num_boards = 10
    chiplets_per_board = routing
    chiplet_time = (218e6 * 8 * 2) / (chiplets_per_board * chiplet_TOPS * 1e12) * 1e6 # in us
    link_time = (17024*2) / (link_GBs * 1e9) * 1e6 # in us
    board_delay = chiplets_per_board * (chiplet_time + link_time)
    if link_time > chiplet_time:
      bottleneck = link_time
      constraints = 'link'
    else:
      bottleneck = chiplet_time
      constraints = 'compute'
    if verbose:
      print(app, 'num of chiplets', chiplets_per_board, 'chiplet_time is ', chiplet_time, 'link time is', link_time)
  elif app == 'GPT3':
    num_boards = 96
    
    # Now assume Q, K, V have the same routings
    if routing['K'] != routing['Q'] or routing['V'] != routing['Q']:
      print('QKV routing error!')

    D = 12288
    # model = {'layer name': [C, M]}
    model = {'Q': [D, D], 'K': [D, D], 'V': [D, D], 'Atten_FC': [D, D], 'FC1': [D, 4*D], 'FC2': [4*D, D]} 
  
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
    board_delay = 0
    constraints = None
    for layer in ['Q', 'Atten_FC', 'FC1', 'FC2']:
      num_chiplets = routing[layer][0] * routing[layer][1]
      if layer == 'Q':
        input_delay = all_in_traffics[layer]*2/(link_GBs*1e9) *1e6 # in us
      else:
        input_delay = all_in_traffics[layer]*num_chiplets*2/(pre_out_chiplets*link_GBs*1e9) *1e6 # in us
    
      mid_link_delay = all_mid_traffics[layer]*2/(link_GBs*1e9) *1e6 # in us
      mid_links_delay = (all_compute_stages[layer]-1)*mid_link_delay
    
      stage_delay = model[layer][0]*model[layer][1]*2/num_chiplets/(chiplet_TOPS*1e12) * 1e6 # in us
      compute_delay = all_compute_stages[layer]*stage_delay
    
      if input_delay > bottleneck:
        bottleneck = input_delay
        constraints = 'link'
      if mid_link_delay > bottleneck:
        bottleneck = mid_link_delay
        constraints = 'link'
      if stage_delay > bottleneck:
        bottleneck = stage_delay
        constraints = 'compute'
      board_delay += (input_delay+mid_links_delay+compute_delay)
    
      pre_out_chiplets = all_out_chiplets[layer]
    
      if verbose:
        print('GPT3 ', layer, 'Layers: ')
        print(input_delay,  mid_links_delay, compute_delay)
  elif app == 'MT-NLG-Atten':
    num_boards = 105
    # Now assume Q, K, V have the same routings
    if routing['K'] != routing['Q'] or routing['V'] != routing['Q']:
      print('QKV routing error!')

    D = 20480
    model = {'Q': [D, D], 'K': [D, D], 'V': [D, D], 'Atten_FC': [D, D]} 
  
    all_in_traffics    = {}
    all_mid_traffics   = {}
    all_compute_stages = {}
    all_out_chiplets   = {}
    for layer in ['Q', 'Atten_FC']:
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
    board_delay = 0
    constraints = None
    for layer in ['Q', 'Atten_FC']:
      num_chiplets = routing[layer][0] * routing[layer][1]
      if layer == 'Q':
        input_delay = all_in_traffics[layer]*2/(link_GBs*1e9) *1e6 # in us
      else:
        input_delay = all_in_traffics[layer]*num_chiplets*2/(pre_out_chiplets*link_GBs*1e9) *1e6 # in us
    
      mid_link_delay = all_mid_traffics[layer]*2/(link_GBs*1e9) *1e6 # in us
      mid_links_delay = (all_compute_stages[layer]-1)*mid_link_delay
    
      stage_delay = model[layer][0]*model[layer][1]*2/num_chiplets/(chiplet_TOPS*1e12) * 1e6 # in us
      compute_delay = all_compute_stages[layer]*stage_delay
    
      if input_delay > bottleneck:
        bottleneck = input_delay
        constraints = 'link'
      if mid_link_delay > bottleneck:
        bottleneck = mid_link_delay
        constraints = 'link'
      if stage_delay > bottleneck:
        bottleneck = stage_delay
        constraints = 'compute'
      board_delay += (input_delay+mid_links_delay+compute_delay)
    
      pre_out_chiplets = all_out_chiplets[layer]

  elif app == 'MT-NLG-FC':
    num_boards = 210
    D = 20480
    n_C = routing[0]
    n_M = routing[1]
    num_chiplets = n_C * n_M
    in_traffics = D/n_C
    out_traffics = 4*D/n_M
    out_chiplets = n_M
    compute_stages = n_C
    mid_traffics = out_traffics
  
    input_delay = in_traffics*2/(link_GBs*1e9) *1e6 # in us
    mid_link_delay = mid_traffics*2/(link_GBs*1e9) *1e6 # in us
    mid_links_delay = (compute_stages-1)*mid_link_delay
    stage_delay = D*4*D*2/num_chiplets/(chiplet_TOPS*1e12) * 1e6 # in us
    compute_delay = compute_stages*stage_delay
    bottleneck = 0.0
    if input_delay > bottleneck:
      bottleneck = input_delay
      constraints = 'link'
    if mid_link_delay > bottleneck:
      bottleneck = mid_link_delay
      constraints = 'link'
    if stage_delay > bottleneck:
      bottleneck = stage_delay
      constraints = 'compute'
    board_delay = (input_delay+mid_links_delay+compute_delay)
  else:
    print('Error!')
    

  throughput = 1e6/bottleneck
  latency = num_boards * board_delay
  if verbose:
    print('Throughput is', throughput, 'latency is ', latency, 'bottleneck is', bottleneck, constraints, 'constraints')

  return throughput, board_delay, bottleneck

def opt_routings(app, chiplet_size, TOPS, link_GBs, chiplets_per_board):
  routings = generate_routings(app, chiplet_size, chiplets_per_board)

  opt_thru = 0
  opt_thru_delay = 1e12
  opt_thru_routing = None
  opt_delay = 1e12
  opt_delay_thru = 0
  opt_delay_routing = None
  for routing in routings:
    new_thru, new_delay, bottleneck = analysis(app, routing, link_GBs, TOPS, verbose=False)
    if new_thru > opt_thru:
      opt_thru = new_thru
      opt_thru_delay = new_delay
      opt_thru_routing = routing
    elif new_thru == opt_thru:
      if new_delay < opt_thru_delay:
        opt_thru_delay = new_delay
        opt_thru_routing = routing
  
    if new_delay < opt_delay:
      opt_delay = new_delay
      opt_delay_thru = new_thru
      opt_delay_routing = routing
    elif new_delay == opt_delay:
      if new_thru > opt_delay_thru:
        opt_delay_thru = new_thru
        opt_delay_routing = routing

  return [opt_thru, opt_thru_delay, opt_thru_routing], [opt_delay, opt_delay_thru, opt_delay_routing]
