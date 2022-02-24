import math
import csv

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
  
  return routings

def analysis(model, routing, link_GBs, chiplet_TOPS, verbose=False):
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
  constraints = None
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
      constraints = 'link'
    if mid_link_delay > bottleneck:
      bottleneck = mid_link_delay
      constraints = 'link'
    if stage_delay > bottleneck:
      bottleneck = stage_delay
      constraints = 'compute'
    layer_delay += (input_delay+mid_links_delay+compute_delay)
  
    pre_out_chiplets = all_out_chiplets[layer]
  
    if verbose:
      print(layer, 'Layers: ')
      print(input_delay,  mid_links_delay, compute_delay)
  
  if verbose:
    print('Throughput is', 1e6/bottleneck, 'latency is ', 96*layer_delay, 'bottleneck is', bottleneck, constraints, 'constraints')

  return 1e6/bottleneck, 96*layer_delay, bottleneck

def opt_routings(model, chiplet_size, TOPS, link_GBs):
  routings = generate_routings(model, chiplet_size)

  opt_thru = 0
  opt_thru_delay = 1e12
  opt_thru_routing = None
  opt_delay = 1e12
  opt_delay_thru = 0
  opt_delay_routing = None
  for routing in routings:
    new_thru, new_delay, bottleneck = analysis(model, routing, link_GBs, TOPS)
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


D = 12288
# model = {'layer name': [C, M]}
model = {'Q': [D, D], 'K': [D, D], 'V': [D, D], 'Atten_FC': [D, D], 'FC1': [D, 4*D], 'FC2': [4*D, D]} 


csvf = open('../asic_cloud_sim/results.csv')
csvreader = csv.reader(csvf)

headers = next(csvreader)[:-1]
header_index = {'sram_per_asic': None, 'tops_per_asic': None, 'io_bw': None, 'server_cost': None, 'server_power': None, 'life_time_tco': None}
for i in range(len(headers)):
  for h in header_index:
    if h in headers[i]:
      header_index[h] = i

o_file = open('routing_opt_results'+'.csv', 'w')

headers.append('opt_thru')
headers.append('opt_thru_delay')
#  headers.append('opt_thru_routing')
headers.append('watts_per_opt_thru')
headers.append('cost_per_opt_thru')
headers.append('tco_per_opt_thru')

headers.append('opt_delay')
headers.append('opt_delay_thru')
#  headers.append('opt_delay_routing')
headers.append('watts_opt_delay')
headers.append('cost_opt_delay')
headers.append('tco_opt_delay')
for h in headers[:]:
  o_file.write("%s,"% h)
o_file.write('\n')

best_tco_per_thru = 1e12
best_tco_delay = 1e12
best_tco_per_thru_design = None
best_tco_delay_design = None
for row in csvreader:
  new_row = row[:-1]
  chiplet_size = float(row[header_index['sram_per_asic']])
  TOPS = float(row[header_index['tops_per_asic']])
  link_GBs = float(row[header_index['io_bw']])
  server_cost = float(row[header_index['server_cost']])
  server_power = float(row[header_index['server_power']])
  life_time_tco = float(row[header_index['life_time_tco']])

  opt_thru_results, opt_delay_results = opt_routings(model, chiplet_size, TOPS, link_GBs)
  [opt_thru, opt_thru_delay, opt_thru_routing] = opt_thru_results
  [opt_delay, opt_delay_thru, opt_delay_routing] = opt_delay_results

  
  new_row.append(math.floor(opt_thru))
  new_row.append(opt_thru_delay/1000)
  #  new_row.append(opt_thru_routing)
  new_row.append(server_power/math.floor(opt_thru)*1000)
  new_row.append(server_cost/math.floor(opt_thru)*1000)
  tco_per_thru = life_time_tco/math.floor(opt_thru)*1000
  new_row.append(tco_per_thru)

  new_row.append(opt_delay/1000)
  new_row.append(math.floor(opt_delay_thru))
  #  new_row.append(opt_delay_routing)
  new_row.append(server_power * opt_delay/1000000)
  new_row.append(server_cost * opt_delay/1000000)
  tco_delay = life_time_tco * opt_delay/1000000
  new_row.append(tco_delay)
  for r in new_row:
    o_file.write("%s,"% r)
  o_file.write('\n')

  # if tco_per_thru < best_tco_per_thru:
  #   best_tco_per_thru_design= [chiplet_size, TOPS, best_thru_routing]
  #   best_tco_per_thru = tco_per_thru
  # if tco_delay < best_tco_delay:
  #   best_tco_delay_design = [chiplet_size, TOPS, best_thru_routing]
  #   best_tco_delay = tco_delay

  #  print('========',chiplet_size,'MB  ', TOPS, 'TOPS  =========')
  #  print(int(best_thru), int(best_thru_lat)/1000)
o_file.close()

#  print(best_tco_per_thru_design)
#  print(best_tco_delay_design)
