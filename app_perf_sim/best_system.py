import pandas as pd
from opt_routing import model, opt_routings, analysis

def analyze_optmial_system(design, target):
  design = design[design.life_time_tco == design.life_time_tco.min()]
  sram = design.iloc[0]['sram_per_asic']
  tops = design.iloc[0]['tops_per_asic']
  link_GBs = design.iloc[0]['io_bw']

  area = design.iloc[0]['die_area']
  asic_power = design.iloc[0]['watts_per_asic']
  num_chiplets = design.iloc[0]['asics_per_server']
  server_power = design.iloc[0]['server_power']
  tco = design.iloc[0]['life_time_tco']

  [opt_thru, opt_thru_delay, opt_thru_routing], [opt_delay, opt_delay_thru, opt_delay_routing] = opt_routings(model, sram, tops, link_GBs)
  
  if 'thru' in target:
    routing =  opt_thru_routing
  elif 'latency' in target:
    routing = opt_delay_routing
  else:
    print('Target Error')
    return None

  analysis(model, routing, link_GBs, tops, True)

  print(link_GBs, sram, tops, area, asic_power, num_chiplets, server_power, tco)
  print(routing)
  system = [link_GBs, sram, tops, area, asic_power, num_chiplets, server_power, tco, routing]
  return system
  
csvfile = 'routing_opt_results.csv'
df = pd.read_csv(csvfile)
df.drop(df.columns[-1], axis=1, inplace=True)
df.columns = df.columns.str.replace('.*\]', '')

valid_df = df[df['asic_hot']==0.0]
valid_df = valid_df[valid_df['server_hot']==0.0]

designs = {'thru':None, 'latency':None, 'thru_tco':None, 'latency_tco':None}
designs['thru'] = valid_df[valid_df.opt_thru == valid_df.opt_thru.max()]
designs['latency'] = valid_df[valid_df.opt_delay == valid_df.opt_delay.min()]
designs['thru_tco'] = valid_df[valid_df.tco_opt_delay == valid_df.tco_opt_delay.min()]
designs['latency_tco'] = valid_df[valid_df.tco_per_opt_thru == valid_df.tco_per_opt_thru.min()]


o_file = open('best_systems'+'.csv', 'w')
o_file.write('opt_target,bandwidth(GB/s),memory(MB),tops,chip_area(mm2),chip_power,num_chiplets,server_power,tco,routing\n')

for target in designs:
  print('================', target, 'opt==================')
  #  [link_GBs, sram, tops, area, asic_power, num_chiplets, server_power, tco, routing] = analyze_optmial_system(designs[target], target)
  system = analyze_optmial_system(designs[target], target)

  o_file.write("%s,"% target)
  for spec in system[:-1]:
    o_file.write("%s,"% spec)
  o_file.write("\"%s \""% system[-1])
  o_file.write("\n")

o_file.close()


