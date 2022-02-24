import pandas as pd
import graphviz
from gpt_analysis import model, opt_routings, analysis

csvfile = 'routing_opt_results.csv'
df = pd.read_csv(csvfile)
df.drop(df.columns[-1], axis=1, inplace=True)
df.columns = df.columns.str.replace('.*\]', '')

valid_df = df[df['asic_hot']==0.0]
valid_df = valid_df[valid_df['server_hot']==0.0]

opt_thru_design = valid_df[valid_df.opt_thru == valid_df.opt_thru.max()]
opt_delay_design = valid_df[valid_df.opt_delay == valid_df.opt_delay.min()]
tco_opt_delay_design = valid_df[valid_df.tco_opt_delay == valid_df.tco_opt_delay.min()]
tco_per_opt_thru_design = valid_df[valid_df.tco_per_opt_thru == valid_df.tco_per_opt_thru.min()]


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
  
  if target == 'thru':
    routing =  opt_thru_routing
  elif target == 'delay':
    routing = opt_delay_routing
  else:
    print('Target Error')
    return None

  analysis(model, routing, link_GBs, tops, True)

  return sram, tops, area, asic_power, num_chiplets, server_power, tco, routing
  

print(analyze_optmial_system(opt_thru_design, 'thru'))
print('========================================')
print(analyze_optmial_system(opt_delay_design, 'delay'))
print('========================================')
print(analyze_optmial_system(tco_per_opt_thru_design, 'thru'))
print('========================================')
print(analyze_optmial_system(tco_opt_delay_design, 'delay'))
print('========================================')

# digraph G {
#     #splines=ortho
#   fontname="Helvetica,Arial,sans-serif"
#   node [fontname="Helvetica,Arial,sans-serif"]
#   edge [fontname="Helvetica,Arial,sans-serif"]
#   graph [center=1 rankdir=LR]
#   edge [arrowsize=0.2, penwidth=0.2]
#   node [width=0.3 height=0.3 label=""]
#   { node [shape=circle style=invis]
#     1 2 3 4 5 6
#   }
#   { node [shape=square]
#     Q1 Q2 V1 V2 K1 K2 A1 A2 FC11 FC12 FC11 FC12 FC13 FC14 FC15 FC16 FC17 FC18 FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28
#   }
# 
#   1 -> Q1 -> {A1 A2} [color="#000000"]
#   2 -> Q2 -> {A1 A2} [color="#000000"]
#   3 -> V1 -> {A1 A2} [color="#000000"]
#   4 -> V2 -> {A1 A2} [color="#000000"]
#   5 -> K1 -> {A1 A2} [color="#000000"]
#   6 -> K2 -> {A1 A2} [color="#000000"]
# 
#   A1 -> {FC11 FC12 FC13 FC14 FC15 FC16 FC17 FC18} 
#   A2 -> {FC11 FC12 FC13 FC14 FC15 FC16 FC17 FC18}
#     
#   FC11 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC12 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC13 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC14 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC15 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC16 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC17 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC18 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
# 
# }
