import CONSTANTS
from DieCost import die_yield_calc, die_cost_calc, lgc_sram_area_calc
import TCO
from Heatsink import evalHS
import ServerCost
import utils
import math

def chiplet_elaborator(design): 
  [tech, lanes_per_server, IO_bandwidth, silicon_per_lane, chiplets_per_lane, mac_area_ratio] = design
  
  asic_spec = {'tech_node':tech, 'lgc_vdd':0.8, 'sram_vdd':0.8, 'frequency': 1, 'other_area': 5.0, 'dram_count':0}
    
  # calculate area
  asic_spec['die_area'] = silicon_per_lane / chiplets_per_lane
  asic_spec['io_bw'] = IO_bandwidth
  IO_count = 2*(IO_bandwidth/12.5) # transmitter and receiver
  asic_spec['io_area'] = IO_count * CONSTANTS.real_io_size[tech]

  if not lgc_sram_area_calc(asic_spec, mac_area_ratio):
    return

  tops = asic_spec['lgc_area'] / CONSTANTS.real_macs_size[tech]
  mem_size = asic_spec['sram_area'] / CONSTANTS.real_sram_size[tech]
  asic_spec['tops_per_asic'] = tops
  asic_spec['sram_per_asic'] = mem_size

  die_area = asic_spec['die_area']

  # calculate die yield and cost
  die_yield = die_yield_calc(die_area, tech)
  (die_cost, dpw) = die_cost_calc(die_area, die_yield, tech)
  
  # Calculate power
  asic_spec['w_lgc']  = tops * CONSTANTS.real_total_power[tech]
  asic_spec['w_sram'] = 0.0
  asic_spec['w_io']   = IO_count * CONSTANTS.real_io_power[tech]
  asic_spec['watts_per_asic'] = asic_spec['w_lgc'] + asic_spec['w_sram'] + asic_spec['w_io'] + CONSTANTS.others_power[tech]
  
  asic_spec['die_cost'] = die_cost
  asic_spec['die_si_cost'] = 0.0
  asic_spec['si_num'] = 0
  
  
  # Heat sink elaboration
  evalHS_area = die_area
  (max_die_power, hs_cost) = evalHS(evalHS_area, chiplets_per_lane)

  keep_large_power = False
  if 1.0*max_die_power <= asic_spec['watts_per_asic']:
    if keep_large_power:
      asic_spec['asic_hot'] = 1.0
    else:
      print('TOO HOT!! in tech of', tech, tops,'TOPS, ', mem_size, 'MB', ', die area is', die_area, 'has power of ', asic_spec['watts_per_asic'], '(',                        (asic_spec['watts_per_asic']/die_area), 'W/mm2) and the max power is',max_die_power)
      return
  else:
    asic_spec['asic_hot'] = 0.0
      
  asic_spec['watts_per_mm2'] = asic_spec['watts_per_asic'] / asic_spec['die_area']

  # Server cost and power calculation
  srv_chiplets = lanes_per_server * chiplets_per_lane
  srv_tops = tops * srv_chiplets
  srv_spec = ServerCost.evalServerCost(asic_spec, die_cost, hs_cost, chiplets_per_lane, srv_chiplets, srv_tops, lanes_per_server)
  srv_spec['sram_per_server'] = mem_size * srv_chiplets
  
  srv_spec['server_chip_power'] = asic_spec['watts_per_asic']*srv_spec['asics_per_server']
  if srv_spec['server_chip_power'] > CONSTANTS.board_max_power:
    if keep_large_power:
      srv_spec['server_hot'] = 1.0
    else:
      #  print 'Board power limit exceeded!! In tech of', tech, tops,'TOPS, ', mem_size, 'MB', ', has total board chip power of ', srv_spec['server_chip_power'], 'and the max    power is 1000'
      return
  else:
    srv_spec['server_hot'] = 0.0

  # Evaluate TCO
  dc_spec = TCO.evalTCO(srv_spec['server_power'],
                        srv_spec['server_cost'],
                        CONSTANTS.SrvLife)
  
  extra_spec = utils.extra_specs_calculator(dc_spec, srv_spec, 
                                       dpw, max_die_power,
                                       die_yield)
  return asic_spec, srv_spec, dc_spec, extra_spec
