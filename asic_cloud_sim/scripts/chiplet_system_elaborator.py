import CONSTANTS
from DieCost import die_yield_calc, die_cost_calc, die_area_calc
import cStringIO
import TCO
import FabCost
from Heatsink import evalHS
import ServerCost
import utils
import math

def chiplet_elaborator(design): 
  # CONSTANTS
  # SRAM density mm2/MB, data from real implementations
  real_sram_size = {'16nm': 1.28, '7nm': 0.45, '5nm': 0.35}
  # SRAM dynamic power W/MB, fake data
  real_sram_power = {'7nm': 0.0, '5nm': 0.0}
  # MACs density mm2/Tera BF16 ops, data from real implementations
  real_macs_size = {'7nm': 0.29, '5nm': 0.22}
  # MACs dynamic power W/Tera BF16 ops, fake data
  real_macs_power = {'7nm': 0.8, '5nm': 0.6}
  # IOs density mm2/count, each one is 12.5GB/s, 7 and 5nm is fake
  real_io_size = {'16nm': 0.3, '7nm': 0.3, '5nm': 0.3}
  # IOs power W/count, each one is 12.5GB/s, 7 and 5nm is fake
  real_io_power = {'16nm': 0.175, '7nm': 0.175, '5nm': 0.175}

  # Needed memory for each board in MB, 
  mem_per_board = 3840 # GPT3

  [tech, tops, mem_size, IO_count, liquid_cool] = design
  print_spec = {'tech_node': tech}
  results = cStringIO.StringIO() 

  # calculate chiplet MAC, memory and IO area per die
  asic_spec = {'tech_node':tech, 'lgc_vdd':0.8, 'sram_vdd':0.8, 'frequency': 1, 'other_area': 10.0}
  asic_spec['lgc_area'] = tops * real_macs_size[tech]
  asic_spec['sram_area'] = mem_size * real_sram_size[tech]
  asic_spec['io_area'] = IO_count * real_io_size[tech]
  asic_spec['tops_per_asic'] = tops
  asic_spec['sram_per_asic'] = mem_size

  # calculate die area and cost (performance is dummy)
  die_area = die_area_calc(asic_spec) 
  if (die_area > 600.0):
    print 'in tech of',tech,'depth of',depth,'cannot be fitted', ',', ' die area is:', die_area
    return
  
  asic_spec['die_area'] = die_area

  # calculate die yield and cost
  die_yield = die_yield_calc(die_area, tech)
  (die_cost, dpw) = die_cost_calc(die_area, die_yield, tech)
  print_spec['die_cost_init'] = die_cost
  print_spec['chiplet_yield'] = die_yield
  
  # calculate # of chiplets per board
  chiplets_per_board = math.ceil(mem_per_board / mem_size)
  chiplets_per_lane = math.ceil(chiplets_per_board / CONSTANTS.lanes_per_server)
  used_lanes = math.ceil(chiplets_per_board / chiplets_per_lane)

  print_spec['chiplets_per_lane'] = chiplets_per_lane
  print_spec['die_area_init'] = die_area
  print_spec['total_silicon_area_init'] = die_area * chiplets_per_board
  print_spec['total_silicon_cost_init'] = die_cost * chiplets_per_board
 
  # Calculate power and performance based on provisioning
  asic_spec['w_lgc']  = tops * real_macs_power[tech]
  asic_spec['w_sram'] = mem_size * real_sram_power[tech]
  asic_spec['w_io']   = IO_count * real_io_power[tech]
  asic_spec['watts_per_asic'] = asic_spec['w_lgc'] + asic_spec['w_sram'] + asic_spec['w_io']
  asic_spec['joules_per_tops'] = asic_spec['watts_per_asic'] / \
                                  asic_spec['tops_per_asic'] 

  print_spec['chiplet_power'] = asic_spec['watts_per_asic']
  if liquid_cool:
    # Fake water cooling
    hs_cost = 'water'
    print_spec['max_power_per_lane'] = 500.0
    print_spec['max_die_power_init'] = 500.0/chiplets_per_lane
    max_die_power = 500.0/chiplets_per_lane
  else:
    # Heat sink elaboration
    flag = False
    (max_die_power, hs_cost) = evalHS(die_area, chiplets_per_lane)
    print_spec['max_die_power_init'] = max_die_power
    print_spec['max_power_per_lane'] = max_die_power*chiplets_per_lane
    while (max_die_power <= asic_spec['watts_per_asic']) and (die_area <= 600.0):
      die_area += 5
      (max_die_power, hs_cost) = evalHS(die_area, chiplets_per_lane)
      flag = True   

    if hs_cost == 0:
      die_area = asic_spec['die_area']
      (max_die_power, hs_cost) = evalHS(die_area, chiplets_per_lane)
      flag = False
      if hs_cost == 0:
        return
    
    if (flag):
      #  print 'in tech of',tech,'depth of',depth,',','for ', chiplets_per_lane, \
            #  ' ASICs per lane dark silicon is used (',die_area-asic_spec['die_area'],'mm)'
      asic_spec['die_area'] = die_area
      die_yield = die_yield_calc(die_area, tech)
      (die_cost, dpw) = die_cost_calc(die_area, die_yield, tech)

  print_spec['die_area_last'] = die_area
  print_spec['dark_silicon_used'] = print_spec['die_area_last'] - print_spec['die_area_init']
  print_spec['die_cost_last'] = die_cost
  print_spec['total_silicon_area_last'] = die_area * chiplets_per_board
  print_spec['total_silicon_cost_last'] = die_cost * chiplets_per_board
    
  # Server cost and power calculation  
  srv_spec = ServerCost.evalServerCost(asic_spec, die_cost, hs_cost, chiplets_per_lane, used_lanes)

  # Evaluate TCO
  dc_spec = TCO.evalTCO(srv_spec['server_power'],
                        srv_spec['server_cost'],
                        CONSTANTS.SrvLife)

  # ----------------------------------------------------------------------------
  extra_spec = utils.extra_specs_calculator (asic_spec, srv_spec, 
                                             dpw, max_die_power, tech,
                                             die_yield)
  
  utils.fprintData([asic_spec, srv_spec, dc_spec, extra_spec], results)

  print asic_spec['tech_node'], asic_spec['tops_per_asic'], asic_spec['sram_per_asic'], chiplets_per_board, asic_spec['die_area'], asic_spec['watts_per_asic']
    
  return results.getvalue() 

