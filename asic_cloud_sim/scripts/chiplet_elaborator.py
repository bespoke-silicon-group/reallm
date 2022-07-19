import CONSTANTS
from DieCost import die_yield_calc, die_cost_calc, die_area_calc, si_cost_calc
import TCO
from Heatsink import evalHS
import ServerCost
import utils
import math

def chiplet_elaborator(design): 
  [app, tech, srv_mem, srv_tops, srv_chiplets, IO_BW, \
   keep_large_power, use_total_power, use_dram, \
   TPU, SI, organic_sub] = design
  
  asic_spec = {'tech_node':tech, 'lgc_vdd':0.8, 'sram_vdd':0.8, 'frequency': 1, 'other_area': 5.0}
    
  # tops and mem per chiplet
  tops = float(srv_tops) / srv_chiplets
  if use_dram:
    # mem_size = 900.0 / srv_chiplets
    mem_size = 1800.0 / srv_chiplets
    asic_spec['dram_count'] = float(srv_mem) / CONSTANTS.hbm_die_cap / srv_chiplets
  else:
    mem_size = float(srv_mem) / srv_chiplets
    asic_spec['dram_count'] = 0.0

  if TPU:
    use_dram = True
    mem_size = 144.0
    tops = 138
    asic_spec['dram_count'] = 1

  # calculate chiplet MAC, memory and IO area per die
  asic_spec['lgc_area'] = tops * CONSTANTS.real_macs_size[tech]
  asic_spec['sram_area'] = mem_size * CONSTANTS.real_sram_size[tech]
  asic_spec['io_bw'] = IO_BW
  IO_count = 2*(IO_BW/12.5) # transmitter and receiver
  asic_spec['io_area'] = IO_count * CONSTANTS.real_io_size[tech]
  asic_spec['io_area'] = 0.0
  asic_spec['tops_per_asic'] = tops
  asic_spec['sram_per_asic'] = mem_size

  # calculate die area and cost
  die_area = die_area_calc(asic_spec)
  if TPU:
    die_area = 380.0
  if (die_area > 1000.0):
    print('in tech of',tech,'cannot be fitted', ',', ' die area is:', die_area)
    return

  asic_spec['die_area'] = die_area

  # calculate die yield and cost
  die_yield = die_yield_calc(die_area, tech)
  (die_cost, dpw) = die_cost_calc(die_area, die_yield, tech)
  
  # Calculate power
  if use_total_power:
    if use_dram:
      asic_spec['w_lgc']  = tops * CONSTANTS.real_dram_total_power[tech]
    else:
      asic_spec['w_lgc']  = tops * CONSTANTS.real_total_power[tech]
    asic_spec['w_sram'] = 0.0
    asic_spec['w_io']   = IO_count * CONSTANTS.real_io_power[tech]
    asic_spec['watts_per_asic'] = asic_spec['w_lgc'] + asic_spec['w_sram'] + \
                                  asic_spec['w_io'] + CONSTANTS.others_power[tech]
    if TPU:
      asic_spec['w_io'] = 0.0
      asic_spec['w_sram'] = 0.0
      asic_spec['watts_per_asic'] = 175.0
  else:
    asic_spec['w_lgc']  = tops * CONSTANTS.real_macs_power[tech]
    asic_spec['w_reg'] = tops * CONSTANTS.real_reg_power[tech]
    asic_spec['w_sram'] = tops * CONSTANTS.real_sram_power[tech]
    asic_spec['w_dram'] = tops * CONSTANTS.real_hbm_power
    asic_spec['w_io']   = IO_count * CONSTANTS.real_io_power[tech]
    asic_spec['watts_per_asic'] = asic_spec['w_lgc'] + asic_spec['w_sram'] + \
                                  asic_spec['w_io'] + asic_spec['w_dram']

  asic_spec['joules_per_tops'] = asic_spec['watts_per_asic'] / asic_spec['tops_per_asic']
  
  
  chiplets_per_lane = math.ceil(srv_chiplets / CONSTANTS.lanes_per_server)
  if TPU:
    chiplets_per_lane = 2

  asic_spec['die_cost'] = die_cost
  if SI and chiplets_per_lane > 1:
    si_total_die_area = chiplets_per_lane * asic_spec['die_area']
    si_area = si_total_die_area / 0.9
    if si_area > CONSTANTS.SIMaxSize:
      print('Chiplets area exceed SI size!')
      print('     ', srv_tops, srv_chiplets, si_area)
    extra_cost_per_die = si_cost_calc(si_area, die_cost, chiplets_per_lane)
    asic_spec['die_si_cost'] = extra_cost_per_die
  else:
    asic_spec['die_si_cost'] = 0.0
  
  
  if False: # No water cooling for now
    # Fake water cooling
    hs_cost = 'water'
    max_die_power = 500.0/chiplets_per_lane
  else:
    # Heat sink elaboration
    if SI and chiplets_per_lane > 1:
      evalHS_area = si_area 
      (max_si_power, hs_cost) = evalHS(evalHS_area, 1)
      max_die_power = max_si_power / chiplets_per_lane
      # print('max si power is ', max_si_power, 'in', chiplets_per_lane, 'chiplets')
      # print('chiplet power is ', asic_spec['watts_per_asic'])
    else:
      evalHS_area = die_area + asic_spec['dram_count'] * CONSTANTS.real_hbm_size
      (max_die_power, hs_cost) = evalHS(evalHS_area, chiplets_per_lane)

    if 1.0*max_die_power <= asic_spec['watts_per_asic']:
      if keep_large_power:
        asic_spec['asic_hot'] = 1.0
      else:
        # print 'TOO HOT!! in tech of', tech, tops,'TOPS, ', mem_size, 'MB', ', die area is', die_area, 'has power of ', asic_spec['watts_per_asic'], '(',                        (asic_spec['watts_per_asic']/die_area), 'W/mm2) and the max power is',max_die_power
        return
    else:
      asic_spec['asic_hot'] = 0.0
      
  asic_spec['watts_per_mm2'] = asic_spec['watts_per_asic'] / asic_spec['die_area']

  # Server cost and power calculation
  srv_spec = ServerCost.evalServerCost(asic_spec, die_cost, hs_cost, chiplets_per_lane, srv_chiplets, srv_tops, TPU)
  
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
  dc_spec['num_boards'] = CONSTANTS.num_boards[app]
  
  extra_spec = utils.extra_specs_calculator(dc_spec, srv_spec, 
                                       dpw, max_die_power,
                                       die_yield)
  return asic_spec, srv_spec, dc_spec, extra_spec
