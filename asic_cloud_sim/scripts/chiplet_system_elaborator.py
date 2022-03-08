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

  results = cStringIO.StringIO() 
  [app, tech, tops, mem_size, IO_BW, keep_large_power, use_total_power] = design

  #  mem_per_stage = CONSTANTS.mem_per_stage[app]
  #  stage_per_board = CONSTANTS.stage_per_board[app]
  mem_per_board = CONSTANTS.mem_per_board[app]

  # calculate chiplet MAC, memory and IO area per die
  asic_spec = {'tech_node':tech, 'lgc_vdd':0.8, 'sram_vdd':0.8, 'frequency': 1, 'other_area': 10.0}
  asic_spec['lgc_area'] = tops * CONSTANTS.real_macs_size[tech]
  asic_spec['sram_area'] = mem_size * CONSTANTS.real_sram_size[tech]
  asic_spec['io_bw'] = IO_BW
  IO_count = 2*(IO_BW/12.5) # transmitter and receiver
  asic_spec['io_area'] = IO_count * CONSTANTS.real_io_size[tech]
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
  
  # calculate # of chiplets per board
  #  chiplets_per_stage = math.ceil(mem_per_stage / mem_size)
  #  chiplets_per_board = chiplets_per_stage * stage_per_board
  if mem_per_board[0] < mem_size: # BERT or GPT2 or some T-NLG
    stages_per_chiplet = math.floor(mem_size / mem_per_board[0])
    chiplets_per_board = math.ceil(len(mem_per_board) / stages_per_chiplet)
  else:
    chiplets_per_board = 0
    for mem_per_stage in mem_per_board:
      chiplets_per_board += math.ceil(mem_per_stage / mem_size)
  chiplets_per_lane = math.ceil(math.sqrt(chiplets_per_board))

  # Calculate power and performance based on provisioning
  if use_total_power:
    asic_spec['w_lgc']  = tops * CONSTANTS.real_total_power[tech]
    asic_spec['w_sram'] = 0.0
    asic_spec['w_io']   = 0.0
  else:
    asic_spec['w_lgc']  = tops * CONSTANTS.real_macs_power[tech]
    asic_spec['w_sram'] = mem_size * CONSTANTS.real_sram_power[tech]
    asic_spec['w_io']   = IO_count * CONSTANTS.real_io_power[tech]
  asic_spec['watts_per_asic'] = asic_spec['w_lgc'] + asic_spec['w_sram'] + asic_spec['w_io']

  asic_spec['joules_per_tops'] = asic_spec['watts_per_asic'] / \
                                  asic_spec['tops_per_asic'] 

  if False: # No water cooling for now
    # Fake water cooling
    hs_cost = 'water'
    max_die_power = 500.0/chiplets_per_lane
  else:
    # Heat sink elaboration
    (max_die_power, hs_cost) = evalHS(die_area, chiplets_per_lane)

    if 1.1*max_die_power <= asic_spec['watts_per_asic']:
      if keep_large_power:
        asic_spec['asic_hot'] = 1.0
      else:
        # print 'TOO HOT!! in tech of', tech, tops,'TOPS, ', mem_size, 'MB', ', die area is', die_area, 'has power of ', asic_spec['watts_per_asic'], '(', (asic_spec['watts_per_asic']/die_area), 'W/mm2) and the max power is',max_die_power
        return
    else:
      asic_spec['asic_hot'] = 0.0

    # flag = False
    # while (max_die_power <= asic_spec['watts_per_asic']) and (die_area <= 600.0):
    #   die_area += 5
    #   (max_die_power, hs_cost) = evalHS(die_area, chiplets_per_lane)
    #   flag = True   
    #   if die_area/asic_spec['die_area'] > 1.1:
    #     print 'TOO HOT!! in tech of', tech, tops,'TOPS, ', mem_size, 'MB', ', die area is', die_area, 'has power of ', asic_spec['watts_per_asic'], '(', (asic_spec['watts_per_asic']/die_area), 'W/mm2) and the max power is',max_die_power
    #     return
    # if (flag):
    #   print 'in tech of',tech, \
    #         ' dark silicon is used (',die_area-asic_spec['die_area'],'mm)'
    #   asic_spec['die_area'] = die_area
    #   die_yield = die_yield_calc(die_area, tech)
    #   (die_cost, dpw) = die_cost_calc(die_area, die_yield, tech)

  asic_spec['watts_per_mm2'] = asic_spec['watts_per_asic'] / asic_spec['die_area']
  asic_spec['die_cost'] = die_cost
    
  # Server cost and power calculation  
  srv_spec = ServerCost.evalServerCost(asic_spec, die_cost, hs_cost, chiplets_per_lane, chiplets_per_board)
  srv_spec['server_chip_power'] = asic_spec['watts_per_asic']*srv_spec['asics_per_server']
  if srv_spec['server_chip_power'] > CONSTANTS.board_max_power:
    if keep_large_power:
      srv_spec['server_hot'] = 1.0
    else:
      #  print 'Board power limit exceeded!! In tech of', tech, tops,'TOPS, ', mem_size, 'MB', ', has total board chip power of ', srv_spec['server_chip_power'], 'and the max power is 1000'
      return
  else:
    srv_spec['server_hot'] = 0.0

  # Evaluate TCO
  dc_spec = TCO.evalTCO(srv_spec['server_power'],
                        srv_spec['server_cost'],
                        CONSTANTS.SrvLife)
  dc_spec['num_boards'] = CONSTANTS.num_boards[app]
  # ----------------------------------------------------------------------------
  extra_spec = utils.extra_specs_calculator (dc_spec, srv_spec, 
                                             dpw, max_die_power, tech,
                                             die_yield)
  
  utils.fprintData([asic_spec, srv_spec, dc_spec, extra_spec], results, True)

  return results.getvalue() 

