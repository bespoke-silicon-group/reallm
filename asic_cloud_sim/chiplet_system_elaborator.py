import CONSTANTS
from DieCost import die_yield_calc, die_cost_calc, die_area_calc, fill_yield, average_working_RCAs
import cStringIO
import TCO
from Heatsink import evalHS
import VoltageScaling
import ServerCost
import utils

def chiplet_elaborator(design): 
  [default_spec, io_spec, depth, width, asics_per_col, RCA_performance, tech, MPW, liquid_cool] = design
  print_spec = {'tech_node': tech, 'PEs_per_lane': depth*asics_per_col}
  # calculate chiplet units and IO units per die
  N = depth * width
  asic_spec = default_spec.copy()
  asic_spec['IO_count'] = 2
  results = cStringIO.StringIO() 

  # calculate die area and cost (performance is dummy)
  die_area = die_area_calc (asic_spec, N) 
  if (die_area > 600.0):
    print 'in tech of',tech,'depth of',depth,'cannot be fitted', ',', ' die area is:', die_area
    return
  
  # update asic_spec
  asic_spec['die_area'] = die_area
  print_spec['PEs_per_chiplet'] = depth
  print_spec['chiplets_per_lane'] = asics_per_col
  print_spec['die_area_init'] = die_area
  print_spec['total_silicon_area_init'] = die_area* asics_per_col
  
  # calculate die yield and cost
  (die_yield, RCA_accept_rate) = die_yield_calc (asic_spec, N, N)
  (die_cost, dpw) = die_cost_calc (die_area, die_yield, tech, True, MPW)
  print_spec['die_cost_init'] = die_cost
  print_spec['total_silicon_cost_init'] = die_cost * asics_per_col
  print_spec['chiplet_yield'] = die_yield

  
  used_units = depth * asics_per_col
  # used RCA in each ASIC
  provisioned_RCA = width * (used_units * 1.0 / asics_per_col)
  
  # filter out systems with total chip powered off
  if ((depth*asics_per_col - used_units) >= depth) :
    return
 
  # Calculate power and performance based on provisioning
  VoltageScaling.asic_power_performance(asic_spec, None, io_spec, 
                                        N, provisioned_RCA, 0)
  asic_spec['mhash_per_asic']   = RCA_performance * provisioned_RCA 
  asic_spec['joules_per_mhash'] = asic_spec['watts_per_asic'] / \
                                  asic_spec['mhash_per_asic'] 

  print_spec['chiplet_power'] = asic_spec['watts_per_asic']
  if liquid_cool:
    # Fake water cooling
    hs_cost = 'water'
    print_spec['max_power_per_lane'] = 500.0
    print_spec['max_die_power_init'] = 500.0/asics_per_col
    max_die_power = 500.0/asics_per_col
  else:
    # Heat sink elaboration
    flag = False
    (max_die_power, hs_cost) = evalHS(die_area, asics_per_col)
    print_spec['max_die_power_init'] = max_die_power
    print_spec['max_power_per_lane'] = max_die_power*asics_per_col 
    while (max_die_power <= asic_spec['watts_per_asic']) and (die_area <= 600.0):
        die_area += 5
        (max_die_power, hs_cost) = evalHS(die_area, asics_per_col)
        flag = True   

    #  if (die_area>600.0):
      #  print 'in tech of',tech,'depth of',depth,',', 'for ', asics_per_col,' ASICs per lane chiplet is too hot'
      #  return

    if hs_cost == 0:
      die_area = asic_spec['die_area']
      (max_die_power, hs_cost) = evalHS(die_area, asics_per_col)
      flag = False
      if hs_cost == 0:
        return
    
    if (flag):
      #  print 'in tech of',tech,'depth of',depth,',','for ', asics_per_col, \
            #  ' ASICs per lane dark silicon is used (',die_area-asic_spec['die_area'],'mm)'
      asic_spec['die_area'] = die_area
      (die_yield, RCA_accept_rate) = die_yield_calc (asic_spec, N, N)
      (die_cost, dpw) = die_cost_calc (die_area, die_yield, tech, True, MPW)

  print_spec['die_area_last'] = die_area
  print_spec['dark_silicon_used'] = print_spec['die_area_last'] - print_spec['die_area_init']
  print_spec['die_cost_last'] = die_cost
  print_spec['total_silicon_area_last'] = die_area * asics_per_col
  print_spec['total_silicon_cost_last'] = die_cost * asics_per_col

    
  # Server cost and power calculation  
  srv_spec = ServerCost.evalServerCost(asic_spec, None, io_spec, die_cost, hs_cost, 
                                       asics_per_col, False)

  # Evaluate TCO
  dc_spec = TCO.evalTCO(srv_spec['server_power'],
                        srv_spec['server_cost'],
                        CONSTANTS.SrvLife)

  # ----------------------------------------------------------------------------
  extra_spec = utils.extra_specs_calculator (asic_spec, srv_spec, 
                                             N, provisioned_RCA, N, 
                                             dpw, max_die_power, tech,
                                             die_yield)
  extra_spec['chiplet_depth'] = depth
  extra_spec['chiplet_width'] = width
  extra_spec['used_units'] = used_units
  
  utils.fprintData([print_spec, dc_spec, srv_spec, asic_spec, extra_spec], results, area_csv=True)
    
  return results.getvalue() 


