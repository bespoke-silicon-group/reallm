import sys
import math
import CONSTANTS

# Reads logic and sram power as well as their vdd from asic_spec
def evalServerCost(asic_spec, die_cost, hs_cost, asics_per_col, chiplets_per_board, tops_per_server, TPU = False):

   num_of_asics = chiplets_per_board
   used_lanes = math.ceil(chiplets_per_board / asics_per_col)

   if TPU:
     used_lanes = 2

   srv_spec = {
      'asics_per_col'   : asics_per_col,
      'lanes_per_server': used_lanes,
      'asics_per_server': num_of_asics,
      'tops_per_server' : tops_per_server,
   }

   # Calculating power and current
   w_fan = CONSTANTS.FanPower * used_lanes
   c_app  = CONSTANTS.APPPower / 1.0

   w_IO = asic_spec['w_io']
   c_lgc  = (asic_spec['w_lgc']  / asic_spec['lgc_vdd'])  * num_of_asics
   c_sram = (asic_spec['w_sram'] / asic_spec['sram_vdd']) * num_of_asics
   c_dcdc = c_app + c_lgc + c_sram
   w_dcdc = (asic_spec['watts_per_asic']*num_of_asics + w_IO +
                                 CONSTANTS.APPPower) / CONSTANTS.DCDCEfficiency
   output_of_psu = w_dcdc + w_fan

   server_power = output_of_psu / CONSTANTS.PSUEfficiency
   if TPU:
     server_power = 275*4

   num_of_dcdc = math.ceil(c_dcdc / CONSTANTS.DCDCMaxCurrent)

   # Calculating costs
   cost_dcdc = CONSTANTS.DCDCCostPerAmp * c_dcdc
   cost_psu = server_power * CONSTANTS.PSUCostPerW

   cost_per_package = CONSTANTS.evalPackageCost(die_cost)
   cost_all_package = cost_per_package * num_of_asics

   cost_all_silicon = (die_cost + asic_spec['die_si_cost']) * num_of_asics

   if hs_cost == 'water':
      cost_all_heatsinks = 10.0 * num_of_asics
   else:
      cost_all_heatsinks = hs_cost[0] * num_of_asics
   cost_all_fans = CONSTANTS.FanCost * CONSTANTS.columns_per_lane * CONSTANTS.lanes_per_server
   cost_all_ethernet = 2 * CONSTANTS.EthernetCost

   pcb_parts_cost = CONSTANTS.PCBPartsCost
   chassis_cost = CONSTANTS.ChassisCost
   pcb_cost = CONSTANTS.PCBCost * 2

   if (asic_spec['dram_count'] == 0):
     cost_all_drams = 0.0
   else:
     cost_all_drams = num_of_asics * asic_spec['dram_count'] * \
         CONSTANTS.real_hbm_price

   system_cost = pcb_cost + pcb_parts_cost + chassis_cost

   server_cost = cost_all_silicon + \
                 cost_all_package + \
                 cost_all_heatsinks + \
                 cost_all_drams + \
                 cost_all_fans + \
                 cost_all_ethernet + \
                 cost_dcdc + \
                 cost_psu + \
                 system_cost

   srv_spec['silicon_cost'] = cost_all_silicon
   srv_spec['package_cost'] = cost_all_package
   srv_spec['heatsink_cost'] = cost_all_heatsinks
   srv_spec['fan_cost'] = cost_all_fans
   srv_spec['dcdc_cost'] = cost_dcdc
   srv_spec['psu_cost'] = cost_psu
   srv_spec['system_cost'] = system_cost
   srv_spec['server_cost'] = server_cost
   srv_spec['server_power'] = server_power
   srv_spec['die_cost'] = die_cost
   if hs_cost == 'water':
      srv_spec['al_cost'] = 0.0
      srv_spec['al_weight'] = 0.0
      srv_spec['cu_cost'] = 0.0
      srv_spec['cu_weight'] = 0.0
   else:
      srv_spec['al_cost'] = hs_cost[1] * num_of_asics
      srv_spec['al_weight'] = hs_cost[2] * num_of_asics
      srv_spec['cu_cost'] = hs_cost[3] * num_of_asics
      srv_spec['cu_weight'] = hs_cost[4] * num_of_asics

   srv_spec['pcb_cost'] = pcb_cost
   srv_spec['pcb_parts_cost'] = pcb_parts_cost
   srv_spec['chassis_cost'] = chassis_cost
   if hs_cost == 'water':
      srv_spec['cost_per_heatsink'] = 10.0
   else:
      srv_spec['cost_per_heatsink'] = hs_cost[0]
   srv_spec['cost_per_fan'] = CONSTANTS.FanCost
   srv_spec['power_per_fan'] = CONSTANTS.FanPower
   srv_spec['cost_per_package'] = cost_per_package
   srv_spec['dcdc_current'] = c_dcdc
   srv_spec['num_of_dcdc'] = num_of_dcdc
   srv_spec['cost_all_ethernet'] = cost_all_ethernet

   return srv_spec
# end of evalServer
