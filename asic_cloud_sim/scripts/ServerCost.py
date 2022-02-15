import sys
import math
import DieCost
import CONSTANTS

# In voltage stacking scenario just read watts_per_asic from asic_spec
# Otherwise, Reads logic and sram power as well as their vdd from asic_spec as well
# Finally reads DRAM properties for each asic 
# Reads Voltage_stacking flag for DCDC cost
def evalServerCost(asic_spec, dram_spec, io_spec, die_cost, hs_cost, asics_per_col, voltage_stacking):
   
   num_of_asics = asics_per_col * CONSTANTS.columns_per_lane * CONSTANTS.lanes_per_server
   chain_length = 0
   num_of_chains = 0
   srv_spec = {
      'asics_per_col'   : asics_per_col,
      'columns_per_lane': CONSTANTS.columns_per_lane,
      'lanes_per_server': CONSTANTS.lanes_per_server,
      'asics_per_server': num_of_asics,
      'mhash_per_server': asic_spec['mhash_per_asic'] * num_of_asics,
      'num_of_chains'   : num_of_chains,
      'chain_length'    : chain_length,
   }

   # Calculating power and current 
   w_fan = CONSTANTS.FanPower * CONSTANTS.columns_per_lane * CONSTANTS.lanes_per_server 
   c_app  = CONSTANTS.APPPower / 1.0
   
   if (asic_spec['dram_count'] == 0):
      w_dram = 0
      c_dram_asic = 0
      c_dram = 0
   else:
      w_dram = num_of_asics * asic_spec['dram_count'] * dram_spec['dram_power']
      c_dram_asic = (asic_spec['dram_mc_power'] / dram_spec['MC_vdd']) * num_of_asics
      if (dram_spec['type'] == 'SDR'): #dram_vdd is different than MC_vdd
        c_dram =  asic_spec['dram_count'] * num_of_asics * \
               (dram_spec['dram_power'] / dram_spec['dram_vdd'])
      else:
        c_dram =  asic_spec['dram_count'] * num_of_asics * \
               (dram_spec['dram_power'] / dram_spec['MC_vdd'])
    
   if (asic_spec['IO_count'] == 0):
      w_IO = 0 
   else:
      w_IO = asic_spec['IO_count'] * io_spec['IO_power']

   # in case of voltage stacking no DCDC current is considered for logic and sram
   if(voltage_stacking):
      c_dcdc = c_app
      w_dcdc = CONSTANTS.APPPower / CONSTANTS.DCDCEfficiency
      output_of_psu = w_dcdc + asic_spec['watts_per_asic']*num_of_asics + w_fan + w_dram + w_IO
   else:
      c_lgc  = (asic_spec['w_lgc']  / asic_spec['lgc_vdd'])  * num_of_asics 
      c_sram = (asic_spec['w_sram'] / asic_spec['sram_vdd']) * num_of_asics 
      c_dcdc = c_app + c_lgc + c_sram + c_dram_asic + c_dram
      w_dcdc = (asic_spec['watts_per_asic']*num_of_asics + w_dram + w_IO + 
                                    CONSTANTS.APPPower) / CONSTANTS.DCDCEfficiency 
      output_of_psu = w_dcdc + w_fan
   
   server_power = output_of_psu / CONSTANTS.PSUEfficiency

   num_of_dcdc = math.ceil(c_dcdc / CONSTANTS.DCDCMaxCurrent)

   # Calculating costs
   cost_dcdc = CONSTANTS.DCDCCostPerAmp * c_dcdc 
   cost_psu = server_power * CONSTANTS.PSUCostPerW
   cost_per_package = CONSTANTS.evalPackageCost(die_cost)

   cost_all_silicon = die_cost * num_of_asics
   cost_all_package = cost_per_package * num_of_asics
   if hs_cost == 'water':
      cost_all_heatsinks = 3.0 * num_of_asics
   else:
      cost_all_heatsinks = hs_cost[0] * num_of_asics
   cost_all_fans = CONSTANTS.FanCost * CONSTANTS.columns_per_lane * CONSTANTS.lanes_per_server 
   cost_all_ethernet = asic_spec['ethernet_count'] * CONSTANTS.EthernetCost

   pcb_parts_cost = CONSTANTS.PCBPartsCost
   chassis_cost = CONSTANTS.ChassisCost
   if (asic_spec['dram_count'] == 0) and (asic_spec['IO_count'] == 0):
      pcb_cost = CONSTANTS.PCBCost
   else:
      pcb_cost = CONSTANTS.PCBCost * 2
      
   if (asic_spec['dram_count'] == 0):
      cost_all_drams = 0
   else:
      cost_all_drams = num_of_asics * asic_spec['dram_count'] * \
                       dram_spec['cost']
   system_cost = pcb_cost + pcb_parts_cost + chassis_cost


   server_cost = cost_all_silicon + \
                 cost_all_package + \
                 cost_all_heatsinks + \
                 cost_all_drams +\
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
      srv_spec['cost_per_heatsink'] = 3.0
   else:
      srv_spec['cost_per_heatsink'] = hs_cost[0]
   srv_spec['cost_per_fan'] = CONSTANTS.FanCost
   srv_spec['power_per_fan'] = CONSTANTS.FanPower
   srv_spec['cost_per_package'] = cost_per_package
   srv_spec['dcdc_current'] = c_dcdc
   srv_spec['num_of_dcdc'] = num_of_dcdc
   srv_spec['cost_all_ethernet'] = cost_all_ethernet

   srv_spec['cost_all_drams'] = cost_all_drams
   
   return srv_spec
# end of evalServer
