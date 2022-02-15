import os
import math
import CONSTANTS
import pandas as pd
import cStringIO

#hs_header = [
#       'q_asic', 'r_total', 'r_si', 'r_tim',
#       'r_sp', 'r_fins', 'die_length', 'die_width',
#       'fin_height', 'fin_thermal_cond', 'fin_thickness', 'fin_air_volume',
#       'n_of_fins', 'base_thickness', 'base_width', 'base_length',
#       'base_thermal_cond']

asic_header = [
   'lgc_vdd', 'sram_vdd', 'f_scale', 'mhash_per_asic', 'frequency',
   'watts_per_asic', 'w_lgc', 'w_sram', 'dram_count', 'die_area','RCA_y']

srv_header = [
   'mhash_per_server', 'server_cost', 'silicon_cost', 'package_cost',
   'heatsink_cost', 'fan_cost', 'dcdc_cost', 'psu_cost', 'system_cost',
   'die_cost', 'server_power', 'asics_per_col', 'columns_per_lane',
   'lanes_per_server', 'al_weight', 'al_cost', 'cu_weight', 'cu_cost',
   'asics_per_server', 'num_of_chains', 'chain_length']

tco_header = ['life_time_tco', 'DCAmortization', 'DCInterest', 'DCOpex', 
              'SrvAmortization', 'SrvInterest', 'SrvOpex', 'SrvPower', 'PUEOverhead']

extra_header = ['units_per_die','threshold_voltage','pcb_cost','pcb_parts_cost',
                'chassis_cost','cost_per_heatsink','cost_per_fan','power_per_fan',
                'cost_per_package','dcdc_current','num_of_dcdc','cost_all_ethernet', 
                'lgc_dyn_pwr', 'lgc_leak_pwr', 'sram_dyn_pwr', 'sram_leak_pwr',
                'dram_mc_power','dies_per_wafer', 'joules_per_mhash', 'cost_all_drams',
                'q_asic','at_least_working','provisioned_for', 'die_yield']

csv_header = [tco_header, srv_header, asic_header, extra_header]
area_csv_header = [['tech_node', 'PEs_per_lane',
                    'PEs_per_chiplet', 'chiplets_per_lane','chiplet_power',
                    'die_area_init', 'chiplet_yield', 
                    'max_die_power_init', 'max_power_per_lane', 
                    'die_cost_init', 'total_silicon_area_init', 'total_silicon_cost_init',
                    'dark_silicon_used', 'die_area_last',
                    'die_cost_last', 'total_silicon_area_last', 'total_silicon_cost_last'], 
                    tco_header, srv_header, asic_header, extra_header]
                    # last is  considered max die power and add dark silicon

def gerometric_list (start, stop, step):
  result = []
  temp = start
  while (temp < stop):
    result.append(temp)
    temp *= step
  return result

#
# dd: decimal digits
# dtz: drop trailing zeros from decimal
#
def float_format(value, valtype="Regular", dtz=True, dd=1):
  if (value == float("Infinity")):
    strval = "Inf"
  elif (value == float("-Infinity")):
    strval = "-Inf"
  else:
    if (valtype == "Regular"):
      thousand = 1000.0
      suffixes = ['', 'K', 'M', 'B', 'T']
    elif (valtype == "Computer"):
      thousand = 1024.0
      suffixes = ['', 'K', 'M', 'G', 'T']
    else:
      thousand = 1.0 # no conversion
      suffixes = ['']

    for suffix in suffixes:
      if (value < thousand) or (suffix == suffixes[-1]):
        break
      else:
        value /= thousand

    strval = "{:0,.{p}f}".format(value, p=dd)
    if dtz: # drop trailing zeros from decimal
      strval = strval.rstrip('0').rstrip('.') if '.' in strval else strval
    strval += suffix

  return strval

def fprintHeader(o_file, area_csv=False):
   if area_csv:
      headers = area_csv_header
   else:
      headers = csv_header
   i = 1

   o_file.write('# ')
   for header in headers:
      for h in header:
         o_file.write('[{0:d}]{1:s}, '.format(i, h))
         i += 1
   o_file.write('\n')
# end of fprintHeader

def fprintData(data_list,o_file, area_csv=False):
   if area_csv:
      headers = area_csv_header
   else:
      headers = csv_header
   i = 0
   for header in headers:
      data = data_list[i]
      for h in header:
         o_file.write('{0}, '.format(data[h]),)
      i += 1
   o_file.write('\n')
# end of fprintData

#
# This function operates according to fprintHeader() and fprintData().
# Modify the CSV parsing accordinly if these two fprint functions change.
#
def csv2df(csvfile):
  df = pd.DataFrame.from_csv(csvfile, index_col=False, parse_dates=False)
  # drop the useless column caused by a comma at the end of each row, 
  # according to fprintHeader() fprintData() 
  df.drop(df.columns[-1], axis=1, inplace=True) 
  # remove every character before ']', according to fprintHeader() 
  df.columns = df.columns.str.replace('.*\]', '') 
  return df
# end of csv2df

def printHeader(header_list):
   i = 1

   print '#',
   for header in header_list:
      for h in header:
         print '[{0:d}]{1:s},'.format(i, h),
         i += 1
   print
# end of printHeader

def printData(header_list, data_list):
   i = 0
   for header in header_list:
      data = data_list[i]
      for h in header:
         print '{0},'.format(data[h]),
      i += 1
   print
# end of printData


# selects extra specs to be saved
def extra_specs_calculator (asic_spec, srv_spec, j,k,N, dpw, max_die_power, tech, die_y):
    # Extra Specs
    threshold_voltage = CONSTANTS.TechData.loc[tech, "Vth"]
    extra_spec = {
      'units_per_die'    : N,
      'threshold_voltage': threshold_voltage,
      'pcb_cost'         : srv_spec['pcb_cost'],
      'pcb_parts_cost'   : srv_spec['pcb_parts_cost'],
      'chassis_cost'     : srv_spec['chassis_cost'],
      'cost_per_heatsink': srv_spec['cost_per_heatsink'],
      'cost_per_fan'     : srv_spec['cost_per_fan'],
      'power_per_fan'    : srv_spec['power_per_fan'],
      'cost_per_package' : srv_spec['cost_per_package'],
      'dcdc_current'     : srv_spec['dcdc_current'],
      'num_of_dcdc'      : srv_spec['num_of_dcdc'],
      'cost_all_ethernet': srv_spec['cost_all_ethernet'],
      'lgc_dyn_pwr'      : asic_spec['lgc_dyn_pwr']   * asic_spec['f_scale'] * N, 
      'lgc_leak_pwr'     : asic_spec['lgc_leak_pwr']  * asic_spec['f_scale'] * N,
      'sram_dyn_pwr'     : asic_spec['sram_dyn_pwr']  * asic_spec['f_scale'] * N,
      'sram_leak_pwr'    : asic_spec['sram_leak_pwr'] * asic_spec['f_scale'] * N,
      'dram_mc_power'    : asic_spec['dram_mc_power'], 
      'dies_per_wafer'   : dpw,
      'joules_per_mhash' : asic_spec['joules_per_mhash'],
      'cost_all_drams'   : srv_spec['cost_all_drams'],
      'q_asic'           : max_die_power,
      'at_least_working' : j,  
      'provisioned_for'  : k,
      'die_yield'        : die_y
    }

    return extra_spec 

def delay_loader (techs):
  delays = {}
  for tech in techs:
    size = filter(str.isdigit, tech)
    table_path = os.environ['SCRIPTS_PATH']+'/'+size+'.table'
    CoreVdd = CONSTANTS.TechData.loc[tech, 'CoreVdd']
    df = pd.DataFrame.from_csv(table_path, index_col=False, parse_dates=False) #, sep=', ')
    df.rename(columns= {'units: volts' : 'Vdd' , ' uamps' : 'I', ' V/I (relative delay)': 'Rdelay'}, inplace=True) 
    df['Rdelay'] = df['Rdelay'] / df.loc[df['Vdd']==round(CoreVdd,3),'Rdelay'].values[0]
    #df['freq'] = 1.0 / df['Rdelay']
    delays[tech] = dict(zip(df.Vdd , df.Rdelay))
  return delays


