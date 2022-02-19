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
   'tech_node',
   'lgc_vdd', 'sram_vdd', 'frequency',
   'tops_per_asic', 'sram_per_asic',
   'die_area', 'lgc_area', 'sram_area', 'io_area',
   'watts_per_asic', 'w_lgc', 'w_sram', 'w_io', 
   'watts_per_mm2', 'joules_per_tops']

asic_header_simple = [
   'tech_node', 'sram_per_asic', 'tops_per_asic', 
   'die_area', 'watts_per_asic',
   'watts_per_mm2', 'die_cost']

srv_header = [
   'die_cost', 'server_power', 
   'asics_per_col', 'lanes_per_server', 'asics_per_server',
   'tops_per_server', 
   'server_cost', 'silicon_cost', 'package_cost',
   'heatsink_cost', 'fan_cost', 'dcdc_cost', 'psu_cost', 'system_cost']

srv_header_simple = [
   'asics_per_server',
   'server_power', 
   'tops_per_server',
   'server_cost'
   ]

tco_header = ['life_time_tco', 'DCAmortization', 'DCInterest', 'DCOpex', 
              'SrvAmortization', 'SrvInterest', 'SrvOpex', 'SrvPower', 'PUEOverhead']

tco_header_simple = ['life_time_tco', 'num_boards']

extra_header = ['threshold_voltage','pcb_cost','pcb_parts_cost',
                'chassis_cost','cost_per_heatsink','cost_per_fan','power_per_fan',
                'cost_per_package','dcdc_current','num_of_dcdc','cost_all_ethernet', 
                'dies_per_wafer',
                'q_asic', 'die_yield']
extra_header_simple = ['cost_per_tops', 'watts_per_tops', 'tco_per_tops']

csv_header = [asic_header, srv_header, tco_header, extra_header]
simple_csv_header = [asic_header_simple, srv_header_simple, tco_header_simple, extra_header_simple]

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

def fprintHeader(o_file, simple_csv=False):
   if simple_csv:
      headers = simple_csv_header
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

def fprintData(data_list, o_file, simple_csv=False):
   if simple_csv:
      headers = simple_csv_header
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
def extra_specs_calculator (dc_spec, srv_spec, dpw, max_die_power, tech, die_y):
    # Extra Specs
    threshold_voltage = CONSTANTS.TechData.loc[tech, "Vth"]
    extra_spec = {
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
      'dies_per_wafer'   : dpw,
      'q_asic'           : max_die_power,
      'die_yield'        : die_y,
      'cost_per_tops'    : srv_spec['server_cost']/srv_spec['tops_per_server'],
      'watts_per_tops'   : srv_spec['server_power']/srv_spec['tops_per_server'],
      'tco_per_tops'     : dc_spec['life_time_tco']/srv_spec['tops_per_server']
    }

    return extra_spec 



