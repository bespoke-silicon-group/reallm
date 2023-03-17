#hs_header = [
#       'q_asic', 'r_total', 'r_si', 'r_tim',
#       'r_sp', 'r_fins', 'die_length', 'die_width',
#       'fin_height', 'fin_thermal_cond', 'fin_thickness', 'fin_air_volume',
#       'n_of_fins', 'base_thickness', 'base_width', 'base_length',
#       'base_thermal_cond']

asic_header = ['tech_node',
               'sram_per_asic', 'tops_per_asic',
               'watts_per_asic',
               'die_area',
               'die_cost']

srv_header = ['asics_per_server',
              'sram_per_server',
              'server_power',
              'tops_per_server',
              'server_cost'
             ]

# tco_header = ['life_time_tco', 'num_boards',
tco_header = ['life_time_tco', 
              'DCAmortization', 'DCInterest', 'DCOpex',
              'SrvAmortization', 'SrvInterest',
              'SrvOpex', 'SrvPower', 'PUEOverhead']

extra_header = ['cost_per_tops', 'watts_per_tops', 'tco_per_tops',
                'max_die_power_per_server', 'die_yield']

csv_header = [asic_header, srv_header, tco_header, extra_header]

def fprintHeader(o_file):
  headers = csv_header
  i = 1

  o_file.write('# ')
  for header in headers:
    for h in header:
      o_file.write('[{0:d}]{1:s}, '.format(i, h))
      i += 1
  o_file.write('\n')

def fprintData(data_list, o_file):
  headers = csv_header
  i = 0
  for header in headers:
    data = data_list[i]
    for h in header:
      o_file.write('{0}, '.format(data[h]),)
    i += 1
  o_file.write('\n')

# selects extra specs to be saved
def extra_specs_calculator (dc_spec, srv_spec, dpw, max_die_power, die_y):
  # Extra Specs
  extra_spec = {
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
    'tco_per_tops'     : dc_spec['life_time_tco']/srv_spec['tops_per_server'],
    'max_die_power_per_server': max_die_power * srv_spec['asics_per_server']
  }

  return extra_spec


