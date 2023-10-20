from structs.Server import Server
from typing import Any, Dict, List, Optional

chip_header = ['sram', 
               'perf',
               'sram_bw',
               'tdp',
               'area',
               'cost',
               ]
pkg_header = []

srv_header = ['num_packages',
              'sram',
              'tdp',
              'perf',
              'cost',
             ]

tco_header = ['total',
              'dc_amortization', 'dc_interest', 'dc_opex',
              'srv_amortization', 'srv_interest',
              'srv_opex', 'srv_power', 'pue_overhead']

extra_header = ['cost_per_tops', 'watts_per_tops', 'tco_per_tops',
                'sram_density', 'die_yield', 'sparse_weight', 'chip_id']

csv_header = [chip_header, srv_header, tco_header, extra_header]

def fprintHeader(o_file):
  i = 1
  o_file.write('# ')
  for h in chip_header:
    o_file.write('[{0:d}]{1:s}, '.format(i, 'chip_'+h))
    i += 1
  for h in pkg_header:
    o_file.write('[{0:d}]{1:s}, '.format(i, 'pkg_'+h))
    i += 1
  for h in srv_header:
    o_file.write('[{0:d}]{1:s}, '.format(i, 'srv_'+h))
    i += 1
  for h in tco_header:
    o_file.write('[{0:d}]{1:s}, '.format(i, 'tco_'+h))
    i += 1
  for h in extra_header:
    o_file.write('[{0:d}]{1:s}, '.format(i, h))
    i += 1
  o_file.write('\n')

def fprintData(o_file, servers: List[Server]) -> None:
  i = 0
  for server in servers:
    chip_dict = server.package.chip.to_dict(chip_header)
    for h in chip_header:
      o_file.write('{0}, '.format(chip_dict[h]))
    pkg_dict = server.package.to_dict(pkg_header)
    for h in pkg_header:
      o_file.write('{0}, '.format(pkg_dict[h]))
    srv_dict = server.to_dict(srv_header)
    for h in srv_header:
      o_file.write('{0}, '.format(srv_dict[h]))
    tco_dict = server.tco.to_dict(tco_header)
    for h in tco_header:
      o_file.write('{0}, '.format(tco_dict[h]))
    extra_spec = extra_specs_calculator(server)
    for h in extra_header:
      o_file.write('{0}, '.format(extra_spec[h]))
      i += 1
    o_file.write('\n')

def extra_specs_calculator (server: Server) -> Dict[str, Any]:
  extra_spec = {
    'cost_per_tops'    : server.cost / (server.perf / 1e12),
    'watts_per_tops'   : server.tdp / (server.perf / 1e12),
    'tco_per_tops'     : server.tco.total / (server.perf / 1e12),
    'sram_density'     : server.package.chip.sram / 1e6 / server.package.chip.sram_area,
    'die_yield'        : server.package.chip.die_yield,
    'sparse_weight'    : False,
    'chip_id'          : server.package.chip.chip_id
  }

  return extra_spec

def to_csv(file_path: str, servers: List[Server]) -> None:
  o_file = open(file_path, 'w')
  fprintHeader(o_file)
  fprintData(o_file, servers)


