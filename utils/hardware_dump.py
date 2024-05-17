from structs.Server import Server
from typing import Any, Dict, List, Optional

chip_header = [
               'area',
               'sram_mb', 
               'tops',
               'sram_bw_TB_per_sec',
               'sram_area',
               'mac_area',
               'tdp',
               'cost',
               ]
pkg_header = []

srv_header = ['num_packages',
              'sram_mb',
              'tdp',
              'tops',
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
  for h in chip_header:
    o_file.write('{0:s}, '.format('chip_'+h))
  for h in pkg_header:
    o_file.write('{0:s}, '.format('pkg_'+h))
  for h in srv_header:
    o_file.write('{0:s}, '.format('srv_'+h))
  for h in tco_header:
    o_file.write('{0:s}, '.format('tco_'+h))
  for h in extra_header:
    o_file.write('{0:s}, '.format(h))
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
    'sram_density'     : server.package.chip.sram_mb / server.package.chip.sram_area,
    'die_yield'        : server.package.chip.die_yield,
    'sparse_weight'    : False,
    'chip_id'          : server.package.chip.chip_id,
    'hs_max_power'     : server.hs.max_power,
    'cost_all_package' : server.package.cost * server.num_packages,
    'cost_all_hs'      : server.hs.cost * server.num_packages,
    'cost_all_fans'    : server.constants.FanCost * server.num_lanes,
    'cost_ethernet'    : server.constants.EthernetCost,
    'cost_dcdc'        : server.cost_dcdc,
    'cost_psu'         : server.cost_psu,
    'system_cost'      : server.constants.PCBPartsCost + server.constants.ChassisCost + server.constants.PCBCost * 2
  }

  return extra_spec

def to_csv(file_path: str, servers: List[Server]) -> None:
  o_file = open(file_path, 'w')
  fprintHeader(o_file)
  fprintData(o_file, servers)


