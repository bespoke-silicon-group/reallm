import utils
import argparse
from chiplet_elaborator import chiplet_elaborator
from chiplet_system_config import set_config

def run(config, set_app=None, set_tech=None, set_use_dram=None, set_keep_large_power=None, set_use_total_power=None, set_srv_mem=None, set_IO_bandwidth=None, set_srv_tops_options=None, set_srv_chiplets_options=None, set_TPU=None, set_SI=None, set_organic_sub=None, set_lanes_per_server=None):
  app, tech, use_dram, keep_large_power, use_total_power, srv_mem, IO_bandwidth, srv_tops_options, srv_chiplets_options, TPU, SI, organic_sub = set_config(config)

  if set_app is not None:
    app = set_app
  if set_tech is not None:
    tech = set_tech
  if set_use_dram is not None:
    use_dram = set_use_dram
  if set_keep_large_power is not None:
    keep_large_power = set_keep_large_power
  if set_use_total_power is not None:
    use_total_power = set_use_total_power
  if set_srv_mem is not None:
    srv_mem = set_srv_mem
  if set_IO_bandwidth is not None:
    IO_bandwidth= set_IO_bandwidth
  if set_srv_tops_options is not None:
    srv_tops_options = set_srv_tops_options
  if set_srv_chiplets_options is not None:
    srv_chiplets_options = set_srv_chiplets_options
  if set_TPU is not None:
    TPU = set_TPU
  if set_SI is not None:
    SI = set_SI
  if set_organic_sub is not None:
    organic_sub = set_organic_sub
  if set_lanes_per_server is not None:
    lanes_per_server = set_lanes_per_server
  else:
    lanes_per_server = 6

  all_data = []
  for srv_tops in srv_tops_options:
    for num_chiplets in srv_chiplets_options:
      design = [app, tech, srv_mem, srv_tops, num_chiplets,
                IO_bandwidth, keep_large_power, use_total_power,
                use_dram, TPU, SI, organic_sub]
      results = chiplet_elaborator(design, lanes_per_server)
      if results != None:
        all_data.append(list(results))

  o_file = open(config+'.csv', 'w')
  utils.fprintHeader(o_file)

  for data in all_data:
    utils.fprintData(data, o_file)

  o_file.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='exploration')
  args = parser.parse_args()
  config = args.config
  run(config)
  print('write results of', config)

