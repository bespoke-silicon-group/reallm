import utils
import argparse
from chiplet_elaborator import chiplet_elaborator
from chiplet_system_config import set_config

def run(config):
  app, tech, use_dram, keep_large_power, use_total_power, srv_mem, IO_bandwidth,               srv_tops_options, srv_chiplets_options, TPU, SI, organic_sub = set_config(config)

  all_data = []
  for srv_tops in srv_tops_options:
    for num_chiplets in srv_chiplets_options:
      design = [app, tech, srv_mem, srv_tops, num_chiplets,
                IO_bandwidth, keep_large_power, use_total_power,
                use_dram, TPU, SI, organic_sub]
      results = chiplet_elaborator(design)
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

