import utils
import argparse
from chiplet_elaborator import chiplet_elaborator
from chiplet_system_config import set_config

def run(config):
  tech, IO_bandwidth, lanes_per_server, lane_silicon_options, lane_chiplets_options, mac_area_ratio_options = set_config(config)


  all_data = []
  for silicon_per_lane in lane_silicon_options:
    for chiplets_per_lane in lane_chiplets_options:
      for mac_area_ratio in mac_area_ratio_options:
        design = [tech, lanes_per_server, IO_bandwidth, silicon_per_lane, chiplets_per_lane, mac_area_ratio]
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

