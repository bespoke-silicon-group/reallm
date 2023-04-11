import hardware_exploration.scripts.utils as utils
from hardware_exploration.scripts.chiplet_elaborator import chiplet_elaborator
from hardware_exploration.chiplet_system_config import set_config
import argparse

def run(config, results_dir):

  all_data = []

  if config == 'exploration':
    tech, IO_bandwidth, lanes_per_server, lane_silicon_options, lane_chiplets_options, mac_area_ratio_options = set_config(config)
    chip_id_dict = {}
    chip_num = 0
    #for silicon_per_lane in lane_silicon_options:
    for chip_area in range(20, 801, 20):
     for chiplets_per_lane in lane_chiplets_options:
        silicon_per_lane = chip_area*chiplets_per_lane
        if silicon_per_lane > 6000 or silicon_per_lane < 200:
          continue
        # chip_area = silicon_per_lane / chiplets_per_lane
        for mac_area_ratio in mac_area_ratio_options:
          chip = f'{chip_area}_{mac_area_ratio}'
          if chip in chip_id_dict:
            chip_id = chip_id_dict[chip]
          else:
            chip_id = chip_num
          design = [tech, lanes_per_server, IO_bandwidth, silicon_per_lane, chiplets_per_lane, mac_area_ratio, chip_id]
          results = chiplet_elaborator(design)
          if results != None:
            if chip not in chip_id_dict:
              chip_id_dict[chip] = chip_num
              chip_num += 1
            all_data.append(list(results))
  elif config == 'HBM_large_chip':
    # Based on DGX A100
    tech = '7nm'
    lanes_per_server = 8
    IO_bandwidth = 50
    chip_area = 826
    chiplets_per_lane = 1
    silicon_per_lane = chiplets_per_lane * chip_area
    mac_area_ratio = None
    chip_id = -1
    design = [tech, lanes_per_server, IO_bandwidth, silicon_per_lane, chiplets_per_lane, mac_area_ratio, chip_id]
    results = chiplet_elaborator(design)
    if results != None:
      all_data.append(list(results))
  elif config == 'HBM_chiplet':
    # Half of TPUv4i, 
    tech = '7nm'
    lanes_per_server = 8
    IO_bandwidth = 50
    chip_area = 200
    chiplets_per_lane = 4
    silicon_per_lane = chiplets_per_lane * chip_area
    mac_area_ratio = None
    chip_id = -2
    design = [tech, lanes_per_server, IO_bandwidth, silicon_per_lane, chiplets_per_lane, mac_area_ratio, chip_id]
    results = chiplet_elaborator(design)
    if results != None:
      all_data.append(list(results))
  else:
    print('Wrong Configs')

  o_file_path = results_dir+'/'+config+'.csv'
  o_file = open(o_file_path, 'w')
  utils.fprintHeader(o_file)

  for data in all_data:
    utils.fprintData(data, o_file)

  o_file.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='exploration')
  parser.add_argument('--results-dir', type=str)
  args = parser.parse_args()
  config = args.config
  results_dir = args.results_dir
  run(config, results_dir)
  print('write results of', config)

