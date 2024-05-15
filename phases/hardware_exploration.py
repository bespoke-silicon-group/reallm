import argparse
from structs.HardwareConfig import ChipConfig, PackageConfig, ServerConfig
from utils.utils import to_csv
import pickle
import yaml
import os
import time

def hardware_exploration(config: dict, results_dir: str, hardware_name: str, verbose: bool):
  chip_config = ChipConfig(config['Chip'])
  pkg_config = PackageConfig(config['Package'])
  srv_config = ServerConfig(config['Server'])

  start_time = time.time()
  chips = chip_config.explore(verbose)
  packges = pkg_config.explore(chips, verbose)
  servers = srv_config.explore(packges, verbose)
  print(f'Finished {hardware_name} hardware exploration in {time.time() - start_time} seconds')

  results_dir = results_dir + '/' + hardware_name
  if os.path.exists(results_dir) == False:
    os.mkdir(results_dir)
  # print results to csv file
  o_file_path = results_dir + '/' + hardware_name + '.csv'
  to_csv(o_file_path, servers)
  # save results to pickle file
  with open(results_dir + '/' + hardware_name + '.pkl', 'wb') as f:
    pickle.dump(servers, f)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config-file', type=str)
  parser.add_argument('--results-dir', type=str, default='outputs')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()
  hardware_name = args.config_file.split('/')[-1].split('.')[0]
  config = yaml.safe_load(open(args.config_file, 'r'))
  hardware_exploration(config, args.results_dir, hardware_name, args.verbose)

