import argparse
from pathlib import Path
from structs.HardwareConfig import ChipConfig, PackageConfig, ServerConfig
from utils.utils import to_csv
import pickle
import yaml
import os
import time

def run(config, results_dir: str, verbose: bool):
  name = config['Name']

  chip_config = ChipConfig(config['Chip'])
  pkg_config = PackageConfig(config['Package'])
  srv_config = ServerConfig(config['Server'])

  start_time = time.time()
  chips = chip_config.explore(verbose)
  packges = pkg_config.explore(chips, verbose)
  servers = srv_config.explore(packges, verbose)
  print(f'Finished {name} hardware exploration in {time.time()-start_time} seconds.')

  if os.path.exists(results_dir) == False:
    os.mkdir(results_dir)
  # print results to csv file
  o_file_path = results_dir+'/'+name+'.csv'
  to_csv(o_file_path, servers)
  # save results to pickle file
  with open(results_dir+'/'+name+'.pkl', 'wb') as f:
    pickle.dump(servers, f)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=Path)
  parser.add_argument('--results-dir', type=str, default='results')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()
  config = yaml.safe_load(open(args.config, 'r'))
  run(config, args.results_dir, args.verbose)

