import argparse
from structs.HardwareConfig import ChipConfig, PackageConfig, ServerConfig
from structs.Constants import ChipConstants, PackageConstants, ServerConstants, TCOConstants, EnergyConstants
from utils.hardware_dump import to_csv
import pickle
import yaml
import os
import time

def hardware_exploration(config: dict, constants: dict, results_dir: str, hardware_name: str, verbose: bool):
  chip_constants = ChipConstants(**constants['Chip'])
  pkg_constants = PackageConstants(**constants['Package'])
  srv_constants = ServerConstants(**constants['Server'])
  tco_constants = TCOConstants(**constants['TCO'])
  energy_constants = EnergyConstants(**constants['Energy'])

  chip_config = ChipConfig(config['Chip'])
  pkg_config = PackageConfig(config['Package'])
  srv_config = ServerConfig(config['Server'])

  start_time = time.time()
  chips = chip_config.explore(constants=chip_constants, verbose=verbose)
  packges = pkg_config.explore(chips=chips, constants=pkg_constants, verbose=verbose)
  servers = srv_config.explore(pkgs=packges, constants=srv_constants,
                               tco_constants=tco_constants, energy_constants=energy_constants,
                               verbose=verbose)
  print(f'Finished {hardware_name} hardware exploration in {time.time() - start_time} seconds')

  results_dir = results_dir + '/' + hardware_name
  if os.path.exists(results_dir) == False:
    os.makedirs(results_dir)
  # print results to csv file
  o_file_path = results_dir + '/' + hardware_name + '.csv'
  to_csv(o_file_path, servers)
  # save results to pickle file
  with open(results_dir + '/' + hardware_name + '.pkl', 'wb') as f:
    pickle.dump(servers, f)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config-file', type=str)
  parser.add_argument('--constants-file', type=str)
  parser.add_argument('--results-dir', type=str, default='outputs')
  parser.add_argument('--verbose', action='store_true')
  args = parser.parse_args()
  hardware_name = args.config_file.split('/')[-1].split('.')[0]
  config = yaml.safe_load(open(args.config_file, 'r'))
  constants = yaml.safe_load(open(args.constants_file, 'r'))
  hardware_exploration(config, constants, args.results_dir, hardware_name, args.verbose)
