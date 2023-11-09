import argparse
from pathlib import Path
from structs.HardwareConfig import ChipConfig, PackageConfig, ServerConfig
from utils.utils import to_csv
import pickle
import yaml

# def run(mode: str, name: str, results_dir: str):
def run(config, results_dir: str):
  mode = config['Mode']
  name = config['Name']

  if mode == 'exploration':
    # exploration
    verbose = False
    chip_config = ChipConfig(config['Chip'])
    pkg_config = PackageConfig(config['Package'])
    srv_config = ServerConfig(config['Server'])
    chips = chip_config.explore(verbose)
    packges = pkg_config.explore(chips, verbose)
    servers = srv_config.explore(packges, verbose)
  elif mode == 'fixed':
    # fixed design
    verbose = True
    raise NotImplementedError
  else:
    raise ValueError(f'Invalid mode: {mode}')

  # print results to csv file
  o_file_path = results_dir+'/'+mode+'_'+name+'.csv'
  to_csv(o_file_path, servers)
  # save results to pickle file
  with open(results_dir+'/'+mode+'_'+name+'.pkl', 'wb') as f:
    pickle.dump(servers, f)

  print(f'Finished {mode} of {name}.')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--yaml', type=Path)
  parser.add_argument('--results-dir', type=str, default='results')
  args = parser.parse_args()
  config = yaml.safe_load(open(args.yaml, 'r'))
  results_dir = args.results_dir
  run(config, results_dir)

