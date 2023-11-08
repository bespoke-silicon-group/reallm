import argparse
from structs.Chip import Chip
from structs.Package import Package
from structs.IO import IO
from tests.sram_only_single_chip_configs import chip_config, package_config, server_config
from utils.utils import to_csv
import pickle

def run(config, results_dir):

  if config == 'exploration':
    chips = chip_config.explore()
    packages = package_config.explore(chips)
    servers = server_config.explore(packages)
    # print results to csv file
    o_file_path = results_dir+'/'+config+'.csv'
    to_csv(o_file_path, servers)
    # save results to pickle file
    with open(results_dir+'/'+config+'.pkl', 'wb') as f:
      pickle.dump(servers, f)
    print(f'Found {len(servers)} valid hardware designs.')

  elif config == 'test':
    chip_num = 0
    chip_area_options = [140]
    mac_area_ratio_options = [0.16]
    operational_intensity_options = [2.0]
    packages_per_lane_options = range(17, 18)
    chip_io = IO(io_type='p2p', num=4, bandwidth_per_io=12.5e9, 
                 area_per_io=0.3, tdp_per_io=0.125)
    server_io = IO(io_type='s2s', num=2, bandwidth_per_io=12.5e9)
    
    server_specs = []
    for chip_area in chip_area_options:
      for mac_area_ratio in mac_area_ratio_options:
        for operational_intensity in operational_intensity_options:
          chip = Chip(chip_id=chip_num, pkg2pkg_io=chip_io, area=chip_area, mac_ratio=mac_area_ratio, 
                      operational_intensity=operational_intensity)
          chip_num += 1
          if chip.valid:
            package = Package(chip=chip, num_chips=1)
            for packages_per_lane in packages_per_lane_options:
              silicon_per_lane = (package.heatsource_length * package.heatsource_width) * packages_per_lane 
              if silicon_per_lane <= 6000 and silicon_per_lane >= 400:
                server_specs.append((package, packages_per_lane, server_io))
    print(chip_num)
    print('Start to evaluate servers')
    servers = []
    num_cores = 2
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(explore_servers, server_specs)
    for result in results:
      if result != None:
        servers.append(result)
    # print results to csv file 
    o_file_path = results_dir+'/'+config+'.csv'
    to_csv(o_file_path, servers) 
    # save results to pickle file
    with open(results_dir+'/'+config+'.pkl', 'wb') as f:
      pickle.dump(servers, f)
    print(f'Found {len(servers)} valid hardware designs.')


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='exploration')
  parser.add_argument('--results-dir', type=str, default='results')
  args = parser.parse_args()
  config = args.config
  results_dir = args.results_dir
  run(config, results_dir)
  print('write results of', config)

