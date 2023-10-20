import argparse
from structs.Chip import Chip
from structs.Package import Package
from structs.Server import Server
from structs.IO import IO
from utils.utils import to_csv
from typing import Optional
import multiprocessing
from itertools import product

def explore_servers(package: Package, packages_per_lane: int, io: IO) -> Optional[Server]:
  server = Server(package=package, packages_per_lane=packages_per_lane, io=io)
  if server.valid:
    return server
  else:
    return None

def run(config, results_dir):

  if config == 'exploration':
    chip_num = 0
    chip_area_options = [20, 30, 40, 50, 60, 80, 100, 120, 140, 160, 180, 200, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680, 720, 760, 800]
    mac_area_ratio_options = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.40, 0.50]
    operational_intensity_options = [1.0, 1.25, 1.5, 1.75, 2.0, 4.0, 8.0, 16.0, 32.0]
    packages_per_lane_options = range(3, 21)
    chip_io = IO(io_type='p2p', num=4, bandwidth_per_io=12.5e9, 
                 area_per_io=0.3, tdp_per_io=0.125)
    server_io = IO(io_type='s2s', num=2, bandwidth_per_io=10e9)
    
    server_specs = []
    for chip_area in chip_area_options:
      for mac_area_ratio in mac_area_ratio_options:
        for operational_intensity in operational_intensity_options:
          chip = Chip(chip_id=chip_num, pkg2pkg_io=chip_io, area=chip_area, mac_ratio=mac_area_ratio, 
                      operational_intensity=operational_intensity)
          chip_num += 1
          if chip.valid:
            # print(chip.area, chip.sram, chip.sram_area, chip.perf/1e12, chip.sram_bw/1e12, chip.dies_per_wafer, chip.cost)
            package = Package(chip=chip, num_chips=1)
            for packages_per_lane in packages_per_lane_options:
              silicon_per_lane = (package.heatsource_length * package.heatsource_width) * packages_per_lane 
              if silicon_per_lane < 6000 and silicon_per_lane > 400:
                server_specs.append((package, packages_per_lane, server_io))
    print(chip_num)
    print('Start to evaluate servers')
    servers = []
    num_cores = multiprocessing.cpu_count() # 144 on kk7
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(explore_servers, server_specs)
    for result in results:
      if result != None:
        servers.append(result)

    o_file_path = results_dir+'/'+config+'.csv'
    to_csv(o_file_path, servers)
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

