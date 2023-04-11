def set_config(config):
  if config == 'exploration':
    # Chiplet Size Exploration
    tech = '7nm'
    lanes_per_server = 8
    IO_bandwidth = 25.0
    
    lane_silicon_min = 100
    lane_silicon_max = 6000
    lane_silicon_step = 100
    lane_silicon_options = range(lane_silicon_min, lane_silicon_max+1, lane_silicon_step)
    lane_silicon_options = [100, 200, 500, 750, 1000, 2000, 3000, 4500, 6000]
    
    lane_chiplets_min = 1
    lane_chiplets_max = 20
    lane_chiplets_step = 2
    lane_chiplets_options = range(lane_chiplets_min, lane_chiplets_max+1, lane_chiplets_step)
    lane_chiplets_options = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]

    mac_area_ratio_options = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
  else:
    print('Wrong Tasks!')
    return -1

  return tech, IO_bandwidth, lanes_per_server, lane_silicon_options, lane_chiplets_options, mac_area_ratio_options
