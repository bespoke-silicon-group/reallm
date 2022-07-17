from opt_routing import opt_routing

if __name__ == '__main__': 

  algo_spec = {
    'num_layers': 48,
    'd': 1600,
  }

  sys_spec = {
    'chips_per_pkg': 1,
    'pkgs_per_srv': 24,
    'num_srvs': 1,
    'chip_tops': 36.7,
    'c2c_bw': 100, # in GB/s
    'p2p_bw': 25,  # in GB/s
    's2s_bw': 10,  # in GB/s
    'T_start': 0.01,  # us, init time for each data transfer
    'hbm_bw': None
  }

  best_routing, best_delay, detail_delay = opt_routing(sys_spec, algo_spec)
  print(best_routing, best_delay)

