def set_config(config):
  if config == 'exploration':
    # Chiplet Size Exploration
    app = 'GPT3'
    tech = '7nm'
    
    use_dram = False
    keep_large_power = False
    use_total_power = True
    
    srv_mem = 3840.0
    IO_bandwidth = 25.0
    
    srv_tops_min = 400
    srv_tops_max = 1200
    srv_tops_step = 100
    srv_tops_options = range(srv_tops_min, srv_tops_max+1, srv_tops_step)
    
    srv_chiplets_min = 6
    srv_chiplets_max = 90
    srv_chiplets_step = 6
    srv_chiplets_options = range(srv_chiplets_min, 
                                 srv_chiplets_max+1, 
                                 srv_chiplets_step)
    TPU = False
    SI = False
    organic_sub = False
  elif config == 'HBM':
    # HBM
    app = 'GPT3'
    tech = '7nm'
    
    use_dram = True
    keep_large_power = False
    use_total_power = True
    
    srv_mem = 192.0
    # srv_mem = 32.0
    IO_bandwidth = 25.0
    
    srv_tops_min = 400
    srv_tops_max = 1200
    srv_tops_step = 100
    srv_tops_options = range(srv_tops_min, srv_tops_max+1, srv_tops_step)
    
    srv_chiplets_min = 6
    srv_chiplets_max = 24
    srv_chiplets_step = 6
    srv_chiplets_options = range(srv_chiplets_min, 
                                 srv_chiplets_max+1, 
                                 srv_chiplets_step)
    
    TPU = False
    SI = False
    organic_sub = False
  elif config == 'TPU':
    # TPU
    app = 'GPT3'
    tech = '7nm'
  
    use_dram = False
    keep_large_power = True
    use_total_power = True
  
    srv_mem = 576.0
    IO_bandwidth = 25.0
  
    srv_tops_options = [552.0]
  
    srv_chiplets_options = [4]
  
    TPU = True
    SI = False
    organic_sub = False
  elif config == 'SI':
    app = 'GPT3'
    tech = '7nm'
    
    use_dram = False
    keep_large_power = False
    use_total_power = True
    
    srv_mem = 3840.0
    IO_bandwidth = 25.0
    
    srv_tops_min = 400
    srv_tops_max = 1200
    srv_tops_step = 100
    srv_tops_options = range(srv_tops_min, srv_tops_max+1, srv_tops_step)
    
    srv_chiplets_min = 12
    srv_chiplets_max = 90
    srv_chiplets_step = 6
    srv_chiplets_options = range(srv_chiplets_min, 
                                 srv_chiplets_max+1, 
                                 srv_chiplets_step)
    TPU = False
    SI = True
    organic_sub = False
  elif config == 'organic_sub':
    app = 'GPT3'
    tech = '7nm'
    
    use_dram = False
    keep_large_power = False
    use_total_power = True
    
    srv_mem = 3840.0
    IO_bandwidth = 25.0
    
    srv_tops_min = 400
    srv_tops_max = 1200
    srv_tops_step = 100
    srv_tops_options = range(srv_tops_min, srv_tops_max+1, srv_tops_step)
    
    srv_chiplets_min = 6
    srv_chiplets_max = 90
    srv_chiplets_step = 6
    srv_chiplets_options = range(srv_chiplets_min, 
                                 srv_chiplets_max+1, 
                                 srv_chiplets_step)
    TPU = False
    SI = False
    organic_sub = True
  else:
    print('Wrong Tasks!')
    return -1

  return app, tech, use_dram, keep_large_power, use_total_power, srv_mem, IO_bandwidth, srv_tops_options, srv_chiplets_options, TPU, SI, organic_sub
