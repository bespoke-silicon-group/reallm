import time
import multiprocessing
import utils
import numpy as np

from chiplet_system_elaborator import chiplet_elaborator

if __name__ == '__main__':
  number_of_cores = multiprocessing.cpu_count()
  #print "Number of cores %d" % (number_of_cores)
  p = multiprocessing.Pool(number_of_cores)
  
  app='GPT3'

  techs = ['7nm']
  # MB memory per chiplet
  MEM_per_chiplet = np.arange(40.0, 340.0, 20.0)

  IO_bandwidth = 12.5 # GB/s

  keep_large_power = False
  use_total_power = True

  start_time = time.time()
  for tech in techs:
    designs = []
    for MEM in MEM_per_chiplet:
      # BF16 tera ops per second per chiplet
      TOPS_per_chiplet = np.arange(0.01*MEM, 0.5*MEM, 0.02*MEM, dtype=float)
      for TOPS in TOPS_per_chiplet:
        designs.append([app, tech, TOPS, MEM, IO_bandwidth, keep_large_power, use_total_power])
      
    results = p.map(chiplet_elaborator, designs)
    o_file = open('results'+'.csv', 'w')
    utils.fprintHeader(o_file, True)
    for result in results:
     if result is not None:
       o_file.write("%s" % result)
    o_file.close()
  
  p.close()
  p.join()
 
  print ('#Run time is %s seconds' % (time.time() - start_time))

