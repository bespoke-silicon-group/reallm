import time
import multiprocessing
import utils

from chiplet_system_elaborator import chiplet_elaborator

if __name__ == '__main__':
  number_of_cores = multiprocessing.cpu_count()
  #print "Number of cores %d" % (number_of_cores)
  p = multiprocessing.Pool(number_of_cores)
  
  techs = ['7nm']
  # MB memory per chiplet
  MEM_per_chiplet = [40, 80, 160, 320] 

  IO_bandwidth = 50.0 # GB/s
  IO_count = 2*(IO_bandwidth/12.5) # transmitter and receiver

  liquid_cool = False
  use_total_power = True

  start_time = time.time()
  app='Chiplet'
 
  for tech in techs:
    designs = []
    for MEM in MEM_per_chiplet:
      # BF16 tera ops per second per chiplet
      TOPS_per_chiplet = [0.1*MEM, 0.15*MEM, 0.2*MEM, 0.25*MEM, 0.3*MEM, 0.35*MEM, 0.4*MEM, 0.45*MEM]
      for TOPS in TOPS_per_chiplet:
        #  chiplet_elaborator([tech, TOPS, MEM, IO_count, liquid_cool, use_total_power])
        designs.append([tech, TOPS, MEM, IO_count, liquid_cool, use_total_power])
      
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

