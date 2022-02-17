import time
import multiprocessing
import utils

from chiplet_system_elaborator import chiplet_elaborator

if __name__ == '__main__':
  number_of_cores = multiprocessing.cpu_count()
  #print "Number of cores %d" % (number_of_cores)
  p = multiprocessing.Pool(number_of_cores)
  
  techs = ['7nm']
  # BF16 tera ops per second per chiplet
  TOPS_per_chiplet = [40, 80, 160, 320] 
  # MB memory per chiplet
  MEM_per_chiplet = [40, 80, 160, 320] 

  IO_bandwidth = 100.0 # GB/s
  IO_count = 2*(IO_bandwidth/12.5) # transmitter and receiver

  liquid_cool = True

  start_time = time.time()
  app='Chiplet'

  
 
  print 'tech_node', 'tops_per_asic', 'sram_per_asic', 'chiplets_per_board', 'die_area', 'watts_per_asic'
  for tech in techs:
    designs = []
    for TOPS in TOPS_per_chiplet:
      for MEM in MEM_per_chiplet:
        chiplet_elaborator([tech, TOPS, MEM, IO_count, liquid_cool])
        #  designs.append([tech, TOPS, MEM, IO_count, liquid_cool])
      
    # results = p.map(chiplet_elaborator, designs)
    # if liquid_cool:
    #   o_file = open(tech+'_liquid_cool'+'.csv', 'w')
    # else:
    #   o_file = open(tech+'_PEs_per_lane'+'.csv', 'w')
    # utils.fprintHeader(o_file)
    # for result in results:
    #   if result is not None:
    #     o_file.write("%s" % result)
    #     #  print "%s" % result
    # o_file.close()
  
  p.close()
  p.join()
 
  print ('#Run time is %s seconds' % (time.time() - start_time))

