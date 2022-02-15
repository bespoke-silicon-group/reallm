import time
import CONSTANTS
import multiprocessing
import utils
import ASICSpecs
import IOScaling
import VoltageScaling
import TechScaling
from DieCost import fill_yield

from chiplet_system_elaborator import chiplet_elaborator

if __name__ == '__main__':
  utils.extra_header.append('chiplet_depth')
  utils.extra_header.append('chiplet_width')
  utils.extra_header.append('used_units')
  number_of_cores = multiprocessing.cpu_count()
  #print "Number of cores %d" % (number_of_cores)
  p = multiprocessing.Pool(number_of_cores)
  
  PEs_per_col_constraints = [64]
  techs = ['22nm']
  PEs_on_chiplet =[4, 8, 16, 32, 64, 128, 256, 512, 1024] 
  liquid_cool = False

  start_time = time.time()
  app='Chiplet'
 
  for tech in techs:
    default_spec=TechScaling.Classical(eval('ASICSpecs.'+app),tech, TechNorm="22nm")
    Vth = CONSTANTS.TechData.loc[tech, 'Vth']
    CoreVdd = CONSTANTS.TechData.loc[tech, 'CoreVdd']
    Vdd = CoreVdd 
    size = CONSTANTS.TechData.at[tech, "FeatureSize"]
    asic_spec = VoltageScaling.RCAVoltageScaling(Vdd, Vdd, default_spec, tech)
    
    # Voltage scaling to get 1000Mhz frequency
    if (size > 22) :
      while (asic_spec['frequency'] < 1000) and (Vdd < 1.5 * CoreVdd):
        Vdd += 0.005
        asic_spec = VoltageScaling.RCAVoltageScaling(Vdd, Vdd, default_spec, tech)
      
    elif (size < 22): 
      while (asic_spec['frequency'] > 1000) and (Vdd >= Vth):
        Vdd -= 0.005
        asic_spec = VoltageScaling.RCAVoltageScaling(Vdd, Vdd, default_spec, tech)

      
    io_spec = IOScaling.GetSpec(tech)
    RCA_performance = 1000 # GFLOPS/s
    asic_spec['IO_count'] = 1 # just non zero for now
    asic_spec['dram_count'] = 0
    fill_yield(asic_spec, None, io_spec, tech)

    #  for spec in asic_spec:
      #  print spec, asic_spec[spec]
      
    for PEs_per_col in PEs_per_col_constraints:
     designs = []
     for x in PEs_on_chiplet:
       if x > PEs_per_col:
         continue
       designs.append([asic_spec, io_spec, x, 1, PEs_per_col/x, RCA_performance, tech, False, liquid_cool])

     results = p.map(chiplet_elaborator, designs)

     if liquid_cool:
       o_file = open(tech+'_'+str(PEs_per_col)+'PEs_per_lane_liquid_cool'+'.csv', 'w')
     else:
       o_file = open(tech+'_'+str(PEs_per_col)+'PEs_per_lane'+'.csv', 'w')
     utils.fprintHeader(o_file, area_csv=True)
     for result in results:
       if result is not None:
         o_file.write("%s" % result)
         print "%s" % result
     o_file.close()
  
  p.close()
  p.join()
 
  print ('#Run time is %s seconds' % (time.time() - start_time))

