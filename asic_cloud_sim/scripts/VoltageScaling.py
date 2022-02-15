import os
import sys
import math
import CONSTANTS
import utils

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dfplot
import HTMLTable

pd.set_option('precision', 3)
#  delays = utils.delay_loader (['16nm','65nm','130nm', '180nm','250nm'])
delays = utils.delay_loader (['130nm'])

def delay_interpolator (voltage , tech):
   x0 = round(voltage,3)
   x1 = round(voltage+0.001,3)
   y0 = delays[tech][x0]
   y1 = delays[tech][x1]
   y = y0 + (voltage-x0) * ((y1-y0)/(x1-x0))
   return y 
 
def getDelay(voltage, tech):
   if tech == '16nm' or tech == '10nm' or tech == '7nm':
     delay = lambda (x): 0.9043676 + (55200190 - 0.9043676)/(1 + (x/0.05606959)**7.454929)
     return delay(voltage)
     #  return delay_interpolator (voltage, tech)
     #delays[tech][round(voltage,2)]

   elif tech == '22nm':
     #delay = lambda (x): 0.859+1/(4.5031*x**3+5.2082*x**2-5.2337*x+1.053)
     delay = lambda (x): 0.9511146 + (19091220 - 0.9511146)/(1 + (x/0.03698389)**6.032485) 
     if (voltage <= 0.9):
       return delay(voltage)
     else:
        return (0.9/voltage)
   elif tech == '28nm':
     #delay = lambda (x): 0.859+1/(4.5031*x**3+5.2082*x**2-5.2337*x+1.053)
     delay = lambda (x): 0.9511146 + (19091220 - 0.9511146)/(1 + (x/0.03698389)**6.032485) 
     if (voltage <= 0.9):
       return delay(voltage)
     else:
        return (0.9/voltage)
   # delay = lambda (x): x/((x - CONSTANTS.TechData.loc[tech, 'Vth'])**1.05) # alpha-power delay model
   # return delay(voltage)
   elif tech == '40nm': # similar Vdd and Vth, sharing the same delay curve
     # Vdd (V):     0.7  0.8   0.9   1.0   1.1   1.2
     # Delay (ps): 10.24 9.166 8.502 8.092 7.824 7.656
     #delay = lambda x: 7.225274 + (20.88606 - 7.225274)/(1 + (x/0.5113651)**4.018167) # fitted curve, making 40nm pareto curve of BC disappear
     #delay = lambda x: 7.135465 + (16779965 - 7.135465)/(1 + (x/0.008727068)**3.207474) # essentially 65nm curve div 2
     delay = lambda x: 0.7512032 + (10696000 - 0.7512032)/(1 + (x/0.00883598)**3.799659)
     if (voltage <= 0.9):
       return delay(voltage)
     else:
        return (0.9/voltage)
   elif tech == '65nm':
     # Vdd (V):     0.7   0.8   0.9   1.0   1.1   1.2
     # Delay (ps): 40.49 31.23 25.99 22.70 20.45 18.84
     #delay = lambda x: 14.27093 + (33559930 - 14.27093)/(1 + (x/0.008727068)**3.207474)
     #return delay(voltage)
     return delay_interpolator (voltage, tech)
   elif tech == '90nm':
     # Vdd (V):     0.7   0.8   0.9   1.0   1.1   1.2
     # Delay (ps): 51.19 40.39 34.10 30.02 27.22 25.20
     #delay = lambda x: 18.84715 + (28150600 - 18.84715)/(1 + (x/0.007286879)**2.99623)
     #return delay(voltage)/29.9432429558
     return delay_interpolator (voltage, '65nm')
   elif tech == '130nm':
     # Vdd (V):     0.7   0.8   0.9   1.0   1.1   1.2   1.3
     # Delay (ps): 69.81 55.84 47.53 42.07 38.29 35.52 33.44
     #delay = lambda x: 25.83747 + (51057880 - 25.83747)/(1 + (x/0.004833614)**2.807056)
     #return delay(voltage)/35.507281579
     return delay_interpolator (voltage, tech)
   elif tech == '180nm':
     # Vdd (V):      0.9   1.0   1.1   1.2    1.3   1.8
     # Delay (ps): 176.1 152.2 134.8 121.2  110.3  80.03
     #delay = lambda x: 43.73323 + (8196212 - 43.73323)/(1 + (x/0.002427462)**1.865289)
     #return delay(voltage)
     return delay_interpolator (voltage, tech)
   elif tech == '250nm':
     #delay = lambda x: 37.46+12228.17*math.exp(-4.79*x)
     #return delay(voltage)
     return delay_interpolator (voltage, tech)
   else:
     print 'ERROR, unknown tech'
     #delay = lambda x: 37.46+12228.17*math.exp(-4.79*x)
     #voltage_factor = nominal_voltage / CONSTANTS.TechData.loc["250nm", "CoreVdd"]
     return -1
     #delay_factor * delay(voltage/voltage_factor)

def getUnscaledClkLatency(spec):
   return max(spec['cp_lgc_path'], \
              spec['cp_mixed_lgc_path'] + spec['cp_mixed_sram_path'])
# End of evalMinClkLatency

def getClkLatency(spec):
   return max(spec['cp_lgc_path'], \
              spec['cp_mixed_lgc_path'] + spec['cp_mixed_sram_path']) / \
              spec['f_scale']
# End of evalClkLatency

def RCAVoltageScaling(lgc_vdd, sram_vdd, nominal_spec, tech = "28nm"):
   nom_lgc_delay  = nominal_spec['nom_lgc_delay']
   nom_sram_delay = nominal_spec['nom_sram_delay']

   new_lgc_delay = getDelay(lgc_vdd, tech)
   new_sram_delay = getDelay(sram_vdd, tech)

   if((nom_lgc_delay <= 0) or
      (nom_sram_delay <= 0) or
      (new_lgc_delay <= 0) or
      (new_sram_delay <= 0)):
      print >> sys.stderr, 'evalASICScaling: VDD out of range'
      sys.exit(-1)

   new_lgc_vdd_scale = lgc_vdd / nominal_spec['lgc_vdd']
   new_sram_vdd_scale = sram_vdd / nominal_spec['sram_vdd']

   # New Clock Latency
   lgc_delay_scale = new_lgc_delay / nom_lgc_delay
   sram_delay_scale = new_sram_delay / nom_sram_delay

   new_cp_lgc_delay = nominal_spec['cp_lgc_path'] * lgc_delay_scale
   new_cp_mixed_lgc_delay = nominal_spec['cp_mixed_lgc_path'] * lgc_delay_scale
   new_cp_mixed_sram_delay = nominal_spec['cp_mixed_sram_path'] * sram_delay_scale
   new_clk_latency = max(new_cp_lgc_delay, \
                         new_cp_mixed_lgc_delay + new_cp_mixed_sram_delay)
   new_clk_scale = getUnscaledClkLatency(nominal_spec) / new_clk_latency

   # New Dynamic Power
   lgc_dyn_pwr = nominal_spec['lgc_dyn_pwr'] * new_clk_scale * new_lgc_vdd_scale**2
   sram_dyn_pwr = nominal_spec['sram_dyn_pwr'] * new_clk_scale * new_sram_vdd_scale**2

   # New Static Power
   lgc_leak_pwr = nominal_spec['lgc_leak_pwr'] * new_lgc_vdd_scale**2
   sram_leak_pwr = nominal_spec['sram_leak_pwr'] * new_sram_vdd_scale**2
   asic_spec = nominal_spec.copy()

   asic_spec['f_scale'] = 1.0
   asic_spec['lgc_vdd'] = lgc_vdd
   asic_spec['sram_vdd'] = sram_vdd
   asic_spec['cp_lgc_path'] = new_cp_lgc_delay
   asic_spec['cp_mixed_lgc_path'] = new_cp_mixed_lgc_delay
   asic_spec['cp_mixed_sram_path'] = new_cp_mixed_sram_delay
   asic_spec['lgc_dyn_pwr'] = lgc_dyn_pwr
   asic_spec['lgc_leak_pwr'] = lgc_leak_pwr
   asic_spec['sram_dyn_pwr'] = sram_dyn_pwr
   asic_spec['sram_leak_pwr'] = sram_leak_pwr
   asic_spec['dram_bw'] = nominal_spec['dram_bw'] * new_clk_scale
   asic_spec['frequency'] = 1 / getClkLatency(asic_spec) * 1e3
   asic_spec['unit_perf'] = nominal_spec['unit_perf'] * new_clk_scale

   return asic_spec
# end of evalASICVoltageScaling

# Calculates ASIC power and performance
def asic_power_performance(asic_spec, dram_spec, io_spec, total_RCA, active_RCA, performance_drop):
   RCA_lgc_pwr  = (asic_spec['lgc_dyn_pwr'] * asic_spec['f_scale'] + \
                   asic_spec['lgc_leak_pwr'])
   RCA_sram_pwr = (asic_spec['sram_dyn_pwr'] * asic_spec['f_scale'] + \
                   asic_spec['sram_leak_pwr'])
   RCA_power = active_RCA * (RCA_lgc_pwr + RCA_sram_pwr)
  
  
   dram_count = asic_spec['dram_count'] 
   IO_count   = asic_spec['IO_count']  

   if dram_count > 0:
      DRAM_power = dram_count * dram_spec['MC_power']
   else:
      DRAM_power = 0
   
   if IO_count > 0:
      IO_power = IO_count * io_spec['IO_power']
   else:
      IO_power = 0

   asic_spec['w_lgc']  = RCA_lgc_pwr * active_RCA 
   asic_spec['w_sram'] = RCA_sram_pwr * active_RCA
   asic_spec['watts_per_asic'] = RCA_power + DRAM_power + IO_power
   asic_spec['dram_mc_power'] = DRAM_power
   asic_spec['IO_power'] = IO_power
   asic_spec['mhash_per_asic'] = asic_spec['unit_perf'] * total_RCA * performance_drop 
   asic_spec['joules_per_mhash'] = asic_spec['watts_per_asic'] / asic_spec['mhash_per_asic'] 

   return asic_spec
# end of asic_power_performance

if __name__ == '__main__':
  TechNodes =  ['16nm', '28nm', '40nm', '65nm', '90nm', '130nm', '180nm', '250nm']
  Marker = ['ro', 'bv', 'g^', 'k<', 'y>', 'c*', 'm+', 'rx']

  print '40nm', getDelay(0.9,'40nm')

  DfList = []
  fig, ax = plt.subplots(figsize=(15,9))
  for m, Tech in zip(Marker, TechNodes):
    Vth = CONSTANTS.TechData.loc[Tech, 'Vth']
    CoreVdd = CONSTANTS.TechData.loc[Tech, 'CoreVdd']
    VddLimit = CoreVdd * 1.5

    VddRange = np.linspace(Vth, VddLimit, 20) # 20 points
    Delay = map(lambda v: 1000000.0/getDelay(v, Tech), VddRange)

    ax.plot(VddRange, Delay, m, label=Tech, markersize=10)

  ax.grid()
  ax.legend(loc=4, borderaxespad=0, numpoints=1)
  ax.set_xlim(0.3, 1.5)

  plt.ylabel('Frequency')
  plt.xlabel('Voltage')
  plt.legend

  imf = os.environ.get('IMAGE_FORMAT')
  fig.savefig('VoltageScaling.'+imf)


#if __name__ == '__main__':
#  TechNodes =  ['16nm', '28nm', '40nm', '65nm', '90nm', '130nm', '180nm', '250nm']
#
#  DfList = []
#  for Tech in TechNodes:
#    Vth = CONSTANTS.TechData.loc[Tech, 'Vth']
#    CoreVdd = CONSTANTS.TechData.loc[Tech, 'CoreVdd']
#    VddLimit = CoreVdd * 1.5
#
#    VddRange = np.linspace(Vth, VddLimit, 20) # 20 points
#    Delay = map(lambda v: 1000000.0/getDelay(v, Tech), VddRange)
#
#    if Tech in ['28nm']:
#      DelayUnit = 'a.u.' # normalized
#    else:
#      DelayUnit = 'ps'
#
#    DelayColName = 'Delay ('+DelayUnit+')'
#
#    df = pd.DataFrame(np.matrix([VddRange, Delay]).T, columns = ['Vdd (V)', DelayColName])
#
#    DfList.append(df)
#
#    ax = dfplot.line(df, x='Vdd (V)', y=DelayColName,
#                     xlabel='Vdd (V)', ylabel=DelayColName,
#                     linestyle=':', legend=None, figsize=(10,10))
#
#    ax.set_xlim(Vth-0.1, VddLimit+0.1)
#  # ax.set_title('Voltage-Delay Curve of '+Tech+' Technology')
#
#    fig = ax.get_figure()
#    imf = os.environ.get('IMAGE_FORMAT')
#    fig.savefig('DelayCurve_'+Tech+'.'+imf, bbox_inches = 'tight')
#
#    plt.close(fig)
#
#  DfTable = pd.concat(DfList, axis=1, keys=TechNodes)
#
#  HTMLTableTitle = '''
#  <b>Voltage-Delay Curves of Process Nodes</b>
#  '''
#
#  HTMLBody = '''
#  <img src='DelayCurve_16nm.svg' height='650px', width='650px'/>
#  <img src='DelayCurve_28nm.svg' height='650px', width='650px'/>
#  <img src='DelayCurve_40nm.svg'  height='650px', width='650px'/>
#  <img src='DelayCurve_65nm.svg' height='650px', width='650px'/>
#  <img src='DelayCurve_90nm.svg'  height='650px', width='650px'/>
#  <img src='DelayCurve_130nm.svg' height='650px', width='650px'/>
#  <img src='DelayCurve_250nm.svg' height='650px', width='650px'/>
#  '''
#
#  HTMLTable.df2html(DfTable, 'DelayCurve', 'Regular', HTMLTableTitle, HTMLBody)
