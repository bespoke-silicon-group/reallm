import math
import TR
import CONSTANTS
import FAN

# Convert from CFM(cubic feet per minute) to CMS(cubic meters per second)
def CFM2CMS(cfm):
   return cfm * 0.3048**3 / 60

# Convert from CMS(cubic meters per second) to CFM(cubic feet per minute)
def CMS2CFM(cms):
   return cms / (0.3048**3) * 60

def PowerBudget(hs_spec, n_of_asics):
   return (CONSTANTS.MaxDieTemp - CONSTANTS.Ambient) / \
            (TR.evalRTotal(hs_spec) + (n_of_asics - 1) / \
            (hs_spec['air_specific_heat'] * hs_spec['air_density'] * hs_spec['fin_air_volume']))
# end of PowerBudget

# Maximum number of fins
def MaxNofFins(hs_spec):
   return int((hs_spec['base_width'] + CONSTANTS.MinFinPitch) / (hs_spec['fin_thickness'] + CONSTANTS.MinFinPitch))
# end of MaxNofFins

def findBestThickAndN(hs_tmpl, asics_per_col, full_lane_hs=True):
   base_thickness = CONSTANTS.BaseThickPrecision
   best_spec_at_length = None

   n_of_fins = 4
   fin_step = 2
   max_fins = MaxNofFins(hs_tmpl);
   air_vol_step = CFM2CMS(CONSTANTS.AirVolumePrecision)
   initial_air_vol = air_vol_step 

   while(base_thickness < CONSTANTS.HSMaxHeight):
      n_of_fins = 2
      best_spec_at_thick = None
      

      while(n_of_fins <= max_fins):
         hs_spec = hs_tmpl.copy()
         hs_spec['n_of_fins'] = n_of_fins
         hs_spec['base_thickness'] = base_thickness
         hs_spec['fin_height'] = CONSTANTS.HSMaxHeight - base_thickness

         # Estimate suppliable air volume assuming one big heat sink
         # whose length is base_length * asics_per_col
         if (full_lane_hs):
            hs_spec['base_length'] *= asics_per_col
         fin_air_volume = FAN.evalAirVolume(hs_spec, initial_air_vol,
                                            FAN.fan_9GA0312P3K001, air_vol_step)
         initial_air_vol = fin_air_volume - 5*air_vol_step
         if (full_lane_hs):
            hs_spec['base_length'] /= asics_per_col

         if(fin_air_volume <= 0.0):
            # Pressure drop too high
            break
      

         hs_spec['fin_air_volume'] = fin_air_volume
         # Estimate each thermal resistance
         TR.evalDIEThermalResistance(hs_spec)
         TR.evalTIMThermalResistance(hs_spec)
         TR.evalFinThermalResistance(hs_spec)
         TR.evalBaseSpreadingResistance(hs_spec)

         hs_spec['r_total'] = TR.evalRTotal(hs_spec)
         hs_spec['q_asic'] = PowerBudget(hs_spec, asics_per_col)

         # Store the Best Result
         if(best_spec_at_thick == None or
            best_spec_at_thick['q_asic'] < hs_spec['q_asic']):
            best_spec_at_thick = hs_spec
         else:
            # q_asic would be worse if the number of fins is further increased
            # due to higher pressure drop.
            break
         
         n_of_fins += fin_step
         

      if(best_spec_at_thick != None):
         if(best_spec_at_length == None or
            best_spec_at_length['q_asic'] < best_spec_at_thick['q_asic']):
            best_spec_at_length = best_spec_at_thick
         else:
            # q_asic would be worse with a thicker spreader
            break

      base_thickness += CONSTANTS.BaseThickPrecision
      fin_step = 1 
   return best_spec_at_length
# end of findBestThickAndN

# find best HS for die
def findHS(hs_tmpl, asics_per_col, max_length):
   if hs_tmpl['die_width'] > hs_tmpl['base_width']:
      print 'ERROR, Die width is larger than Heat Sink width'
      return None
   
   # Find the best result by considering single heatsink covering the lane 
   hs_spec = hs_tmpl.copy()
   hs_spec['base_length'] = max_length
   best_result = findBestThickAndN(hs_spec, asics_per_col, False)
   #print "best:", best_result['q_asic'], best_result['base_length']

   # Find the 95% of the best or shorter one
   base_length = math.ceil(hs_tmpl['die_length'] / CONSTANTS.BaseLengthPrecision) * CONSTANTS.BaseLengthPrecision
   hs_to_use = None

   while(base_length <= max_length / asics_per_col):
      hs_spec = hs_tmpl.copy()
      hs_spec['base_length'] = base_length

      best_spec_at_length = findBestThickAndN(hs_spec, asics_per_col)

      # Register the result
      if(best_spec_at_length != None):
         if(hs_to_use == None or
            hs_to_use['q_asic'] < best_spec_at_length['q_asic']):
            if(hs_to_use == None or
               best_spec_at_length['q_asic'] <= best_result['q_asic'] * 0.95):
               hs_to_use = best_spec_at_length
            else:
               # The result exceeds 95% of the best. Too good.
               break

      base_length += CONSTANTS.BaseLengthPrecision
   #print "results:", hs_to_use['q_asic'],hs_to_use['base_length'], hs_to_use['base_length']*asics_per_col
   return hs_to_use
# end of findHSToUseFast

def evalHS(die_area, asics_per_col):
  hs_spec = TR.getInputTemplate()
  # converting from mm to m as well
  hs_spec['die_length'] = (die_area**0.5) * 1e-3
  hs_spec['die_width']  = (die_area**0.5) * 1e-3
  hs_spec = findHS(hs_spec, asics_per_col, CONSTANTS.total_length)
  if hs_spec is None:
    print 'No Heatsink can be found for diea area of',die_area,'asics_per_col of', asics_per_col
    return (0,0)
  else:
    hs_cost = CONSTANTS.evalHeatSinkCost(hs_spec)
    return (hs_spec['q_asic'],hs_cost)
 
