import math
import CONSTANTS
from decimal import *

#######################################
# Calculate dies per wafer            #
#######################################

# given circle radious and square edge length, and distance between
# diameter and center row of squares (longest one), calculates the 
# number of squares that can be fited
def max_fit(r,a,d):
	D1 = a/2.0 + d
	# we don't need to calculate the longest line close to diameter
	D2 = (3.0*a)/2.0 - d
	R2 = r*r
	summ = 0
	while (D1<r):
		l = 2.0*math.sqrt(R2-D1*D1)
		summ += math.floor(l/a)
		D1 += a
		
	while (D2<r):
		l = 2.0*math.sqrt(R2-D2*D2)
		summ += math.floor(l/a)
		D2 += a
	
	return int(summ)
	
# we want to find the optimal value by binary searching the different values
# for center row distance to diameter. The optimal solution would have 
# the longest row farthest from the center, having two long lines is the best if possible
def max_square (r,a):
	start = 0.0
	end   = a/2.0
	max_start = max_fit(r,a,start)
	max_end   = max_fit(r,a,end)
	# if we can fit two long rows, it's the optimal point
	if max_end >= max_start:
		return max_end	
	step    = a/8.0
	end     = end/2.0
	max_end = max_fit(r,a,end)
	while (max_start != max_end):
		if (max_start>max_end):
			end -= step
		else:
			start = end
			max_start = max_end
			end += step
			
		max_end = max_fit(r,a,end)
		step = step / 2.0
		#print "max_start , max_end: ", max_start , max_end
	return max_end
	
# Converts the problem of dies per wafer to squares per circle
# by adding the dicing gap to die width as square edge and calculate
# radious instead of diameter
def dies_per_wafer (area, wafer_diameter=300):
	return max_square(wafer_diameter/2.0,math.sqrt(area)+(CONSTANTS.DicingGap*1e3))

##############################
# Yield model                #
##############################

# Negative binomial model for yield
def yield_negative_binomial (area, D0, alpha):
  return math.pow((1.0+(D0*area/alpha)),alpha*-1)
  
# wrapper that gets area and tech and caculates yield using negative binomial model
def die_yield_calc(area, tech): 
  if (area == 0):
    return 1.0
  else:
    return yield_negative_binomial(area,CONSTANTS.D0s[tech],CONSTANTS.alphas[tech])

def die_area_calc(asic_spec):
  #calculating die area
  die_area = asic_spec['lgc_area'] + asic_spec['sram_area'] + \
             asic_spec['io_area'] + \
             asic_spec['other_area']
  die_area = die_area / 0.7
  side = math.sqrt(die_area)
  L = 0.35
  patterning_area = 4.0 * L * (side + L)
  return die_area+patterning_area

# calculates die cost
def die_cost_calc(die_area, die_yield, tech):
  # calculate silicon and testing cost
  dpw = dies_per_wafer(die_area,CONSTANTS.TechData.loc[tech, "WaferDiameter"]*1e3)
  die_cost = (CONSTANTS.FMWC[tech] *1.0 / dpw) / die_yield
  # testing cost model of 1% overhead
  testing_cost_per_die = 0.01 * die_cost
  
  return (die_cost + testing_cost_per_die, dpw) 


