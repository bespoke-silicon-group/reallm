#!/usr/bin/env python
import math
import FabCost
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
def RCA_yield(area,tech): 
  if (area == 0):
    return 1.0
  else:
    return yield_negative_binomial(area,CONSTANTS.D0s[tech],CONSTANTS.alphas[tech])

# select function in math
def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

# calculates probability for a case of binomial
def p_binomial (RCA_count, healthy_count, RCA_yield):
  occurences =   nCr(RCA_count,healthy_count)
  # If result is too large we should use Decimal library for intermediate results
  if isinstance(occurences, long): 
    return float(Decimal(occurences) * \
                 Decimal(pow(RCA_yield,healthy_count))* \
                 Decimal(pow(1.0-RCA_yield,RCA_count-healthy_count)))
  else:
    return (occurences* \
            pow(RCA_yield,healthy_count)* \
            pow(1.0-RCA_yield,RCA_count-healthy_count))

# calculates probability of dies that have the required working RCAs
def at_least_yield (required_RCA, total_RCA, RCA_yield):
  if (required_RCA==0):
    return 1.0
  elif ((total_RCA-required_RCA)>required_RCA):
    sum_bad = 0.0
    for i in range(0,required_RCA):
      sum_bad += p_binomial(total_RCA,i,RCA_yield)
    return 1.0 - sum_bad
  else:
    sum_good = 0.0
    for i in range(required_RCA,total_RCA+1):
      sum_good += p_binomial(total_RCA,i,RCA_yield)
    # Floating point addition error fix
    if (sum_good > 1.0):
      sum_good = 1.0
    return sum_good

# calculates expected value for number of working RCAs per die
def average_working_RCAs (RCAs_per_die, provisioned_RCAs, 
                          at_least_working_RCAs, RCA_yield,
                          all_RCA_y = -1):
  sum_working = 0.0
  if (at_least_working_RCAs <= int(math.floor(provisioned_RCAs))):
    for i in range (at_least_working_RCAs,int(math.floor(provisioned_RCAs))+1):
      sum_working += i * p_binomial(RCAs_per_die,i,RCA_yield)
  
  for i in range (int(math.floor(provisioned_RCAs))+1,RCAs_per_die+1):
    sum_working +=  provisioned_RCAs * p_binomial(RCAs_per_die,i,RCA_yield)
  
  if (all_RCA_y == -1):
    return sum_working/at_least_yield (at_least_working_RCAs, RCAs_per_die, RCA_yield)
  else:
    return sum_working/all_RCA_y

# testing cost model of 1% overhead
def testing_cost_ratio (die_size, test_time, to_be_tested= True):
  if (not to_be_tested):
    return 0.0
  else:
    return 0.01

###################################
# Die Cost                        #
###################################
def patterning_area (die_area, outwards = True):
  side = math.sqrt(die_area)
  L = 0.35
  if outwards:
    return 4.0 * L * (side + L)
  else:
    return 4.0 * L * (side - L)

def fill_yield (asic_spec,dram_spec,io_spec,tech):
  dram_count = asic_spec['dram_count'] 
  IO_count   = asic_spec['IO_count']  
  
  if dram_count > 0:
    dram_area = dram_spec['MC_area']
  else:
    dram_area = 0.0
  
  if IO_count > 0:
    IO_area = io_spec['IO_area']
  else:
    IO_area = 0.0
  RCA_area   = asic_spec['unit_area']

  asic_spec['unit_dram_area'] = dram_area
  asic_spec['unit_io_area']   = IO_area
  asic_spec['dram_y']         = RCA_yield(dram_area,tech)
  asic_spec['RCA_y']          = RCA_yield(RCA_area,tech)
  asic_spec['IO_y']           = RCA_yield(IO_area,tech)
  asic_spec['network_y']      = 1.0

  return

# calculates area, cost, performance drop and dies_per_wafer for a design
def die_area_calc (asic_spec, N): 
  #calculating die area
  die_area = N * 1.0 * (asic_spec['unit_area'] + asic_spec['sram_area']) + \
             asic_spec['dram_count'] * asic_spec['unit_dram_area'] + \
             asic_spec['IO_count'] * asic_spec['unit_io_area']

  return die_area+patterning_area(die_area)
  
# Calculates die yield
def die_yield_calc (asic_spec, at_least_working_RCA, N):
  # calculate all RCAs acceptance rate
  RCA_accept_rate = at_least_yield (at_least_working_RCA, N, asic_spec['RCA_y']) 
  
  # calculate total die yield based on binomial model
  die_y      = (asic_spec['dram_y']**asic_spec['dram_count']) * \
               (asic_spec['IO_y']**asic_spec['IO_count']) * asic_spec['network_y'] * \
               RCA_accept_rate 
  #die_y = 1.0 # for testing without added die cost due to yield

  return (die_y, RCA_accept_rate)

# calculates die cost
def die_cost_calc (die_area, die_yield, tech, to_be_tested, MPW = False):
  # calculate silicon and testing cost
  if (MPW):
    dpw = FabCost.CostBook[tech]['rpw']
    die_cost = (FabCost.CostBook[tech]['cpw'] *1.0 / dpw) / die_yield
  else:
    dpw = dies_per_wafer(die_area,CONSTANTS.TechData.loc[tech, "WaferDiameter"]*1e3)
    die_cost = (FabCost.CostBook[tech]['fmwc'] *1.0 / dpw) / die_yield
  testing_cost_per_die = testing_cost_ratio (0,0,to_be_tested) * die_cost
  
  return (die_cost + testing_cost_per_die, dpw) 
