import math
# Core functions:
# 1) evalFinThermalResistance
#    -> Evaluate the heat resistance of a heatsink (r_fins)
# 2) evalFinPressureDrop 
#    -> Evaluate the pressure drop (fins_pd)
# 3) evalBaseSpreadingResistance 
#    -> Evaluate the spreading resistance of a heatsink (r_sp)
# 4) evalHSIncomingAir
#    -> Evaluate the amount of air required at the opening of the duct to 
#       supply certain amount of air to the heatsink (duct_inair_volume)
#
# You need inputs listed below to run those functions.
#                    | (1) | (2) | (3) | (4) |
# base_length        | x   | x   | x   |     |
# fin_height         | x   | x   |     | x   |
# fin_thickness      | x   | x   |     | x   |
# fin_air_volume     | x   | x   |     | x   |
# fin_thermal_cond   | x   |     |     |     |
# n_of_fins          | x   | x   |     | x   |
# die_width          |     |     | x   |     |
# die_length         |     |     | x   |     |
# base_width         | x   | x   | x   | x   | 
# base_thickness     |     |     | x   |     |
# base_thermal_cond  |     |     | x   |     |
# fins_pd            |     |     |     | x   |
## duct_width         |     |     |     | x   |
## duct_height        |     |     |     | x   |
#
# Input Parameters (All values must be in the MKS unit system)
# base_width         : Width of the heatsink
# base_length        : Depth of the heatsink
# fin_height         : Height of the heatsink, not including base thickness
# n_of_fins          : Number of fins in the heatsink
# fin_thickness      : Thickness of each fin
# fin_air_volume     : Air volume that will be supplied to the heatsink
#                      (cubic meters per second)
# fin_thermal_cond   : Thermal conductivity of the heatsink
# die_width          : Width of the silicon die (heat source)
# die_length         : Depth of the silicon die (heat source)
# base_thickness     : Thickness of the bottom part of the heatsink
# base_thermal_cond  : Thermal conductivity of the bottom part of the heatsink
# duct_width         : Width of the duct in which the heatsink is
# duct_height        : Height of the duct in which the heatsink is
# fins_pd            : Pressure drop induced by the heatsink
#                      This value is the output of evalFinPressureDrop 
# tim_thermal_cond   : Thermal conductivity of thermal interface material (TIM)
# tim_thickness      : Thickness of TIM in m
# die_thermal_cond   : Thermal conductivity of silicon
# die_thickness      : Thickness of Silicon in m

input_template = {
   # Heat Source
   'die_length'             : 0.01,       # Heat source
   'die_width'              : 0.01,

   # Heat Sink
   'fin_height'             : 30e-3,
   'fin_thermal_cond'       : 200.0,      # Thermal conductivity, W/(mK)
   'fin_thickness'          : 0.5e-3,
   'fin_air_volume'         : 0.009439,   # m^3/s
   'n_of_fins'              : 20.0,
   'base_thickness'         : 5e-3,
   'base_width'             : 0.04,
   'base_length'            : 0.04,
   'base_thermal_cond'      : 200.0,      # Thermal conductivity, W/(mK)

   # TIM and Silicon
   'tim_thermal_cond'       : 4.0,        # Thermal conductivity W/(mK)
   'tim_thickness'          : 0.1e-3,     # 0.1mm
   'die_thermal_cond'       : 149.0,      # Thermal conductivity W/(mK)
   'die_thickness'          : 0.70e-3,    # 0.70mm

#   'duct_width'            : 0.055,      # Duct
#   'duct_height'           : 0.037895

   # Constants
   'air_specific_heat'      : 1004.9,     # J/(kgK)
   'air_thermal_cond'       : 0.02624,    # W/(mK)
   'air_density'            : 1.1644,     # kg/(m^3)
   'air_dynamic_viscostly'  : 0.00001886, # kg/(ms)
}
 
def getInputTemplate():
   return input_template.copy()

# Inputs (All inputs must be in the MKS unit system)
# - base_width       : Width of a heatsink
# - base_length      : Depth of a heatsink
# - fin_height       : Height of a heatsink
# - n_of_fins        : Number of fins in a heatsink
# - fin_thickness    : Thickness of each fin
# - fin_air_volume   : The mount of air supplied to a heatsink
# - fin_thermal_cond : Thermal conductivity of a heatsink
# Outputs
# - r_fins           : Thermal Resistance
def evalFinThermalResistance(inputs):
   outputs = {}

   outputs['hs_fin_gap'] = \
      (inputs['base_width']-inputs['n_of_fins']*inputs['fin_thickness']) \
      / (inputs['n_of_fins'] - 1)

   outputs['hs_exposed_base'] = \
      (inputs['n_of_fins']-1)*outputs['hs_fin_gap']*inputs['base_length']

   outputs['hs_area_per_fin'] = \
      2*inputs['fin_height']*inputs['base_length']

   outputs['air_velocity'] = \
      inputs['fin_air_volume'] \
      / ((inputs['n_of_fins']-1)*outputs['hs_fin_gap']*inputs['fin_height'])

   outputs['prandtl_number'] = \
      inputs['air_dynamic_viscostly']*inputs['air_specific_heat'] \
      / inputs['air_thermal_cond']
   
   outputs['reynolds_number'] = \
      inputs['air_density']*outputs['air_velocity']*outputs['hs_fin_gap']**2 \
      / (inputs['air_dynamic_viscostly']*inputs['base_length'])

   outputs['nu1'] = (outputs['reynolds_number']*outputs['prandtl_number']/2)**-3
   outputs['nu2'] = (0.664*(outputs['reynolds_number']**0.5)*(outputs['prandtl_number']**(1.0/3))*(1+3.65*(outputs['reynolds_number']**(-0.5)))**0.5)**-3

   outputs['nusselt_number'] = (outputs['nu1'] + outputs['nu2'])**(-1.0/3)
   
   outputs['hs_heat_trans_coef'] = \
      outputs['nusselt_number']*inputs['air_thermal_cond']/outputs['hs_fin_gap']

   outputs['m'] = (2*outputs['hs_heat_trans_coef'] \
                     / (inputs['fin_thermal_cond']*inputs['fin_thickness']))**0.5

   outputs['eta_fin'] = math.tanh(outputs['m']*inputs['fin_height']) \
                       / (outputs['m']*inputs['fin_height'])
   
   outputs['r_fins'] = \
      (outputs['hs_heat_trans_coef']*(outputs['hs_exposed_base']+inputs['n_of_fins']*outputs['eta_fin']*outputs['hs_area_per_fin']))**-1

   outputs['fins_heat_trans_coeff'] = \
      1 / (outputs['r_fins'] * inputs['base_length'] * inputs['base_width'])

   inputs['r_fins'] = outputs['r_fins']
   inputs['fins_heat_trans_coeff'] = outputs['fins_heat_trans_coeff']

   return outputs

# Inputs (All inputs must be in the MKS unit system)
# - base_width     : Width of a heatsink
# - base_length    : Depth of a heatsink
# - fin_height     : Height of a heatsink
# - n_of_fins      : Number of fins in a heatsink
# - fin_thickness  : Thickness of each fin
# - fin_air_volume : The mount of air supplied to a heatsink
# Outputs
# - fins_pd        : Pressure Drop
def evalFinPressureDrop(inputs):
   outputs = {}
   
   outputs['hs_fin_gap'] = \
      (inputs['base_width']-inputs['n_of_fins']*inputs['fin_thickness']) \
      / (inputs['n_of_fins'] - 1)
      
   outputs['Dh'] = 2 * outputs['hs_fin_gap']

   outputs['lambda'] = outputs['hs_fin_gap'] / inputs['fin_height']

   outputs['air_velocity'] = \
      inputs['fin_air_volume'] \
      / ((inputs['n_of_fins']-1)*outputs['hs_fin_gap']*inputs['fin_height'])

   outputs['sigma'] = \
      1 - (inputs['n_of_fins'] * inputs['fin_thickness']) \
      / inputs['base_width']

   outputs['Kc'] = \
      0.42 * (1 - outputs['sigma']**2)

   outputs['Ke'] = \
      (1 - outputs['sigma']**2)**2

   outputs['Re'] = \
      inputs['air_density'] * outputs['air_velocity'] * outputs['Dh'] \
      / inputs['air_dynamic_viscostly']

   outputs['L*'] = inputs['base_length'] / outputs['Dh'] / outputs['Re']

   outputs['f'] = \
      (24 - 32.527 * outputs['lambda'] + 46.721 * outputs['lambda']**2 \
       - 40.829 * outputs['lambda']**3 + 22.954 * outputs['lambda']**4 \
       - 6.089 * outputs['lambda']**5) / outputs['Re']

   outputs['fapp'] = \
      ((3.44/outputs['L*']**0.5)**2 + (outputs['f']*outputs['Re'])**2)**0.5 \
      / outputs['Re']

   outputs['fins_pd'] = \
      (outputs['Kc']+4*outputs['fapp']*inputs['base_length']/outputs['Dh']+outputs['Ke']) \
      * inputs['air_density'] * outputs['air_velocity']**2 / 2

   inputs['fins_pd'] = outputs['fins_pd']

   return outputs

# Inputs (All inputs must be in the MKS unit system)
# - base_width     : Width of a heatsink
# - fin_height     : Height of a heatsink
# - n_of_fins      : Number of fins in a heatsink
# - fin_thickness  : Thickness of each fin
# - duct_width     : Width of a duct in which the heatsink is inside
# - duct_height    : Height of a duct in which the heatsink is inside
# - fin_air_volume : The mount of air supplied to the heatsink
# - fins_pd        : Pressure Drop induced by the heatsink
# Outputs
# - duct_inair_volume : The amount of air required at the opening of the duct
#                       to supply fin_air_volume to the heatsink
def evalHSIncomingAir(inputs):
   outputs = {}

   outputs['hs_fin_gap'] = \
      (inputs['base_width']-inputs['n_of_fins']*inputs['fin_thickness']) \
      / (inputs['n_of_fins'] - 1)
 
   outputs['air_velocity'] = \
      inputs['fin_air_volume'] \
      / ((inputs['n_of_fins']-1)*outputs['hs_fin_gap']*inputs['fin_height'])
  
   outputs['Ad'] = inputs['duct_width'] * inputs['duct_height']

   outputs['Ahs'] = \
      inputs['fin_thickness'] * inputs['fin_height'] * inputs['n_of_fins']

   outputs['Af'] = \
      inputs['base_width'] * inputs['fin_height'] - outputs['Ahs']

   outputs['Ab'] = \
      outputs['Ad'] - inputs['base_width'] * inputs['fin_height']

   outputs['a'] = outputs['Ad']**2

   outputs['b'] = -2 * outputs['Ad'] * outputs['Af'] * outputs['air_velocity']

   outputs['c'] = \
      outputs['air_velocity']**2 * (outputs['Af']**2 - outputs['Ab']**2) \
      -2 * inputs['fins_pd'] * outputs['Ab']**2 / inputs['air_density']

   outputs['duct_inair_velocity'] = \
      (-outputs['b'] + (outputs['b']**2 - 4 * outputs['a'] * outputs['c'])**0.5) \
      / (2 * outputs['a'])
   outputs['duct_inair_volume'] = outputs['duct_inair_velocity'] * outputs['Ad']

   inputs['duct_inair_volume'] = outputs['duct_inair_volume']

   return outputs

# Inputs (All inputs must be in the MKS unit system)
# - die_width         : Width of a silicon die
# - die_length        : Depth of a silicon die
# - base_width        : Width of a heatsink
# - base_length       : Depth of a heatsink
# - base_thickness    : Thickness of the bottom part of the heatsink
# - base_thermal_cond : Thermal conductivity of the bottom part of the heatsink
# Outputs
# - r_sp              : Spreading resistance of the bottom part of the heatsink
def evalBaseSpreadingResistance(inputs):
   outputs = {}
   
   outputs['r1'] = \
      (inputs['die_length'] * inputs['die_width'] / math.pi) ** 0.5

   outputs['r2'] = \
      (inputs['base_length'] * inputs['base_width'] / math.pi) ** 0.5

   outputs['epsilon'] = outputs['r1'] / outputs['r2']

   outputs['tau'] = inputs['base_thickness'] / outputs['r2']

   outputs['Bi'] = \
      inputs['fins_heat_trans_coeff'] * outputs['r2'] / inputs['base_thermal_cond']

   outputs['lambda'] = \
      math.pi + 1 / (outputs['epsilon'] * math.pi**0.5)

   outputs['phi'] = \
      (math.tanh(outputs['lambda']*outputs['tau'])+outputs['lambda']/outputs['Bi']) \
      / (1 + outputs['lambda'] / outputs['Bi'] * math.tanh(outputs['lambda']*outputs['tau']))

   outputs['psi_max'] = \
      outputs['epsilon'] * outputs['tau'] / math.pi**0.5 \
      + 1 / math.pi**0.5 * (1 - outputs['epsilon']) * outputs['phi']

   outputs['r_sp'] = \
      outputs['psi_max'] / (inputs['base_thermal_cond'] * outputs['r1'] * math.pi**0.5)

   inputs['r_sp'] = outputs['r_sp']

   return outputs

# Inputs (All inputs must be in the MKS unit system)
# - die_width       : Width of a silicon die
# - die_length      : Depth of a silicon die
# Outputs
# - r_tim           : Thermal resistance induced by TIM
def evalTIMThermalResistance(inputs):
   r_tim = 1 / inputs['tim_thermal_cond'] * inputs['tim_thickness'] \
            / (inputs['die_width'] * inputs['die_length'])

   inputs['r_tim'] = r_tim

# Inputs (All inputs must be in the MKS unit system)
# - die_width       : Width of a silicon die
# - die_length      : Depth of a silicon die
# Outputs
# - r_si            : Thermal resistance induced by TIM
def evalDIEThermalResistance(inputs):
   r_si = 1 / inputs['die_thermal_cond'] * inputs['die_thickness'] \
            / (inputs['die_width'] * inputs['die_length'])

   inputs['r_si'] = r_si

def evalRTotal(inputs):
   return inputs['r_si'] + inputs['r_tim'] + inputs['r_sp'] + inputs['r_fins']

