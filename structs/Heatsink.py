from dataclasses import dataclass
from typing import Optional
from .Base import Base

import math

# Based on ASIC Clouds Headsink evaluation
@dataclass
class Heatsink(Base):
    # Heat Source
    heatsource_length: float # mm
    heatsource_width: float  # mm
    packages_per_lane: int
    # Heat Sink
    fin_height: float = 30e-3
    fin_thermal_cond: float = 200.0 # Thermal conductivity, W/(mK)
    fin_thickness: float = 0.5e-3
    fin_air_volume: float = 0.009439   # m^3/s
    n_of_fins: float = 20.0
    base_thickness: float = 5e-3
    base_thermal_cond: float = 200.0 # Thermal conductivity, W/(mK)
    # TIM and Silicon
    tim_thermal_cond: float = 4.0 # Thermal conductivity W/(mK)
    tim_thickness: float = 0.1e-3 # 0.1mm
    die_thermal_cond: float = 149.0 # Thermal conductivity W/(mK)
    die_thickness: float = 0.70e-3 # 0.70mm

    # Auto-generate
    # Validness
    valid: Optional[bool] = None
    # Cost
    cost: Optional[float] = None
    # max power
    max_power: Optional[float] = None

    # Constants
    base_width = 0.05 # meter, DOUBLE CHECK THIS
    base_length = 0.05 # meter
    # base_width = 0.04 # meter, need to update this for hetergeneous integration
    # base_length = 0.04 # meter
    max_length = 450.0 * 1e-3 # meter
    air_specific_heat = 1004.9 # J/(kgK)
    air_thermal_cond = 0.02624 # W/(mK)
    air_density = 1.1644 # kg/(m^3)
    air_dynamic_viscostly = 0.00001886 # kg/(ms)
    BaseLengthPrecision = 1e-3
    BaseThickPrecision = 1e-3
    HSMaxHeight = 35e-3
    AirVolumePrecision = 0.1
    MaxDieTemp = 90.0
    Ambient = 30.0
    MinFinPitch = 1e-3
    AlCost = 4122.68            # $/m3
    CuCost = 40707.66           # $/m3
    HSBaseCost = 1.0            # $/each

    def update(self) -> None:
        hs_spec = getInputTemplate()
        # converting from mm to m as well
        hs_spec['die_length'] = self.heatsource_length * 1e-3
        hs_spec['die_width'] = self.heatsource_width * 1e-3
        hs_spec = self._find_heatsink(hs_spec)
        if hs_spec is None:
            self.valid = False
            self.cost = math.inf
            self.max_power = 0.0
        else:
            self.valid = True
            self.cost = self._get_cost(hs_spec)
            # self.max_power = hs_spec['q_asic']
            # TODO: FIX THIS LATER, consider more advanced air cooling
            self.max_power = hs_spec['q_asic'] * 2.0
            self.fin_height = hs_spec['fin_height']
            self.fin_thermal_cond = hs_spec['fin_thermal_cond'] 
            self.fin_thickness = hs_spec['fin_thickness']
            self.fin_air_volume = hs_spec['fin_air_volume'] 
            self.n_of_fins = hs_spec['n_of_fins']
            self.base_thickness = hs_spec['base_thickness']
            self.base_thermal_cond = hs_spec['base_thermal_cond'] 
            self.tim_thermal_cond = hs_spec['tim_thermal_cond'] 
            self.tim_thickness = hs_spec['tim_thickness'] 
            self.die_thermal_cond = hs_spec['die_thermal_cond']
            self.die_thickness = hs_spec['die_thickness'] 

    def _find_heatsink(self, hs_tmpl) -> Optional[dict]:
        if hs_tmpl['die_width'] > hs_tmpl['base_width']:
            print(f'Die width {hs_tmpl["die_width"]} is larger than max Heat Sink width {hs_tmpl["base_width"]}')
            return None
   
        # Find the best result by considering single heatsink covering the lane 
        hs_spec = hs_tmpl.copy()
        hs_spec['base_length'] = self.max_length
        best_result = self._find_best_thick_and_N(hs_spec, self.packages_per_lane, False)

        # Find the 95% of the best or shorter one
        base_length = math.ceil(hs_tmpl['die_length'] / self.BaseLengthPrecision) * self.BaseLengthPrecision
        hs_to_use = None

        while(base_length <= self.max_length / self.packages_per_lane):
           hs_spec = hs_tmpl.copy()
           hs_spec['base_length'] = base_length

           best_spec_at_length = self._find_best_thick_and_N(hs_spec, self.packages_per_lane)

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

           base_length += self.BaseLengthPrecision
        return hs_to_use

    def _find_best_thick_and_N(self, hs_tmpl, asics_per_col, full_lane_hs=True) -> Optional[dict]:
        base_thickness = self.BaseThickPrecision
        best_spec_at_length = None

        n_of_fins = 4
        fin_step = 2
        max_fins = self._max_no_fins(hs_tmpl);
        air_vol_step = CFM2CMS(self.AirVolumePrecision)
        initial_air_vol = air_vol_step 

        while(base_thickness < self.HSMaxHeight):
            n_of_fins = 2
            best_spec_at_thick = None

            while(n_of_fins <= max_fins):
                hs_spec = hs_tmpl.copy()
                hs_spec['n_of_fins'] = n_of_fins
                hs_spec['base_thickness'] = base_thickness
                hs_spec['fin_height'] = self.HSMaxHeight - base_thickness

                # Estimate suppliable air volume assuming one big heat sink
                # whose length is base_length * asics_per_col
                if (full_lane_hs):
                    hs_spec['base_length'] *= asics_per_col
                fin_air_volume = evalAirVolume(hs_spec, initial_air_vol, 
                                               fan_9GA0312P3K001, air_vol_step)
                initial_air_vol = fin_air_volume - 5*air_vol_step
                if (full_lane_hs):
                    hs_spec['base_length'] /= asics_per_col

                if(fin_air_volume <= 0.0):
                    # Pressure drop too high
                    break

                hs_spec['fin_air_volume'] = fin_air_volume
                # Estimate each thermal resistance
                evalDIEThermalResistance(hs_spec)
                evalTIMThermalResistance(hs_spec)
                evalFinThermalResistance(hs_spec)
                evalBaseSpreadingResistance(hs_spec)

                hs_spec['r_total'] = evalRTotal(hs_spec)
                hs_spec['q_asic'] = self._power_budget(hs_spec, asics_per_col)

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

            base_thickness += self.BaseThickPrecision
            fin_step = 1 
        return best_spec_at_length
    
    def _power_budget(self, hs_spec, n_of_asics):
        return (self.MaxDieTemp - self.Ambient) / \
            (evalRTotal(hs_spec) + (n_of_asics - 1) / \
             (hs_spec['air_specific_heat'] * hs_spec['air_density'] * hs_spec['fin_air_volume']))

    def _max_no_fins(self, hs_spec):
        return int((hs_spec['base_width'] + self.MinFinPitch) / (hs_spec['fin_thickness'] + self.MinFinPitch))
    
    def _get_cost(self, hs_spec) -> float:
        base_volume = hs_spec['base_width'] * hs_spec['base_length'] * hs_spec['base_thickness']
        fin_volume = hs_spec['fin_thickness'] * hs_spec['fin_height'] * hs_spec['base_length'] * hs_spec['n_of_fins']

        al_weight = 0.0
        al_cost = 0.0
        cu_weight = 0.0
        cu_cost = 0.0

        for spec in [[hs_spec['base_thermal_cond'], base_volume], \
                     [hs_spec['fin_thermal_cond'], fin_volume]]:
           if(spec[0] == 200.0):
              al_cost += self.AlCost * spec[1]
              al_weight += 2700.0 * spec[1] # kg
           elif(spec[0] == 400.0):
              cu_cost += self.CuCost * spec[1]
              cu_weight += 8940.0 * spec[1] # kg
           else:
              print('Unknown heat sink material. Disable cost estimation')

        hs_cost = self.HSBaseCost + al_cost + cu_cost
        return hs_cost


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
   'base_width'             : 0.05,       # the original is 0.04, updated for A100 GPU
   'base_length'            : 0.05,
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

def fan_9GA0312P3K001(air):
   spec = [0, 7.4, '9GA0312P3K001']
   
   if((air >= 0) and (air <= 0.01)):
      spec[0] = \
           1.49e14   * air**5 \
         - 4.099e12  * air**4 \
         + 3.833e10  * air**3 \
         - 1.375e8   * air**2 \
         + 70736.364 * air    \
         + 799.481

   return spec

def fan_9GA0312P3G001(air):
   spec = [0, 4.0, '9GA0312P3G001']

   if((air >= 0) and (air <= 0.0075)):
      spec[0] = \
           4.255e14 * air**5 \
         - 8.901e12 * air**4 \
         + 6.39e10  * air**3 \
         - 1.754e8  * air**2 \
         + 84550    * air    \
         + 465

   return spec

def fan_9GV0312K301(air):
   spec = [0, 9.48, '9GV0312K301']

   if((air >= 0) and (air <= 0.0108)):
      spec[0] =  \
           9.733e11  * air**5 \
         - 4.693e10  * air**4 \
         - 4.499e8   * air**3 \
         + 1.296e7   * air**2 \
         - 80383.612 * air    \
         + 425.123

   return spec

def fan_9GV0312J301(air):
   spec = [0, 7.2, '9GV0312J301']

   if((air >= 0) and (air <= 0.0093)):
      spec[0] = \
           4.978e13 * air**5 \
         - 1.116e12 * air**4 \
         + 7.221e9  * air**3 \
         - 8.785e6  * air**2 \
         - 52006.7  * air    \
         + 319.946

   return spec

def fan_9GV0312E301(air):
   spec = [0, 2.52, '9GV0312E301']

   if((air >= 0) and (air <= 0.0061)):
      spec[0] = \
          -4.546e12  * air**5 \
          + 4.19e11  * air**4 \
          - 6.754e9  * air**3 \
          + 3.494e7  * air**2 \
          - 71375.19 * air    \
          + 130

   return spec

def fan_9GV0312H301(air):
   spec = [0, 1.92, '9GV0312H301']

   if((air >= 0) and (air <= 0.00416)):
      spec[0] = \
           3.11e14  * air**5 \
         - 3.024e12 * air**4 \
         + 6.84e9   * air**3 \
         + 5.7e6    * air**2 \
         - 31900    * air    \
         + 60

   return spec

def get38mmFans():
   return [fan_9GV0312H301,   # 1.92W \
           fan_9GV0312E301,   # 2.52W \
           fan_9GA0312P3G001, # 4.0W \
           fan_9GV0312J301,   # 7.2W \
           fan_9GA0312P3K001, # 7.4W \
           fan_9GV0312K301    # 9.48W \
           ]


def fan_9GV0612P1G03(air):
   if((air < 0) or (air > 0.04)):
      return 0
   else:
      return 1.455e11 * air**5 \
           - 1.574e10 * air**4 \
           + 5.749e8  * air**3 \
           - 7.815e6  * air**2 \
           + 9026.989 * air    \
           + 749.481

def fan_9GA0712P1G001(air):
   if((air < 0) or (air > 0.042)):
      return 0
   else:
      return 3.488e10  * air**5 \
           - 4.12e9    * air**4 \
           + 1.544e8   * air**3 \
           - 2.008e6   * air**2 \
           - 11159.754 * air    \
           + 858.007

def fan_9GV0812P1F03(air):
   if((air < 0) or (air > 0.05)):
      return 0
   else:
      return 7.776e9 * air**5 \
           - 9.288e8 * air**4 \
           + 3.294e7 * air**3 \
           - 3.195e5 * air**2 \
           - 4875    * air    \
           + 299.702

def fan_9G0912G101(air):
   if((air < 0) or (air > 0.05)):
      return 0
   else:
      return 4.666e9  * air**5 \
           - 7.599e8  * air**4 \
           + 4.233e7  * air**3 \
           - 9.41e5   * air**2 \
           + 4251.364 * air    \
           + 149.843

def evalAirVolume(hs_tmpl, initial_vol, fan, vol_step):
   air_vol = initial_vol
   hs_spec = hs_tmpl.copy()

   while(1):
      hs_spec['fin_air_volume'] = air_vol

      evalFinPressureDrop(hs_spec)
      
      # Fan's maximum allowable static pressure to suppy this air volume
      max_static_pd = fan(air_vol)

      if(hs_spec['fins_pd'] > max_static_pd[0]):
         # This air volume is infeasible.
         # Previous result was the best result
         return air_vol - vol_step

      air_vol += vol_step

# Convert from CFM(cubic feet per minute) to CMS(cubic meters per second)
def CFM2CMS(cfm):
   return cfm * 0.3048**3 / 60

# Convert from CMS(cubic meters per second) to CFM(cubic feet per minute)
def CMS2CFM(cms):
   return cms / (0.3048**3) * 60
