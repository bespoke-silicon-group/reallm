import sys
import pandas as pd

MaxDieTemp = 90.0
Ambient = 30.0
MinFinPitch = 1e-3

BaseLengthPrecision = 1e-3
BaseThickPrecision = 1e-3
HSMaxHeight = 35e-3

lanes_per_server = 6
columns_per_lane = 1
total_length = 450.0 * 1e-3

DicingGap = 0.1e-3
WaferDiameter = 300e-3

SRAMVdd = 1.0
VddPrecision = -3

AirVolumePrecision = 0.1

################################################################################
# Package-Level Integration
################################################################################
SIWaferCost = 1937.0
SIMaxSize = 900.0
SID0 = 0.0005
BondCost = 1.0
BondYield = 0.99

################################################################################
# Server Cost
################################################################################
WaferCost = 12000.0
PCBCost = 50.0            # $/each (parts not included)x2 if there is DRAM
PCBPartsCost = 50.0       # (w/o DCDC, michael's sheet)
DCDCCostPerAmp = 0.33     # $/amp
DCDCMaxCurrent = 30.0     # Amps
DCDCEfficiency = 0.95     # 5% loss
FanPower = 7.4            # W/each
FanCost = 15.0
APPPower = 10.0           # W (Application Processor + DRAM)
APPCost = 10.0            # $ (Application Processor + DRAM)

# PSU
PSUCostPerW = 0.13        # $/W
PSUEfficiency = 0.95      # 5% loss
PSUOutputVoltage = 12.0

# Chassis
ChassisCost = 30.0        # $

# Heat sink
#HeatSinkPerM2 = 3750.0     # $1.5/(0.04x0.01)
AlCost = 4122.68            # $/m3
CuCost = 40707.66           # $/m3
HSBaseCost = 1.0            # $/each

# Ethernet
EthernetCost = 10.0        #$ for 1 GigE

# Package
def evalPackageCost(die_cost):
   # http://electroiq.com/blog/2001/03/technology-comparisons-and-the-economics-of-flip-chip-packaging/
   return 0.103 * die_cost + 1.185
# end of evalPackageCost

def evalHeatSinkCost(hs_spec):
   base_volume = hs_spec['base_width'] * hs_spec['base_length'] * hs_spec['base_thickness']
   fin_volume = hs_spec['fin_thickness'] * hs_spec['fin_height'] * hs_spec['base_length'] * hs_spec['n_of_fins']

   al_weight = 0.0
   al_cost = 0.0
   cu_weight = 0.0
   cu_cost = 0.0

   for spec in [[hs_spec['base_thermal_cond'], base_volume], \
                [hs_spec['fin_thermal_cond'], fin_volume]]:
      if(spec[0] == 200.0):
         al_cost += AlCost * spec[1]
         al_weight += 2700.0 * spec[1] # kg
      elif(spec[0] == 400.0):
         cu_cost += CuCost * spec[1]
         cu_weight += 8940.0 * spec[1] # kg
      else:
         print >> sys.stderr, 'Unknown heat sink material. Disable cost estimation'

   hs_cost = HSBaseCost + al_cost + cu_cost

   return [hs_cost, al_cost, al_weight, cu_cost, cu_weight]
# end of HeatSinkCost

################################################################################
# TCO
################################################################################
SrvLife = 1.5              # years
ElectricityCost = 0.067    # $/kWh
InterestRate = 0.08        # Annual Interest
DCCapex = 10.0             # $/W
DCAmortPeriod = 12.0       # amortizatio period in year
DCOpex = 0.04              # $/kW/month
PUE = 1.5
SrvOpexRate = 0.05         # % of server amortization
SrvAvgPwr = 1.0            # Server Average Power Relative to Peak,,

# ---------------------------------------------------------------------------- #
# Technology
# ---------------------------------------------------------------------------- #

# According to ITRS 2012, a 6T SRAM cell in 90nm is 1 um2, and so the transistor size is 1/6 um2.
# We scale this area to other tech nodes using S^2 factor.

TechNodes  = ['5nm', '7nm', '10nm', '16nm', '22nm', '28nm', '40nm', '65nm', '90nm', '130nm', '180nm', '250nm']
TechParams =               ['FeatureSize', 'Vth', 'CoreVdd', 'TrSize', 'WaferDiameter', 'CPP']
TechUnits  =               [         'nm',   'V',       'V',    'um2',             'm', 'nm']
TechData   = pd.DataFrame([[          5.0,  0.35,      0.75,    0.004,          300e-3,    57.0],  #   5nm
                           [          7.0,  0.35,      0.75,    0.004,          300e-3,    57.0],  #   7nm
                           [         10.0,  0.35,      0.75,    0.005,          300e-3,    64.0],  #  10nm
                           [         16.0,  0.40,      0.80,    0.005,          300e-3,    90.0],  #  16nm
                           [         22.0,  0.35,      0.80,    0.015,          300e-3,   108.0],  #  22nm
                           [         28.0,  0.40,      0.90,    0.016,          300e-3,   117.0],  #  28nm
                           [         40.0,  0.23,      0.90,    0.033,          300e-3,   162.0],  #  40nm
                           [         65.0,  0.40,      1.00,    0.087,          300e-3,   160.0],  #  65nm
                           [         90.0,  0.35,      1.00,    0.167,          300e-3,   240.0],  #  90nm
                           [        130.0,  0.34,      1.20,    0.349,          300e-3,   310.0],  # 130nm
                           [        180.0,  0.42,      1.80,    0.668,          200e-3,   430.0],  # 180nm
                           [        250.0,  0.53,      2.50,    1.289,          200e-3,   640.0]], # 250nm
                           index = TechNodes, columns = TechParams)

FMWC = {'7nm': 9346.0, '65nm': 1937.0}

# ---------------------------------------------------------------------------- #
# Yield 
# ---------------------------------------------------------------------------- #

alphas = {'5nm': 25.0
        , '7nm': 26.0 # ????
        , '10nm': 24.06209 # ????
        , '16nm': 22.06209
        , '22nm' : 18.56382 # ????
        , '28nm' : 14.56382
        , '40nm'  : 13.09522
        , '65nm' : 12.20765
        , '90nm'  : 11.70291
        , '130nm' : 12.68713
        , '180nm' :  7.440977
        , '250nm' : 5.700713
        }
    
D0s = {'5nm'  : 0.001 # ????
     , '7nm'  : 0.001 # ????
     , '10nm' : 0.001 # ????
     , '16nm' : 0.003418466
     , '22nm'  : 0.002806611 # ????
     , '28nm'  : 0.002206611
     , '40nm'   : 0.001611654
     # , '65nm'  : 0.001519241
     , '65nm'  : 0.0005 # For SI
     , '90nm'   : 0.001274998
     , '130nm'  : 0.001355531
     , '180nm'  : 0.001270288
     , '250nm'  : 0.001084176
     }
# This is defects/mm2 = defects/cm2 /100


# ---------------------------------------------------------------------------- #
# ML Chips
# ---------------------------------------------------------------------------- #

# hbm density (mm2) and price ($) per 4-stack die, 8GB
hbm_die_cap = 8.0
real_hbm_size = 92.0
real_hbm_price = 150.0 * (hbm_die_cap / 8.0)

# SRAM density mm2/MB, data from real implementations
real_sram_size = {'16nm': 1.28, '7nm': 0.45, '5nm': 0.35}
# MACs density mm2/Tera BF16 ops, data from real implementations
real_macs_size = {'7nm': 0.29, '5nm': 0.22}
# IOs density mm2/count, each one is 12.5GB/s, 7 and 5nm is fake
real_io_size = {'16nm': 0.3, '7nm': 0.3, '5nm': 0.3}

# Total Power Model, W/Tera BF16 ops
real_total_power = {'7nm': 1.3, '5nm': 1.1}
real_dram_total_power = {'7nm': 1.6}

# Seprate Power Model
# MACs dynamic power W/Tera BF16 ops
real_macs_power = {'7nm': 0.16, '5nm': 0.12}
# REG read modify write dynamic power W/TOPS,
real_reg_power = {'7nm': 0.4, '5nm': 0.3}
# SRAM dynamic power W/TOPS,
real_sram_power = {'7nm': 1.75, '5nm': 1.2}
# DRAM dynamic power W/TOPS,
real_hbm_power = 40.0

# IOs power W/count, each one is 12.5GB/s, 7 and 5nm is fake
real_io_power = {'16nm': 0.175, '7nm': 0.125, '5nm': 0.11}

others_power = {'7nm': 1.0, '5nm': 0.7}

# Board Max Power
board_max_power = 2000.0

# ---------------------------------------------------------------------------- #
# Applications
# ---------------------------------------------------------------------------- #
# Name   Mem/Stage(MB) Mem/Layer(MB) Layers Stages/Board Mem/Board(MB) Boards
# BERT   8             25            24     72           600           1
# GPT2   20            60            48     144          3024          1
# T-NLG  150           450           78     24           3600          10
# GPT3   1280          3840          96     3            3840          96
# MT-NLG 3400          10200         105    1            3400          315
# Memory needed for each stage, in MB
mem_per_stage = {'GPT3': 3840/3}
# Stage per board
stage_per_board = {'GPT3': 3}
# Memory needed for each board
mem_per_board = {'BERT': [25]*24,
                 #  'GPT2': [378, 378, 378, 378, 378, 378, 378, 378],
                 'GPT2': [20]*144,
                 'T-NLG': [150]*24,
                 'GPT3': [320, 320, 320, 320, 1280, 1280],
                 'MT-NLG-Atten': [850, 850, 850, 850],
                 'MT-NLG-FC': [3400]
                 }
# Boards needed
num_boards = {'BERT': 1, 'GPT2': 1, 'T-NLG': 10, 'GPT3': 96, 'MT-NLG-Atten': 105, 'MT-NLG-FC':210}

