import CONSTANTS
  
  
# Used values for ISCA paper
#HT_area     = 17620440/4.0
#HT_power    = 8.01/4.0

def GetSpec(Tech="28nm"):
  TechSize = CONSTANTS.TechData.at[Tech, "FeatureSize"]

  S = TechSize / 28.0


  io_spec = {'IO_area' : 3.4 + 1.0 * (S**2),
             'IO_power' : 0.55 * S + 0.85
             }

  return io_spec



