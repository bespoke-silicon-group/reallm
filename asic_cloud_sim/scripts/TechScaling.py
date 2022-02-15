import sys
import utils
import FabCost
import CONSTANTS
import ASICSpecs
import VoltageScaling
import HTMLTable
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:0,.3f}'.format # 1,234.567

def sram_area_scaling(sram_area, Tech, TechNorm="22nm"):
  # data from https://fuse.wikichip.org/news/3398/tsmc-details-5-nm/
  if TechNorm == "22nm":
    if Tech == "22nm":
      return sram_area
    elif Tech == "16nm":
      return sram_area / 0.092 * 0.074
    elif Tech == "14nm":
      return sram_area / 0.092 * 0.064
    elif Tech == "10nm":
      return sram_area / 0.092 * 0.042
    elif Tech == "7nm":
      return sram_area / 0.092 * 0.027
    elif Tech == "5nm":
      return sram_area / 0.092 * 0.021
    else:
      print 'ERROR, unknown tech for sram scaling'
      return -1
  else:
    print 'ERROR, unknown technorm for sram scaling'
    return -1
    

def Classical(AsicSpec, Tech="28nm", TechNorm="28nm"): # Classical Dennard Scaling
  asic_spec = AsicSpec.copy()
  if (Tech != TechNorm):
    if (int(TechNorm[:-2]) > 22):
      TechSize = CONSTANTS.TechData.at[Tech, "FeatureSize"]
      TechNormSize = CONSTANTS.TechData.at[TechNorm, "FeatureSize"]
    else: # Scaling with contacted gate pitch size
      TechSize = CONSTANTS.TechData.at[Tech, "CPP"]
      TechNormSize = CONSTANTS.TechData.at[TechNorm, "CPP"]
    S = TechSize / TechNormSize

    TechVdd = CONSTANTS.TechData.at[Tech, "CoreVdd"]
    TechNormVdd = CONSTANTS.TechData.at[TechNorm, "CoreVdd"]

    asic_spec['unit_area']          = AsicSpec['unit_area'] * S * S
    asic_spec['lgc_vdd']            = TechVdd
    asic_spec['sram_vdd']           = TechVdd
    asic_spec['cp_lgc_path']        = AsicSpec['cp_lgc_path'] * S
    asic_spec['cp_mixed_lgc_path']  = AsicSpec['cp_mixed_lgc_path'] * S
    asic_spec['cp_mixed_sram_path'] = AsicSpec['cp_mixed_sram_path'] * S
    asic_spec['dram_bw']            = AsicSpec['dram_bw'] *1.0 / S 
    asic_spec['lgc_dyn_pwr']        = AsicSpec['lgc_dyn_pwr'] * ( (TechVdd / AsicSpec['lgc_vdd']) ** 2 ) # C*V^2*f, C and f scaling cancels each other.
    asic_spec['sram_dyn_pwr']       = AsicSpec['sram_dyn_pwr'] * ( (TechVdd / AsicSpec['lgc_vdd']) ** 2 ) # C*V^2*f, C and f scaling cancels each other.
    asic_spec['lgc_leak_pwr']       = AsicSpec['lgc_leak_pwr']
    asic_spec['sram_leak_pwr']      = AsicSpec['sram_leak_pwr']
    asic_spec['unit_perf']          = AsicSpec['unit_perf'] / S # / (S * S) # MHz -> 1/S, MHash/s -> 1/S, mm2 -> S^2

    asic_spec['sram_area']          =  \
             sram_area_scaling(AsicSpec['sram_area'], Tech, TechNorm)

  asic_spec['nre']              = utils.float_format( FabCost.CostBook[Tech]['nre'] )

  asic_spec['vth']              = CONSTANTS.TechData.at[Tech, "Vth"]
  asic_spec['nominal_vdd']      = CONSTANTS.TechData.at[Tech, "CoreVdd"]

  asic_spec['cp_mixed_path']    = asic_spec['cp_mixed_lgc_path'] + asic_spec['cp_mixed_sram_path']

  asic_spec['frequency']        = (1.0 / VoltageScaling.getClkLatency(asic_spec)) * 1e3 # GHz -> MHz
  
  asic_spec['watts_per_asic']   = asic_spec['f_scale'] * (asic_spec['lgc_dyn_pwr'] + asic_spec['sram_dyn_pwr']) \
                                + asic_spec['lgc_leak_pwr'] + asic_spec['sram_leak_pwr'] # Refer to lgc_sram_power_density of VoltageScaling
  asic_spec['power_density']    = asic_spec['watts_per_asic'] / (asic_spec['unit_area'] + asic_spec['sram_area'])
  asic_spec['mhash_per_asic']   = asic_spec['unit_perf'] * asic_spec['frequency'] # * asic_spec['unit_area']
  asic_spec['joules_per_mhash'] = asic_spec['watts_per_asic'] / asic_spec['mhash_per_asic'] # (Watts * 1sec) / (MHash/s * 1s)
  
  asic_spec['nom_lgc_delay']    = VoltageScaling.getDelay(asic_spec['lgc_vdd'],Tech)
  asic_spec['nom_sram_delay']   = VoltageScaling.getDelay(asic_spec['sram_vdd'],Tech)

  return asic_spec

if __name__ == '__main__':
  BitcoinSpecs = ASICSpecs.Bitcoin
  TechNodes =  ["28nm", "65nm", "130nm", "250nm"]
  Keys = ['unit_area', 'sram_area', 'vth', 'nominal_vdd', 'lgc_vdd', 'sram_vdd', 'cp_lgc_path', 'cp_mixed_path', 'cp_mixed_lgc_path', 'cp_mixed_sram_path', 'frequency', 'f_scale', 
          'lgc_dyn_pwr', 'lgc_leak_pwr', 'sram_dyn_pwr', 'sram_leak_pwr', 'unit_perf', 'mhash_per_asic', 'power_density', 'watts_per_asic', 'joules_per_mhash',
          'nre', 'dram_bw', 'dram_type', 'dram_count', 'dram_mc_area', 'dram_mc_power', 'ethernet_count']
  Series = []
  for Tech in TechNodes:
    Spec = Classical(BitcoinSpecs, Tech, "28nm")
    Serie = pd.Series(Spec, Keys)
    Series.append(Serie)

  DataFrame = pd.DataFrame(np.matrix(Series).T, columns = TechNodes, index = Keys)

  # Rename Index
  DataFrame.index = ['Unit Area (mm2)', 'Vth (V)', 'Nominal Vdd (V)', 'Logic Vdd (V)', 'SRAM Vdd (V)',
                     'Logic-only Critical Path (ns)', 'Mixed Critical Path (ns)', 'Mixed Critical Path - Logic (ns)', 'Mixed Critical Path - SRAM (ns)',
                     'Chip Frequency (MHz)', 'Frequency Scale', 'Logic Dynamic Power (W)', 'Logic Leakage Power (W)', 'SRAM Dynamic Power (W)', 'SRAM Leakage Power (W)',
                     'MHash Density (MHash/s/mm2/MHz)', 'MHash per Chip', 'Power Density (W/mm2)', 'Chip Power (W)', 'Operation Energy (J/MHash)', 'Full Mask NRE ($)',
                     'Required DRAM Bandwidth (GB/s)', 'DRAM Type', 'Number of DRAMs', 'DRAM Controller Area (mm2)', 'DRAM Controller Power (W)', 'Number of 20GigE Cards']

  print DataFrame

  HTMLTable.df2html(DataFrame, "BitcoinSpecs", "Regular", "<b>Bitcoin Specs Scaled across Process Nodes</b>")
