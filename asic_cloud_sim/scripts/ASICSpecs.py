# Those values are taken from https://docs.google.com/a/eng.ucsd.edu/spreadsheets/d/1ew2oabbH2WwEldH5UU7RG6xsY2nROVlVTHukSBE6738/edit?usp=sharing
Bitcoin = {                    # design spec @ nominal_vdd
   'unit_area': 0.660,         # The area of the minimum unit (mm2)
   'lgc_vdd': 0.9,             # nominal VDD which following values are
                               # defined by (V)
   'sram_vdd': 0.9,            # nominal VDD which following values are
                               # defined by (V)
   'f_scale': 1.0,             # Slow down factor applied to the worst
                               # critical path. This is used for
                               # Frequency Scaling
   'cp_lgc_path': 1.21,        # The critical path made of logic only (ns)
   'cp_mixed_lgc_path': 0.72,  # The logic part of the critical path made of
                               # SRAM and logic (ns)
   'cp_mixed_sram_path': 0.52, # The SRAM part of the critical path made of
                               # SRAM and logic (ns)
   'lgc_dyn_pwr': 0.970,        # Dynamic Power of Logic (W)
   'lgc_leak_pwr': 6.72e-4,    # Leakage of Logic (W)
   'sram_dyn_pwr': 3.64e-2,    # Dynamic Power of SRAM (W)
   'sram_leak_pwr': 3.46e-5,   # Leakage of SRAM (W)
   'unit_perf': 806.1 ,        # MHash/s
   'nre': 4450000,             # NRE of this node
   'dram_type': 'None',        # DRAM type, would be changed if required
   'dram_count': 0,            # Number of DRAMs used, one per MC
   'dram_bw': 0,               # Required DRAM Bandwidth GB/s for RCA
   'ethernet_count': 0,        # number of (20GigE) Ethernet cards needed per server, a.k.a. off-pcb interface
}

Litecoin = {                   # design spec @ nominal_vdd
   'unit_area': 0.558,         # The area of the minimum unit (mm2)
   'lgc_vdd': 0.9,             # nominal VDD which following values are
                               # defined by
   'sram_vdd': 0.9,            # nominal VDD which following values are
                               # defined by
   'f_scale': 1.0,             # Slow down factor applied to the worst
                               # critical path. This is used for
                               # Frequency Scaling
   'cp_lgc_path': 1.19,        # The critical path made of logic only (ns)
   'cp_mixed_lgc_path': 0.34,   # The logic part of the critical path made of
                               # SRAM and logic (ns)
   'cp_mixed_sram_path': 0.85, # The SRAM part of the critical path made of
                               # SRAM and logic (ns)
   'lgc_dyn_pwr': 5.51e-2,     # Dynamic Power of Logic
   'lgc_leak_pwr': 2.45e-4,    # Leakage of Logic
   'sram_dyn_pwr': 8.59e-3,    # Dynamic Power of SRAM
   'sram_leak_pwr': 5.07e-4,   # Leakage of SRAM
   'unit_perf': 1.852e-2,      # MHash/s
   'nre': 4450000,             # NRE of this node
   'dram_bw': 0,               # Required DRAM Bandwidth GB/s
   'dram_type': 'None',        # DRAM type, would be changed if required
   'dram_count': 0,            # Number of DRAMs used, one per MC
   'ethernet_count': 0,        # number of (20GigE) Ethernet cards needed per server, a.k.a. off-pcb interface
}

H265 =     {                   # design spec @ nominal_vdd
   'unit_area': 2.90,          # The area of the minimum unit (mm2)
   'lgc_vdd': 0.9,             # nominal VDD which following values are
                               # defined by
   'sram_vdd': 0.9,            # nominal VDD which following values are
                               # defined by
   'f_scale': 1.0,             # Slow down factor applied to the worst
                               # critical path. This is used for
                               # Frequency Scaling
   'cp_lgc_path': 2.02,        # The critical path made of logic only (ns)
   'cp_mixed_lgc_path': 1.01,  # The logic part of the critical path made of
                               # SRAM and logic (ns)
   'cp_mixed_sram_path': 1.01, # The SRAM part of the critical path made of
                               # SRAM and logic (ns)
   'lgc_dyn_pwr': 9.78e-2,     # Dynamic Power of Logic
   'lgc_leak_pwr': 0,          # Leakage of Logic
   'sram_dyn_pwr': 2.89e-2,    # Dynamic Power of SRAM
   'sram_leak_pwr': 0,         # Leakage of SRAM
   'unit_perf': 3.000e-2,      # KHash/s
   'nre': 4450000,             # NRE of this node
   'dram_bw': 0.5813,          # Required DRAM Bandwidth GB/s
   'dram_type': 'lpddr3',      # DRAM type, would be changed if required
   'dram_count': 1,            # Number of DRAMs used, one per MC
   'ethernet_count': 1,        # number of (20GigE) Ethernet cards needed per server, a.k.a. off-pcb interface
}

DDN =     {                    # design spec @ nominal_vdd
   'unit_area': 50.11,         # The area of the minimum unit (mm2)
   'lgc_vdd': 0.9,             # nominal VDD which following values are
                               # defined by
   'sram_vdd': 0.9,            # nominal VDD which following values are
                               # defined by
   'f_scale': 1.0,             # Slow down factor applied to the worst
                               # critical path. This is used for
                               # Frequency Scaling

   'cp_lgc_path': 1.65,           
   'cp_mixed_lgc_path': 0.1,   # Not used 
   'cp_mixed_sram_path': 0.1,  # Not used 
   
   'lgc_dyn_pwr': 7.96,        # Total power of RCA 
   'lgc_leak_pwr': 0,          # Not used 
   'sram_dyn_pwr': 0,          # Not used 
   'sram_leak_pwr': 0,         # Not used 
   'unit_perf': 1,             # Not used 
   'nre': 4450000,             # NRE of this node
   'dram_bw': 0,               # Required DRAM Bandwidth GB/s
   'dram_type': 'None',        # DRAM type, would be changed if required
   'dram_count': 0,            # Number of DRAMs used, one per MC
   'ethernet_count': 1,        # number of (20GigE) Ethernet cards needed per server, a.k.a. off-pcb interface
}

#NeuralNet = {                  # design spec @ nominal_vdd
#   'unit_area': 0.046724,      # The area of the minimum unit (mm2)
#   'lgc_vdd': 1.05,            # nominal VDD which following values are
#                               # defined by
#   'sram_vdd': 1.05,           # nominal VDD which following values are
#                               # defined by
#   'f_scale': 1.0,             # Slow down factor applied to the worst
#                               # critical path. This is used for
#                               # Frequency Scaling
#   'cp_lgc_path': 1.00,        # The critical path made of logic only (ns)
#   'cp_mixed_lgc_path': 0.0,   # The logic part of the critical path made of
#                               # SRAM and logic (ns)
#   'cp_mixed_sram_path': 1.00, # The SRAM part of the critical path made of
#                               # SRAM and logic (ns)
#   'lgc_dyn_pwr': 5.0165e-3,   # Dynamic Power of Logic
#   'lgc_leak_pwr': 0,          # Leakage of Logic
#   'sram_dyn_pwr': 6.9241e-3,  # Dynamic Power of SRAM
#   'sram_leak_pwr': 0,         # Leakage of SRAM
#   'unit_perf': 2.09e-8,       # MHash/s
#   'nre': 4450000,             # NRE of this node
#   'dram_bw': 0,               # Required DRAM Bandwidth GB/s
#   'dram_type': 'None',        # DRAM type, would be changed if required
#   'dram_count': 0,            # Number of DRAMs used, one per MC
#   'dram_mc_area': 0,          # Dram Memory Controller Area (mm2)
#   'dram_mc_power': 0,         # DRAM Memory Controller Power (W)
#   'ethernet_count': 1,        # number of Ethernet cards needed per server
#}
#

Chiplet =     {                # design spec @ nominal_vdd
   'unit_area':0.6,            # The area of MACs in the minimum unit (mm2)
   'sram_area':0.7,            # The area of SRAM the minimum unit (mm2)
   'lgc_vdd': 0.8,             # nominal VDD which following values are
                               # defined by
   'sram_vdd': 0.8,            # nominal VDD which following values are
                               # defined by
   'f_scale': 1.0,             # Slow down factor applied to the worst
                               # critical path. This is used for
                               # Frequency Scaling

   'cp_lgc_path': 1.0,           
   'cp_mixed_lgc_path': 0.1,   # Not used 
   'cp_mixed_sram_path': 0.1,  # Not used 
   
   'lgc_dyn_pwr': 1.00,        # Total power of RCA 
   'lgc_leak_pwr': 0,          # Not used 
   'sram_dyn_pwr': 0,          # Not used 
   'sram_leak_pwr': 0,         # Not used 
   'unit_perf': 1,             # Not used 
   'nre': 100000,              # NRE of this node
   'dram_bw': 0,               # Required DRAM Bandwidth GB/s
   'dram_type': 'None',        # DRAM type, would be changed if required
   'dram_count': 0,            # Number of DRAMs used, one per MC
   'ethernet_count': 1,        # number of (1GigE) Ethernet cards needed per server, a.k.a. off-pcb interface
}

