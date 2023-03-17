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

