#######################################################
## model spec
#######################################################
gpt2_spec =  {'num_layers': 54,  'd': 1600} # layer number from 48 to 54
tnlg_spec =  {'num_layers': 80,  'd': 4256} # layer number from 78 to 80
gpt3_spec =  {'num_layers': 96,  'd': 12288}
mtnlg_spec = {'num_layers': 105, 'd': 20480}

model_spec = {'gpt2':  gpt2_spec,
              'tnlg':  tnlg_spec,
              'gpt3':  gpt3_spec,
              'mtnlg': mtnlg_spec}

#######################################################
## hardware spec
#######################################################
ts = 0.01

# bandwidth, all in GB/s
c2c_bw = 100 
p2p_bw = 50 
s2s_bw = 10

gpu_p2p_bw = 600
gpu_s2s_bw = 25

# chiplet cloud 
gpt2_cc = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 9,
  'num_srvs': 1,
  'chip_tops': 43.8,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

tnlg_cc = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 28,
  'num_srvs': 4,
  'chip_tops': 43.8,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

gpt3_cc = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 36,
  'num_srvs': 32,
  'chip_tops': 43.8,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

mtnlg_cc = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 30,
  'num_srvs': 105,
  'chip_tops': 43.8,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

chiplet_cloud_spec = {'gpt2':   gpt2_cc,
                      'tnlg':   tnlg_cc,
                      'gpt3':   gpt3_cc,
                      'mtnlg': mtnlg_cc}

# gpu 
gpu_tops = 312
gpu_hbm_bw = 2000
gpt2_gpu = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 1,
  'num_srvs': 1,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

tnlg_gpu = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 1,
  'num_srvs': 1,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

gpt3_gpu = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 8,
  'num_srvs': 1,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

mtnlg_gpu = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 8,
  'num_srvs': 2,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

gpu_spec = {'gpt2':   gpt2_gpu,
            'tnlg':   tnlg_gpu,
            'gpt3':   gpt3_gpu,
            'mtnlg': mtnlg_gpu}

# hbm chiplet, 320 GB/srv, 320/24=13.3 GB/chip
hbm_tops = 1200.0/24
hbm_hbm_bw = 1555*8/24
gpt2_hbm = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 1,
  'num_srvs': 1,
  'chip_tops': hbm_tops,
  'c2c_bw': p2p_bw,
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': hbm_hbm_bw
}

tnlg_hbm = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 3,
  'num_srvs': 1,
  'chip_tops': hbm_tops,
  'c2c_bw': p2p_bw,
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': hbm_hbm_bw
}

gpt3_hbm = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 28,
  'num_srvs': 1,
  'chip_tops': hbm_tops,
  'c2c_bw': p2p_bw,
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': hbm_hbm_bw
}

mtnlg_hbm = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 28,
  'num_srvs': 3,
  'chip_tops': hbm_tops,
  'c2c_bw': p2p_bw,
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': hbm_hbm_bw
}

hbm_spec = {'gpt2':   gpt2_hbm,
            'tnlg':   tnlg_hbm,
            'gpt3':   gpt3_hbm,
            'mtnlg': mtnlg_hbm}

# silicon interposer chiplet
si_tops = 1100/32
gpt2_si = {
  'chips_per_pkg': 2,
  'pkgs_per_srv': 4,
  'num_srvs': 1,
  'chip_tops': si_tops,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

tnlg_si = {
  'chips_per_pkg': 2,
  'pkgs_per_srv': 14,
  'num_srvs': 4,
  'chip_tops': si_tops,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

gpt3_si = {
  'chips_per_pkg': 2,
  'pkgs_per_srv': 18,
  'num_srvs': 32,
  'chip_tops': si_tops,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

mtnlg_si = {
  'chips_per_pkg': 2,
  'pkgs_per_srv': 15,
  'num_srvs': 105,
  'chip_tops': si_tops,
  'c2c_bw': c2c_bw, 
  'p2p_bw': p2p_bw,
  's2s_bw': s2s_bw,
  'T_start': ts,
  'hbm_bw': None
}

si_spec = {'gpt2':   gpt2_si,
           'tnlg':   tnlg_si,
           'gpt3':   gpt3_si,
           'mtnlg': mtnlg_si}

# gpu_same_tops
gpt2_gpu_same_tops = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 1,
  'num_srvs': 1,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

tnlg_gpu_same_tops = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 8,
  'num_srvs': 2,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

gpt3_gpu_same_tops = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 8,
  'num_srvs': 20,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

mtnlg_gpu_same_tops = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 8,
  'num_srvs': 55,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

gpu_same_tops_spec = {'gpt2':   gpt2_gpu_same_tops,
                      'tnlg':   tnlg_gpu_same_tops,
                      'gpt3':   gpt3_gpu_same_tops,
                      'mtnlg': mtnlg_gpu_same_tops}

# gpu_same_tco
gpt2_gpu_same_tco = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 1,
  'num_srvs': 1,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

tnlg_gpu_same_tco = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 2,
  'num_srvs': 1,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

gpt3_gpu_same_tco = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 8,
  'num_srvs': 3,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

mtnlg_gpu_same_tco = {
  'chips_per_pkg': 1,
  'pkgs_per_srv': 8,
  'num_srvs': 7,
  'chip_tops': gpu_tops,
  'c2c_bw': gpu_p2p_bw,
  'p2p_bw': gpu_p2p_bw,
  's2s_bw': gpu_s2s_bw,
  't_start': ts,
  'hbm_bw': gpu_hbm_bw
}

gpu_same_tco_spec = {'gpt2':   gpt2_gpu_same_tco,
                     'tnlg':   tnlg_gpu_same_tco,
                     'gpt3':   gpt3_gpu_same_tco,
                     'mtnlg': mtnlg_gpu_same_tco}
