import csv
from software_evaluation.opt_mapping import opt_mapping
from tqdm import tqdm
import argparse

if __name__ == '__main__': 

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str)
  parser.add_argument('--hw-csv', type=str)
  parser.add_argument('--results-dir', type=str)
  args = parser.parse_args()

  model_name = args.model
  hw_csv = args.hw_csv
  results_dir = args.results_dir

  gpt2 = {
    'num_layers': 48,
    'd': 1600,
    'max_ctx_len': 1024,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  tnlg = {
    'num_layers': 78,
    'd': 4256,
    'max_ctx_len': 1024,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  gpt3 = {
    'num_layers': 96,
    'd': 12288,
    'max_ctx_len': 2048,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  mtnlg = {
    'num_layers': 105,
    'd': 20480,
    'max_ctx_len': 2048,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  palm = {
    'num_layers': 118,
    'd': 18432,
    'max_ctx_len': 2048,
    'batch_size': 1,
    'bytes_per_number': 2,
    'num_heads': 48
  }

  gpt3_ctx_8K = {
    'num_layers': 96,
    'd': 12288,
    'max_ctx_len': 1024*8,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  gpt3_ctx_32K = {
    'num_layers': 96,
    'd': 12288,
    'max_ctx_len': 1024*32,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  gpt3_ctx_128K = {
    'num_layers': 96,
    'd': 12288,
    'max_ctx_len': 1024*128,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  gpt_340B = {
    'num_layers': 105,
    'd': 128*128,
    'max_ctx_len': 2048,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  gpt_700B = {
    'num_layers': 96,
    'd': 96*128*2,
    'max_ctx_len': 4096,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  gpt_1360B = {
    'num_layers': 105,
    'd': 128*128*2,
    'max_ctx_len': 2048,
    'batch_size': 1,
    'bytes_per_number': 2
  }

  all_models = {'gpt2': gpt2, 'gpt3': gpt3, 
                'tnlg': tnlg,
                'palm': palm,
                # 'mtnlg':mtnlg,
                # 'gpt3_ctx_8K':  gpt3_ctx_8K, 
                # 'gpt3_ctx_32K': gpt3_ctx_32K, 
                # 'gpt3_ctx_128K': gpt3_ctx_128K, 
                # 'gpt_340B': gpt_340B,
                # 'gpt_700B': gpt_700B,
                # 'gpt_1360B': gpt_1360B,
                }

  sys_spec = {
    'chips_per_pkg': 1,
    'pkgs_per_srv': 36,
    'num_srvs': 1,
    'tops_per_chip': 36.7,
    'sram_per_chip': 360,   # in MB
    'power_per_chip': 67,   # in Watt
    'c2c_bw': 100,      # in GB/s
    'p2p_bw': 25,       # in GB/s
    's2s_bw': 1.25,     # in GB/s
    'T_start': 0.0001,  # us, init time for each data transfer
    'hbm_bw': None
  }

  out_header = ['srv_id', 'chip_id',
                'tops_per_chip', 'sram_per_chip', 'power_per_chip', 
                'power_per_srv',
                'chips_per_srv', 'srv_tco', 'num_srvs', 
                't', 'p', 'batch', 'micro_batch',  
                'micro_batch_latency', 'batch_pipeline_latency',
                'compute_latency', 'communicate_latency', 
                'real_ctx_len',
                'latency',
                'tput', 'all_tco', 'all_srv_cost', 
                'real_tops', 'peak_tops', 'utilization',
                'all_tco/tops', 'all_tco/tput',
                'W/tput', '$/tput',
                'tco/token', 'tco/1ktoken',
                'real_w', 'tops/mb', 'all_area',
                'latency_ms', '1/tput',
                'latency_best', 'tput_best',
                ]

  # for model_name in tqdm(all_models):
  print('Generated design points for:', model_name)
  if 'exploration' in hw_csv:
    csv_out = open(results_dir+'/'+model_name+'_all.csv', 'w')
  elif 'HBM_chiplet' in hw_csv:
    csv_out = open(results_dir+'/'+model_name+'_HBM_chiplet.csv', 'w')
  else:
    print('Wrong HW csv', hw_csv)

  csv_writer = csv.DictWriter(csv_out, fieldnames=out_header)
  csv_writer.writeheader()

  with open(hw_csv) as f:
    csv_reader = csv.reader(f, delimiter=',')
    line = 0
    for row in tqdm(csv_reader):
      if line > 0:
        server = sys_spec.copy()
        algo = all_models[model_name].copy()
        chip_id = int(row[25])
        server['pkgs_per_srv']     = int(row[6])
        server['tops_per_chip']    = float(row[2])
        server['sram_per_chip']    = float(row[1])
        server['power_per_chip']   = float(row[3])
        server['power_per_srv']    = float(row[8])
        if chip_id == -1:
          server['hbm_bw'] = 2039
        if chip_id == -2:
          server['hbm_bw'] = 614
        
        best_latency, best_tput, all_results = opt_mapping(server, algo)
        for r in all_results:
          new_data = {}
          new_data['srv_id']         = line
          new_data['chip_id']        = int(row[25]) 
          new_data['tops_per_chip']  = server['tops_per_chip']
          new_data['sram_per_chip']  = server['sram_per_chip']
          new_data['power_per_chip'] = server['power_per_chip']
          new_data['chips_per_srv']  = server['pkgs_per_srv']
          new_data['power_per_srv']  = server['power_per_srv']
          new_data['srv_tco']        = row[11]

          new_data['num_srvs'] = r[0]['srvs']
          new_data['t'] = r[0]['t']
          new_data['p'] = r[0]['p']
          new_data['batch'] = r[0]['batch']
          new_data['real_ctx_len'] = r[0]['real_ctx_len']
          new_data['micro_batch'] = r[0]['micro_batch']
          new_data['micro_batch_latency'] = r[0]['micro_batch_latency']
          new_data['batch_pipeline_latency'] = r[0]['batch_pipeline_latency']
          new_data['latency'] = r[1]
          new_data['latency_ms'] = r[1] / 1000.0
          new_data['compute_latency'] = r[2][0]
          new_data['communicate_latency'] = r[2][1]
          new_data['tput'] = r[3]
          new_data['1/tput'] = 1/r[3]

          if new_data['latency'] == best_latency:
            new_data['latency_best'] = '1'
          else:
            new_data['latency_best'] = '0'
          if new_data['tput'] == best_tput:
            new_data['tput_best'] = '1'
          else:
            new_data['tput_best'] = '0'
            
          total_tops = new_data['tput'] * algo['num_layers'] * algo['d'] * algo['d'] * 24 / 1e12 # tera operations per sec
          theory_peak_tops = new_data['num_srvs'] * new_data['chips_per_srv'] * new_data['tops_per_chip']
          new_data['real_tops'] = total_tops
          new_data['peak_tops'] = theory_peak_tops
          new_data['utilization'] = float(total_tops/theory_peak_tops)
          # TCO fix portion: SrvAmortization, SrvInterest
          srv_tco_fix = float(row[15]) + float(row[16])
          srv_tco_power = (float(row[11]) - srv_tco_fix)*new_data['utilization']
          new_data['all_srv_cost'] = new_data['num_srvs'] * srv_tco_fix
          new_data['all_tco'] = new_data['num_srvs'] * (srv_tco_fix + srv_tco_power)
          new_data['all_tco/tops'] = new_data['all_tco'] / (total_tops/(new_data['latency']/1e6))
          new_data['all_tco/tput'] = new_data['all_tco'] / new_data['tput']
          # new_data['real_w'] = new_data['num_srvs'] * new_data['chips_per_srv'] * new_data['power_per_chip'] * new_data['utilization']
          new_data['real_w'] = new_data['num_srvs'] * new_data['power_per_srv'] * new_data['utilization']
          new_data['W/tput'] = new_data['real_w'] / new_data['tput']
          new_data['$/tput'] = new_data['all_srv_cost'] / new_data['tput']
          total_sec= 1.5 * 365 * 24 * 3600
          new_data['tco/token'] = new_data['all_tco'] / total_sec / new_data['tput']
          new_data['tco/1ktoken'] = 1000 * new_data['tco/token'] 
          new_data['tops/mb'] = new_data['tops_per_chip'] / new_data['sram_per_chip']
          new_data['all_area'] = new_data['num_srvs'] * new_data['chips_per_srv'] * float(row[4])

          csv_writer.writerow(new_data)

      line += 1

  csv_out.close()
