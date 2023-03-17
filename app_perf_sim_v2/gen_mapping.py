import csv

from opt_mapping import opt_mapping, generate_mappings

if __name__ == '__main__': 

  algo_spec = {
    'num_layers': 96,
    'd': 12288,
    'max_ctx_len': 2048,
    'batch_size': 1,
    'bytes_per_number': 2
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
    's2s_bw': 10,       # in GB/s
    'T_start': 0.0001,  # us, init time for each data transfer
    'hbm_bw': None
  }

  out_header = ['srv_id', 'tops_per_chip', 'sram_per_chip', 'power_per_chip', 'chips_per_srv', 'srv_tco', 'num_srvs', 
      't', 'p', 'batch', 'latency', 'compute_latency', 'communicate_latency', 'tput', 'latency_best', 'tput_best',
      'all_tco', 'all_srv_cost', 
      'real_tops', 'peak_tops', 'utilization',
      'all_tco/tops', 'all_tco/tput']
  csv_out = open('all.csv', 'w')
  csv_writer = csv.DictWriter(csv_out, fieldnames=out_header)
  csv_writer.writeheader()

  with open('exploration.csv') as f:
    csv_reader = csv.reader(f, delimiter=',')
    line = 0
    for row in csv_reader:
      if line > 0:
        print('Evaluating chip ', line)
        if(float(row[4])>800.0):
          print('Chip too big')
          continue
        if(float(row[2])<0.01): # TOPS
          print('MACs too few')
          continue

        server = sys_spec.copy()
        server['pkgs_per_srv']     = int(row[6])
        server['tops_per_chip']    = float(row[2])
        server['sram_per_chip']    = float(row[1])
        server['power_per_chip']   = float(row[3])

        best_latency, best_tput, all_results = opt_mapping(server, algo_spec)
        for r in all_results:
          new_data = {}
          new_data['srv_id']         = line
          new_data['tops_per_chip']  = server['tops_per_chip']
          new_data['sram_per_chip']  = server['sram_per_chip']
          new_data['power_per_chip'] = server['power_per_chip']
          new_data['chips_per_srv']  = server['pkgs_per_srv']
          new_data['srv_tco']        = row[11]

          new_data['num_srvs'] = r[0]['srvs']
          new_data['t'] = r[0]['t']
          new_data['p'] = r[0]['p']
          new_data['batch'] = r[0]['batch']
          new_data['latency'] = r[1]
          new_data['compute_latency'] = r[2][0]
          new_data['communicate_latency'] = r[2][1]
          new_data['tput'] = r[3]

          if new_data['latency'] == best_latency:
            new_data['latency_best'] = '1'
          else:
            new_data['latency_best'] = '0'
          if new_data['tput'] == best_tput:
            new_data['tput_best'] = '1'
          else:
            new_data['tput_best'] = '0'
            
          total_tops = new_data['tput'] * algo_spec['num_layers'] * algo_spec['d'] * algo_spec['d'] * 24 / 1e12 # tera operations per sec
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

          csv_writer.writerow(new_data)

      line += 1

  csv_out.close()
