import math
import csv
from opt_routing import opt_routings

if __name__ == '__main__': 
  apps = ['BERT', 'GPT2', 'T-NLG', 'GPT3', 'MT-NLG-Atten', 'MT-NLG-FC']
  techs = ['7nm']
  for tech in techs:
    for app in apps:
      csvf = open('../asic_cloud_sim/'+app+'_'+tech+'_results.csv')
      csvreader = csv.reader(csvf)
      
      headers = next(csvreader)[:-1]
      header_index = {'sram_per_asic': None, 'tops_per_asic': None, 'io_bw': None, 'server_cost': None, 'server_power': None, 'life_time_tco': None, 'asics_per_server': None}
      for i in range(len(headers)):
        for h in header_index:
          if h in headers[i]:
            header_index[h] = i
      
      o_file = open(app+'_'+tech+'_routing_opt_results'+'.csv', 'w')
      
      headers.append('opt_thru')
      headers.append('opt_thru_delay')
      #  headers.append('opt_thru_routing')
      headers.append('watts_per_opt_thru')
      headers.append('cost_per_opt_thru')
      headers.append('tco_per_opt_thru')
      
      headers.append('opt_delay')
      headers.append('opt_delay_thru')
      #  headers.append('opt_delay_routing')
      headers.append('watts_opt_delay')
      headers.append('cost_opt_delay')
      headers.append('tco_opt_delay')
      for h in headers[:]:
        o_file.write("%s,"% h)
      o_file.write('\n')
      
      best_tco_per_thru = 1e12
      best_tco_delay = 1e12
      best_tco_per_thru_design = None
      best_tco_delay_design = None
      for row in csvreader:
        new_row = row[:-1]
        chiplet_size = float(row[header_index['sram_per_asic']])
        TOPS = float(row[header_index['tops_per_asic']])
        link_GBs = float(row[header_index['io_bw']])
        server_cost = float(row[header_index['server_cost']])
        server_power = float(row[header_index['server_power']])
        life_time_tco = float(row[header_index['life_time_tco']])
        chiplets_per_board = int(float(row[header_index['asics_per_server']]))
      
        opt_thru_results, opt_delay_results = opt_routings(app, chiplet_size, TOPS, link_GBs, chiplets_per_board)
        [opt_thru, opt_thru_delay, opt_thru_routing] = opt_thru_results
        [opt_delay, opt_delay_thru, opt_delay_routing] = opt_delay_results
      
        
        new_row.append(math.floor(opt_thru))
        new_row.append(opt_thru_delay/1000)
        #  new_row.append(opt_thru_routing)
        new_row.append(server_power/math.floor(opt_thru)*1000)
        new_row.append(server_cost/math.floor(opt_thru)*1000)
        tco_per_thru = life_time_tco/math.floor(opt_thru)*1000
        new_row.append(tco_per_thru)
      
        new_row.append(opt_delay/1000)
        new_row.append(math.floor(opt_delay_thru))
        #  new_row.append(opt_delay_routing)
        new_row.append(server_power * opt_delay/1000000)
        new_row.append(server_cost * opt_delay/1000000)
        tco_delay = life_time_tco * opt_delay/1000000
        new_row.append(tco_delay)
        for r in new_row:
          o_file.write("%s,"% r)
        o_file.write('\n')
      
      o_file.close()

