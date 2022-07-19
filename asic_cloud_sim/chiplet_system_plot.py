import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import math
import argparse

def read_csv(csv_name, verbose=False):
  with open(csv_name) as csv_file:
    reader = csv.reader(csv_file)
    header = next(reader)
    if verbose:
      print(header)
    rows = []
    for row in reader:
      rows.append(row)

  x = []
  for row in rows:
    x.append(row[2][1:])
  if verbose:
    print(x)
  return rows, x

def gen_pareto_chiplets(csv_name):
    
  rows, x = read_csv(csv_name)
  chiplets = []

  for row in rows:
    sram = float(row[1])
    chiplet = {'tops_per_asic': float(row[2]), 'die_area': float(row[4]), 'watts_per_asic': float(row[5]),
               'die_cost': float(row[6]), 'asics_per_server': float(row[8]), 'server_power': float(row[9]), 
               'tops_per_server': float(row[10]), 'server_cost': float(row[11]), 'tco': float(row[13]), 
               'DCAmortization': float(row[15]), 'DCInterest': float(row[16]), 'DCOpex': float(row[17]), 
               'SrvAmortization': float(row[18]), 'SrvInterest': float(row[19]), 'SrvOpex': float(row[20]), 
               'SrvPower': float(row[21]), 'PUEOverhead': float(row[22]),
               'cost_per_tops': float(row[23]), 'watts_per_tops': float(row[24]), 'tco_per_tops': float(row[25]),
               'max_die_power_per_server': float(row[26]), 'sram_per_asic': float(row[1])
              }
    # if sram in chiplets:
    #     if chiplets[sram]['tops_per_asic'] < chiplet['tops_per_asic']:
    #         chiplets[sram] = chiplet
    # else:
    #     chiplets[sram] = chiplet

    chiplets.append(chiplet)
    chiplet['$_per_tops'] = chiplet['server_cost'] / chiplet['tops_per_server']
    chiplet['w_per_tops'] = chiplet['server_power'] / chiplet['tops_per_server']
      
  return chiplets

def plot_pareto(chiplet_systems, x_label='w_per_tops', y_label='$_per_tops', draw_tco=None, p1_x=1.5, p2_x=1.55, base=8.7):
  figure(figsize=(5, 6), dpi=200)
  plt.rcParams.update({'font.size': 15})
    
  markers=['.', '*', '+', '^', '>', '.', '<', '2', 'o', 's', '3' ]
  for sys in chiplet_systems:
    config = sys['config']
    chiplets = sys['chiplets']
    y = []
    x = []
    for c in chiplets:
      y.append(c[y_label])
      x.append(c[x_label])
  
    if config == 'exploration':
      label = 'Chiplet Cloud'
    elif config == 'SI':
      label = 'Si Interposer'
    elif config == 'organic_sub':
      label = 'Organic Sub'
    else:
      label = config
    plt.scatter(x, y, s=50, label=label, marker=markers.pop(0))

  plt.xlabel('W/TOPS', fontsize=15)
  plt.ylabel('$/TOPS', fontsize=15)
  plt.grid(which="major")

  plt.legend(bbox_to_anchor=(1.0, 0.0), loc='lower right', fontsize=10, ncol=1)
  
  if draw_tco is not None:
    tco_line(plt, chiplets, draw_tco, p1_x, p2_x, base)
  return plt

def tco_line(plt, chiplets, tco_tops_list, p1_x=1.5, p2_x=1.55, base=8.7):
  a = np.array([[float(chiplets[0]['server_power']), float(chiplets[0]['server_cost'])], [float(chiplets[3]['server_power']), float(chiplets[3]['server_cost'])]])
  b = np.array([float(chiplets[0]['tco']), float(chiplets[3]['tco'])])
  x = np.linalg.solve(a, b)
  A = x[0]
  B = x[1]

  for tco_tops in tco_tops_list:
    p1_y = (tco_tops-A*p1_x)/B
    p2_y = (tco_tops-A*p2_x)/B
    if tco_tops == base:
      plt.axline((p1_x,p1_y), (p2_x,p2_y), linewidth=1, color='r',label='Same TCO/TOPS')
    plt.axline((p1_x,p1_y), (p2_x,p2_y), linewidth=1, color='r')
    plt.text(p1_x, p1_y, "{:3.1f}".format(tco_tops), fontsize=10, color='r', )

  return plt

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='exploration')
  args = parser.parse_args()
  config = args.config

  chiplets = gen_pareto_chiplets(config+'.csv')
  systems = [{'config': config, 'chiplets':chiplets}]
  plt = plot_pareto(systems, draw_tco=[7.95])

  plt.savefig('pareto.pdf', bbox_inches='tight', pad_inches=0.1)

