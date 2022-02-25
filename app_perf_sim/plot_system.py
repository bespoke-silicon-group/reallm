import graphviz
import csv
import re
import math

def plot_routing(routing, output):
  layer_routing = re.findall(r"\[(.*?)\]", routing)
  
  Q_r = layer_routing[0]
  K_r = layer_routing[1]
  V_r = layer_routing[2]
  A_r = layer_routing[3]
  FC1_r = layer_routing[4]
  FC2_r = layer_routing[5]
  
  real_routing = {'Q':[None,None], 'K':[None,None], 'V':[None,None], 'FC0':[None,None], 'FC1':[None,None], 'FC2':[None,None]}
  
  real_routing['Q'][0] = int(Q_r.split(',')[0])
  real_routing['Q'][1] = int(Q_r.split(',')[1])
  real_routing['K'][0] = int(Q_r.split(',')[0])
  real_routing['K'][1] = int(Q_r.split(',')[1])
  real_routing['V'][0] = int(Q_r.split(',')[0])
  real_routing['V'][1] = int(Q_r.split(',')[1])
  real_routing['FC0'][0] = int(A_r.split(',')[0])
  real_routing['FC0'][1] = int(A_r.split(',')[1])
  real_routing['FC1'][0] = int(FC1_r.split(',')[0])
  real_routing['FC1'][1] = int(FC1_r.split(',')[1])
  real_routing['FC2'][0] = int(FC2_r.split(',')[0])
  real_routing['FC2'][1] = int(FC2_r.split(',')[1])
  
  G = graphviz.Digraph(output+'_system', comment=output)
  
  pre_out_nodes = []
  for node_type in ['Q', 'K', 'V', 'FC0', 'FC1', 'FC2']:
  #  for node_type in ['Q', 'K', 'V', 'FC0', 'FC1']:
    C = real_routing[node_type][0]
    M = real_routing[node_type][1]
    num_nodes = C*M
    for i in range(num_nodes):
      G.node(node_type+str(i), label=node_type, shape='square')
    for i in range(M):
      for j in range(C-1):
        G.edge(node_type+str(i*C+j), node_type+str(i*C+j+1))
    
    if node_type in ['Q', 'K', 'V']:
      for i in range(M):
        pre_out_nodes.append(node_type+str((i+1)*C-1))
    else:
      for i in range(M):
        if len(pre_out_nodes) >= C:
          Nout_to_1in = math.ceil(len(pre_out_nodes)/C)
          for j in range(C):
            in_node = node_type+str(i*C+j)
            for k in range(Nout_to_1in):
              out_node = pre_out_nodes[math.floor(j*(len(pre_out_nodes)/C))+k]
              G.edge(out_node, in_node)
        if len(pre_out_nodes) < C:
          out_to_Nin = math.ceil(C/len(pre_out_nodes))
          for j in range(len(pre_out_nodes)):
            out_node = pre_out_nodes[j]
            for k in range(out_to_Nin):
              in_node_index = math.floor(j*C/len(pre_out_nodes))+k
              in_node = node_type+str(in_node_index)
              G.edge(out_node, in_node)
      pre_out_nodes = []
      for i in range(M):
        pre_out_nodes.append(node_type+str((i+1)*C-1))
  
  
  G.render(directory='./').replace('\\', '/')


f = open('best_systems.csv')
csvreader = csv.reader(f)

headers = next(csvreader)

for row in csvreader:
  opt_target = row[0]
  link_GBs = row[1]
  sram = row[2]
  tops = row[3]
  area = row[4]
  asic_power = row[5]
  num_chiplets = row[6]
  server_power = row[7]
  tco = row[8]
  routing = row[9]

  # print(opt_target, sram, tops, routing)
  plot_routing(routing, opt_target)

f.close()


# digraph G {
#     #splines=ortho
#   fontname="Helvetica,Arial,sans-serif"
#   node [fontname="Helvetica,Arial,sans-serif"]
#   edge [fontname="Helvetica,Arial,sans-serif"]
#   graph [center=1 rankdir=LR]
#   edge [arrowsize=0.2, penwidth=0.2]
#   node [width=0.3 height=0.3 label=""]
#   { node [shape=circle style=invis]
#     1 2 3 4 5 6
#   }
#   { node [shape=square]
#     Q1 Q2 V1 V2 K1 K2 A1 A2 FC11 FC12 FC11 FC12 FC13 FC14 FC15 FC16 FC17 FC18 FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28
#   }
# 
#   1 -> Q1 -> {A1 A2} [color="#000000"]
#   2 -> Q2 -> {A1 A2} [color="#000000"]
#   3 -> V1 -> {A1 A2} [color="#000000"]
#   4 -> V2 -> {A1 A2} [color="#000000"]
#   5 -> K1 -> {A1 A2} [color="#000000"]
#   6 -> K2 -> {A1 A2} [color="#000000"]
# 
#   A1 -> {FC11 FC12 FC13 FC14 FC15 FC16 FC17 FC18} 
#   A2 -> {FC11 FC12 FC13 FC14 FC15 FC16 FC17 FC18}
#     
#   FC11 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC12 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC13 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC14 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC15 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC16 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC17 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
#   FC18 -> {FC21 FC22 FC23 FC24 FC25 FC26 FC27 FC28}
# 
# }
