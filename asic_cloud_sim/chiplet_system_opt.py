import csv

opt_goal = 'cost_per_tops'

f = open('results.csv')
csvreader = csv.reader(f)

headers = next(csvreader)
header_index = {'sram_per_asic': None, 'tops_per_asic': None, 'cost_per_tops': None, 'watts_per_tops': None, 'tco_per_tops': None}
for i in range(len(headers)):
  for h in header_index:
    if h in headers[i]:
      header_index[h] = i

best_cost_per_tops = float('inf')
best_watts_per_tops = float('inf')
best_tco_per_tops = float('inf')

for row in csvreader:
  cpt = float(row[header_index['cost_per_tops']])
  wpt = float(row[header_index['watts_per_tops']])
  tpt = float(row[header_index['tco_per_tops']])
  if cpt < best_cost_per_tops:
    best_cost_per_tops_design = row[:]
    best_cost_per_tops = cpt
  if wpt < best_watts_per_tops:
    best_watts_per_tops_design = row[:]
    best_watts_per_tops = wpt
  if tpt < best_tco_per_tops:
    best_tco_per_tops_design = row[:]
    best_tco_per_tops = tpt

headers.insert(0, '[0] opt_goal')
best_cost_per_tops_design.insert(0, 'cost_per_tops')
best_watts_per_tops_design.insert(0, 'watts_per_tops')
best_tco_per_tops_design.insert(0, 'tco_per_tops')

f.close()

o_file = open('opt_results'+'.csv', 'w')
for h in headers[:-1]:
  o_file.write("%s,"% h)
o_file.write('\n')
for d in best_cost_per_tops_design[:-1]:
  o_file.write("%s,"% d)
o_file.write('\n')
for d in best_watts_per_tops_design[:-1]:
  o_file.write("%s,"% d)
o_file.write('\n')
for d in best_tco_per_tops_design[:-1]:
  o_file.write("%s,"% d)
o_file.write('\n')
o_file.close()


