import os
import configparser
import json
import math

def get_bw_from_dramsim3(num_bytes: int, config_path: str)-> float:
    config = configparser.ConfigParser()
    config.read(config_path)
    channels = int(config.get('system', 'channels'))
    bus_bits = int(config.get('system', 'bus_width'))
    total_bus_bytes = channels * bus_bits / 8
    columns = int(config.get('dram_structure', 'columns'))

    num_cycles = math.ceil(num_bytes / total_bus_bytes) * 100

    addr = 0
    base_cycle = 0

    f = open('trace.txt', 'w')
    for cyc in range(num_cycles):
        for ch in range(channels):
            f.write(f'{hex(addr)} READ {base_cycle + cyc}\n')
            addr += columns
    f.close()
    
    sim_cycles = num_cycles + 1
    os.system(f'make -s single_run CONFIG={config_path} TRACE=trace.txt CYCLE={sim_cycles}')

    with open('dramsim3.json') as json_file:
        data = json.load(json_file)
        tot_bw = 0.0
        for ch in range(channels):
            tot_bw += data[str(ch)]['average_bandwidth']

    return tot_bw

config_dir = 'configs'

data_bytes = 2
sa_width_min = 4
sa_width_max = 512

bw_dict = dict()

for root, subdirs, files in os.walk(config_dir):
    for f in files:
        config_path = os.path.join(root, f)
        bw_dict[config_path] = dict()
        for sa_width in range(sa_width_min, sa_width_max + 1):
            num_bytes = sa_width * sa_width * data_bytes
            bw = get_bw_from_dramsim3(num_bytes, config_path)
            bw_dict[config_path][sa_width] = bw

with open('bw_dict.json', 'w') as outfile:
    json.dump(bw_dict, outfile, indent=4)