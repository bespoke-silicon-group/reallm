# %%
# from structs.HardwareConfig import ChipConfig, PackageConfig
import yaml, json

# %%
def gen_llmcompass_config(cc_yaml: str, template_json: str): 
    cc_config = yaml.safe_load(open(cc_yaml, 'r'))
    template_config = json.load(open(template_json, 'r'))
    template_config['name'] = cc_config['Name']

    chip_config = cc_config['Chip']
    pkg_config = cc_config['Package']
    io = chip_config['pkg2pkg_io']
    hbm = pkg_config['hbm'][0]

    template_config['interconnect']['link']['name'] = cc_config['Name'] + ' Link'
    template_config['interconnect']['link']['bandwidth_both_directions_byte'] = io['bandwidth_per_io']
    template_config['interconnect']['link']['bandwidth_per_direction_byte'] = io['bandwidth_per_io'] / 2
    template_config['interconnect']['link']['latency_second'] = io['init_time']
    template_config['interconnect']['link_count_per_device'] = io['num']

    if 'core' not in chip_config:
        raise ValueError('No core config found in yaml file')
    core = chip_config['core']

    template_config['device']['frequency_Hz'] = chip_config['freq']
    template_config['device']['compute_chiplet_count'] = pkg_config['num_chips']
    template_chip_conifg = template_config['device']['compute_chiplet']
    template_chip_conifg['core_count'] = core['core_count']
    template_chip_conifg['process_node'] = chip_config['tech']
    template_chip_conifg['core']['sublane_count'] = core['sublane_count']
    template_chip_conifg['core']['systolic_array']['array_width'] = core['sa_width']
    template_chip_conifg['core']['systolic_array']['array_height'] = core['sa_height']
    template_chip_conifg['core']['vector_unit']['vector_width'] = core['vector_width']
    template_chip_conifg['core']['vector_unit']['flop_per_cycle'] = core['vector_flop_per_cycle']
    template_chip_conifg['core']['register_file']['num_registers'] = core['num_registers']
    template_chip_conifg['core']['SRAM_KB'] = core['SRAM_KB']

    template_config['device']['memory_protocol'] = hbm['config']

    template_config['device']['io']['global_buffer_MB'] = chip_config['sram'] / 1e6
    template_config['device']['io']['global_buffer_bandwidth_per_cycle_byte'] = chip_config['sram_bw'] / chip_config['freq']
    template_config['device']['io']['memory_channel_active_count'] = chip_config['hbm_channels'] // hbm['num_channels']
    template_config['device']['io']['pin_count_per_channel'] = hbm['channel_width'] * hbm['num_channels']
    template_config['device']['io']['bandwidth_per_pin_bit'] = hbm['bit_rate']

    template_config['device']['memory']['total_capacity_GB'] = hbm['channel_bytes'] * chip_config['hbm_channels'] / 1024 / 1024 / 1024

    # write new config to file
    with open(f'outputs/{cc_config["Name"]}/llmcompass.json', 'w') as f:
        json.dump(template_config, f, indent=4)
# %%
# cc_yaml = '../inputs/hardware/config/hw_example.yaml'
# template_json = '../LLMCompass/configs/template.json'
# gen_llmcompass_config(cc_yaml, template_json)
