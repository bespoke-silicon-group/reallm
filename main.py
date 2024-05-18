import argparse, yaml, os
from phases.hardware_exploration import hardware_exploration
from phases.software_evaluation import software_evaluation

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-hw', '--hw-config', type=str, dest='hw_config',
                      help='Path to the hardware configuration file')
  parser.add_argument('-c', '--constants', type=str, dest='constants',
                      default='inputs/hardware/constant/7nm_default.yaml', 
                      help='Path to the constants file')
  parser.add_argument('-m', '--model-config', type=str, dest='model_config',
                      help='Path to the languge model configuration file')
  parser.add_argument('-s', '--sys-config', type=str, dest='sys_config',
                      help='Path to the system configuration file')
  parser.add_argument('-o', '--outputs-dir', type=str, dest='outputs_dir',
                      default='outputs', help='Path to the outputs directory')
  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                      help='Print verbose output')
  args = parser.parse_args()
  hardware_name = args.hw_config.split('/')[-1].split('.')[0]
  hw_config = yaml.safe_load(open(args.hw_config, 'r'))
  constants = yaml.safe_load(open(args.constants, 'r'))
  hardware_exploration(hw_config, constants, args.outputs_dir, hardware_name, args.verbose)
  
  model_config = yaml.safe_load(open(args.model_config, 'r'))
  if args.sys_config == None:
    args.sys_config = 'inputs/software/system/sys_default.yaml'
    print(f'Warning: System configuration file not found. Using default configuration: {args.sys_config}')
  if os.path.exists(args.sys_config) == False:
    args.sys_config = 'inputs/software/system/sys_default.yaml'
    print(f'Warning: System configuration file not found. Using default configuration: {args.sys_config}')

  sys_config = yaml.safe_load(open(args.sys_config, 'r'))
  hw_pickle = f'{args.outputs_dir}/{hardware_name}/{hardware_name}.pkl'
  software_evaluation(model_config, sys_config, hw_pickle, args.outputs_dir, args.verbose)

if __name__ == '__main__':
  main()
