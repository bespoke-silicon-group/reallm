import argparse, yaml
from phases.hardware_exploration import hardware_exploration
from phases.software_evaluation import software_evaluation

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-hw', '--hw-config', type=str, dest='hw_config',
                      help='Path to the hardware configuration file')
  parser.add_argument('-m', '--model-config', type=str, dest='model_config',
                      help='Path to the languge model configuration file')
  parser.add_argument('-o', '--outputs-dir', type=str, dest='outputs_dir',
                      default='outputs', help='Path to the outputs directory')
  parser.add_argument('-v', '--verbose', action='store_true', dest='verbose',
                      help='Print verbose output')
  args = parser.parse_args()
  hardware_name = args.hw_config.split('/')[-1].split('.')[0]
  hw_config = yaml.safe_load(open(args.hw_config, 'r'))
  hardware_exploration(hw_config, args.outputs_dir, hardware_name, args.verbose)
  
  hw_pickle = f'{args.outputs_dir}/{hardware_name}/{hardware_name}.pkl'
  software_evaluation(args.model_config, args.hw_config, hw_pickle, args.outputs_dir, args.verbose)


if __name__ == '__main__':
  main()
