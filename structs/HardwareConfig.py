import multiprocessing, itertools, copy
from structs.Base import Base
from structs.Chip import Chip
from structs.Package import Package
from structs.Server import Server
from structs.IO import IO
from structs.Memory import HBM, Memory_3D_Vault
from typing import Optional, List, Dict
from dataclasses import dataclass

# Expand dict
def expand_dict(input_dict:  Dict) -> List[Dict]:
  '''
  Given a dict, return a list of all combinations of the dict.
  '''
  for key in input_dict:
    if isinstance(input_dict[key], Dict):
      input_dict[key] = expand_dict(input_dict[key])
    if not isinstance(input_dict[key], List):
      input_dict[key] = [input_dict[key]]

  all_dicts = []
  keys, values = zip(*input_dict.items())
  for v in itertools.product(*values):
    all_dicts.append(dict(zip(keys, v)))

  return all_dicts

@dataclass
class ChipConfig(Base):
  '''
  To define a chip, you should either give the perf, sram and bandwidth, 
  or area and mac_ratio and operational intensity
  '''
  yaml_config: dict
  all_configs: List[dict] = None

  def update(self) -> None:
    self.all_configs = expand_dict(self.yaml_config)

  def explore(self, verbose: bool = False) -> List[Chip]:
    chips = []
    chip_id = 0
    for cfg in self.all_configs:
      config = copy.deepcopy(cfg)
      if 'chip_id' not in config:
        config['chip_id'] = chip_id
      if 'pkg2pkg_io' in config:
        config['pkg2pkg_io'] = IO(**config['pkg2pkg_io'])
      if 'chip2chip_io' in config:
        config['chip2chip_io'] = IO(**config['chip2chip_io'])
      chip = Chip(**config)
      if chip.valid:
        chips.append(chip)
        chip_id += 1
      elif verbose:
        print(f'Invalid chip design: {chip.invalid_reason}')

    print(f'Found {len(chips)} valid chip designs.')

    return chips

@dataclass
class PackageConfig(Base):
  '''
  Package configuration given a chip.
  '''
  yaml_config: dict
  all_configs: List[dict] = None

  def update(self) -> None:
    # if 'mem_3d' in self.yaml_config:
    #   if isinstance(self.yaml_config['mem_3d'], Dict):
    #     self.yaml_config['mem_3d'] = expand_dict(self.yaml_config['mem_3d'])
    #   else:
    #     # list of 3d mem configs
    #     self.yaml_config['mem_3d'] = [expand_dict(mem) for mem in self.yaml_config['mem_3d']][0]
    self.all_configs = expand_dict(self.yaml_config)

  def explore(self, chips: List[Chip], verbose: bool = False) -> List[Package]:
    pkgs = []
    pkg_id = 0
    for chip in chips:
      for cfg in self.all_configs:
        config = copy.deepcopy(cfg)
        config['chip'] = chip
        if pkg_id not in config:
          config['package_id'] = pkg_id
        if 'mem_3d' in config:
          config['mem_3d'] = Memory_3D_Vault(**config['mem_3d'])
        if 'hbm' in config:
          config['hbm'] = HBM(**config['hbm'])
        pkg = Package(**config)
        if pkg.valid:
          pkgs.append(pkg)
          pkg_id += 1
        elif verbose:
          print(f'Invalid package design: {pkg.invalid_reason}')

    print(f'Found {len(pkgs)} valid package designs.')

    return pkgs

@dataclass
class ServerConfig(Base):
  '''
  Server configuration given a package.
  '''
  yaml_config: dict
  all_configs: List[dict] = None

  def update(self) -> None:
    self.all_configs = expand_dict(self.yaml_config)

  def explore(self, pkgs: List[Package], verbose: bool = False) -> List[Server]:
    srv_specs = []
    srv_id = 0
    for pkg in pkgs:
      for cfg in self.all_configs:
        config = copy.deepcopy(cfg)
        if 'server_id' not in config:
          config['server_id'] = srv_id
        config['package'] = pkg
        config['io'] = IO(**config['io'])
        srv_specs.append((config,verbose))
        srv_id += 1

    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_cores) as pool:
      results = pool.starmap(self._eval_server, srv_specs)

    valid_servers = [srv for srv in results if srv != None]
    print(f'Found {len(valid_servers)} valid server designs.')

    return valid_servers
  
  def _eval_server(self, config: dict,
                   verbose: bool = False) -> Optional[Server]:
    srv = Server(**config)
    if srv.valid:
      return srv
    else:
      if verbose:
        print(f'Invalid server design: {srv.invalid_reason}')
      return None
  
