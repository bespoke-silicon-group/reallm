from dataclasses import dataclass, replace
from typing import Optional
import math
import json

from .Base import Base
from .IO import IO
from .Chip import Chip, dies_per_wafer, get_die_yield
from .Memory import Memory, HBM, Memory_3D_Vault
from .Constants import PackageConstants, PackageConstantsCommon

@dataclass
class Package(Base):
    package_id: int
    chip: Chip
    num_chips: int = 1 # number of chips per package
    mem_3d: Optional[Memory_3D_Vault] = None # memory stacked on the top of the chip, can be SRAM or DRAM
    hbm: Optional[HBM] = None # memory on the side of the chip, usually is HBM
    io: Optional[IO] = None # package to package links
    si: bool = None # whether to use silicon interposer

    constants: PackageConstants = PackageConstantsCommon
    custom_max_power_density: Optional[float] = None # W/mm2

    num_hbm_stacks: int = 0

    heatsource_length: Optional[float] = None # mm
    heatsource_width: Optional[float] = None # mm

    total_die_area: Optional[float] = None # mm2
    area: Optional[float] = None # mm2
    length: Optional[float] = None # mm
    width: Optional[float] = None # mm

    tdp: Optional[float] = None # Watt
    cost: Optional[float] = None # $
    chip_cost: Optional[float] = None # $
    pkg_cost: Optional[float] = None # $
    perf: Optional[float] = None # #OPS
    sram: Optional[float] = None # Byte
    dram: Optional[float] = None # Byte
    total_mem: Optional[int] = None # sram and dram, Byte
    dram_bw_per_chip: Optional[float] = None # Byte/sec

    valid: Optional[bool] = None
    invalid_reason: Optional[str] = None

    def update(self) -> None:
        if self.chip.hbm_channels > 0:
            # HBM requires silicon interposer
            self.si = True
            if self.hbm:
                self.num_hbm_stacks = self.num_chips * (self.chip.hbm_channels // self.hbm.num_channels)
            else:
                raise ValueError('HBM is required but not provided')
        elif self.num_chips == 1:
            # 1 chip per package doesn't need silicon interposer
            self.si = False

        if self.chip.mem_3d_vaults > 0:
            if not self.mem_3d:
                raise ValueError('Memory 3D vault is required but not provided')

        self.io = replace(self.chip.pkg2pkg_io)
        self.io.num = self.num_chips * self.chip.pkg2pkg_io.num
        self.io.update()
        self.perf = self.num_chips * self.chip.perf
        self.tdp = self.num_chips * self.chip.tdp
        self.sram = self.num_chips * self.chip.sram
        if self.chip.mem_3d_vaults > 0:
            self.tdp += self.num_chips * self.chip.mem_3d_vaults * self.mem_3d.tdp
            if 'sram' in self.mem_3d.mem_type or 'SRAM' in self.mem_3d.mem_type:
                self.sram += self.num_chips * self.chip.mem_3d_vaults * self.mem_3d.cap
                self.dram = 0
                self.dram_bw_per_chip = 0
            elif 'dram' in self.mem_3d.mem_type or 'DRAM' in self.mem_3d.mem_type:
                self.dram = self.num_chips * self.chip.mem_3d_vaults * self.mem_3d.cap
                self.dram_bw_per_chip = self.mem_3d.bandwidth * self.chip.mem_3d_vaults
            else:
                raise ValueError('Memory 3D vault type is not supported')
        elif self.num_hbm_stacks > 0:
            self.tdp += self.num_hbm_stacks * self.hbm.tdp
            self.dram = self.num_hbm_stacks * self.hbm.cap
            if self.hbm.simulator:
                total_dram_bw = self._get_bw_from_simulator()
            else:
                total_dram_bw = self.hbm.bandwidth * self.num_hbm_stacks
            self.dram_bw_per_chip = total_dram_bw / self.num_chips
        else:
            self.dram = 0
            self.dram_bw_per_chip = 0
        self.total_mem = self.sram + self.dram
        # print(f'Package {self.package_id}: {self.num_chips} chips, {self.num_hbm_stacks} HBM stacks, {self.perf/1e12} TFLOPS, {self.sram/1e6} MB SRAM, {self.dram/1024/1024/1024} GB DRAM, {self.dram_bw_per_chip/1024/1024/1024} GB/s DRAM BW per chip')

        self._update_dimension()
        if self.check_area():
            if self.check_thermal():
                self.valid = True
                self.cost = self._get_cost()
            else:
                self.valid = False
        else:
            self.valid = False
    
    def _update_dimension(self) -> None:
        self.total_die_area = self.num_chips * self.chip.area
        if self.hbm:
            self.total_die_area += self.num_hbm_stacks * self.hbm.area
        if self.si: 
            # silicon interposer
            self.area = self.total_die_area * self.constants.si_area_scale_factor
        else: 
            # organic substrate
            self.area = self.total_die_area * self.constants.os_area_scale_factor
        self.length = math.sqrt(self.area)
        self.width = self.length

    def check_area(self) -> bool:
        if self.chip.mem_3d_vaults > 0:
            if self.chip.mem_3d_vaults * self.mem_3d.area > self.chip.area:
                self.invalid_reason = f'Not enough space to place {self.chip.mem_3d_vaults} 3D memory vaults'
                return False
            if self.chip.mem_3d_vault_tsvs:
                if self.chip.mem_3d_vault_tsvs != self.mem_3d.tsvs:
                    self.invalid_reason = f'Chip has {self.chip.mem_3d_vault_tsvs} TSVs per 3D memory vault, but the memory has {self.mem_3d.tsvs} TSVs per vault'
                    return False
            self.heatsource_length = max(self.chip.x, self.chip.y) * math.ceil(math.sqrt(self.num_chips))
        if self.num_hbm_stacks > 0:
            # check the physical layout of HBM to see if there is enough space to place it
            # now we only allow HBM to be placed on the long side of the chip
            # and the long side of HBM is parallel to the side of the chip
            # this is the same as GPU and TPU
            chip_long_side = max(self.chip.x, self.chip.y)
            hbm_long_side = max(self.hbm.x, self.hbm.y)
            max_num_stacks_per_chip = 2 * math.ceil(chip_long_side / hbm_long_side)
            max_num_stacks = max_num_stacks_per_chip * self.num_chips
            if self.num_hbm_stacks > max_num_stacks:
                self.invalid_reason = f'Not enough space to place {self.num_hbm_stacks} HBM stacks in total chip area of {self.num_chips * self.chip.area} mm2'
                return False
            elif self.num_hbm_stacks > max_num_stacks / 2:
                # If place on both sides
                num_stacks_per_side = math.ceil(self.num_hbm_stacks / self.num_chips / 2)
                chip_hbm_long_side = max(chip_long_side, num_stacks_per_side * hbm_long_side)
                chip_hbm_short_side = min(self.chip.x, self.chip.y) + 2 * min(self.hbm.x, self.hbm.y)
            else:
                # If place on one side only
                num_stacks_per_side = math.ceil(self.num_hbm_stacks / self.num_chips)
                chip_hbm_long_side = max(chip_long_side, num_stacks_per_side * hbm_long_side)
                chip_hbm_short_side = min(self.chip.x, self.chip.y) + min(self.hbm.x, self.hbm.y)
            self.heatsource_length = max(chip_hbm_long_side, chip_hbm_short_side) * math.floor(math.sqrt(self.num_chips)) * 1.2
        else:
            if self.total_die_area > self.constants.max_die_area:
                self.invalid_reason = f'Not enough space to place {self.num_chips} chips, the area is {self.total_die_area} mm2, but the max area is {self.constants.max_die_area} mm2'
                return False
            self.heatsource_length = max(self.chip.x, self.chip.y) * math.ceil(math.sqrt(self.num_chips)) * 1.2
        self.heatsource_width = self.heatsource_length

        return True
    
    def check_thermal(self) -> bool:
        self.power_density = self.tdp / (self.heatsource_length * self.heatsource_width)
        if self.custom_max_power_density:
            if self.power_density <= self.custom_max_power_density:
                return True
            else:
                self.invalid_reason = f'Power density {self.power_density} W/mm2 is higher than the custom max power density {self.custom_max_power_density} W/mm2'
                return False
        elif self.power_density <= self.constants.max_power_density:
            return True
        else:
            self.invalid_reason = f'Power density {self.power_density} W/mm2 is higher than the max power density {self.constants.max_power_density} W/mm2'
            return False
        
    def _get_cost(self) -> float:
        total_cost = self.num_chips * self.chip.cost
        if self.hbm:
            total_cost += self.num_hbm_stacks * self.hbm.cost
        if self.mem_3d:
            total_cost += (self.num_chips * self.mem_3d.cost * self.chip.mem_3d_vaults)
        total_cost += self._eval_package_cost()
        return total_cost
    
    def _eval_package_cost(self) -> float:
        # From ASIC Cloud
        # http://electroiq.com/blog/2001/03/technology-comparisons-and-the-economics-of-flip-chip-packaging/
        # return 0.103 * (self.num_chips * self.chip.cost) + 1.185

        # From 'Chiplet Actuary: A Quantitative Cost Model and Multi-Chiplet Architecture Exploration', DAC '22
        # Paper: https://dl.acm.org/doi/pdf/10.1145/3489517.3530428
        # GitHub: https://github.com/Yinxiao-Feng/DAC2022

        if self.si:
            # silicon interposer
            # we're using chip-last approach here
            cost_os = self.constants.os_cost_per_mm2 * self.area
            num_si_per_wafer = dies_per_wafer(die_area=self.area, wafer_diameter=300.0, wafer_dicing_gap=0.1)
            cost_si = self.constants.si_wafer_cost / num_si_per_wafer
            cost_c4_bump = self.constants.c4_bump_cost_per_mm2 * self.area
            si_yield = get_die_yield(self.area, self.constants.si_D0, self.constants.si_alpha)
            si_bonding_yield = self.constants.si_bonding_yield ** self.num_chips
            cost_pkg = (cost_si + cost_c4_bump) / si_yield / si_bonding_yield / self.constants.os_bonding_yield + \
                cost_os / self.constants.os_bonding_yield
            cost_wasted_chips = self.num_chips * self.chip.cost * (1 / si_bonding_yield / self.constants.os_bonding_yield - 1)
        else:
            # organic substrate
            if self.num_chips == 1:
                pkg_layer_cost_factor = 1
            else:
                for area_threshold, layer_cost_factor in self.constants.os_layer_scale_factor.items():
                    if self.area > area_threshold:
                        pkg_layer_cost_factor = layer_cost_factor
                        break
            bonding_yield = self.constants.os_bonding_yield ** self.num_chips
            cost_pkg = self.area * self.constants.os_cost_per_mm2 * pkg_layer_cost_factor / bonding_yield
            cost_wasted_chips = self.num_chips * self.chip.cost * (1 / bonding_yield - 1)

        return cost_pkg + cost_wasted_chips
    
    def _get_bw_from_simulator(self) -> int:
        '''
        Get the bandwidth from a json file generate by DRAMSim3 simulator.
        Return the bandwidth in byte/sec.
        '''
        with open(self.hbm.bw_dict_path) as json_file:
            bw_dict = json.load(json_file)
        config = f'configs/{self.hbm.config}/{self.num_hbm_stacks}_stack.ini'
        bw_GBps = bw_dict[config][str(self.chip.sa_width)]
        bw = int(bw_GBps * 1024 * 1024 * 1024)
        return bw




