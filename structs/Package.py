from dataclasses import dataclass, replace
from typing import Optional
import math

from .Base import Base
from .IO import IO
from .Chip import Chip, dies_per_wafer, get_die_yield
from .Memory import Memory, HBM
from .Constants import PackageConstants, PackageConstantsCommon

@dataclass
class Package(Base):
    package_id: int
    chip: Chip
    num_chips: int = 1 # number of chips per package
    mem_3d: Optional[Memory] = None # memory stacked on the top of the chip, can be SRAM or DRAM
    hbm: Optional[HBM] = None # memory on the side of the chip, usually is HBM
    io: Optional[IO] = None # package to package links
    si: bool = None # whether to use silicon interposer

    constants: PackageConstants = PackageConstantsCommon

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

    valid: Optional[bool] = None
    invalid_reason: Optional[str] = None

    def update(self) -> None:
        # HBM requires silicon interposer
        if self.si == None and self.hbm:
            self.si = True
        self.io = replace(self.chip.pkg2pkg_io)
        self.io.num = self.num_chips * self.chip.pkg2pkg_io.num
        self.io.update()
        self.perf = self.num_chips * self.chip.perf
        self.tdp = self.num_chips * self.chip.tdp
        self.sram = self.num_chips * self.chip.sram
        self.dram = 0.0

        self._update_dimension()
        if self.check_area():
            self.valid = True
            if self.mem_3d:
                raise NotImplementedError('3D memory is not supported yet')
                # self.tdp += (self.num_chips * self.mem_3d.tdp)
                # if self.mem_3d.mem_type == '3d_dram':
                #     self.dram += (self.num_chips * self.mem_3d.size)
                # elif self.mem_3d.mem_type == '3d_sram':
                #     self.sram += (self.num_chips * self.mem_3d.size)
            if self.hbm:
                self.tdp += self.hbm.total_tdp
                self.dram += self.hbm.total_bytes
            self.total_mem = self.sram + self.dram
            self.cost = self._get_cost()
        else:
            self.valid = False
    
    def _update_dimension(self) -> None:
        self.total_die_area = self.num_chips * self.chip.area
        if self.hbm:
            self.total_die_area += self.hbm.total_area
        if self.si: 
            # silicon interposer
            self.area = self.total_die_area * self.constants.si_area_scale_factor
        else: 
            # organic substrate
            self.area = self.total_die_area * self.constants.os_area_scale_factor
        self.length = math.sqrt(self.area)
        self.width = self.length

    def check_area(self) -> bool:
        if self.mem_3d:
            raise NotImplementedError('3D memory is not supported yet')
        if self.hbm:
            # check the physical layout of HBM to see if there is enough space to place it
            # now we only allow HBM to be placed on the long side of the chip
            # and the long side of HBM is parallel to the side of the chip
            # this is the same as GPU and TPU
            chip_long_side = max(self.chip.x, self.chip.y)
            hbm_long_side = max(self.hbm.stack_x, self.hbm.stack_y)
            max_num_stacks = 2 * math.ceil(chip_long_side / hbm_long_side)
            if self.hbm.num_stacks > max_num_stacks:
                self.invalid_reason = f'Not enough space to place {self.hbm.num_stacks} HBM stacks'
                return False
            elif self.hbm.num_stacks > max_num_stacks / 2:
                # If place on both sides
                num_stacks_per_side = math.ceil(self.hbm.num_stacks / 2)
                chip_hbm_long_side = max(chip_long_side, num_stacks_per_side * hbm_long_side)
                chip_hbm_short_side = min(self.chip.x, self.chip.y) + 2 * min(self.hbm.stack_x, self.hbm.stack_y)
            else:
                # If place on one side only
                num_stacks_per_side = self.hbm.num_stacks
                chip_hbm_long_side = max(chip_long_side, num_stacks_per_side * hbm_long_side)
                chip_hbm_short_side = min(self.chip.x, self.chip.y) + min(self.hbm.stack_x, self.hbm.stack_y)
            self.heatsource_length = max(chip_hbm_long_side, chip_hbm_short_side)
        else:
            if self.area > self.constants.max_die_area:
                self.invalid_reason = f'Not enough space to place {self.num_chips} chips'
                return False
            self.heatsource_length = max(self.chip.x, self.chip.y) * math.ceil(math.sqrt(self.num_chips))
        self.heatsource_width = self.heatsource_length

        return True
        
    def _get_cost(self) -> float:
        total_cost = self.num_chips * self.chip.cost
        if self.mem_3d:
            raise NotImplementedError('3D memory is not supported yet')
        if self.hbm:
            total_cost += self.hbm.total_cost
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




