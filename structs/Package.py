from dataclasses import dataclass, replace
from typing import Optional
import math

from .Base import Base
from .IO import IO
from .Chip import Chip
from .Memory import Memory, HBM

@dataclass
class Package(Base):
    package_id: int
    chip: Chip
    num_chips: int = 1 # number of chips per package
    mem_3d: Optional[Memory] = None # memory stacked on the top of the chip, can be SRAM or DRAM
    hbm: Optional[HBM] = None # memory on the side of the chip, usually is HBM

    io: Optional[IO] = None # package to package links

    heatsource_length: Optional[float] = None # mm
    heatsource_width: Optional[float] = None # mm

    area: Optional[float] = None # mm2
    length: Optional[float] = None # mm
    width: Optional[float] = None # mm

    tdp: Optional[float] = None # Watt
    cost: Optional[float] = None # $
    perf: Optional[float] = None # #OPS
    sram: Optional[float] = None # Byte
    dram: Optional[float] = None # Byte
    total_mem: Optional[int] = None # sram and dram, Byte

    valid: Optional[bool] = None
    invalid_reason: Optional[str] = None

    def update(self) -> None:
        self.io = replace(self.chip.pkg2pkg_io)
        self.io.num = self.num_chips * self.chip.pkg2pkg_io.num
        self.io.update()
        self.cost = self._get_cost()
        self.perf = self.num_chips * self.chip.perf
        self.tdp = self.num_chips * self.chip.tdp
        self.sram = self.num_chips * self.chip.sram
        self.dram = 0.0

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
            self._update_dimension()
        else:
            self.valid = False
    
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
            self.heatsource_length = max(self.chip.x, self.chip.y)
        self.heatsource_width = self.heatsource_length

        return True
        
    def _update_dimension(self) -> None:
        # To CONFIRM!
        if self.hbm:
            self.area = (self.chip.area + self.hbm.total_area) / 0.18
        else:
            self.area = self.chip.area / 0.18
        self.length = math.sqrt(self.area)
        self.width = self.length
        
    def _get_cost(self) -> float:
        total_cost = self.num_chips * self.chip.cost
        if self.mem_3d:
            raise NotImplementedError('3D memory is not supported yet')
        if self.hbm:
            total_cost += self.hbm.total_cost
        total_cost += self._eval_package_cost()
        return total_cost
    
    def _eval_package_cost(self) -> float:
        # TODO: add advanced packaing cost evaluation
        # http://electroiq.com/blog/2001/03/technology-comparisons-and-the-economics-of-flip-chip-packaging/
        return 0.103 * (self.num_chips * self.chip.cost) + 1.185



