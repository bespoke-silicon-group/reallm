from dataclasses import dataclass, replace
from typing import Optional
import math

from .Base import Base
from .IO import IO
from .Chip import Chip
from .Memory import Memory, HBM

@dataclass
class Package(Base):
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
            self.heatsource_length = math.sqrt(self.chip.area + self.hbm.total_area)
        else:
            self.heatsource_length = math.sqrt(self.chip.area)
        self.heatsource_width = self.heatsource_length

        self.total_mem = self.sram + self.dram

        self._update_dimension()

        self.valid = self.check_area()
    
    def check_area(self) -> bool:
        # TODO: add more physical layout constraints
        if True:
            return True
        else:
            self.invalid_reason = 'Package area violation'
            return False
        
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



