from dataclasses import dataclass, replace
from typing import Optional

from .Base import Base

@dataclass
class Memory(Base):
    vdd: float = 1.2 # in volt
    pj_per_byte: Optional[float] = None # in pJ/byte per access

    mem_type: Optional[str] = None # sram, hbm, 3d_sram, 3d_dram
    total_bytes: Optional[int] = None # in byte
    total_bandwidth: Optional[int] = None # in byte/sec
    total_area: Optional[float] = None # in mm2
    total_tdp: Optional[float] = None # in watt
    total_cost: Optional[float] = None # in $

    def update(self) -> None:
        pass

@dataclass
class HBM(Memory):
    num_stacks: Optional[int] = None # need to be specified
    stack_bytes: int = 16*1024*1024*1024 # in byte 
    stack_bandwidth: int = 410*1024*1024*1024 # in byte/sec
    stack_cost: float = 120 # in $
    stack_x: float = 7.75 # in mm
    stack_y: float = 11.87 # in mm
    stack_tdp: Optional[float] = None # in watt
    stack_area: Optional[float] = None # in mm

    def update(self) -> None:
        if not self.mem_type:
            self.mem_type = 'hbm2'
        if not self.pj_per_byte:
            self.pj_per_byte = 31.2 # HBM2
        self.stack_area = self.stack_x * self.stack_y
        self.stack_tdp = self.stack_bandwidth * self.pj_per_byte * 1e-12

        self.total_bytes = self.num_stacks * self.stack_bytes
        self.total_bandwidth = self.num_stacks * self.stack_bandwidth
        self.total_area = self.num_stacks * self.stack_area
        self.total_tdp = self.num_stacks * self.stack_tdp
        self.total_cost = self.num_stacks * self.stack_cost