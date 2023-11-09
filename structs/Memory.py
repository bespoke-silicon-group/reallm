from dataclasses import dataclass, replace
from typing import Optional

from .Base import Base

@dataclass
class Memory(Base):
    mem_type: str # sram, hbm, 3d_sram, 3d_dram
    size: float # in byte
    bandwidth: float # in byte/sec
    area: float # in mm2
    tdp: float # in watt
    cost: float # in $
    vdd: float = 1.2 # in volt
