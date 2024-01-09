from dataclasses import dataclass, replace
from typing import Optional

from .Base import Base

@dataclass
class Memory(Base):
    vdd: Optional[float] = None # in volt
    pj_per_byte: Optional[float] = None # in pJ/byte per access

    mem_type: Optional[str] = None # sram, hbm, 3d_sram, 3d_dram
    cap: Optional[int] = None # in byte
    bandwidth: Optional[int] = None # in byte/sec
    area: Optional[float] = None # in mm2
    tdp: Optional[float] = None # in watt
    cost: Optional[float] = None # in $

    def update(self) -> None:
        pass

@dataclass
class HBM(Memory):
    # One HBM Stack
    config_dir: str = 'configs/HBM2_8GB'
    channel_bytes: int = 1024 * 1024 * 1024 # in byte
    channel_width: int = 128 # in bit
    bit_rate: int = 2 * 1024 * 1024 * 1024 # in bit/sec
    num_channels: int = 8
    cost_per_gb: float = 7.5 # $120 for 16GB
    pj_per_byte: float = 31.2 # HBM2
    vdd: float = 1.2 # in volt
    x: float = 7.75 # in mm
    y: float = 11.87 # in mm

    def update(self) -> None:
        self.cap = self.channel_bytes * self.num_channels
        self.bandwidth = self.bit_rate * self.channel_width * self.num_channels / 8
        self.tdp = self.bandwidth * self.pj_per_byte * 1e-12
        self.area = self.x * self.y
        self.cost = self.cap / 1024 / 1024 / 1024 * self.cost_per_gb