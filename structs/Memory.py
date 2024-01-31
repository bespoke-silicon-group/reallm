from dataclasses import dataclass, replace
from typing import Optional

from .Base import Base

@dataclass
class Memory(Base):
    simulator: Optional[bool] = None
    vdd: Optional[float] = None # in volt
    pj_per_byte: Optional[float] = None # in pJ/byte per access

    mem_type: Optional[str] = None # sram, hbm, 3d_sram, 3d_dram
    cap: Optional[int] = None # in byte
    bandwidth: Optional[int] = None # in byte/sec
    area: Optional[float] = None # in mm2
    tdp: Optional[float] = None # in watt
    cost: Optional[float] = None # in $
    bandwidth_efficiency: Optional[float] = None # the ratio of the actual bandwidth to the theoretical bandwidth

    def update(self) -> None:
        pass

@dataclass
class HBM(Memory):
    # One HBM Stack
    channel_bytes: Optional[int] = None # in byte
    channel_width: Optional[int] = None # in bit
    bit_rate: Optional[int] = None  # in bit/sec
    num_channels: Optional[int] = None

    cost_per_gb: float = 7.5 # $120 for 16GB
    pj_per_byte: float = 31.2 # HBM2
    vdd: float = 1.2 # in volt
    x: float = 7.75 # in mm
    y: float = 11.87 # in mm
    stack_cost: Optional[float] = None # in $

    config: str = 'HBM2_4GB'
    bw_dict_path: str = 'mem_sim/bw_dict.json'

    def update(self) -> None:
        self.mem_type = 'hbm'
        self.cap = self.channel_bytes * self.num_channels
        self.bandwidth = self.bit_rate * self.channel_width * self.num_channels / 8
        self.tdp = self.bandwidth * self.pj_per_byte * 1e-12
        self.area = self.x * self.y
        if self.stack_cost:
            self.cost = self.stack_cost
        else:
            self.cost = self.cap / 1024 / 1024 / 1024 * self.cost_per_gb

@dataclass
class Memory_3D_Vault(Memory):
    layer_bytes: Optional[int] = None
    layer_area: Optional[float] = None
    layer_cost: Optional[float] = None
    num_layers: Optional[int] = None

    vdd: float = 1.2 # in volt

    config: Optional[str] = None

    def update(self) -> None:
        self.cap = self.layer_bytes * self.num_layers
        self.tdp = self.bandwidth * self.pj_per_byte * 1e-12
        self.area = self.layer_area
        self.cost= self.layer_cost * self.num_layers