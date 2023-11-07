from typing import Optional
from dataclasses import dataclass

from .Base import Base

@dataclass
class IO(Base):
   io_type: str # 'c2c': chip 2 chip, 'p2p': package to package, 's2s': server to server
   num: int # number of IOs
   bandwidth_per_io: int # peak bandwidth per IO, in byte/s
   area_per_io: float = 0.0 # in mm2
   tdp_per_io: float = 0.0 # watts
   joules_per_byte: float = 0.0 # joules per byte
   init_time: float = 1e-10 # the time to initialize a data transfer, in sec

   bandwidth: Optional[int] = None# peak total bandwidth, in byte/s, per direction
   area: Optional[float] = None
   tdp: Optional[float] = None
   def update(self) -> None:
      # This is how it works in ASPLOS submission apprently, fix this later
      self.bandwidth = self.num * self.bandwidth_per_io / 2
      # self.bandwidth = self.num * self.bandwidth_per_io 
      self.area = self.num * self.area_per_io
      self.tdp = self.num * self.tdp_per_io