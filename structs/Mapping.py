import math
from dataclasses import dataclass
from .Base import Base

@dataclass
class Mapping(Base):
    t: int = 1 # tensor parallelism size, number of chips per pipeline stage
    p: int = 1 # pipeline paralelism size
    micro_batch: int = 1 # micro batch size