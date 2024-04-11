import math
from dataclasses import dataclass
from .Base import Base

@dataclass
class Mapping(Base):
    t: int = 1 # tensor parallelism size, number of chips per pipeline stage
    p: int = 1 # pipeline paralelism size
    micro_batch: int = 1 # micro batch size
    prefill_micro_batch: int = 1 # micro batch size for prefill
    hybrid: bool = False
    prefill_batch: int = 0
    prefill_t: int = 0
    prefill_p: int = 0
    def update(self) -> None:
        if not self.hybrid:
            self.prefill_t = self.t
            self.prefill_p = self.p
