import math
from dataclasses import dataclass
from .Base import Base

@dataclass
class Mapping(Base):
    t: int = 1 # tensor parallelism size, number of chips per pipeline stage
    p: int = 1 # pipeline paralelism size
    c: int = 1 # context parallelism size
    micro_batch: int = 1 # micro batch size
    prefill_micro_batch: int = 1 # micro batch size for prefill

    dynamic: bool = False
    num_sub_sys: int = 0
    sub_batch: int = 0
    sub_micro_batch: int = 0
    sub_prefill_micro_batch: int = 0
    sub_t: int = 0
    sub_p: int = 0
    sub_ctx_len: int = 0
