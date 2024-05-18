# =============================================================================
# WARNING: This file is a template for the constants related to the process 
# node that used for an accurate area estimation. 
# If you want to use this model, please set the area_model to 'micro_arch' and 
# fill in the values for the process node that you are targeting in this file.
# =============================================================================

from typing import List
from micro_arch_sim.vlsi import VLSI
from micro_arch_sim.sram import SRAM
import math

# =============================================================================
# VLSI Constants
# =============================================================================

vlsi_7nm = VLSI(
    process=7,
    fo4=math.inf,
    V_wire=math.inf,
    W_track=math.inf,
    A_track=math.inf,
    bf16_fma=math.inf,
    fp32_adder=math.inf,
)

vlsi_constants = {'7nm': vlsi_7nm}


# =============================================================================
# SRAMs
# =============================================================================

available_srams_7nm: List[SRAM] = [
    SRAM(process=7, depth=1024, width=128, x=math.inf, y=math.inf),
    SRAM(process=7, depth=1024, width=32,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=1024, width=64,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=2048, width=128, x=math.inf, y=math.inf),
    SRAM(process=7, depth=2048, width=32,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=2048, width=64,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=4096, width=128, x=math.inf, y=math.inf),
    SRAM(process=7, depth=4096, width=32,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=4096, width=64,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=512,  width=128, x=math.inf, y=math.inf),
    SRAM(process=7, depth=512,  width=32,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=512,  width=64,  x=math.inf, y=math.inf),
    SRAM(process=7, depth=8192, width=32,  x=math.inf, y=math.inf),
]

available_srams = {'7nm': available_srams_7nm }
