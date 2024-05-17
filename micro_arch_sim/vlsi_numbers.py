from typing import List
from vlsi import VLSI
from sram import SRAM
import math

# =============================================================================
# This file is a template for the constants related to the process node that 
# used for an accurate area estimation. 
# If you want to use this model, please set the area_model to 'micro_arch' and 
# fill in the values for the process node that you are targeting in this file.
# =============================================================================

# =============================================================================
# VLSI Constants
# =============================================================================

vlsi_12nm = VLSI(
    process=12,
    fo4=50,
    V_wire=10,
    W_track=0.2,
    A_track=0.2,
    bf16_fma=math.inf,
    fp32_adder=math.inf,
)

vlsi_7nm = VLSI(
    process=7,
    fo4=50,
    V_wire=10,
    W_track=0.2,
    A_track=0.2,
    bf16_fma=math.inf,
    fp32_adder=math.inf,
)

vlsi_constants = {'12nm': vlsi_12nm, '7nm': vlsi_7nm}


# =============================================================================
# SRAMs
# =============================================================================

available_srams_12nm: List[SRAM] = [
    SRAM(process=12, depth=1024, width=128, x=100.0, y=200.0),
    SRAM(process=12, depth=1024, width=32,  x=100.0, y=200.0),
    SRAM(process=12, depth=1024, width=64,  x=100.0, y=200.0),
    SRAM(process=12, depth=2048, width=128, x=100.0, y=200.0),
    SRAM(process=12, depth=2048, width=32,  x=100.0, y=200.0),
    SRAM(process=12, depth=2048, width=64,  x=100.0, y=200.0),
    SRAM(process=12, depth=4096, width=128, x=100.0, y=200.0),
    SRAM(process=12, depth=4096, width=32,  x=100.0, y=200.0),
    SRAM(process=12, depth=4096, width=64,  x=100.0, y=200.0),
    SRAM(process=12, depth=512,  width=128, x=100.0, y=200.0),
    SRAM(process=12, depth=512,  width=32,  x=100.0, y=200.0),
    SRAM(process=12, depth=512,  width=64,  x=100.0, y=200.0),
    SRAM(process=12, depth=8192, width=32,  x=100.0, y=200.0),
]

# =============================================================================
# 7nm Components (scalled from 12nm)
# =============================================================================

scaling_12nm_to_7nm = (7 / 12) ** 2

# alu_7nm = ALU(
#    process=7,
#    bf16_fma=alu_12nm.bf16_fma * scaling_12nm_to_7nm,
#    fp32_adder=alu_12nm.fp32_adder * scaling_12nm_to_7nm,
# )

# =============================================================================
# 7nm SRAMs (obtained by converting 12nm SRAMs)
# =============================================================================

# Source: https://fuse.wikichip.org/news/7343/iedm-2022-did-we-just-witness-the-death-of-sram/
#       16NM = 0.074
#       7NM = 0.027
#       RATIO = 0.027 / 0.074 = 0.365
#       SIDE RATIO = sqrt(0.365) = 0.604
sram_side_ratio_16nm_to_7nm = 0.604

available_srams_7nm: List[SRAM] = [
    SRAM(
        process=7,
        depth=s.depth,
        width=s.width,
        x=s.x * sram_side_ratio_16nm_to_7nm,
        y=s.y * sram_side_ratio_16nm_to_7nm,
    )
    for s in available_srams_12nm
]

available_srams = {'12nm': available_srams_12nm,  '7nm': available_srams_7nm, }
