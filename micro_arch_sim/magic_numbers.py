from typing import List
from vlsi import VLSI
from sram import SRAM
import math

# =============================================================================
# Fake 12nm VLSI Constants
# =============================================================================

vlsi_12nm = VLSI(
    process=12,
    fo4=30.0,
    V_wire=10.0,
    W_track=0.1,
    A_track=0.1,
    bf16_fma=1000.0,
    fp32_adder=1000.0,
)

vlsi_7nm = VLSI(
    process=7,
    fo4=math.inf,
    V_wire=math.inf,
    W_track=0.1,
    A_track=math.inf,
    bf16_fma=math.inf,
    fp32_adder=math.inf,
)

# =============================================================================
# Fake 12nm SRAMs
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
    SRAM(process=12, depth=512, width=128,  x=100.0, y=200.0),
    SRAM(process=12, depth=512, width=32,   x=100.0, y=200.0),
    SRAM(process=12, depth=512, width=64,   x=100.0, y=200.0),
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
