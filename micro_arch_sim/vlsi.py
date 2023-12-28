from dataclasses import dataclass


@dataclass
class VLSI:
    process: int  # tech node (nm)

    fo4: float
    W_track: float  # average routing track width (nm)
    A_track: float  # average routing track width (nm)
    V_wire: float  # um / ps

    bf16_fma: float  # um^2
    fp32_adder: float  # um^2
