#!/usr/bin/env python3
import math
from typing import List
from micro_arch_sim.sram import SRAM, MEMORY
from micro_arch_sim.vlsi import VLSI

from vlsi_numbers.magic_numbers import available_srams_12nm, available_srams_7nm, vlsi_12nm, vlsi_7nm


# return capacity in MB
def design_memory(
    area: float,  # um^2
    bandwidth: int,  # bits per cycle
    word_width: int = 16,  # bits per word
    vlsi_params: VLSI = vlsi_12nm,
    available_srams: List[SRAM] = available_srams_12nm,
) -> float:
    area_sorted_srams = sorted(available_srams, key=lambda x: x.area)
    crossbar_ports = math.ceil(bandwidth / word_width)

    xbar_double_track = 2 * vlsi_params.W_track  # use a double track for SI
    xbar_word_width = (
        xbar_double_track * word_width
    )  # width per word for just the parallel wires
    xbar_total_width = xbar_word_width * crossbar_ports

    best_memory: MEMORY = None
    best_noc_area: float = None
    best_noc_padding_only: float = None
    best_noc_noc_only: float = None
    for sram in area_sorted_srams:
        banks_x = math.ceil(bandwidth / sram.width)
        banks_y = 1
        best_memory_for_sram: MEMORY = None
        best_noc_area_for_sram: float = None
        best_noc_padding_only_for_sram: float = None
        best_noc_area_for_sram: float = None
        while True:
            memory = MEMORY(
                base=sram,
                banks_x=banks_x,
                banks_y=banks_y,
            )
            min_side_len = min(memory.x, memory.y)
            max_side_len = max(memory.x, memory.y)

            if xbar_total_width < min_side_len:
                noc_area = xbar_total_width * min_side_len
                memory_padding_area = 0.0
            elif xbar_total_width < max_side_len:
                noc_area = xbar_total_width * max_side_len
                memory_padding_area = 0.0
            else:
                noc_area = xbar_total_width * xbar_total_width
                memory_padding_area = min_side_len * (xbar_total_width - max_side_len)

            if memory.area + noc_area + memory_padding_area > area:
                break
            else:
                best_memory_for_sram = memory
                best_noc_noc_only_for_sram = noc_area
                best_noc_padding_only_for_sram = memory_padding_area
                best_noc_area_for_sram = noc_area + memory_padding_area
                banks_y += 1
        # end while

        if best_memory_for_sram is not None:
            if best_memory is None or best_memory.bits < best_memory_for_sram.bits:
                best_memory = best_memory_for_sram
                best_noc_area = best_noc_area_for_sram
                best_noc_noc_only = best_noc_noc_only_for_sram
                best_noc_padding_only = best_noc_padding_only_for_sram
    # end for

    # print (best_memory)
    # print(best_noc_area)
    # print(best_noc_noc_only)
    # print(best_noc_padding_only)

    # total_area = best_memory.area + best_noc_area

    # print((best_memory.bits/8/1024/1024) / (total_area/1000000))
    if best_memory is None:
        return None
    else:
        return best_memory.bits / 8 / 1024 / 1024


if __name__ == "__main__":
    import sys
    cap_MB: float = design_memory(
        area=float(sys.argv[1]),
        bandwidth=int(sys.argv[2]),
        word_width=int(sys.argv[3]),
        vlsi_params=vlsi_7nm,
        available_srams=available_srams_7nm,
    )
    print("Capacity in MB " + str(cap_MB))
