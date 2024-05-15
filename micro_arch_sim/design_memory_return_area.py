#!/usr/bin/env python3
import math
from typing import List
from micro_arch_sim.sram import SRAM, MEMORY
from micro_arch_sim.vlsi import VLSI

from vlsi_numbers.magic_numbers import available_srams_12nm, available_srams_7nm, vlsi_12nm, vlsi_7nm


# return area in um2
def design_memory_return_area(
    capacity: float, # MB
    bandwidth: int,  # bits per cycle
    word_width: int = 16,  # bits per word
    vlsi_params: VLSI = vlsi_12nm,
    available_srams: List[SRAM] = available_srams_12nm,
) -> float:
    capacity_bits = capacity * 1024 * 1024 * 8
    capacity_sorted_srams = sorted(available_srams, key=lambda x: x.bits, reverse=True)
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
    for sram in capacity_sorted_srams:
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

            if memory.bits > capacity_bits:
                break
            else:
                best_memory_for_sram = memory
                best_noc_noc_only_for_sram = noc_area
                best_noc_padding_only_for_sram = memory_padding_area
                best_noc_area_for_sram = noc_area + memory_padding_area
                banks_y += 1
        # end while

        if best_memory_for_sram is not None:
            if best_memory is None or (best_memory.area + best_noc_area) > (best_memory_for_sram.area + best_noc_area_for_sram):
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
        # return best_memory.bits / 8 / 1024 / 1024
        return best_memory.area + best_noc_area


if __name__ == "__main__":
    import sys
    area_um2: float = design_memory_return_area(
        capacity=float(sys.argv[1]),
        bandwidth=int(sys.argv[2]),
        word_width=int(sys.argv[3]),
        vlsi_params=vlsi_7nm,
        available_srams=available_srams_7nm,
    )
    print("Area in um2 " + str(area_um2))
    print("Area in mm2 " + str(area_um2 / 1e6))
