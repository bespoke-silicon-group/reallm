from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass
from Base import Base
from Constants import PicoJoules, Joules, EnergyConstants

if TYPE_CHECKING:
    from .System import System

@dataclass
class TokenEnergy(Base):
    system: System
    ctx_len: int

    joules_per_token: Optional[Joules] = None
    picojoules_per_token: Optional[PicoJoules] = None

    constants: EnergyConstants = EnergyConstants()

    def update(self) -> None:
        num_chips = self.system.num_chips
        d_model = self.system.model.d
        n_layers = self.system.model.num_layers
        bytes_per_word = self.system.model.bytes_per_number
        if self.system.server.package.hbm:
            # if there is HBM, we will use HBM for kv cache and weight
            weight_mem_energy = self.system.server.package.hbm.pj_per_byte
            kvcache_mem_energy = self.system.server.package.hbm.pj_per_byte
        else:
            # otherwise, we will use SRAM for weight and DRAM for kv cache
            weight_mem_energy = self.constants.sram
            kvcache_mem_energy = self.constants.sram

        num_weights_per_layer = 12 * d_model * d_model
        num_weights_total = n_layers * num_weights_per_layer
        num_weights_per_chip = num_weights_total / num_chips
        weights_GB = num_weights_per_chip * bytes_per_word / (2**30)

        num_kvcache_per_layer = 2 * d_model * (self.ctx_len - 1)
        num_kvcache_total = n_layers * num_kvcache_per_layer
        num_kvcache_per_chip = num_kvcache_total / num_chips
        kvcache_GB = num_kvcache_per_chip * bytes_per_word / (2**30)

        weight_mem_energy_GB = weight_mem_energy * (2**30)
        kvcache_mem_energy_GB = kvcache_mem_energy * (2**30)

        gemm_fma_energy = num_weights_per_chip * self.constants.fma_fp16
        matmul_fma_energy = num_kvcache_per_chip * self.constants.fma_fp16
        weight_mem_energy = weight_mem_energy_GB * weights_GB
        kvcache_mem_energy = kvcache_mem_energy_GB * kvcache_GB

        total_energy:PicoJoules = gemm_fma_energy + matmul_fma_energy + weight_mem_energy + kvcache_mem_energy
        self.picojoules_per_token = total_energy
        self.joules_per_token = total_energy / 1e12
