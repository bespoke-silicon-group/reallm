from dataclasses import dataclass

Joules = float
PicoJoules = float


@dataclass
class EnergyModelParams:
    sram: PicoJoules = 1.25
    dram: PicoJoules = 80.0
    hbm2: PicoJoules = 31.2
    stacked_dram: PicoJoules = 18.72

    fma_fp16: PicoJoules = 2.75


def model_energy_per_token(
        d_model: int,
        bytes_per_word: int,
        n_layers: int,
        ctx_len: int,
        batch: int,
        k_parallel: int,
        t_parallel: int,

        weight_mem_energy: PicoJoules,
        kvcache_mem_energy: PicoJoules,
        fma_energy: PicoJoules,
) -> Joules:

    num_chips = k_parallel * t_parallel

    num_weights_per_layer = 12 * d_model * d_model
    num_weights_total = n_layers * num_weights_per_layer
    num_weights_per_chip = num_weights_total / num_chips
    weights_GB = num_weights_per_chip * bytes_per_word / (2**30)

    num_kvcache_per_layer = 2 * batch * d_model * (ctx_len - 1)
    num_kvcache_total = n_layers * num_kvcache_per_layer
    num_kvcache_per_chip = num_kvcache_total / num_chips
    kvcache_GB = num_kvcache_per_chip * bytes_per_word / (2**30)

    weight_mem_energy_GB = weight_mem_energy * (2**30)
    kvcache_mem_energy_GB = kvcache_mem_energy * (2**30)

    gemm_fma_energy = num_weights_per_chip * fma_energy
    matmul_fma_energy = num_kvcache_per_chip * fma_energy
    weight_mem_energy = weight_mem_energy_GB * weights_GB
    kvcache_mem_energy = kvcache_mem_energy_GB * kvcache_GB

    total_energy:PicoJoules = gemm_fma_energy + matmul_fma_energy + weight_mem_energy + kvcache_mem_energy
    return total_energy / 1e12


## Example
if __name__ == "__main__":
    energy = EnergyModelParams()
    print(
        model_energy_per_token(
            d_model=12288,
            bytes_per_word=2,
            n_layers=96,
            ctx_len=100,
            batch=64,
            k_parallel=1,
            t_parallel=1,
            weight_mem_energy=energy.sram,
            kvcache_mem_energy=energy.stacked_dram,
            fma_energy=energy.fma_fp16,
        )
    )
