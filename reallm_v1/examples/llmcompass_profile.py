# %%
import sys
import time
sys.path.append("./LLMCompass")
from LLMCompass.software_model.utils import Tensor, data_type_dict
from LLMCompass.software_model.matmul import Matmul as Matmul
from LLMCompass.software_model.fast_matmul import Matmul as FastMatmul
from LLMCompass.hardware_model.device import device_dict


from line_profiler import profile

@profile
def profile_matmul():
    K = 12288
    N = K
    M = 2**8
    pcb = device_dict["A100_80GB_fp16"]
    compile_mode="heuristic-GPU"
    data_type=data_type_dict["fp16"]
    input1 = Tensor([M, K], data_type)
    input2 = Tensor([K, N], data_type)

    # mm = Matmul(data_type)
    # _ = mm(input1, input2)
    # uarch_lat = mm.compile_and_simulate(pcb, compile_mode)

    mm = FastMatmul(data_type)
    _ = mm(input1, input2)
    start = time.perf_counter()
    fast_uarch_lat = mm.compile_and_simulate(pcb, compile_mode)
    fast_sim_time = time.perf_counter() - start

    # print(f"Matrix size {M}x{K}x{N} finished")
    # print(f"uarch latency: {uarch_lat:.3e}")
    # print(f"fast uarch latency: {fast_uarch_lat:.3e}")

profile_matmul()
# %%
if __name__ == "__main__":
    profile_matmul()