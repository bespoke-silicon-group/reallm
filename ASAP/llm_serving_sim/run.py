from simulator import Simulator
from hardware_sim import HardwareSim
from hardware import Hardware, H100, Our_3D, System_Num_Nodes
from scheduler import Scheduler
from model import Model, llama405, llama70, opt175
from typing import List, Optional
import os
import logging

def run_simulator(eval_model: Model, 
                  hardware_node: Hardware,
                  scheduler_algos: List[str],
                  workloads: List[str],
                  req_rates: List[int],
                  ctx_lens: List[str],
                  sim_method: str,
                  prefill_chunk: Optional[int] = None,
                  num_nodes: Optional[int] = None,
                  end_time: int = 500,
                  start_reqs: int = 0,
                  end_reqs: int = 30000,
                  overwrite: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)

    if sim_method == "roofline":
        io_algo = ""
        result_dir = "results"
    elif sim_method == "llmcompass":
        io_algo = "multishot"
        result_dir = "lc_results"

    for workload in workloads:
        for ctx_len in ctx_lens:
            for req_rate in req_rates:
                if 'simple' in workload:
                    ctx_len = workload.split("_")[1]
                    ratio = workload.split("_")[2]
                    trace = f"traces/simple/rr_{ctx_len}_{ratio}.csv"
                elif 'synthetic' in workload:
                    ctx_len = workload.split("_")[1]
                    ratio = workload.split("_")[2]
                    trace = f"traces/synthetic/{ctx_len}_{ratio}_{req_rate}.csv"
                else:
                    if ctx_len == '8k' or 'k' in workload:
                        trace = f"traces/rr_{workload}_{req_rate}.csv"
                    else:
                        trace = f"traces/rr_{workload}{ctx_len}_{req_rate}.csv"
                        workload = f"{workload}{ctx_len}"
                for scheduler_algo in scheduler_algos:
                    if num_nodes is None:
                        num_nodes = System_Num_Nodes[eval_model.name][ctx_len]
                    hw_name = f"{num_nodes}-{hardware_node.name}-{scheduler_algo}"
                    if 'synthetic' in workload:
                        output_dir = f"{result_dir}/{eval_model.name}/{ctx_len}_{ratio}_{req_rate}/{hw_name}"
                    else:
                        output_dir = f"{result_dir}/{eval_model.name}/rr_{workload}_{req_rate}/{hw_name}"
                    if not overwrite and os.path.exists(output_dir):
                        logging.info(f"{output_dir} exists, skipping")
                        continue

                    eval_hardware = Hardware(node=hardware_node, num_nodes=num_nodes, io_algo=io_algo)
                    hardware_sim = HardwareSim(hardware=eval_hardware,
                                               method=sim_method,
                                               scheduler_algo=scheduler_algo,
                                               max_ctx_len=int(ctx_len[:-1]) * 1000)
                    if "mixed-sarathi" in scheduler_algo:
                        if len(scheduler_algo.split("-")) == 3:
                            chunk_size = scheduler_algo.split("-")[2]
                            if chunk_size == "dynamic":
                                # key is req_rate, value is chunk size
                                prefill_chunks = {0: 384, 15: 512, 16: 576, 17: 640, 
                                                  18: 896, 19: 1024, 20: 1536, 21: 2048}
                                prefill_chunk = 2048
                                for key, value in prefill_chunks.items():
                                    if req_rate <= key:
                                        prefill_chunk = value
                                        break
                            else:
                                prefill_chunk = int(chunk_size)
                        else:
                            if prefill_chunk is None:
                                if '128k' in workload:
                                    prefill_chunk = 1280
                                elif '64k' in workload:
                                    prefill_chunk = 1024
                                else:
                                    prefill_chunk = 384
                        scheduler = Scheduler(algo="mixed-sarathi", prefill_chunk=prefill_chunk)
                    elif "prefetch-mixed" in scheduler_algo:
                        # chunk size to able to prefetch 1 layer of KV
                        if len(scheduler_algo.split("-")) == 3:
                            chunk_size = scheduler_algo.split("-")[2]
                            if chunk_size == "dynamic":
                                # key is req_rate, value is chunk size
                                prefill_chunks = {0: 384, 15: 512, 16: 576, 17: 640, 
                                                  18: 896, 19: 1024, 20: 1536, 21: 2048}
                                prefill_chunk = 2048
                                for key, value in prefill_chunks.items():
                                    if req_rate <= key:
                                        prefill_chunk = value
                                        break
                            else:
                                prefill_chunk = int(chunk_size)
                        else:
                            if prefill_chunk is None:
                                prefill_chunks = {16: 384, 32: 1024, 48: 1280, 64: 1536}
                                prefill_chunk = prefill_chunks[eval_hardware.num_nodes]
                        scheduler = Scheduler(algo="prefetch-mixed",
                                              prefill_chunk=prefill_chunk,
                                              mem_3d_size=eval_hardware.mem_3d_size)
                    elif scheduler_algo == "prefetch-thread":
                        # num_nodes, chunk size to able to prefetch 3D memory fully (900tflops)
                        # 16,        640
                        # 32,        1024
                        # 48,        1280
                        # 64,        1536
                        if prefill_chunk is None:
                            prefill_chunks = {16: 480, 32: 1024, 48: 1280, 64: 1536}
                            prefill_chunk = prefill_chunks[eval_hardware.num_nodes]
                        scheduler = Scheduler(algo="prefetch-thread", 
                                              prefill_chunk=prefill_chunk,
                                              mem_3d_size=eval_hardware.mem_3d_size)
                    else:
                        scheduler = Scheduler(algo=scheduler_algo)

                    simulator = Simulator(model=eval_model,
                                          trace=trace,
                                          scheduler=scheduler, 
                                          hardware_sim=hardware_sim, 
                                          end_time=end_time,
                                          start_reqs=start_reqs,
                                          end_reqs=end_reqs,
                                          result_dir=result_dir)

                    simulator.run()

    print("========== All Simulations Done! ==========")