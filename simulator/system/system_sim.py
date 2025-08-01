import os
import multiprocessing
from .simulator import Simulator
from .hardware_sim import HardwareSim
from .hardware import Hardware, H100
from .scheduler import Scheduler

def run_system_sim(model, trace,
                   hw_node_name, num_nodes, parallelism, io_algo, 
                   scheduler_algo, prefill_chunk=2048,
                   sim_method='roofline', # roofline, llmcompass
                   end_reqs=500, # should be set based on the request rate and workload
                   max_ctx_len = 8192*4,
                   workspace_dir = 'workspace/',
                   ):

    # This actually doesn't matter ??
    eval_hardware = Hardware(node=H100, num_nodes=num_nodes, 
                             parallelism=parallelism, io_algo=io_algo,
    )

    hardware_sim = HardwareSim(
        hardware=eval_hardware,
        method=sim_method,
        scheduler_algo=scheduler_algo,
        max_ctx_len = max_ctx_len,
    )

    scheduler = Scheduler(
        algo=scheduler_algo,
        prefill_chunk=prefill_chunk,
    )

    eval_hardware.node.name = hw_node_name

    sim = Simulator(
        model = model,
        trace=trace,
        scheduler=scheduler,
        hardware_sim=hardware_sim,
        end_time=500,
        start_reqs=0,
        end_reqs=end_reqs,
        workspace_dir=workspace_dir,
    )
    sim.run()
