import heapq
import logging

from collections import defaultdict
from scheduler import Scheduler
from hardware_sim import HardwareSim
from request import Request
from model import Model
from task import PrefillTask, DecodeTask
import pickle
import os
import numpy as np
# import utils

class Simulator:
    def __init__(self,
                 model: Model,
                 trace: str,
                 scheduler: Scheduler,
                 hardware_sim: HardwareSim,
                 end_time: int,
                 start_reqs: int,
                 end_reqs: int,
                 exp_dist_path: str = None,
                 result_dir: str = 'results'
                 ):
        self.model = model
        self.trace = trace
        self.scheduler = scheduler
        self.hardware_sim = hardware_sim
        self.end_time = end_time
        self.start_reqs = start_reqs
        self.end_reqs = end_reqs
        self.result_dir = result_dir
        self.accept_new_req = True
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        self.time = 0.0
        self.prefill_fifo = []
        self.decode_fifo = []
        self.requests = dict() # request_id -> Request
        logging.info(f"TraceSimulator initialized for trace {trace} with scheduler {hardware_sim.scheduler_algo}")

        self.load_trace()

        if exp_dist_path is not None:
            self.exp_distribution = np.load(exp_dist_path)
        else:
            self.exp_distribution = None

    def load_trace(self):
        """
        Load requests from the trace file to the prefill FIFO.
        """
        with open(self.trace, "r") as f:
            # request_id,request_type,application_id,arrival_timestamp,batch_size,prompt_size,token_size
            for line in f:
                if line.startswith("request_id"):
                    continue
                request_id, request_type, application_id, arrival_timestamp, batch_size, prompt_size, token_size = line.strip().split(",")
                request = Request(int(request_id), self.model, float(arrival_timestamp), int(prompt_size), int(token_size))
                self.requests[int(request_id)] = request
                prefill_task = PrefillTask(request, float(arrival_timestamp))
                self.prefill_fifo.append(prefill_task)

    def run(self):
        while self.time < self.end_time:
            if self.prefill_fifo == [] and self.decode_fifo == [] and self.scheduler.is_done():
                break
            if self.prefill_fifo != []:
                if self.prefill_fifo[0].req.id >= self.end_reqs:
                    break
            self.step()
        
        self.logging_results()
    
    def step(self):
        """
        Run one step of the simulation.
        """
        kernel, new_time = self.scheduler.run(self.time, self.prefill_fifo, self.decode_fifo, self.accept_new_req)
        self.time = new_time
        if self.exp_distribution is not None:
            latency, accept_new_req = self.hardware_sim.run(kernel, self.exp_distribution)
        else:
            latency, accept_new_req = self.hardware_sim.run(kernel)
        self.accept_new_req = accept_new_req
        self.time += latency

        self.scheduler.update(self.time, self.requests, self.decode_fifo)

    def logging_results(self, detailed=False):
        model_name = self.model.name
        if len(self.trace.split("/")[-1].split(".")) == 2:
            workload_name = self.trace.split("/")[-1].split(".")[0]
        else:
            workload_name = self.trace.split("/")[-1].split(".")[0] + "." + self.trace.split("/")[-1].split(".")[1]
        hw_name = f"{self.hardware_sim.hardware.num_nodes}-{self.hardware_sim.hardware.node.name}-{self.hardware_sim.scheduler_algo}-{self.hardware_sim.hardware.parallelism}"
        # check if the directory exists
        if not os.path.exists(f"{self.result_dir}/{model_name}"):
            os.mkdir(f"{self.result_dir}/{model_name}")
        if not os.path.exists(f"{self.result_dir}/{model_name}/{workload_name}"):
            os.mkdir(f"{self.result_dir}/{model_name}/{workload_name}")
        if not os.path.exists(f"{self.result_dir}/{model_name}/{workload_name}/{hw_name}"):
            os.mkdir(f"{self.result_dir}/{model_name}/{workload_name}/{hw_name}")
        logging.info(f'Final time: {self.time}')
        # Dump the simulation results
        file_name = f"{self.result_dir}/{model_name}/{workload_name}/{hw_name}/sim_results.csv"
        with open(file_name, "w") as f:
            f.write("request_id,arrival_time,finish_time\n")
            for req_id, request in self.requests.items():
                if req_id < self.start_reqs:
                    continue
                if request.t_start > self.end_time:
                    break
                # f.write(f"{req_id},{request.t_start}," + ",".join([str(t) for t in request.t_end]) + "\n")
                f.write(f"{req_id},{request.t_start},")
                for t in request.t_end:
                    if t is None:
                        f.write("None,")
                    else:
                        f.write(f"{t:.4f},")
                f.write("\n")
                if None in request.t_end:
                    # ????
                    # if len(request.t_end) > 100 and None not in request.t_end[:100]:
                    if len(request.t_end) > 1000 and None not in request.t_end[:1000]:
                        continue
                    else:
                        logging.info(f"Request {req_id} did not finish")
                        break
        # Dump the kernel performance to a pickle file
        # kernel_perf = self.hardware_sim.kernels_perf 
        # file_name = f"{self.result_dir}/{model_name}/{workload_name}/{hw_name}/kernel_perf.pkl"
        # with open(file_name, "wb") as f:
        #     pickle.dump(kernel_perf, f)

        # logging.info("========================================")
        # logging.info("Simulation results:")
        # for req_id, request in self.requests.items():
        #     if detailed:
        #         logging.info(f"Request {req_id}:")
        #         for i in range(request.output_len):
        #             logging.info(f"Token {i}: {request.t_end[i]}")
        #     else:
        #         # logging.info(f"Request {req_id}: {request.t_end[-1] - request.t_start}")
        #         logging.info(f"Request {req_id}: {request.t_end[-1]}")
