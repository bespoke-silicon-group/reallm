from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from .Base import Base
from .Mapping import Mapping
from .IO import IO
from .TCO import TCO
from .Energy import TokenEnergy
import math

if TYPE_CHECKING:
    from .System import System

@dataclass
class Performance(Base):
    system: System
    batch: int
    mapping: Mapping
    prefill_len: int
    generate_len: int
    update_on_init: bool = True
    asplos_version: bool = False

    prefill_latency: Optional[float] = None # prefill latency, in sec
    generate_latency: Optional[float] = None # generate latency for generate_len tokens, in sec
    generate_throughput: Optional[float] = None # the peak throughput of generate stage, in tokens/sec
    generate_throughput_per_chip: Optional[float] = None # the peak throughput of generate stage per chip, in tokens/sec
    max_ctx_len: Optional[int] = None # max context length, given the input hardware
    prefill_utilization: Optional[float] = None
    generate_utilization: Optional[float] = None
    overall_utilization: Optional[float] = None
    srv_tco: Optional[TCO] = None # TCO given the generate utilization, per server
    tco_per_token: Optional[float] = None # TCO per token at the peak generate throughput, in $/token


    def update(self) -> None:
        """
        Calculate the latency of prefill and generate stages.
        We adopt the same micro-batch pipeline-parallel schedule as DeepSpeed Inference.
        (https://arxiv.org/pdf/2207.00032.pdf)
        """
        self.asplos_version = self.system.asplos_version
        if self.update_on_init:
            self.prefill_eval()
            self.generate_eval()
            # When calculating the TCO, we only consider the utilization of the generate stage.
            self.tco_eval(self.generate_utilization)
    
    def prefill_eval(self) -> None:
        micro_batch_latency = self._get_micro_batch_latency('prefill')
        self.prefill_latency = micro_batch_latency.total + (self.batch / self.mapping.micro_batch - 1) * micro_batch_latency.pipeline_stage

        # TODO: calculate utilization based on flops
        # total_flops = (12 * 2 * self.system.model.d ** 2 * self.prefill_len) * self.system.model.num_layers * self.batch
        self.prefill_utilization = micro_batch_latency.compute / micro_batch_latency.total

    def generate_eval(self) -> None:
        """
        The throughput of generate stage depends on how we do micro-batch and pipeline parallelism.
        For example, assume we have batch size of 4.

        If we do 4 pipeline stages, and 2 micro-batches (- and =), we have:
        --==    --==
          --==    --==
            --==    --==
              --==    --==
        so on average it takes 8 cycels to generate 1 token for the whole batch.
        This is because to start generating the next token of a micro-batch, 
        we need to finish the previous token of that micro-batch.

        If we do 4 pipeline stages, and 4 micro-batches (-, =, + and *), we have:
        -=+*-=+*
         -=+*-=+*
          -=+*-=+*
           -=+*-=+*
        Now it's 4 cycles per token.

        If we do 2 pipeline stages, and 4 micro-batches (-, =, + and *), we have:
        --==++**--==++**
          --==++**--==++**
        Now it's 8 cycles per token again.
        This is because to start generating the next token of a micro-batch,
        we also need to wait for the first pipline stage to finish the previous token.

        Therefore, the throughput of generate stage is:
        throughput = 1 / max(micro_batch_latency, pipeline_stage_latency * num_micro_batches) * batch
        """
        micro_batch_latency = self._get_micro_batch_latency('generate')
        avg_token_latency = max(micro_batch_latency.total, micro_batch_latency.pipeline_stage * (self.batch / self.mapping.micro_batch))
        self.generate_latency = micro_batch_latency.total + self.generate_len * avg_token_latency
        self.generate_throughput = 1 / avg_token_latency * self.batch
        self.generate_throughput_per_chip = self.generate_throughput / (self.system.num_servers * self.system.server.num_chips)

        sys_peak_flops = self.system.server.perf * self.system.num_servers
        if self.asplos_version:
            throughput_flops = self.generate_throughput * self.system.model.get_generate_flops(0)
        else:
            throughput_flops = self.generate_throughput * self.system.model.get_generate_flops(self.prefill_len + self.generate_len / 2)
        # print(f'system server perf = {self.system.server.perf / 1e12}, number of servers = {self.system.num_servers}')
        # print(f'throughput flops = {throughput_flops / 1e12}, sys peak flops = {sys_peak_flops / 1e12}')
        # print(f'generate throughput = {self.generate_throughput}, batch = {self.batch}, micro_batch = {self.mapping.micro_batch}, micro_batch_latency = {micro_batch_latency.total}')
        # print(f'throughput per chip = {self.generate_throughput_per_chip}')
        # print(f'generate flops = {self.system.model.get_generate_flops(0) / 1e12}')
        # print(f'generate flops = {self.system.model.get_generate_flops(100) / 1e12}')
        # print(f'generate flops = {self.system.model.get_generate_flops(500) / 1e12}')
        # print(f'generate flops = {self.system.model.get_generate_flops(self.prefill_len + self.generate_len / 2) / 1e12}')
        self.generate_utilization = throughput_flops / sys_peak_flops
    
    def tco_eval(self, utilization) -> None:
            """
            When calculating the TCO, we use utilization * TDP as the real power consumption.
            """
            joules_per_token = TokenEnergy(system=self.system, ctx_len=self.prefill_len + self.generate_len / 2).joules_per_token
            energy_model_power = joules_per_token * self.generate_throughput # in watts
            # print(f"TCO evaluation, utilization based on FLOPS = {utilization}, \
            #       server TDP = {self.system.server.tdp * utilization}, \
            #       server core tdp = {self.system.server.core_tdp * utilization}, \
            #       energy model power = {energy_model_power}")

            # TODO: use the real power consumption

            self.srv_tco = TCO(server_tdp=self.system.server.tdp * utilization,
                               server_cost=self.system.server.cost,
                               server_life=self.system.server.constants.SrvLife)
            if self.asplos_version:
                # this is a bug in the asplos version
                self.srv_tco.fix_part -= self.srv_tco.srv_opex
                self.srv_tco.power_part += (self.srv_tco.srv_opex * utilization)
                self.srv_tco.total = self.srv_tco.fix_part + self.srv_tco.power_part
            srv_life_sec = self.system.server.constants.SrvLife * 365 * 24 * 3600
            self.tco_per_token = self.srv_tco.total * self.system.num_servers / srv_life_sec / self.generate_throughput

    def _get_overall_utilization(self) -> float:
        # TODO: calculate utilization based on prefill and generate flops, now assume it's 50%
        return 0.50
    
    def _get_micro_batch_latency(self, stage: str) -> MicroBatchLatency:
        """
        Calculate the latency of one micro-batch inference.

        :param stage: the stage of inference, either 'prefill' or 'generate'
        :return: the latency of one inference, in sec
        """
        d = self.system.model.d
        t = self.mapping.t
        data_bytes = self.system.model.bytes_per_number

        if stage == 'prefill':
            activation_row = self.mapping.micro_batch * self.prefill_len
            atten_activation_row = activation_row
        else:
            activation_row = self.mapping.micro_batch
            atten_activation_row = activation_row * (self.prefill_len + self.generate_len / 2)
            if self.asplos_version:
                atten_activation_row = activation_row * 20
        activation_col = d
        activation_size = activation_row * activation_col

        chip_perf = self.system.server.package.chip.perf # in flops/sec

        t_srv = self.system.num_servers / self.mapping.p # number of servers per pipeline stage
        t_pkg = self.system.server.num_packages * t_srv # number of packages per pipeline stage
        if t_srv > 1:
            # multiple servers per pipeline stage, collective operation includes inter-server communication
            # pipeline stage to stage also uses server to server link
            # collective_links = [self.system.server.io, self.system.server.package.io, self.system.server.package.chip.chip2chip_io]
            collective_links = [self.system.server.io, self.system.server.package.io]
            stage2stage_bw = self.system.server.io.bandwidth
        elif t_srv == 1:
            # single server per pipeline stage
            # pipeline stage to stage still uses server to server link
            # collective_links = [self.system.server.package.io, self.system.server.package.chip.io]
            collective_links = [self.system.server.package.io]
            stage2stage_bw = self.system.server.io.bandwidth
        else:
            # multiple pipeline stages per server
            if t_pkg > 1:
                # multiple packages per pipeline stage
                # collective_links = [self.system.server.package.io, self.system.server.package.chip.io]
                collective_links = [self.system.server.package.io]
            else:
                # single package per pipeline stage or multiple pipeline stages per package
                # collective_links = [self.system.server.package.chip.io]
                collective_links = [self.system.server.package.io]
            stage2stage_bw = self.system.server.package.io.bandwidth
            # pipeline stage to stage uses both package to package link and server to server link
            # assume p_srv stages per server, there are p_srv - 1 pkg to pkg links and 1 srv to srv link
            # p_srv = self.mapping.p / self.system.num_servers # this should be an integer
            # stage2stage_bw = ((p_srv - 1) * self.system.server.package.io.bandwidth + self.system.server.io.bandwidth) / p_srv

        # need to add the initialization time for each data transfer
        stage2stage_latency = activation_size * data_bytes / stage2stage_bw + self.system.server.io.init_time

        ##################################################################
        # We adopt the same partitioning scheme as Megatron-LM
        # (https://arxiv.org/pdf/1909.08053.pdf, Figure 3 (b))
        # The difference is that we do not require the tensor parallelism 
        # size t, which is also the number of chips per pipeline stage,
        # to smaller than number of heads. When t > num of heads,
        # we need one more all-reduce for Q * K_T, both Q and K_T of a 
        # head are partitioned across (t / num_heads) chips.
        # TODO: what if t % num_head != 0 or num_heads % t != 0?
        ##################################################################
        # Attention Layer
        # attention FC to get Q, K, and V matrix
        # all chips get the complete activation of size (activation_row, d)
        # each chip has weight of size (d, 3d/t) --> d rows, 3d/t cols
        # so for each chip, we have (activation_row, d) * (d, 3d/t)
        atten_qkv_latency = self._get_matmul_latency(activation_row, d, math.ceil(3 * d / t), chip_perf, self.system.weight_bw_per_chip, data_bytes)
        ##################################################################
        # attention matmul: Q * K_T, (micro_batch * ctx_len, d / t) * (d / t, micro_batch * ctx_len)
        # atten_matmul1_latency = self._get_matmul_latency(atten_activation_row, d / t, atten_activation_row, chip_perf, self.system.kv_bw_per_chip, data_bytes)
        # in ASPLOS submission
        if self.asplos_version:
            atten_matmul1_latency = self._get_matmul_latency(1, d / t, atten_activation_row, chip_perf, self.system.kv_bw_per_chip, data_bytes)
        else:
            atten_matmul1_latency = self._get_matmul_latency(1, math.ceil(d / t), atten_activation_row, chip_perf, self.system.kv_bw_per_chip, data_bytes)
        if t > self.system.model.num_heads:
            # DOUBLE CHECK HERE
            chips_per_head = int(t / self.system.model.num_heads)
            atten_communication_latency_1 = self._get_ring_all_reduce_latency(chips_per_head, atten_activation_row * data_bytes, collective_links)
        else:
            atten_communication_latency_1 = 0.0
        ##################################################################
        # attention matmul: (Q * K_T) * V, (micro_batch * ctx_len, micro_batch * ctx_len) * (micro_batch * ctx_len, d / t)
        # atten_matmul2_latency = self._get_matmul_latency(atten_activation_row, atten_activation_row, d / t, chip_perf, self.system.weight_bw_per_chip, data_bytes)
        # in ASPLOS submission
        atten_matmul2_latency = self._get_matmul_latency(1, atten_activation_row, math.ceil(d / t), chip_perf, self.system.weight_bw_per_chip, data_bytes)
        # no need for all-reduce even when t > num_heads, since each chip has the complete tensor of score, and part of V, 
        # it is able to compute part of the atten_out O
        ##################################################################
        # attention FC to get the output
        # each chip get the activation of size (activation_row, d/t)
        # each chip has weight of size (d/t, d) --> d/t rows, d cols, 
        # so for each chip, we have (activation_row, d/t) * (d/t, d)
        atten_fc_latency = self._get_matmul_latency(activation_row, math.ceil(d / t), d, chip_perf, self.system.weight_bw_per_chip, data_bytes)
        ##################################################################
        # all-reduce, DOUBLE CHECK HERE
        atten_all_to_all_latency = self._get_ring_all_reduce_latency(t, d * data_bytes, collective_links) / 2 # half of the latency of all-reduce
        atten_all_reduce_latnecy = self._get_ring_all_reduce_latency(t, activation_row * d * data_bytes, collective_links)
        if d == 18432: # PaLM, parallel Atten and FC, ring-all reduce is overlapped, but there is 2 all-to-all on activation (batch size = 1)
          atten_communication_latency_2 = 2 * atten_all_to_all_latency
        elif d == 8192: # Llama, need 2 all-to-all on activation
          atten_communication_latency_2 = atten_all_reduce_latnecy + 2 * atten_all_to_all_latency
        else:
          atten_communication_latency_2 = atten_all_reduce_latnecy

        ##################################################################
        # FC Layer
        ##################################################################
        # FC 1
        # all chips get the whole activation of size (activation_row, d)
        # each chip has weight of size (d, 4d/t) --> d rows, 4d/t cols
        # so for each chip we have (activation_row, d) * (d, 4d/t)
        fc1_latency = self._get_matmul_latency(activation_row, d, math.ceil(4 * d / t), chip_perf, self.system.weight_bw_per_chip, data_bytes)
        ##################################################################
        # FC 2
        # each chip get the activation of size (activation_row, 4d/t)
        # each chip has weight of size (4d/t, d) --> 4d/t rows, d cols, 
        # each for chip we have (activation_row, 4d/t) * (4d/t, d)
        fc2_latency = self._get_matmul_latency(activation_row, math.ceil(4 * d / t), d, chip_perf, self.system.weight_bw_per_chip, data_bytes)
        ##################################################################
        # all-reduce
        # We adopt the same partitioning scheme as EFFICIENTLY SCALING TRANSFORMER INFERENCE
        # (https://arxiv.org/pdf/2211.05102.pdf, Section 3.2.2)
        # the method has a latency of 4 / sqrt(t) * all_reduce_latency
        # TODO: do we need to consider activation stationary and adopt
        # the weight-gathered layout as described in Section 3.2.3?
        if self.asplos_version:
            fc_communication_latency = self._get_ring_all_reduce_latency(t, activation_size * data_bytes * 4 / math.floor(math.sqrt(t)), collective_links)
        else:
            # Double check this
            fc_communication_latency = self._get_ring_all_reduce_latency(t, activation_size * data_bytes * 4 / math.sqrt(t), collective_links)

        micro_batch_latency = MicroBatchLatency(self.mapping.p, 
                                                self.system.model.num_layers, 
                                                stage2stage_latency, 
                                                atten_qkv_latency, 
                                                atten_matmul1_latency, 
                                                atten_communication_latency_1, 
                                                atten_matmul2_latency, 
                                                atten_fc_latency, 
                                                atten_communication_latency_2, 
                                                fc1_latency, 
                                                fc2_latency, 
                                                fc_communication_latency)

        return micro_batch_latency 
    
    def _get_matmul_latency(self, m: int, n: int, k: int, flops: int, weight_bw: int, data_bytes: int = 2, util: float = 1.0) -> float:
        """
        Get the latency of a matrix multiplication (m, n) * (n, k) = (m, k).

        :param m: the number of rows of the first matrix
        :param n: the number of columns of the first matrix and the number of rows of the second matrix
        :param k: the number of columns of the second matrix
        :param flops: the number of floating point operations per second
        :param weight_bw: the bandwidth of the weight transfer (the second matrix (n, k)), in bytes/sec
        :param data_bytes: the number of bytes per number
        :param util: the utilization, assume it's 1.0 for now
        :return: the latency of the matrix multiplication, in sec
        """
        compute_delay = m * n * k * 2 / flops / util # 2 means 2 flops per mac
        if self.asplos_version:
            weight_bw *= 1.024
        memory_delay = n * k * data_bytes / weight_bw
        return max(compute_delay, memory_delay)


    def _get_ring_all_reduce_latency(self, num_nodes: int, num_bytes: int, collective_links: list[IO] | IO) -> float:
        """
        Calculate the latency of ring all-reduce, based on the formula in the Appendix A.1 
        of paper Efficiently Scaling Transformer Inference (https://arxiv.org/pdf/2211.05102.pdf)
        We add the support for multiple different links and the data transfer initialization time.

        :param num_nodes: number of nodes
        :param num_bytes: number of bytes of each node to all-reduce
        :param collective_links: the links used for collective communication
        :return: the latency of ring all-reduce, in sec
        """
        if collective_links is IO:
            collective_links = [collective_links]
        bottleneck_link = min(collective_links, key=lambda link: link.bandwidth)
        bandwidth = bottleneck_link.bandwidth
        init_time = bottleneck_link.init_time

        chunk_bytes = num_bytes / num_nodes
        num_transfers = num_nodes - 1
        all_gather_time = num_transfers * (chunk_bytes / bandwidth + init_time)
        reduce_scatter_time = all_gather_time
        all_reduce_time = all_gather_time + reduce_scatter_time

        # ASPLOS version
        if self.asplos_version:
            if num_nodes == 1:
                return 0
            else:
                all_reduce_time = 2 * (num_bytes / bandwidth + init_time)
        
        return all_reduce_time
    

@dataclass
class MicroBatchLatency(Base):
    p: int
    num_layers: int
    stage2stage: float
    atten_qkv: float
    atten_matmul1: float
    atten_communication1: float
    atten_matmul2: float
    atten_fc: float
    atten_communication2: float
    fc1: float
    fc2: float
    fc_communication: float

    # derived metrics
    pipeline_stage: Optional[float] = None
    total: Optional[float] = None
    compute: Optional[float] = None
    communication: Optional[float] = None
    utilization: Optional[float] = None

    stage2stage_us: Optional[float] = None
    atten_qkv_us: Optional[float] = None
    atten_matmul1_us: Optional[float] = None
    atten_communication1_us: Optional[float] = None
    atten_matmul2_us: Optional[float] = None
    atten_fc_us: Optional[float] = None
    atten_communication2_us: Optional[float] = None
    fc1_us: Optional[float] = None
    fc2_us: Optional[float] = None
    fc_communication_us: Optional[float] = None
    pipeline_stage_us: Optional[float] = None
    total_us: Optional[float] = None
    compute_us: Optional[float] = None
    communication_us: Optional[float] = None

    def update(self) -> None:
        layer_compute = self.atten_qkv + self.atten_matmul1 + self.atten_matmul2 + self.atten_fc + self.fc1 + self.fc2
        layer_communication = self.atten_communication1 + self.atten_communication2 + self.fc_communication
        self.pipeline_stage = (layer_compute + layer_communication) * (self.num_layers / self.p) + self.stage2stage
        self.total = self.pipeline_stage * self.p
        self.compute = layer_compute * self.num_layers
        self.communication = self.total - self.compute
        self.utilization = self.compute / self.total

        self.stage2stage_us = self.stage2stage * 1e6
        self.atten_qkv_us = self.atten_qkv * 1e6
        self.atten_matmul1_us = self.atten_matmul1 * 1e6
        self.atten_communication1_us = self.atten_communication1 * 1e6
        self.atten_matmul2_us = self.atten_matmul2 * 1e6
        self.atten_fc_us = self.atten_fc * 1e6
        self.atten_communication2_us = self.atten_communication2 * 1e6
        self.fc1_us = self.fc1 * 1e6
        self.fc2_us = self.fc2 * 1e6
        self.fc_communication_us = self.fc_communication * 1e6
        self.pipeline_stage_us = self.pipeline_stage * 1e6
        self.total_us = self.total * 1e6
        self.compute_us = self.compute * 1e6
        self. communication_us = self.communication * 1e6