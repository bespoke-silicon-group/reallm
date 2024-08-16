from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, TYPE_CHECKING, Tuple
from .Base import Base, TensorShape
from .Mapping import Mapping
from .IO import IO
from .TCO import TCO
from LLMCompass.design_space_exploration.dse import template_to_system, read_architecture_template
from LLMCompass.software_model.transformer import (
    TransformerBlockInitComputationTP,
    TransformerBlockAutoRegressionTP,
)
from LLMCompass.software_model.utils import data_type_dict, Tensor
from LLMCompass.hardware_model.device import Device
import math

if TYPE_CHECKING:
    from .System import System
    from .Chip import Chip

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
    prefill_throughput: Optional[float] = None # the throughput of prefill stage, in tokens/sec
    prefill_utilization: Optional[float] = None
    prefill_core_energy: Optional[Energy] = None # prefill total core energy
    prefill_power: Optional[float] = None # prefill power, in watts
    prefill_srv_tco: Optional[TCO] = None # TCO given the generate utilization, per server
    prefill_tco_per_token: Optional[float] = None # TCO per token during prefill, in $/token

    generate_latency: Optional[float] = None # generate latency for generate_len tokens, in sec
    generate_throughput: Optional[float] = None # the peak throughput of generate stage, in tokens/sec
    generate_utilization: Optional[float] = None
    generate_throughput_per_chip: Optional[float] = None # the peak throughput of generate stage per chip, in tokens/sec
    generate_core_energy: Optional[Energy] = None # generate total core energy
    generate_power: Optional[float] = None # generate power, in watts
    generate_srv_tco: Optional[TCO] = None # TCO given the generate utilization, per server
    generate_tco_per_token: Optional[float] = None # TCO per token during generate, in $/token

    srv_tco: Optional[TCO] = None # TCO given the generate utilization, per server
    tco_per_token: Optional[float] = None # TCO per token at the peak generate throughput, in $/token

    def update(self) -> None:
        """
        Calculate the latency of prefill and generate stages.
        We adopt the same micro-batch pipeline-parallel schedule as DeepSpeed Inference.
        (https://arxiv.org/pdf/2207.00032.pdf)
        """
        self.asplos_version = self.system.asplos_version
        if self.system.server.package.chip.dataflow == 'llmcompass':
            self.llmcompass = True
            hw_specs = read_architecture_template(f'outputs/{self.system.server.package.chip.chip_id}/llmcompass.json')
            self.lc_system = template_to_system(hw_specs)
        else:
            self.llmcompass = False
        if self.update_on_init:
            if self.mapping.dynamic:
                if self.mapping.sub_ctx_len < self.prefill_len + self.generate_len:
                    sub_generate_len = self.mapping.sub_ctx_len - self.prefill_len
                else:
                    sub_generate_len = self.generate_len
                sub_sys = replace(self.system,
                                  num_servers=self.system.sub_sys_num_servers,
                                  eval_len = [self.prefill_len, sub_generate_len],
                                  sub_sys_num_servers=None,
                                  update_on_init=False
                                  )
                sub_sys_mapping = Mapping(t=self.mapping.sub_t,
                                          p=self.mapping.sub_p,
                                          micro_batch=self.mapping.sub_micro_batch,
                                          prefill_micro_batch=self.mapping.sub_prefill_micro_batch,
                                         dynamic=False)
                sub_sys_perf = Performance(system=sub_sys, 
                                           batch=self.mapping.sub_batch,
                                           mapping=sub_sys_mapping, 
                                           prefill_len=self.prefill_len, 
                                           generate_len=sub_generate_len,
                                           update_on_init=False
                                           )
                sub_sys_perf.prefill_eval()
                sub_sys_perf.generate_eval()
                num_sub_sys = self.mapping.num_sub_sys
                self.prefill_latency = sub_sys_perf.prefill_latency
                self.prefill_throughput = sub_sys_perf.prefill_throughput * num_sub_sys
                self.prefill_utilization = sub_sys_perf.prefill_utilization
                self.prefill_core_energy = Energy(fma=sub_sys_perf.prefill_core_energy.fma * num_sub_sys,
                                                  mem=sub_sys_perf.prefill_core_energy.mem * num_sub_sys,
                                                  comm=sub_sys_perf.prefill_core_energy.comm * num_sub_sys
                                                  )
                self.prefill_power = sub_sys_perf.prefill_power * num_sub_sys
                self.prefill_bottleneck = sub_sys_perf.prefill_bottleneck

                self.mapping.dynamic = False
                # TODO: Add KV all to all latency
            else:
                self.prefill_eval()

            self.generate_eval()

            self.srv_tco, self.prefill_tco_per_token, self.generate_tco_per_token = self._get_tco()

            self.prefill_srv_tco = self.srv_tco
            self.generate_srv_tco = self.srv_tco
            self.tco_per_token = self.generate_tco_per_token
    
    def prefill_eval(self) -> None:
        micro_batch_latency = self._get_micro_batch_latency('prefill')
        self.prefill_latency = micro_batch_latency.total + (self.batch / self.mapping.prefill_micro_batch - 1) * micro_batch_latency.pipeline_stage
        self.prefill_throughput = self.batch * self.prefill_len / self.prefill_latency

        sys_peak_flops = self.system.perf * self.prefill_latency
        real_flops = self.system.model.get_prefill_flops(self.prefill_len) * self.batch
        self.prefill_utilization = real_flops / sys_peak_flops

        self.prefill_core_energy = self._get_core_energy('prefill')
        self.prefill_power = self.prefill_core_energy.total / self.prefill_latency + self.system.other_tdp

        self.prefill_bottleneck = self._get_bottleneck(micro_batch_latency)
        
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
        self.generate_latency = micro_batch_latency.total + (self.generate_len - 1) * avg_token_latency
        self.generate_throughput = 1 / avg_token_latency * self.batch
        self.generate_throughput_per_chip = self.generate_throughput / self.system.num_chips

        sys_peak_flops = self.system.perf
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

        self.generate_core_energy = self._get_core_energy('generate')
        self.generate_power = self.generate_core_energy.total / self.generate_latency + self.system.other_tdp

        self.generate_bottleneck = self._get_bottleneck(micro_batch_latency)

    def _get_bottleneck(self, lat: MicroBatchLatency) -> str:
        t_io = lat.communication
        t_mem = 0
        t_comp = 0
        for k in ['atten_qkv', 'atten_matmul1', 'atten_matmul2', 'atten_fc', 'fc1', 'fc2']:
            mm = getattr(lat, k)
            if mm.block_ldst_time > mm.block_comp_time:
                t_mem += (mm.block_ldst_time * lat.num_layers)
            else:
                t_comp += (mm.block_comp_time * lat.num_layers)
        if t_io > t_mem and t_io > t_comp:
            return 'Interconnect'
        elif t_mem > t_io and t_mem > t_comp:
            return 'Memory'
        elif t_comp > t_io and t_comp > t_mem:
            return 'Compute'
    
    def _get_tco(self) -> Tuple[TCO, float, float]:
        '''
        Calculate the server TCO.
        '''
        prefill_ratio = self.prefill_latency / (self.prefill_latency + self.generate_latency)
        generate_ratio = self.generate_latency / (self.prefill_latency + self.generate_latency)
        utilization = self.generate_utilization * generate_ratio + self.prefill_utilization * prefill_ratio
        srv_power = (self.generate_power * generate_ratio + self.prefill_power * prefill_ratio) / self.system.num_servers
        inference_throughput = self.batch / (self.prefill_latency + self.generate_latency)

        if self.system.energy_model:
            srv_tco = TCO(constants=self.system.server.tco_constants,
                          server_tdp=srv_power,
                          server_cost=self.system.server.cost,
                          server_life=self.system.server.constants.SrvLife)
        else:
            srv_tco = TCO(constants=self.system.server.tco_constants,
                          server_tdp=self.system.server.tdp * utilization,
                          server_cost=self.system.server.cost,
                          server_life=self.system.server.constants.SrvLife)
        if self.asplos_version:
            # this is a bug in the asplos version
            srv_tco.fix_part -= self.srv_tco.srv_opex
            srv_tco.power_part += (self.srv_tco.srv_opex * utilization)
            srv_tco.total = self.srv_tco.fix_part + self.srv_tco.power_part
        srv_life_sec = self.system.server.constants.SrvLife * 365 * 24 * 3600
        tco_per_sec = srv_tco.total * self.system.num_servers / srv_life_sec
        tco_per_inference = tco_per_sec / inference_throughput
        tco_per_input_token = tco_per_inference / (self.batch * self.prefill_len)
        tco_per_output_token = tco_per_inference / (self.batch * self.generate_len)

        return srv_tco, tco_per_input_token, tco_per_output_token
    
    def _get_core_energy(self, stage: str) -> Energy:
        '''
        Calculate the total core energy of prefill or generate.
        It includes the energy of FMA, memory access, and communication. 
        Memory access includes the energy of reading weights and kv from SRAM or HBM for once, 
        and reading activation from SRAM for once.
        Communication only includes the energy of all-reduce on the chip to chip link.
        This will be the lower bound of the real energy consumption.

        :param stage: the stage of inference, either 'prefill' or 'generate'
        :return: in the form of Energy dataclass
        '''
        constants = self.system.server.energy_constants
        d_model = self.system.model.d
        n_layers = self.system.model.num_layers
        bytes_per_word = self.system.model.bytes_per_number

        if self.system.server.package.num_hbm_stacks > 0:
            # if there is HBM, we will use HBM for kv cache and weight
            weight_mem_energy = self.system.server.package.hbm.pj_per_byte
            kvcache_mem_energy = self.system.server.package.hbm.pj_per_byte
        elif self.system.server.package.chip.mem_3d_vaults > 0:
            weight_mem_energy = self.system.server.package.mem_3d.pj_per_byte
            kvcache_mem_energy = self.system.server.package.mem_3d.pj_per_byte
        else:
            # otherwise, we will use SRAM for weight and DRAM for kv cache
            weight_mem_energy = constants.sram_wgt
            kvcache_mem_energy = constants.sram_wgt

        num_weights_per_layer = 12 * d_model * d_model
        num_weights_total = n_layers * num_weights_per_layer
        weight_mem_energy = num_weights_total * bytes_per_word * weight_mem_energy

        # For activation: 
        # FC: 1 in Q,K,V projection, 1 in post-atten, 1 in FF1, 4 in FF2
        # Matmul: 2 in attention matmul
        # TODO: double check, add the case when d_head * d_model != d
        if stage == 'prefill':
            num_acts_per_layer_fc = 7 * d_model * self.prefill_len * self.mapping.prefill_micro_batch
            num_acts_per_layer_matmul = 2 * d_model * self.prefill_len * self.mapping.prefill_micro_batch
            num_acts_per_layer = num_acts_per_layer_fc + num_acts_per_layer_matmul
            num_acts_total = n_layers * num_acts_per_layer

            num_kvcache_per_layer = 2 * d_model * self.prefill_len
            num_kvcache_total = n_layers * num_kvcache_per_layer

            gemm_fma_energy = num_weights_total * self.prefill_len * self.mapping.prefill_micro_batch * constants.fma_fp16
            matmul_fma_energy = num_kvcache_total * self.prefill_len * self.mapping.prefill_micro_batch * constants.fma_fp16

            # 2 all-reduce per layer
            num_allreduce_per_layer = 2 * d_model * self.prefill_len * self.mapping.prefill_micro_batch * self.mapping.t
            num_allreduce_total = n_layers * num_allreduce_per_layer

        elif stage == 'generate':
            num_acts_per_layer_fc = 7 * d_model * 1 * self.mapping.micro_batch
            num_acts_per_layer_matmul = 2 * d_model * 1 * self.mapping.micro_batch
            num_acts_per_layer = num_acts_per_layer_fc + num_acts_per_layer_matmul
            num_acts_total = n_layers * num_acts_per_layer

            num_kvcache_per_layer = 2 * d_model * (self.prefill_len + self.generate_len / 2)
            num_kvcache_total = n_layers * num_kvcache_per_layer

            gemm_fma_energy = num_weights_total * 1 * self.mapping.micro_batch * constants.fma_fp16
            matmul_fma_energy = num_kvcache_total * 1 * self.mapping.micro_batch * constants.fma_fp16

            # 2 all-reduce per layer
            num_allreduce_per_layer = 2 * d_model * 1 * self.mapping.micro_batch * self.mapping.t
            num_allreduce_total = n_layers * num_allreduce_per_layer

        acts_energy = num_acts_total * bytes_per_word * constants.sram_act
        kvcache_mem_energy = num_kvcache_total * bytes_per_word * kvcache_mem_energy

        mem_energy = weight_mem_energy + acts_energy + kvcache_mem_energy
        fma_energy = gemm_fma_energy + matmul_fma_energy

        if self.system.server.package.chip.chip2chip_io:
            link_pj_per_byte = self.system.server.package.chip.chip2chip_io.pj_per_byte
        else:
            link_pj_per_byte = self.system.server.package.chip.pkg2pkg_io.pj_per_byte
        comm_energy = num_allreduce_total * bytes_per_word * link_pj_per_byte

        if stage == 'prefill':
            num_iters = self.batch / self.mapping.prefill_micro_batch
        elif stage == 'generate':
            num_iters = self.batch / self.mapping.micro_batch * self.generate_len

        return Energy(fma=fma_energy * num_iters / 1e12, 
                      mem=mem_energy * num_iters / 1e12 , 
                      comm=comm_energy * num_iters / 1e12)

    def _get_micro_batch_latency(self, stage: str) -> MicroBatchLatency:
        """
        Calculate the latency of one micro-batch inference.
        We adopt weight stationary, all weights/kv-cache will be read only once.

        TODO: According to Figure 3 in 'Efficiently Scaling Transformer Inference' 
        (https://arxiv.org/pdf/2211.05102.pdf), we may need to consider the weight-gathered 
        layout when tokens per micro-batch is large.

        :param stage: the stage of inference, either 'prefill' or 'generate'
        :return: the latency of one inference, in sec
        """
        d = self.system.model.d
        num_heads = self.system.model.num_heads
        d_head = self.system.model.d_head
        if self.system.model.d_ff is None:
            self.system.model.d_ff = 4 * d
        d_ff = self.system.model.d_ff
        t = self.mapping.t
        data_bytes = self.system.model.bytes_per_number

        if stage == 'prefill':
            # By default, micro batch size for prefill is 1
            micro_batch = self.mapping.prefill_micro_batch
            activation_row = self.prefill_len * micro_batch
        else:
            micro_batch = self.mapping.micro_batch
            activation_row = micro_batch

        activation_col = d
        activation_size = activation_row * activation_col


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
        stage2stage_latency = activation_size * data_bytes / (stage2stage_bw * self.system.io_bandwidth_efficiency)+ self.system.server.io.init_time

        # Set the compute performance efficiency
        self.system.server.package.chip.compute_perf_efficiency = self.system.compute_perf_efficiency

        ##################################################################
        # We adopt the same partitioning scheme as Megatron-LM
        # (https://arxiv.org/pdf/1909.08053.pdf, Figure 3 (b))
        # The difference is that we do not require the tensor parallelism 
        # size t, which is also the number of chips per pipeline stage,
        # to smaller than number of heads. When t > num of heads,
        # we need one more all-reduce for Q * K_T, both Q and K_T of a 
        # head are partitioned across (t / num_heads) chips.
        ##################################################################
        # Attention Layer
        # attention FC to get Q, K, and V matrix
        # all chips get the complete activation of size (activation_row, d)
        # each chip has weight of size (d, 3d/t) --> d rows, 3d/t cols
        # so for each chip, we have (activation_row, d) * (d, 3d/t)
        d_atten = d_head * num_heads
        A = (activation_row, d)
        B = (d, math.ceil(3 * d_atten / t))
        atten_qkv_latency = MatmulLatency(A, B, self.system.server.package.chip, self.system.weight_bw_per_chip)
        ##################################################################
        # attention matmul: Q * K_T, 
        # prefill: (micro_batch, ctx_len, d / t) * (micro_batch, d / t, ctx_len)
        # generate: (micro_batch, 1, d / t) * (micro_batch, d / t, ctx_len)
        # in ASPLOS submission
        if self.asplos_version:
            A = (micro_batch, 1, d / t)
            B = (micro_batch, d / t, micro_batch * 20)
        else:
            if stage == 'prefill':
                A = (micro_batch, self.prefill_len, math.ceil(d_atten / t))
                B = (micro_batch, math.ceil(d_atten / t), self.prefill_len)
            else:
                A = (micro_batch, 1, math.ceil(d_atten / t))
                B = (micro_batch, math.ceil(d_atten / t), self.prefill_len + self.generate_len / 2)
        atten_matmul1_latency = MatmulLatency(A, B, self.system.server.package.chip, self.system.kv_bw_per_chip)
        if t > self.system.model.num_heads:
            # DOUBLE CHECK HERE
            chips_per_head = int(t / self.system.model.num_heads)
            if self.asplos_version:
                atten_allreduce_size = micro_batch * 20
            elif stage == 'prefill':
                atten_allreduce_size = self.prefill_len
            else:
                atten_allreduce_size = self.prefill_len + self.generate_len / 2
            atten_communication_latency_1 = self._get_allreduce_latency(chips_per_head, atten_allreduce_size * data_bytes, collective_links, self.system.allreduce_algo)
        else:
            atten_communication_latency_1 = 0.0
        ##################################################################
        # attention matmul: S * V, 
        # prefill: (micro_batch, ctx_len, ctx_len) * (micro_batch, ctx_len, d / t)
        # generate: (micro_batch, 1, ctx_len) * (micro_batch, ctx_len, d / t)
        if stage == 'prefill':
            A = (micro_batch, self.prefill_len, self.prefill_len)
            B = (micro_batch, self.prefill_len, math.ceil(d_atten / t))
        else:
            A = (micro_batch, 1, self.prefill_len + self.generate_len / 2)
            B = (micro_batch, self.prefill_len + self.generate_len / 2, math.ceil(d_atten / t))
        atten_matmul2_latency = MatmulLatency(A, B, self.system.server.package.chip, self.system.kv_bw_per_chip)
        # no need for all-reduce even when t > num_heads, since each chip has the complete tensor of score, and part of V, 
        # it is able to compute part of the atten_out O
        ##################################################################
        # attention FC to get the output
        # each chip get the activation of size (activation_row, d/t)
        # each chip has weight of size (d/t, d) --> d/t rows, d cols, 
        # so for each chip, we have (activation_row, d/t) * (d/t, d)
        A = (activation_row, math.ceil(d_atten / t))
        B = (math.ceil(d_atten / t), d)
        atten_fc_latency = MatmulLatency(A, B, self.system.server.package.chip, self.system.weight_bw_per_chip)
        ##################################################################
        # all-reduce, DOUBLE CHECK HERE
        atten_all_to_all_latency = self._get_allreduce_latency(t, d * data_bytes, collective_links, self.system.allreduce_algo) / 2 # half of the latency of all-reduce
        atten_all_reduce_latnecy = self._get_allreduce_latency(t, activation_row * d * data_bytes, collective_links, self.system.allreduce_algo)
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
        A = (activation_row, d)
        B = (d, math.ceil(d_ff / t))
        if 'glu' in self.system.model.act:
            A = (2, activation_row, d)
            B = (2, d, math.ceil(d_ff / t))
        fc1_latency = MatmulLatency(A, B, self.system.server.package.chip, self.system.weight_bw_per_chip)
        ##################################################################
        # FC 2
        # each chip get the activation of size (activation_row, 4d/t)
        # each chip has weight of size (4d/t, d) --> 4d/t rows, d cols, 
        # each for chip we have (activation_row, 4d/t) * (4d/t, d)
        A = (activation_row, math.ceil(d_ff / t))
        B = (math.ceil(d_ff / t), d)
        fc2_latency = MatmulLatency(A, B, self.system.server.package.chip, self.system.weight_bw_per_chip)
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
            fc_communication_latency = self._get_allreduce_latency(t, activation_size * data_bytes, collective_links, self.system.allreduce_algo)
        
        if self.llmcompass:
            if self.system.model.act == 'gelu':
                activation = 'gelu'
            else:
                activation = 'silu'
            if stage == 'prefill':
                lc_model = TransformerBlockInitComputationTP(
                    d_model=d,
                    n_heads=self.system.model.num_heads,
                    device_count=t,
                    data_type=data_type_dict["fp16"],
                    activation=activation,
                    d_ffn=d_ff,
                    n_kv_heads=self.system.model.num_heads // self.system.model.heads_per_kv_cache,
                    use_flash_attn=True
                )
                _ = lc_model(Tensor([micro_batch, self.prefill_len, d], data_type_dict["fp16"]))
            else:
                lc_model = TransformerBlockAutoRegressionTP(
                    d_model=d,
                    n_heads=self.system.model.num_heads,
                    device_count=t,
                    data_type=data_type_dict["fp16"],
                    activation=activation,
                    d_ffn=d_ff,
                    n_kv_heads=self.system.model.num_heads // self.system.model.heads_per_kv_cache,
                    use_flash_attn=True
                )
                _ = lc_model(Tensor([micro_batch, 1, d], data_type_dict["fp16"]), self.prefill_len + self.generate_len // 2)

            latency = lc_model.compile_and_simulate(self.lc_system, 'heuristic-GPU-fast')
            qkv_latency, q_mul_k_latency, a_mul_v_latency, h_matmul0_latency, h1_matmul1_latency, h2_matmul2_latency, h3_matmul3_latency, swi_mul_latency, softmax_latency, layernorm_latency, _, act_latency, allreduce_latency, _ = lc_model.simluate_log.split(", ")
            atten_qkv_latency.time = float(qkv_latency)
            atten_matmul1_latency.time = float(q_mul_k_latency)
            atten_matmul2_latency.time = float(a_mul_v_latency)
            atten_fc_latency.time = float(h_matmul0_latency)
            fc1_latency.time = float(h1_matmul1_latency) + float(swi_mul_latency) + float(h3_matmul3_latency)
            fc2_latency.time = float(h2_matmul2_latency)
            softmax_latency = float(softmax_latency)
            layernorm_latency = float(layernorm_latency)
            act_latency = float(act_latency)
        else:
            softmax_latency = 0
            layernorm_latency = 0
            act_latency = 0

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
                                                fc_communication_latency,
                                                softmax_latency,
                                                layernorm_latency,
                                                act_latency)

        return micro_batch_latency 

    def _get_allreduce_latency(self, num_nodes: int, num_bytes: int, collective_links: list[IO] | IO, algorithm: str) -> float:
        """""
        Calculate the latency of all-reduce.

        :param num_nodes: number of nodes
        :param num_bytes: number of bytes of each node to all-reduce
        :param collective_links: the links used for collective communication
        :param algorithm: the algorithm used for all-reduce, either 'ring', '2d_ring', or '3d_ring'
        :return: the latency of all-reduce, in sec
        """

        if num_nodes == 1:
            return 0

        if collective_links is IO:
            collective_links = [collective_links]
        bottleneck_link = min(collective_links, key=lambda link: link.bandwidth)
        bandwidth = bottleneck_link.bandwidth * self.system.io_bandwidth_efficiency
        init_time = bottleneck_link.init_time

        alpha = init_time
        beta = 1 / bandwidth
        p = num_nodes
        n = num_bytes

        if algorithm == 'ring':
            t = 2 * (p - 1) * (alpha + beta * n / p)
        elif algorithm == '2d_ring':
            sqrt_p = math.ceil(math.sqrt(p))
            t = 2 * (sqrt_p - 1) * (2 * alpha + beta * n / sqrt_p)
        elif algorithm == '3d_ring':
            cbrt_p = math.ceil(p ** (1/3))
            t = 2 * (cbrt_p - 1) * (3 * alpha + beta * n / cbrt_p)
        elif algorithm == 'local_4d_16':
            t_local_ar = 8 * alpha + beta * n
            if p == 16:
                return t_local_ar
            else:
                p_global = math.ceil(p / 16)
                cbrt_p = math.ceil(p_global ** (1/3))
                t_global = 2 * (cbrt_p - 1) * (3 * alpha + beta * n / cbrt_p) 
                t_local_bc = 16 * alpha + beta * n + 2 * math.sqrt(alpha * beta * n)
                return t_local_ar + t_global + t_local_bc
        elif algorithm == 'local_2d_16':
            t_local_ar = 6 * (2 * alpha + beta * n / 4)
            if p == 16:
                return t_local_ar
            else:
                p_global = math.ceil(p / 16)
                cbrt_p = math.ceil(p_global ** (1/3))
                t_global = 2 * (cbrt_p - 1) * (3 * alpha + beta * n / cbrt_p) 
                t_local_bc = 16 * alpha + beta * n + 2 * math.sqrt(alpha * beta * n)
                return t_local_ar + t_global + t_local_bc
        elif algorithm == 'local_ring_16':
            t_local_ar = 30 * alpha + 1.875 * beta * n
            if p == 16:
                return t_local_ar
            else:
                p_global = math.ceil(p / 16)
                cbrt_p = math.ceil(p_global ** (1/3))
                t_global = 2 * (cbrt_p - 1) * (3 * alpha + beta * n / cbrt_p) 
                t_local_bc = 16 * alpha + beta * n + 2 * math.sqrt(alpha * beta * n)
                return t_local_ar + t_global + t_local_bc
        elif algorithm == 'local_ring_8':
            t_local_ar = 14 * alpha + 1.75 * beta * n
            if p == 8:
                return t_local_ar
            else:
                p_global = math.ceil(p / 8)
                cbrt_p = math.ceil(p_global ** (1/3))
                t_global = 2 * (cbrt_p - 1) * (3 * alpha + beta * n / cbrt_p) 
                t_local_bc = 16 * alpha + beta * n + 2 * math.sqrt(alpha * beta * n)
                return t_local_ar + t_global + t_local_bc
        else:
            raise ValueError('Invalid algorithm for all-reduce')
        return t

    def _get_ring_all_reduce_latency(self, num_nodes: int, num_bytes: int, collective_links: list[IO] | IO) -> float:
        """
        Deprecated, use _get_all_reduce_latency instead.

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
        bandwidth = bottleneck_link.bandwidth * self.system.io_bandwidth_efficiency
        init_time = bottleneck_link.init_time

        chunk_bytes = num_bytes / num_nodes
        num_transfers = num_nodes - 1
        all_gather_time = num_transfers * chunk_bytes / bandwidth + init_time
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
class MatmulLatency(Base):
    """
    Get the latency of a matrix multiplication A * B, based on Scott Davidson's code.
    A: [..., m, n]
    B: [..., n, k]
    The first few dimensions ... in the two tensors should be the same, meaning multiple 2-D matrix multiplications
    """
    A: TensorShape
    B: TensorShape
    chip: Chip
    weight_bw: int
    data_bytes: int = 2

    # derived metrics
    I: Optional[int] = None
    J: Optional[int] = None
    K: Optional[int] = None
    K_per_core: Optional[int] = None

    I_hat: Optional[int] = None
    J_hat: Optional[int] = None
    K_hat: Optional[int] = None

    I_bar: Optional[int] = None
    J_bar: Optional[int] = None
    K_bar: Optional[int] = None

    A_shape: Optional[TensorShape] = None
    B_shape: Optional[TensorShape] = None
    O_shape: Optional[TensorShape] = None

    block_ldst_time: Optional[float] = None
    block_comp_time: Optional[float] = None

    num_ops: Optional[int] = None
    utilization: Optional[float] = None
    time: Optional[float] = None

    def update(self) -> None:
        self.dataflow = self.chip.dataflow

        assert len(self.A) == len(self.B)
        assert len(self.A) >= 2
        assert self.A[-1] == self.B[-2]
        self.I = self.A[-2]
        self.J = self.A[-1]
        self.K = self.B[-1]
        if len(self.A) > 2:
            stacks = math.prod(self.A[:-2])
        else:
            stacks = 1
        if self.dataflow == 'WS':
            if stacks > self.chip.num_sa:
                stacks_per_core = math.ceil(stacks / self.chip.num_sa)
                self.K_per_core = self.K
            else:
                stacks_per_core = 1
                cores_per_stack = math.floor(self.chip.num_sa / stacks)
                self.K_per_core = math.ceil(self.K / cores_per_stack)
            
            self.I_hat = math.ceil(self.I / self.chip.acc_depth / 2)
            self.J_hat = math.ceil(self.J / self.chip.sa_width)
            self.K_hat = math.ceil(self.K_per_core / self.chip.sa_width)

            self.I_bar = math.ceil(self.I / self.I_hat) # used to be called S
            self.J_bar = math.ceil(self.J / self.J_hat) # account for partial block in the SA
            self.K_bar = math.ceil(self.K_per_core / self.K_hat) # account for partial block in the SA

            self.A_shape = (stacks_per_core, self.I_hat, self.J_hat, self.I_bar, self.J_bar)
            self.B_shape = (stacks_per_core, self.J_hat, self.K_hat, self.J_bar, self.K_bar)
            self.O_shape = (stacks_per_core, self.I_hat, self.K_hat, self.I_bar, self.K_bar)

            self.block_ldst_time = (self.J_bar * self.K_bar) / (self.weight_bw / self.chip.num_sa)
            self.block_comp_time = math.ceil(max(2 * self.chip.sa_width + self.I_bar, 2 * self.I_bar) / 2) / self.chip.freq
            max_block_time = max(self.block_ldst_time, self.block_comp_time)

            self.time = stacks_per_core * (self.block_ldst_time + self.block_comp_time + (self.I_hat * self.J_hat * self.K_hat - 1) * max_block_time)
            
            self.num_ops = stacks * self.I * self.J * self.K * 2
            self.utilization = self.num_ops / (self.time * self.chip.perf)
        elif self.dataflow == 'roofline':
            self.num_ops = stacks * self.I * self.J * self.K * 2
            self.block_comp_time = self.num_ops / (self.chip.perf * self.chip.compute_perf_efficiency)
            self.block_ldst_time = stacks * (self.J * self.K) * self.data_bytes / self.weight_bw
            self.time = max(self.block_comp_time, self.block_ldst_time)
            self.utilization = self.num_ops / (self.time * self.chip.perf)
        elif self.dataflow == 'llmcompass':
            self.num_ops = stacks * self.I * self.J * self.K * 2
            self.block_comp_time = 0.0
            self.block_ldst_time = 0.0
            self.time = 0.0
            self.utilization = 0.0
        else:
            raise NotImplementedError

    # def _call_dramsim3(self) -> float:
    #     num_bytes = self.J_bar * self.K_bar * self.data_bytes

    #     config = configparser.ConfigParser()
    #     config.read(self.dram_config)
    #     channels = int(config.get('system', 'channels'))
    #     bus_bits = int(config.get('system', 'bus_width'))
    #     total_bus_bytes = channels * bus_bits / 8
    #     columns = int(config.get('dram_structure', 'columns'))

    #     num_cycles = math.ceil(num_bytes / total_bus_bytes) * 100

    #     addr = 0
    #     base_cycle = 0

    #     f = open('trace.txt', 'w')
    #     for cyc in range(num_cycles):
    #         for ch in range(channels):
    #             f.write(f'{hex(addr)} READ {base_cycle + cyc}\n')
    #             addr += columns
    #     f.close()
        
    #     sim_cycles = num_cycles + 1
    #     os.system(f'make -s -f dramsim3.mak run CONFIG={self.dram_config} TRACE=trace.txt CYCLE={sim_cycles}')
 
    #     with open('dramsim3.json') as json_file:
    #         data = json.load(json_file)
    #         tot_bw = 0.0
    #         for ch in range(channels):
    #             tot_bw += data[str(ch)]['average_bandwidth']

    #     return tot_bw

@dataclass
class MicroBatchLatency(Base):
    p: int
    num_layers: int
    stage2stage: float
    atten_qkv: MatmulLatency
    atten_matmul1: MatmulLatency
    atten_communication1: float
    atten_matmul2: MatmulLatency
    atten_fc: MatmulLatency
    atten_communication2: float
    fc1: MatmulLatency
    fc2: MatmulLatency
    fc_communication: float
    softmax: float = 0.0
    layernorm: float = 0.0
    act: float = 0.0

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
        layer_compute = self.atten_qkv.time + self.atten_matmul1.time + self.atten_matmul2.time + self.atten_fc.time + self.fc1.time + self.fc2.time + self.softmax + self.layernorm + self.act
        layer_communication = self.atten_communication1 + self.atten_communication2 + self.fc_communication
        layers_per_stage = math.ceil(self.num_layers / self.p)
        self.pipeline_stage = (layer_compute + layer_communication) * layers_per_stage + self.stage2stage
        self.total = self.pipeline_stage * self.p
        self.compute = layer_compute * layers_per_stage * self.p
        self.communication = self.total - self.compute
        self.utilization = self.compute / self.total

        self.stage2stage_us = self.stage2stage * 1e6
        self.atten_qkv_us = self.atten_qkv.time * 1e6
        self.atten_matmul1_us = self.atten_matmul1.time * 1e6
        self.atten_communication1_us = self.atten_communication1 * 1e6
        self.atten_matmul2_us = self.atten_matmul2.time * 1e6
        self.atten_fc_us = self.atten_fc.time * 1e6
        self.atten_communication2_us = self.atten_communication2 * 1e6
        self.fc1_us = self.fc1.time * 1e6
        self.fc2_us = self.fc2.time * 1e6
        self.fc_communication_us = self.fc_communication * 1e6
        self.pipeline_stage_us = self.pipeline_stage * 1e6
        self.total_us = self.total * 1e6
        self.compute_us = self.compute * 1e6
        self.communication_us = self.communication * 1e6
        self.softmax_us = self.softmax * 1e6
        self.layernorm_us = self.layernorm * 1e6
        self.act_us = self.act * 1e6

@dataclass
class Energy(Base):
    # all in joules
    fma: float
    mem: float
    comm: float
    other: float = 0.0

    total: Optional[float] = None

    def update(self) -> None:
        self.total = self.fma + self.mem + self.comm + self.other
