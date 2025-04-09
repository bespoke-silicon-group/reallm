import math
from dataclasses import dataclass, field, replace
from typing import Optional
from .Base import Base
from .Server import Server
from .Model import Model
from .Performance import Performance
from .Mapping import Mapping

@dataclass
class System(Base):
    server: Server
    model: Model
    allreduce_algo: str = 'ring'
    sub_sys_num_servers: Optional[int] = None # or number of servers in each sub-system
    update_on_init: bool = True
    sw_update_on_init: bool = True

    # optional inputs, but at least one of them should be provided, the other two will be derived
    num_servers: Optional[int] = None # number of servers
    kv_cache_ratio: Optional[float] = None # kv cache ratio of the total memory
    max_ctx_len_batch_1: Optional[int] = None # max context length at batch size 1
    max_tco: Optional[float] = None # max TCO, running at TDP
    # optional inputs
    max_batch: int = 1024 # max batch size
    # eval_len: list[int] = field(default_factory=[128, 129]) # evaluation length for prefill and generate
    eval_len: list[int] = None # evaluation length for prefill and generate 
    energy_model: bool = True # whether to use the energy model for calculating the TCO
    compute_perf_efficiency: float = 1.0 # the ratio of the actual compute performance to the theoretical performance
    io_bandwidth_efficiency: float = 1.0 # the ratio of the actual IO bandwidth to the theoretical bandwidth
    weight_bandwidth_efficiency: float = 1.0 # the ratio of the actual weight bandwidth to the theoretical bandwidth

    asplos_version: bool = False

    # derived fileds
    valid: Optional[bool] = None # if the system is able to hold all the model parameters
    total_mem: Optional[int] = None # total memory, in Byte
    weight_bw_per_chip: Optional[float] = None # weight bandwidth per chip, in Byte/sec
    kv_bw_per_chip: Optional[float] = None # kv cache bandwidth per chip, in Byte/sec
    num_packages: Optional[int] = None
    num_chips: Optional[int] = None
    perf: Optional[int] = None # #FLOPS
    tdp: Optional[float] = None # total tdp
    core_tdp: Optional[float] = None # core tdp including chips and memories
    other_tdp: Optional[float] = None # other parts tdp

    # Optimized performance for different batch sizes
    batch_opt_prefill_lat: Optional[dict[int, Performance]] = None # batch size to optimized prefill latency performance
    batch_opt_prefill_tco: Optional[dict[int, Performance]] = None # batch size to optimized prefill TCO/Token performance
    batch_opt_generate_lat: Optional[dict[int, Performance]] = None # batch size to optimized generate latency performance
    batch_opt_generate_tco: Optional[dict[int, Performance]] = None # batch size to optimized generate TCO/Token performance

    default_mapping: Optional[Mapping] = None

    def update(self) -> None:
        if self.update_on_init:
            self.valid = self._hardware_update()
            if self.valid and self.sw_update_on_init:
                self._software_update()

    def get_perf(self, prefill_len, generate_len) -> Performance:
        return Performance(system=self, prefill_len=prefill_len, generate_len=generate_len)
    
    def _hardware_update(self) -> None:
        # update the number of servers using the kv cache ratio
        if self.kv_cache_ratio or self.max_ctx_len_batch_1:
            if self.kv_cache_ratio:
                required_mem = self.model.model_size_byte / (1 - self.kv_cache_ratio)
            else:
                required_mem = self.model.model_size_byte + self.max_ctx_len_batch_1 * self.model.kv_cache_size_per_token_byte
            if self.num_servers == None or self.num_servers * self.server.total_mem < required_mem:
                # if the number of servers is not provided, we will use the total memory to derive it
                self.num_servers = math.ceil(required_mem / self.server.total_mem)
                # TODO: Find a smarter way later, make sure the number of layers is a multiple of the number of servers
                if self.model.num_layers > self.num_servers:
                    while self.model.num_layers % self.num_servers != 0:
                        self.num_servers += 1
                else:
                    while self.num_servers % self.model.num_layers != 0:
                        self.num_servers += 1
            self.total_mem = self.num_servers * self.server.total_mem
        # udpate the kv cache ratio using the number of servers, or max TCO
        elif self.num_servers or self.max_tco:
            if not self.num_servers:
                self.num_servers = math.ceil(self.max_tco / self.server.tco.total)
            self.total_mem = self.num_servers * self.server.total_mem
            if self.total_mem < self.model.model_size_byte:
                self.invalid_reason = f'Total memory is {self.total_mem/1024/1024/1024} GB, which is less than the model size {self.model.model_size_byte/1024/1024/1024} GB.'
                return False
        else:
            raise ValueError('Please provide one of kv_cache_ratio, max_ctx_len_batch_a, num_servers or max_tco.')
        self.kv_cache_ratio = 1 - self.model.model_size_byte / self.total_mem
        kv_cache_mem = self.kv_cache_ratio * self.total_mem
        if self.max_ctx_len_batch_1 == None:
            self.max_ctx_len_batch_1 = math.floor(kv_cache_mem / self.model.kv_cache_size_per_token_byte)
        self.max_tco = self.server.tco.total * self.num_servers
        self.num_packages = self.num_servers * self.server.num_packages
        self.num_chips = self.num_servers * self.server.num_chips

        # now we only support either 3D memory, or HBM, or SRAM
        if self.server.package.chip.mem_3d_vaults > 0:
            if 'SRAM' in self.server.package.mem_3d.mem_type or 'sram' in self.server.package.mem_3d.mem_type:
                sram_3d_bw_per_chip = self.server.package.mem_3d.bandwidth * self.server.package.chip.mem_3d_vaults
                sram_3d_cap_per_chip = self.server.package.mem_3d.cap * self.server.package.chip.mem_3d_vaults
                total_sram_per_chip = self.server.package.chip.sram + sram_3d_cap_per_chip
                self.weight_bw_per_chip = sram_3d_bw_per_chip * sram_3d_cap_per_chip / total_sram_per_chip + \
                                          self.server.package.chip.sram_bw * self.server.package.chip.sram / total_sram_per_chip
            elif 'DRAM' in self.server.package.mem_3d.mem_type or 'dram' in self.server.package.mem_3d.mem_type:
                self.weight_bw_per_chip = self.server.package.dram_bw_per_chip
            else:
                raise ValueError('Unsupported 3D memory type.')
        elif self.server.package.num_hbm_stacks > 0:
            self.weight_bw_per_chip = self.server.package.dram_bw_per_chip
        else:
            self.weight_bw_per_chip = self.server.package.chip.sram_bw

        self.weight_bw_per_chip *= self.weight_bandwidth_efficiency
        self.kv_bw_per_chip = self.weight_bw_per_chip
        
        self.perf = self.num_servers * self.server.perf
        self.tdp = self.num_servers * self.server.tdp
        self.core_tdp = self.num_servers * self.server.core_tdp
        self.other_tdp = self.tdp - self.core_tdp
        
        return True

    def _software_update(self) -> None:
        if self.eval_len == None:
            self.eval_len = [128, 129]
        prefill_len = self.eval_len[0]
        generate_len = self.eval_len[1]
        total_len = prefill_len + generate_len

        self.batch_opt_prefill_lat = dict()
        self.batch_opt_prefill_tco = dict()
        self.batch_opt_generate_lat = dict()
        self.batch_opt_generate_tco = dict()

        batch = 1
        while batch <= self.max_batch:
            # from deepspeed inference, we should use large micro batch size for prefill and small micro batch size for generate
            batch_opt_prefill_lat = float('inf')
            batch_opt_prefill_tco = float('inf')
            batch_opt_generate_lat = float('inf')
            batch_opt_generate_tco = float('inf')
            if self.default_mapping:
                mapping = Mapping(**self.default_mapping)
                if mapping.micro_batch == 0:
                    mapping.micro_batch = batch
                if mapping.prefill_micro_batch == 0:
                    mapping.prefill_micro_batch = batch
                if mapping.p == 1 and mapping.t != self.num_chips:
                    mapping.t = self.num_chips
                mappings = [mapping]
            else:
                mappings = self.gen_mappings(batch=batch, min_ctx_len=total_len)
                if self.sub_sys_num_servers and (self.sub_sys_num_servers < self.num_servers):
                    if self.num_servers % self.sub_sys_num_servers != 0:
                        raise ValueError('sub_sys_num_servers should be a divisor of num_servers.')
                    num_sub_sys = self.num_servers // self.sub_sys_num_servers 
                    sub_batch = batch // num_sub_sys
                    if sub_batch >= 1:
                        sub_sys = replace(self, 
                                          num_servers=self.sub_sys_num_servers, 
                                          eval_len=[self.eval_len[0], 1],
                                          sub_sys_num_servers=None,
                                          max_ctx_len_batch_1=None, 
                                          kv_cache_ratio=None,
                                          update_on_init=False)
                        sub_sys._hardware_update()
                        if sub_sys.max_ctx_len_batch_1:
                            max_sub_ctx_len = sub_sys.max_ctx_len_batch_1 // sub_batch
                            if max_sub_ctx_len >= self.eval_len[0]:
                                sub_ctx_len = min(max_sub_ctx_len, total_len)
                                new_mappings = []
                                # sub_mappings = sub_sys.gen_mappings(batch=sub_batch, min_ctx_len=sub_ctx_len)
                                # for sub_mapping in sub_mappings:
                                #     for mapping in mappings:
                                #         new_mapping = replace(mapping, 
                                #                               dynamic=True, 
                                #                               num_sub_sys=num_sub_sys, 
                                #                               sub_batch=sub_batch,
                                #                               sub_micro_batch=sub_mapping.micro_batch,
                                #                               sub_prefill_micro_batch=sub_mapping.prefill_micro_batch,
                                #                               sub_t=sub_mapping.t,
                                #                               sub_p=sub_mapping.p,
                                #                               sub_ctx_len=sub_ctx_len)
                                #         new_mappings.append(new_mapping)
                                for mapping in mappings:
                                    new_mapping = replace(mapping, 
                                                          dynamic=True, 
                                                          num_sub_sys=num_sub_sys, 
                                                          sub_batch=sub_batch,
                                                          sub_micro_batch=sub_batch,
                                                          sub_prefill_micro_batch=sub_batch,
                                                          sub_t=sub_sys.num_chips,
                                                          sub_p=1,
                                                          sub_ctx_len=sub_ctx_len)
                                    new_mappings.append(new_mapping)
                                if len(new_mappings) > 0:
                                    mappings = new_mappings

            for mapping in mappings:
                perf = Performance(system=self, mapping=mapping, batch=batch, 
                                   prefill_len=prefill_len, generate_len=generate_len)
                if perf.prefill_latency < batch_opt_prefill_lat:
                    batch_opt_prefill_lat = perf.prefill_latency
                    self.batch_opt_prefill_lat[batch] = perf
                if perf.prefill_tco_per_token < batch_opt_prefill_tco:
                    batch_opt_prefill_tco = perf.prefill_tco_per_token
                    self.batch_opt_prefill_tco[batch] = perf
                if perf.generate_latency < batch_opt_generate_lat:
                    batch_opt_generate_lat = perf.generate_latency
                    self.batch_opt_generate_lat[batch] = perf
                if perf.generate_tco_per_token < batch_opt_generate_tco: 
                    batch_opt_generate_tco = perf.generate_tco_per_token
                    self.batch_opt_generate_tco[batch] = perf
            batch *= 2

    def gen_mappings(self, batch: int, min_ctx_len: int) -> list[Mapping]:
        valid_mappings = []

        def get_all_p(num_layers):
            all_p = []
            last_layers_per_pipeline = 0
            for p in range(1, num_layers + 1):
                layers_per_pipeline = math.ceil(num_layers / p)
                if layers_per_pipeline == last_layers_per_pipeline:
                    continue
                else:
                    last_layers_per_pipeline = layers_per_pipeline
                    all_p.append(p)
            return all_p

        for p in get_all_p(self.model.num_layers):
            # multiple servers per pipeline stage, t_srv needs to be integer 
            if self.num_servers >= p:
                t_srv = self.num_servers // p # int
                t_pkg = t_srv * self.server.num_packages # int
            # multiple pipeline stages per server, num_pipeline_per_server needs to be integer
            else:
                if p % self.num_servers != 0:
                    continue
                t_srv = self.num_servers / p # float
                num_pipeline_per_server = p / self.num_servers # int
                t_pkg = self.server.num_packages // num_pipeline_per_server # int
            
            # check if there's enough memory for the model parameter
            total_used_mem = t_pkg * self.server.package.total_mem * p
            if total_used_mem < (self.model.model_size_byte + batch * min_ctx_len * self.model.kv_cache_size_per_token_byte):
                continue
            else:
                t_chip = t_pkg * self.server.package.num_chips # int
                # either t_chip should be a divisor of num_heads, or num_heads should be a divisor of t_chip
                # if self.model.num_heads % t_chip != 0 and t_chip % self.model.num_heads != 0:
                #     continue
                # iterate through all possible micro batch sizes
                micro_batch = 1
                while micro_batch <= batch:
                    prefill_micro_batch = 1
                    while prefill_micro_batch <= micro_batch:
                        valid_mappings.append(Mapping(t=t_chip, p=p, 
                                                      micro_batch=micro_batch, 
                                                      prefill_micro_batch=prefill_micro_batch))
                        prefill_micro_batch *= 2
                    micro_batch *= 2
        return valid_mappings
            
