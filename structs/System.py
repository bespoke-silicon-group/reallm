import math
from dataclasses import dataclass
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

    # optional inputs, but at least one of them should be provided, the other two will be derived
    num_servers: Optional[int] = None # number of servers
    kv_cache_ratio: Optional[float] = None # kv cache ratio of the total memory
    max_ctx_len_batch_1: Optional[int] = None # max context length at batch size 1
    # optional inputs
    max_batch: int = 1024 # max batch size
    prefill_eval_ctx_len: int = 2048
    generate_eval_prefill_len: int = 128
    generate_eval_generate_len: int = 128

    asplos_version: bool = False

    # derived fileds
    valid: Optional[bool] = None # if the system is able to hold all the model parameters
    total_mem: Optional[int] = None # total memory, in Byte
    weight_bw_per_chip: Optional[float] = None # weight bandwidth per chip, in Byte/sec
    kv_bw_per_chip: Optional[float] = None # kv cache bandwidth per chip, in Byte/sec
    num_packages: Optional[int] = None
    num_chips: Optional[int] = None

    batches_prefill_latency_opt_mapping: Optional[dict[int, Mapping]] = None # batch size to mapping optimized for prefill latency   
    batches_generate_throughput_opt_mapping: Optional[dict[int, Mapping]] = None # batch size to mapping optimized for token generation throughput
    prefill_latency_opt_perf: Optional[Performance] = None # performance (with mapping and batch size) optimized for prefill latency for 2048 tokens
    generate_throughput_opt_perf: Optional[Performance] = None # performance (with mapping and batch size) optimized for token generation throughput

    def update(self) -> None:
        self.valid = self._hardware_update()
        if self.valid:
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
            self.total_mem = self.num_servers * self.server.total_mem
        # udpate the kv cache ratio using the number of servers
        elif self.num_servers:
            self.total_mem = self.num_servers * self.server.total_mem
            if self.total_mem < self.model.model_size_byte:
                return False
        else:
            raise ValueError('Please provide one of kv_cache_ratio, max_ctx_len_batch_a or num_servers.')
        self.kv_cache_ratio = 1 - self.model.model_size_byte / self.total_mem
        kv_cache_mem = self.kv_cache_ratio * self.total_mem
        self.max_ctx_len_batch_1 = math.floor(kv_cache_mem / self.model.kv_cache_size_per_token_byte)
        self.num_packages = self.num_servers * self.server.num_packages
        self.num_chips = self.num_servers * self.server.num_chips

        # TODO: support weight or KV cache in 3D memory
        if self.server.package.mem_3d:
            raise NotImplementedError('3D memory or side memory is not supported yet.')
        if self.server.package.hbm:
            self.weight_bw_per_chip = self.server.package.hbm.total_bandwidth / self.server.package.num_chips
            self.kv_bw_per_chip = self.server.package.hbm.total_bandwidth / self.server.package.num_chips
        else:
            self.weight_bw_per_chip = self.server.package.chip.sram_bw
            self.kv_bw_per_chip = self.server.package.chip.sram_bw
        
        return True

    def _software_update(self) -> None:
        batch = 1
        overall_opt_prefill_latency = float('inf')
        overall_opt_generate_throughput = 0.0
        self.batches_generate_throughput_opt_mapping = {}
        self.batches_prefill_latency_opt_mapping = {}
        while batch <= self.max_batch:
            # prefill latency optimization
            # from deepspeed inference, we should use large micro batch size for prefill and small micro batch size for generate
            batch_opt_prefill_latency = float('inf')
            mappings = self.gen_mappings(batch=batch, min_ctx_len=self.prefill_eval_ctx_len+1)
            for mapping in mappings:
                perf = Performance(system=self, mapping=mapping, batch=batch, prefill_len=self.prefill_eval_ctx_len, generate_len=1, update_on_init=False)
                perf.prefill_eval()
                prefill_latency = perf.prefill_latency
                if prefill_latency < batch_opt_prefill_latency:
                    batch_opt_prefill_latency = prefill_latency
                    self.batches_prefill_latency_opt_mapping[batch] = mapping   
                    if prefill_latency < overall_opt_prefill_latency:
                        overall_opt_prefill_latency = prefill_latency
                        self.prefill_latency_opt_perf = perf

            # generate throughput optimization
            batch_opt_generate_throughput = 0.0
            for mapping in self.gen_mappings(batch=batch, min_ctx_len=256):
                perf = Performance(system=self, mapping=mapping, batch=batch, prefill_len=self.generate_eval_prefill_len, generate_len=self.generate_eval_generate_len, update_on_init=False)
                perf.generate_eval()
                generate_throughput = perf.generate_throughput
                if generate_throughput > batch_opt_generate_throughput:
                    batch_opt_generate_throughput = generate_throughput
                    self.batches_generate_throughput_opt_mapping[batch] = mapping
                    if generate_throughput > overall_opt_generate_throughput:
                        overall_opt_generate_throughput = generate_throughput
                        self.generate_throughput_opt_perf = perf
            batch *= 2
        
        if self.prefill_latency_opt_perf:
            self.prefill_latency_opt_perf.update_on_init = True
            self.prefill_latency_opt_perf.update()
        if self.generate_throughput_opt_perf:
            self.generate_throughput_opt_perf.update_on_init = True
            self.generate_throughput_opt_perf.update()

    def gen_mappings(self, batch: int, min_ctx_len: int) -> list[Mapping]:
        valid_mappings = []
        for p in range(1, self.model.num_layers + 1):
            # pipeline parallelism should be a divisor of the number of layers
            if self.model.num_layers % p != 0:
                continue

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
                # iterate through all possible micro batch sizes
                micro_batch = 1
                while micro_batch <= batch:
                    valid_mappings.append(Mapping(t=t_chip, p=p, micro_batch=micro_batch))
                    micro_batch *= 2

        return valid_mappings


