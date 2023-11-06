from dataclasses import dataclass
from typing import Optional
from .Base import Base

@dataclass
class Model(Base):
    name: str
    num_layers: int
    d: int
    num_heads: int
    heads_per_kv_cache: int = 1 # number of heads that share the same kv cache, 1 <= heads_per_kv_cache <= num_heads and should be a divisor of num_heads
                                # 1 means each head has its own kv cache, i.e., multihead attention
                                # num_heads means all heads share the same kv cache, i.e., multiquery attention
                                # 1 < heads_per_kv_cache < num_heads means grouped query attention
    bytes_per_number: int = 2

    model_size: Optional[int] = None # number of parameters in the model
    model_size_byte: Optional[int] = None # number of parameters in the model, in bytes
    kv_cache_size_per_token: Optional[int] = None # kv cache size per token
    kv_cache_size_per_token_byte: Optional[int] = None # kv cache size per token, in bytes

    def update(self) -> None:
        self.model_size = self._get_model_size()
        self.model_size_byte = self.model_size * self.bytes_per_number
        self.kv_cache_size_per_token = self._get_kv_cache_size()
        self.kv_cache_size_per_token_byte = self.kv_cache_size_per_token * self.bytes_per_number

    def _get_model_size(self) -> int:
        return self.num_layers * 3 * 4 * self.d * self.d

    def _get_kv_cache_size(self) -> int:
        return self.num_layers * 2 * self.d / self.heads_per_kv_cache
    
    def get_generate_flops(self, ctx_len: int) -> int:
        fc_flops = self.num_layers * 3 * 4 * self.d * self.d * 2
        atten_flops = self.num_layers * 2 * self.d * ctx_len * ctx_len * 2
        return fc_flops + atten_flops
    



    