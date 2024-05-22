from dataclasses import dataclass
from typing import Optional
from .Base import Base

@dataclass
class Model(Base):
    name: str
    num_layers: int
    d: int
    num_heads: int
    d_head: Optional[int] = None
    d_ff: Optional[int] = None
    act: str = 'gelu'
    heads_per_kv_cache: int = 1 # number of heads that share the same kv cache, 1 <= heads_per_kv_cache <= num_heads and should be a divisor of num_heads
                                # 1 means each head has its own kv cache, i.e., multihead attention
                                # num_heads means all heads share the same kv cache, i.e., multiquery attention
                                # 1 < heads_per_kv_cache < num_heads means grouped query attention
    d_lora: int = 0 # low-rank adaptation dimension, used on q,k,v,o
    bytes_per_number: int = 2

    model_size: Optional[int] = None # number of parameters in the model
    model_size_byte: Optional[int] = None # number of parameters in the model, in bytes
    lora_size_byte: Optional[int] = None # number of parameters in the low-rank adaptation, in bytes
    kv_cache_size_per_token: Optional[int] = None # kv cache size per token
    kv_cache_size_per_token_byte: Optional[int] = None # kv cache size per token, in bytes

    def update(self) -> None:
        if self.d_head is None:
            self.d_head = self.d // self.num_heads
        if self.d_ff is None:
            self.d_ff = self.d * 4
        self.model_size = self._get_model_size()
        self.model_size_byte = self.model_size * self.bytes_per_number
        self.kv_cache_size_per_token = self._get_kv_cache_size()
        self.kv_cache_size_per_token_byte = self.kv_cache_size_per_token * self.bytes_per_number
        self.lora_size_byte = self._get_lora_size_byte()

    def _get_model_size(self) -> int:
        # atten
        self.q_size = self.num_heads * self.d * self.d_head
        self.k_size = self.num_heads * self.d * self.d_head / self.heads_per_kv_cache
        self.v_size = self.k_size
        self.o_size = self.num_heads * self.d * self.d_head
        self.atten_size = self.q_size + self.k_size + self.v_size + self.o_size
        # ffn
        self.fc1_size = self.d * self.d_ff
        self.fc2_size = self.d_ff * self.d
        self.ffn_size = self.fc1_size + self.fc2_size
        # gated linear unit, need extra weights 
        if 'glu' in self.act: 
            self.glu_size = (self.d * self.d_ff)
        else:
            self.glu_size = 0
        return self.num_layers * (self.atten_size + self.ffn_size + self.glu_size)

    def _get_kv_cache_size(self) -> int:
        return self.num_layers * 2 * self.num_heads * self.d_head / self.heads_per_kv_cache
    
    def _get_lora_size_byte(self) -> int:
        q_A_size = self.num_layers * self.num_heads * self.d * self.d_lora
        q_B_size = self.num_layers * self.num_heads * self.d_lora * self.d_head
        kv_A_size = 2 * self.num_layers * self.num_heads * self.d * self.d_lora / self.heads_per_kv_cache
        kv_B_size = 2 * self.num_layers * self.num_heads * self.d_lora * self.d_head  / self.heads_per_kv_cache
        o_A_size = self.num_layers * self.num_heads * self.d_head * self.d_lora
        o_B_size = self.num_layers * self.num_heads * self.d_lora * self.d
        lora_size = q_A_size + q_B_size + kv_A_size + kv_B_size + o_A_size + o_B_size
        return lora_size * self.bytes_per_number
    
    def get_prefill_flops(self, ctx_len: int) -> int:
        atten_fc_flops = self.atten_size * ctx_len * 2
        ffn_fc_flops = self.ffn_size * ctx_len * 2
        fc_flops = self.num_layers * (atten_fc_flops + ffn_fc_flops)
        self_atten_flops = self.num_layers * 2 * self.num_heads * self.d_head * ctx_len * ctx_len * 2
        return fc_flops + self_atten_flops
        
    def get_generate_flops(self, ctx_len: int) -> int:
        atten_fc_flops = self.atten_size * 2
        ffn_fc_flops = self.ffn_size * 2
        fc_flops = self.num_layers * (atten_fc_flops + ffn_fc_flops)
        self_atten_flops = self.num_layers * 2 * self.num_heads * self.d_head * ctx_len * 1 * 2
        return fc_flops + self_atten_flops
    
