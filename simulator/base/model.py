from dataclasses import dataclass
from typing import Optional, List
from collections import Counter
import math
import numpy as np

from .base import Base

@dataclass
class Model(Base):
    name: str
    num_layers: int
    d: int
    num_heads: int
    moe: bool = False
    d_head: Optional[int] = None
    d_ff: Optional[int] = None
    act: str = 'gelu'
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
        if self.d_head is None:
            self.d_head = self.d // self.num_heads
        if self.d_ff is None:
            self.d_ff = self.d * 4
        self.model_size = self._get_model_size()
        self.model_size_byte = self.model_size * self.bytes_per_number
        self.kv_cache_size_per_token = self._get_kv_cache_size()
        self.kv_cache_size_per_token_byte = self.kv_cache_size_per_token * self.bytes_per_number

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


eval_kernel_types = ['matmul', 'softmax', 'mul', 'layernorm', 'silu']
class KernelSizes:
    def __init__(self, kernel_type):
        if kernel_type not in eval_kernel_types:
            raise ValueError(f"Kernel type must be one of {eval_kernel_types}")
        self.kernel_type = kernel_type
        self.kernel_sizes = Counter()

    def add_kernel(self, *dims):
        """Adds a kernel size (M,K,N) or (B,M,K,N) to the store."""
        if self.kernel_type == 'matmul':
            if len(dims) not in {3, 4}:  # Ensure only valid sizes are added
                raise ValueError("Kernel size must be (M,K,N) or (B,M,K,N)")
        self.kernel_sizes[tuple(dims)] += 1
    
    def get_all_kernel_sizes(self):
        """Returns all unique kernel sizes."""
        return list(self.kernel_sizes.keys())

    def get_freqency(self, kernel_size):
        """Returns the frequency of a given kernel size."""
        return self.kernel_sizes[kernel_size]

class llama:
    def __init__(self, name, layers, d_model, d_ff, d_head, n_heads,
                 n_kv_heads, vocab_size = None, max_seq_len = None):
        self.name = name
        self.layers = layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_head = d_head
        self.n_heads = n_heads
        self.num_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = layers

        self.head_per_kv_head = math.ceil(self.n_heads / self.n_kv_heads)

        self.moe = False
    
    def get_kernel_sizes(self, prefill_len: int, decode_lens: List[int],
                         parallelism = (1, 1, 1, 1)):
        E, T, P, C = parallelism

        all_kernel_sizes = dict()
        for kernel_type in eval_kernel_types:
            all_kernel_sizes[kernel_type] = KernelSizes(kernel_type)
        
        fc_len = prefill_len + len(decode_lens)
        # Embedding layer
        # (fc_len, vocab_size) * (vocab_size, d_model) = (fc_len, d_model)
        # all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.vocab_size / T), self.d_model)

        # All layers
        for _ in range(math.ceil(self.layers / P)):
            # Attention layer
            # Q proj: (fc_len, d_model) * (d_model, d_head*n_heads) = (fc_len, d_head*n_heads)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_model, self.d_head *  math.ceil(self.n_heads / T))
            # K/V proj: (fc_len, d_model) * (d_model, 2*d_head*n_kv_heads) = (fc_len, 2*d_head*n_kv_heads)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_model, 2*self.d_head * math.ceil(self.n_kv_heads / T))
            # PREFILL
            if prefill_len > 0:
                # Q*K^T: (n_kv_heads, head_per_kv_head * prefill_len, d_head) * (n_kv_heads, d_head, prefill_len) = (n_kv_heads, head_per_kv_head * prefill_len, prefill_len)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_kv_heads / T), self.head_per_kv_head * math.ceil(prefill_len / C), self.d_head, prefill_len)
                # softmax: (n_heads, prefill_len, prefill_len)
                all_kernel_sizes['softmax'].add_kernel(math.ceil(self.n_heads / T), math.ceil(prefill_len / C), prefill_len)
                # S*V: (n_kv_heads, head_per_kv_head * prefill_len, prefill_len) * (n_kv_heads, prefill_len, d_head) = (n_kv_heads, head_per_kv_head * prefill_len, d_head)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_kv_heads / T), self.head_per_kv_head * math.ceil(prefill_len / C), prefill_len, self.d_head)
            # DECODE
            for ctx_len in decode_lens:
                # QK^T: (n_kv_heads, head_per_kv_head, d_head) * (n_kv_heads, d_head, ctx_len) = (n_kv_heads, head_per_kv_head, ctx_len)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_kv_heads / T), self.head_per_kv_head, self.d_head, ctx_len)
                # softmax: (n_heads, 1, ctx_len)
                all_kernel_sizes['softmax'].add_kernel(self.n_heads, 1, ctx_len)
                # S*V: (n_kv_heads, head_per_kv_head, ctx_len) * (n_kv_heads, ctx_len, d_head) = (n_kv_heads, head_per_kv_head, d_head)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_kv_heads / T), self.head_per_kv_head, ctx_len, self.d_head)
            # Attention output proj: (fc_len, d_head*n_heads) * (d_head*n_heads, d_model) = (fc_len, d_model)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_head * math.ceil(self.n_heads / T), self.d_model)
            all_kernel_sizes['layernorm'].add_kernel(math.ceil(fc_len / C), self.d_model)

            # Feedforward layer
            # FF1 and FF3: (fc_len, d_model) * (d_model, d_ff) = (fc_len, d_ff)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_model, math.ceil(self.d_ff / T))
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_model, math.ceil(self.d_ff / T))
            all_kernel_sizes['mul'].add_kernel( math.ceil(fc_len / C), math.ceil(self.d_ff / T))
            all_kernel_sizes['silu'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_ff / T))
            # FF2: (fc_len, d_ff) * (d_ff, d_model) = (fc_len, d_model)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_ff / T), self.d_model)
            all_kernel_sizes['layernorm'].add_kernel(math.ceil(fc_len / C), self.d_model)

        # Output layer
        # (fc_len, d_model) * (d_model, vocab_size) = (fc_len, vocab_size)
        # all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_model / T), self.vocab_size)

        return all_kernel_sizes

    def generate_layer_mermaid(self, info_in_box: bool, prefill_len: int, decode_lens: List[int],
                         parallelism = (1, 1, 1, 1)):
        """
        Generate a Mermaid flowchart for a single transformer layer of LLaMA.
        """
        E, T, P, C = parallelism

        fc_len = prefill_len + len(decode_lens)
        if len(decode_lens) > 0:
            avg_decode_len = math.ceil(sum(decode_lens) / len(decode_lens))
        else:
            avg_decode_len = 0

        d = self.d_model             # total hidden size
        d_ff = self.d_ff              # FFN inner dim 
        n_h = self.num_heads        # number of attention heads
        d_h = self.d_head          # attention head size
        n_kv_h = self.n_kv_heads    # number of kv heads

        h_per_kv_h = self.head_per_kv_head

        node_fc_l = math.ceil(fc_len / C)
        node_prefill_l = math.ceil(prefill_len / C)
        node_n_h = math.ceil(n_h / T)
        node_n_kv_h = math.ceil(n_kv_h / T)
        node_d_ff = math.ceil(d_ff / T)

        def label(info_in_box, name, op_str, shape_str):
            if info_in_box:
                return f'"`{name}\nOp: {op_str}\nShape: {shape_str}`"'
            else:
                return f'"{name}"'


        mermaid = f"""graph LR
                        ln1[{label(info_in_box, "Ln1", "LayerNorm", f'[{node_fc_l}, {d}]')}]
                        q_proj[{label(info_in_box,"Q Proj", "Matmul", f'[{node_fc_l}, {d}] * [{d}, {d_h * node_n_h}]')}]
                        kv_proj[{label(info_in_box,"KV Proj", "Matmul", f'[{node_fc_l}, {d}] * [{d}, {2 * d_h * node_n_kv_h}]')}]

                        attn_o_proj[{label(info_in_box, "Out Proj", "Matmul", f'[{node_fc_l}, {d_h * node_n_h}] * [{d_h * node_n_h}, {d}]')}]

                        res_add1[{label(info_in_box, "ReAdd1", "Add", f'[{node_fc_l}, {d}]')}]

                        ln2[{label(info_in_box, "Ln2", "LayerNorm", f'[{node_fc_l}, {d}]')}]

                        subgraph FeedForward Network
                            ffn_in[{label(info_in_box, "FFN In", "Matmul", f'[{node_fc_l}, {d}] * [{d}, {node_d_ff}]')}]
                            ffn_in2[{label(info_in_box, "FFN In2", "Matmul", f'[{node_fc_l}, {d}] * [{d}, {node_d_ff}]')}]
                            mul[{label(info_in_box, "Mul", "Mul", f'[{node_fc_l}, {node_d_ff}]')}]
                            silu[{label(info_in_box, "SiLU", "SiLU", f'[{node_fc_l}, {node_d_ff}]')}]
                            ffn_out[{label(info_in_box, "FFN Out", "Matmul", f'[{node_fc_l}, {node_d_ff}] * [{node_d_ff}, {d}]')}]
                        end

                        res_add2[{label(info_in_box, "ResAdd2", "Add", f'[{node_fc_l}, {d}]')}]

                        ln1 --> q_proj
                        ln1 --> kv_proj

                        attn_o_proj --> res_add1
                        ln1 --> res_add1

                        res_add1 --> ln2
                        ln2 --> ffn_in
                        ln2 --> ffn_in2
                        ffn_in --> mul
                        ffn_in2 --> mul
                        mul --> silu
                        silu --> ffn_out
                        ffn_out --> res_add2
                        res_add1 --> res_add2

        """
        if prefill_len > 0:
            # classDef prefill fill:#e6f7ff,stroke:#007acc,stroke-width:2px;
            mermaid += f"""
                        subgraph Prefill
                            pfl_qk[{label(info_in_box, "QKᵀ", "Matmul", f'[{node_n_kv_h}, {h_per_kv_h * node_prefill_l}, {d_h}] * [{node_n_kv_h}, {d_h}, {prefill_len}]')}]:::prefill
                            pfl_softmax[{label(info_in_box, "Softmax", "Softmax", f'[{node_n_h}, {node_prefill_l}, {prefill_len}]')}]:::prefill
                            pfl_sv[{label(info_in_box, "SV", "Matmul", f'[{node_n_kv_h}, {h_per_kv_h * node_prefill_l}, {prefill_len}] * [{node_n_kv_h}, {prefill_len}, {d_h}]')}]:::prefill
                        end

                        q_proj --> pfl_qk
                        kv_proj --> pfl_qk
                        pfl_qk --> pfl_softmax
                        pfl_softmax --> pfl_sv
                        kv_proj --> pfl_sv
                        pfl_sv --> attn_o_proj

            """
        if len(decode_lens) > 0:
            mermaid += f"""
                        subgraph Decode x {len(decode_lens)}
                            dec_qk[{label(info_in_box, f"QKᵀ", "Matmul", f'[{node_n_kv_h}, {h_per_kv_h}, {d_h}] * [{node_n_kv_h}, {d_h}, {avg_decode_len}]')}]
                            dec_softmax[{label(info_in_box, f"Softmax", "Softmax", f'[{n_h}, 1, {avg_decode_len}]')}]
                            dec_sv[{label(info_in_box, f"SV", "Matmul", f'[{node_n_kv_h}, {h_per_kv_h}, {avg_decode_len}] * [{node_n_kv_h}, {avg_decode_len}, {d_h}]')}]
                        end

                        q_proj --> dec_qk
                        kv_proj --> dec_qk
                        dec_qk --> dec_softmax
                        dec_softmax --> dec_sv
                        kv_proj --> dec_sv
                        dec_sv --> attn_o_proj
            """
        if info_in_box:
            hover_infos = None
        else:
            hover_infos = {
                "ln1": {
                    "name": "LayerNorm",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": None,
                },
                "q_proj": {
                    "name": "Matmul",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": f'[{d}, {d_h * node_n_h}]',
                },
                "kv_proj": {
                    "name": "Matmul",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": f'[{d}, {2 * d_h * node_n_kv_h}]',
                },
                "attn_o_proj": {
                    "name": "Matmul",
                    "input1": f'[{node_fc_l}, {d_h * node_n_h}]',
                    "input2": f'[{d_h * node_n_h}, {d}]',
                },
                "res_add1": {
                    "name": "Add",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": f'[{node_fc_l}, {d}]',
                },
                "ln2": {
                    "name": "LayerNorm",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": None,
                },
                "ffn_in": {
                    "name": "Matmul",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": f'[{d}, {node_d_ff}]',
                },
                "ffn_in2": {
                    "name": "Matmul",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": f'[{d}, {node_d_ff}]',
                },
                "mul": {
                    "name": "Mul",
                    "input1": f'[{node_fc_l}, {node_d_ff}]',
                    "input2": f'[{node_fc_l}, {node_d_ff}]',
                },
                "silu": {
                    "name": "SiLU",
                    "input1": f'[{node_fc_l}, {node_d_ff}]',
                    "input2": None,
                },
                "ffn_out": {
                    "name": "Matmul",
                    "input1": f'[{node_fc_l}, {node_d_ff}]',
                    "input2": f'[{node_d_ff}, {d}]',
                },
                "res_add2": {
                    "name": "Add",
                    "input1": f'[{node_fc_l}, {d}]',
                    "input2": f'[{node_fc_l}, {d}]',
                },
                "pfl_qk": {
                    "name": "Matmul",
                    "input1": f'[{node_n_kv_h}, {h_per_kv_h * node_prefill_l}, {d_h}]',
                    "input2": f'[{node_n_kv_h}, {d_h}, {prefill_len}]',
                },
                "pfl_softmax": {
                    "name": "Softmax",
                    "input1": f'[{node_n_h}, {node_prefill_l}, {prefill_len}]',
                    "input2": None,
                },
                "pfl_sv": {
                    "name": "Matmul",
                    "input1": f'[{node_n_kv_h}, {h_per_kv_h * node_prefill_l}, {prefill_len}]',
                    "input2": f'[{node_n_kv_h}, {prefill_len}, {d_h}]',
                },
                "dec_qk": {
                    "name": "Matmul",
                    "input1": f'[{node_n_kv_h}, {h_per_kv_h}, {d_h}]',
                    "input2": f'[{node_n_kv_h}, {d_h}, {avg_decode_len}]',
                },
                "dec_softmax": {
                    "name": "Softmax",
                    "input1": f'[{n_h}, 1, {avg_decode_len}]',
                    "input2": None,
                },
                "dec_sv": {
                    "name": "Matmul",
                    "input1": f'[{node_n_kv_h}, {h_per_kv_h}, {avg_decode_len}]',
                    "input2": f'[{node_n_kv_h}, {avg_decode_len}, {d_h}]',
                },
            }

        return mermaid, hover_infos


class deepseek:
    def __init__(self, name, layers, d_model, d_dense, 
                 d_exp, dense_layers, n_heads,
                 n_shared_exp, n_routed_exp, n_activated_exp, 
                 kv_lora_rank, q_lora_rank, 
                 d_qk_nope_head, d_qk_rope_head, d_v_head,
                 vocab_size = None, max_seq_len = None):
        self.name = name
        self.layers = layers
        self.num_layers = layers
        self.d_model = d_model
        self.d_dense = d_dense
        self.d_exp = d_exp
        self.dense_layers = dense_layers
        self.n_heads = n_heads
        self.num_heads = n_heads
        self.n_shared_exp = n_shared_exp
        self.n_routed_exp = n_routed_exp
        self.n_activated_exp = n_activated_exp
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.d_qk_nope_head = d_qk_nope_head
        self.d_qk_rope_head = d_qk_rope_head
        self.d_v_head = d_v_head
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.d_qk_head = self.d_qk_nope_head + self.d_qk_rope_head
        self.sparse_layers = self.layers - self.dense_layers

        self.moe = True

    def get_kernel_sizes(self, prefill_len: int, decode_lens: List[int],
                            parallelism = (1, 1, 1, 1), exp_distribution = None):
        if exp_distribution is None:
            # uniform distribution
            exp_distribution = [1/self.n_routed_exp] * self.n_routed_exp
        E, T, P, C = parallelism
        all_kernel_sizes = dict()
        for kernel_type in eval_kernel_types:
            all_kernel_sizes[kernel_type] = KernelSizes(kernel_type)
        
        fc_len = prefill_len + len(decode_lens)
        # Embedding layer
        # (fc_len, vocab_size) * (vocab_size, d_model) = (fc_len, d_model)
        # all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.vocab_size / T), self.d_model)

        # All layers: Multi Latent Attention
        for _ in range(math.ceil(self.layers / P)):
            # MLA linear - DQ: (fc_len, d_model) * (d_model, q_lora_rank) = (fc_len, q_lora_rank)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_model / T), self.q_lora_rank)
            # MLA linear - UQ: (fc_len, q_lora_rank) * (q_lora_rank, n_heads * d_qk_head) = (fc_len, n_heads * d_qk_head)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.q_lora_rank, math.ceil(self.n_heads / T) * self.d_qk_head)
            # MLA linear - DKV: (fc_len, d_model) * (d_model, kv_lora_rank + d_qk_rope_head) = (fc_len, kv_lora_rank + d_qk_rope_head)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_model / T), self.kv_lora_rank + self.d_qk_rope_head)

            # PREFILL
            if prefill_len > 0:
                # matmul1, absorb WUK into WUQ: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L480
                # matmul2: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L483
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T),  math.ceil(prefill_len / C), self.d_qk_nope_head, self.kv_lora_rank)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T) * math.ceil(prefill_len / C), self.kv_lora_rank, prefill_len)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T) * math.ceil(prefill_len / C), self.d_qk_rope_head, prefill_len)
                # a_mul_v
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T) * math.ceil(prefill_len / C), prefill_len, self.kv_lora_rank)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T),  math.ceil(prefill_len / C), self.kv_lora_rank, self.d_v_head)

            # DECODE
            if len(decode_lens) > 0:
                # matmul1, absorb WUK into WUQ: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L480
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), 1, self.d_qk_nope_head, self.kv_lora_rank)
                for ctx_len in decode_lens:
                    # matmul2: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L483
                    all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), self.kv_lora_rank, ctx_len)
                    all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), self.d_qk_rope_head, ctx_len)
                    # a_mul_v
                    all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), ctx_len, self.kv_lora_rank)
                    all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), 1, self.kv_lora_rank, self.d_v_head)

            # Output proj: (fc_len, n_heads * d_v_head) * (n_heads * d_v_head, d_model) = (fc_len, d_model)
            all_kernel_sizes['matmul'].add_kernel(fc_len, math.ceil(self.n_heads / T) * self.d_v_head, self.d_model)
            all_kernel_sizes['layernorm'].add_kernel(fc_len, self.d_model)

        # Dense layers
        for _ in range(self.dense_layers):
            # FF1 and FF3: (fc_len, d_model) * (d_model, d_dense) = (fc_len, d_dense)
            all_kernel_sizes['matmul'].add_kernel(fc_len, self.d_model, math.ceil(self.d_dense / T))
            all_kernel_sizes['matmul'].add_kernel(fc_len, self.d_model, math.ceil(self.d_dense / T))
            all_kernel_sizes['mul'].add_kernel(fc_len, math.ceil(self.d_dense / T))
            all_kernel_sizes['silu'].add_kernel(fc_len, math.ceil(self.d_dense / T))
            # FF2: (fc_len, d_dense) * (d_dense, d_model) = (fc_len, d_model)
            all_kernel_sizes['matmul'].add_kernel(fc_len, math.ceil(self.d_dense / T), self.d_model)
            all_kernel_sizes['layernorm'].add_kernel(fc_len, self.d_model)
        
        # MoE layers
        for _ in range(math.ceil(self.sparse_layers / P)):
            # shared_exp FF1 and FF3: (fc_len, d_model) * (d_model, d_exp * n_shared_exp) = (fc_len, d_exp * n_shared_exp)
            all_kernel_sizes['matmul'].add_kernel(fc_len, self.d_model, math.ceil(self.d_exp * self.n_shared_exp / T))
            all_kernel_sizes['matmul'].add_kernel(fc_len, self.d_model, math.ceil(self.d_exp * self.n_shared_exp / T))
            all_kernel_sizes['mul'].add_kernel(fc_len, math.ceil(self.d_exp * self.n_shared_exp / T))
            all_kernel_sizes['silu'].add_kernel(fc_len, math.ceil(self.d_exp * self.n_shared_exp / T))
            # shared_exp FF2: (fc_len, d_exp * n_shared_exp) * (d_exp * n_shared_exp, d_model) = (fc_len, d_model)
            all_kernel_sizes['matmul'].add_kernel(fc_len, math.ceil(self.d_exp * self.n_shared_exp / T), self.d_model)
            all_kernel_sizes['layernorm'].add_kernel(fc_len, self.d_model)

            # routed_exp 
            def sample_activated_experts(batch_size):
                expert_ids = np.arange(self.n_routed_exp)
                experts_activations = {k: 0 for k in range(self.n_routed_exp)}
                for _ in range(batch_size):
                    activated = np.random.choice(expert_ids, size=self.n_activated_exp, replace=False, p=exp_distribution)
                    for expert_id in activated:
                        experts_activations[expert_id] += 1
                return experts_activations

            def get_expert_activation_per_node(experts_activations):
                activations_per_node = [0] * E
                for expert_id, count in experts_activations.items():
                    node_id = expert_id // (self.n_routed_exp // E)
                    activations_per_node[node_id] += count
                return activations_per_node

            if prefill_len > 0:
                expert_activations = sample_activated_experts(1)
                activations_per_node = get_expert_activation_per_node(expert_activations)
                max_activated_per_node = max(activations_per_node)
                activated_exp = max_activated_per_node

                # FF1 and FF3: (fc_len, d_model) * (d_model, d_exp * n_routed_exp) = (fc_len, d_exp * n_routed_exp)
                all_kernel_sizes['matmul'].add_kernel(prefill_len, self.d_model, math.ceil(self.d_exp * activated_exp / T))
                all_kernel_sizes['matmul'].add_kernel(prefill_len, self.d_model, math.ceil(self.d_exp * activated_exp / T))
                all_kernel_sizes['mul'].add_kernel(prefill_len,  math.ceil(self.d_exp * activated_exp / T))
                all_kernel_sizes['silu'].add_kernel(prefill_len, math.ceil(self.d_exp * activated_exp / T))
                # routed_exp FF2: (fc_len, d_exp * activated_exp) * (d_exp * activated_exp, d_model) = (fc_len, d_model)
                all_kernel_sizes['matmul'].add_kernel(prefill_len, math.ceil(self.d_exp * activated_exp / T), self.d_model)
                all_kernel_sizes['layernorm'].add_kernel(prefill_len, self.d_model)
            elif len(decode_lens) > 0:
                expert_activations = sample_activated_experts(len(decode_lens))
                activations_per_node = get_expert_activation_per_node(expert_activations)
                # It's non-trivial to determine which node is the critical path, now we just assume the one with the most expert activations
                # This might not be true. For example, one node has the most expert activations but all of them are for the same expert, while another node has less expert activations but they are for different experts. The latter might have longer latency due to the more memory accesses.
                critical_node = activations_per_node.index(max(activations_per_node))
                expert_id_start = critical_node * (self.n_routed_exp // E)
                expert_id_end = expert_id_start + (self.n_routed_exp // E) - 1
                for expert_id in range(expert_id_start, expert_id_end + 1):
                    activated_times = expert_activations[expert_id]
                    if activated_times == 0:
                        continue
                    # FF1 and FF3: (fc_len, d_model) * (d_model, d_exp * n_routed_exp) = (fc_len, d_exp * n_routed_exp)
                    all_kernel_sizes['matmul'].add_kernel(activated_times, self.d_model, math.ceil(self.d_exp / T))
                    all_kernel_sizes['matmul'].add_kernel(activated_times, self.d_model, math.ceil(self.d_exp / T))
                    all_kernel_sizes['mul'].add_kernel(activated_times,  math.ceil(self.d_exp / T))
                    all_kernel_sizes['silu'].add_kernel(activated_times, math.ceil(self.d_exp / T))
                    # routed_exp FF2: (fc_len, d_exp * activated_exp) * (d_exp * activated_exp, d_model) = (fc_len, d_model)
                    all_kernel_sizes['matmul'].add_kernel(activated_times, math.ceil(self.d_exp / T), self.d_model)
                    all_kernel_sizes['layernorm'].add_kernel(activated_times, self.d_model)
                
        # Output layer
        # (fc_len, d_model) * (d_model, vocab_size) = (fc_len, vocab_size)
        # all_kernel_sizes['matmul'].add_kernel(fc_len, math.ceil(self.d_model / T), self.vocab_size)

        return all_kernel_sizes

class gpt:
    def __init__(self, name, layers, d_model, d_head, n_heads,
                 vocab_size = None, max_seq_len = None):
        self.name = name
        self.layers = layers
        self.d_model = d_model
        self.d_ff = d_model * 4
        self.d_head = d_head
        self.n_heads = n_heads
        self.num_heads = n_heads
        self.n_kv_heads = n_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_layers = layers

        self.head_per_kv_head = 1

        self.moe = False
    
    def get_kernel_sizes(self, prefill_len: int, decode_lens: List[int],
                         parallelism = (1, 1, 1, 1)):
        E, T, P, C = parallelism

        all_kernel_sizes = dict()
        for kernel_type in eval_kernel_types:
            all_kernel_sizes[kernel_type] = KernelSizes(kernel_type)
        
        fc_len = prefill_len + len(decode_lens)
        # Embedding layer
        # (fc_len, vocab_size) * (vocab_size, d_model) = (fc_len, d_model)
        # all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.vocab_size / T), self.d_model)

        # All layers
        for _ in range(math.ceil(self.layers / P)):
            # Attention layer
            # Q proj: (fc_len, d_model) * (d_model, d_head*n_heads) = (fc_len, d_head*n_heads)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_model, self.d_head *  math.ceil(self.n_heads / T))
            # K/V proj: (fc_len, d_model) * (d_model, 2*d_head*n_heads) = (fc_len, 2*d_head*n_heads)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_model, 2*self.d_head * math.ceil(self.n_heads / T))
            # PREFILL
            if prefill_len > 0:
                # Q*K^T: (n_heads, prefill_len, d_head) * (n_heads, d_head, prefill_len) = (n_heads, prefill_len, prefill_len)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T),  math.ceil(prefill_len / C), self.d_head, prefill_len)
                # softmax: (n_heads, prefill_len, prefill_len)
                all_kernel_sizes['softmax'].add_kernel(math.ceil(self.n_heads / T), math.ceil(prefill_len / C), prefill_len)
                # S*V: (n_heads, prefill_len, prefill_len) * (n_heads, prefill_len, d_head) = (n_heads, prefill_len, d_head)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), math.ceil(prefill_len / C), prefill_len, self.d_head)
            # DECODE
            for ctx_len in decode_lens:
                # QK^T: (n_heads, 1, d_head) * (n_heads, d_head, ctx_len) = (n_heads, 1, ctx_len)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), 1, self.d_head, ctx_len)
                # softmax: (n_heads, 1, ctx_len)
                all_kernel_sizes['softmax'].add_kernel(self.n_heads, 1, ctx_len)
                # S*V: (n_heads, 1, ctx_len) * (n_heads, ctx_len, d_head) = (n_heads, 1, d_head)
                all_kernel_sizes['matmul'].add_kernel(math.ceil(self.n_heads / T), 1, ctx_len, self.d_head)
            # Attention output proj: (fc_len, d_head*n_heads) * (d_head*n_heads, d_model) = (fc_len, d_model)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_head * math.ceil(self.n_heads / T), self.d_model)
            all_kernel_sizes['layernorm'].add_kernel(math.ceil(fc_len / C), self.d_model)

            # Feedforward layer
            # FF1: (fc_len, d_model) * (d_model, d_ff) = (fc_len, d_ff)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), self.d_model, math.ceil(self.d_ff / T))
            all_kernel_sizes['silu'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_ff / T))
            # FF2: (fc_len, d_ff) * (d_ff, d_model) = (fc_len, d_model)
            all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_ff / T), self.d_model)
            all_kernel_sizes['layernorm'].add_kernel(math.ceil(fc_len / C), self.d_model)

        # Output layer
        # (fc_len, d_model) * (d_model, vocab_size) = (fc_len, vocab_size)
        # all_kernel_sizes['matmul'].add_kernel(math.ceil(fc_len / C), math.ceil(self.d_model / T), self.vocab_size)

        return all_kernel_sizes

# %%
llama70b = llama(name='llama70b',
                 layers=80,
                 d_model=8192,
                 d_ff=28672,
                 d_head=128,
                 n_heads=64,
                 n_kv_heads=8,
                 vocab_size=128256,
                 max_seq_len=8192)
llama405b = llama(name='llama405b',
                  layers=126,
                  d_model=16384,
                  d_ff=53248,
                  d_head=128,
                  n_heads=128,
                  n_kv_heads=8,
                  vocab_size=128256,
                  max_seq_len=8192)
deepseekv2 = deepseek(name='deepseekv2',
                        layers=60,
                        d_model=5120,
                        d_dense=12288,
                        d_exp=1536,
                        dense_layers=1,
                        n_heads=128,
                        n_shared_exp=2,
                        n_routed_exp=160,
                        n_activated_exp=6,
                        kv_lora_rank=512,
                        q_lora_rank=1536,
                        d_qk_nope_head=128,
                        d_qk_rope_head=64,
                        d_v_head=128,
                        vocab_size=102400,)

deepseekv3 = deepseek(name='deepseekv3',
                        layers=61,
                      d_model=7168,
                      d_dense=18432,
                      d_exp=2048,
                      dense_layers=3,
                      n_heads=128,
                      n_shared_exp=1,
                      n_routed_exp=256,
                      n_activated_exp=8,
                      kv_lora_rank=512,
                      q_lora_rank=1536,
                      d_qk_nope_head=128,
                      d_qk_rope_head=64,
                      d_v_head=128,
                      vocab_size=129280,)

opt175b = gpt(name='opt175b',
              layers=96,
                d_model=12288,
                d_head=128,
                n_heads=96,
)
