import logging
from typing import List, Dict, Tuple, Optional

from .task import Task, PrefillTask, DecodeTask
from .request import Request
from ..base.model import Model

class LLMKernel:
    def __init__(self, 
                 phase: str,
                 model: Model,
                 n: int,
                 l_start: int,
                 l_end: int,
                 ctx: List[int] = [],):
        self.phase = phase
        self.model = model
        self.n = n
        self.l_start = l_start
        self.l_end = l_end
        self.ctx = ctx # list of context lengths, for decode kernel only

        self.l = l_end - l_start + 1

    # def get_flops(self) -> Tuple[int, int]:
    #     if self.phase == 'prefill':
    #         attn_flops = self.l * 2 * self.model.num_heads * self.model.d_head * self.n * self.n * 2
    #     else:
    #         attn_flops = 0
    #         for ctx in self.ctx:
    #             attn_flops += self.l * 2 * self.model.num_heads * self.model.d_head * ctx * 1 * 2

    #     fc_flops = self.model.model_size * 2 * self.n * self.l / self.model.num_layers
    #     return attn_flops, fc_flops
    
    # def get_bytes(self) -> Tuple[int, int]:
    #     if self.phase == 'prefill':
    #         kv_bytes = self.n * int(self.model.kv_cache_size_per_token_byte * self.l / self.model.num_layers)
    #         # TODO: add activation load
    #     else:
    #         kv_bytes = 0
    #         for ctx in self.ctx:
    #             kv_bytes += int(self.model.kv_cache_size_per_token_byte * ctx * self.l / self.model.num_layers)
    #     weight_bytes = int(self.model.model_size_byte * self.l / self.model.num_layers)
    #     return kv_bytes, weight_bytes

class SimKernel:
    def __init__(self, 
                 prefill_kernel: LLMKernel,
                 decode_kernel: LLMKernel,
                 prefetch: Optional[str] = None):
        self.prefetch = prefetch
        self.prefill_kernel = prefill_kernel
        if prefill_kernel is not None:
            self.model = prefill_kernel.model
        #     self.prefill_layers = prefill_kernel.l_end - prefill_kernel.l_start + 1
        #     self.prefill_attn_flops, self.prefill_fc_flops = self.prefill_kernel.get_flops()
        #     self.prefill_kv_bytes, self.prefill_weight_bytes = self.prefill_kernel.get_bytes()
        # else:
        #     self.prefill_layers = 0
        #     self.prefill_attn_flops, self.prefill_fc_flops = 0, 0
        #     self.prefill_kv_bytes, self.prefill_weight_bytes = 0, 0

        self.decode_kernel = decode_kernel
        if decode_kernel is not None:
            self.model = decode_kernel.model
        #     self.decode_layers = decode_kernel.l_end - decode_kernel.l_start + 1
        #     assert len(decode_kernel.ctx) == decode_kernel.n
        #     self.decode_attn_flops, self.decode_fc_flops = self.decode_kernel.get_flops()
        #     self.decode_kv_bytes, self.decode_weight_bytes = self.decode_kernel.get_bytes()
        # else:
        #     self.decode_layers = 0
        #     self.decode_attn_flops, self.decode_fc_flops = 0, 0
        #     self.decode_kv_bytes, self.decode_weight_bytes = 0, 0

        # self.weight_bytes = self.prefill_weight_bytes + self.decode_weight_bytes
        # self.flops = self.prefill_attn_flops + self.prefill_fc_flops + self.decode_attn_flops + self.decode_fc_flops
        # if self.prefill_kernel is not None and self.decode_kernel is not None:
        #     # find overlapped layers
        #     l_start = max(self.prefill_kernel.l_start, self.decode_kernel.l_start)
        #     l_end = min(self.prefill_kernel.l_end, self.decode_kernel.l_end)
        #     l_overlap = l_end - l_start + 1
        #     self.weight_bytes = self.weight_bytes - self.model.model_size_byte * l_overlap / self.model.num_layers
    
class PrefillPool:
    def __init__(self, algo, prefill_chunk):
        self.algo = algo
        self.prefill_chunk = prefill_chunk

        self.tasks = []
        self.l = 0
        self.n = 0
        self.l_cur = 0
        self.n_cur = 0

    def new_tasks(self, tasks: List[Task]):
        self.tasks = []
        while len(tasks) > 0:
            self.tasks.append(tasks.pop(0))
        self.l = self.tasks[0].req.model.num_layers
        self.n = sum([task.req.input_len for task in self.tasks])
        self.l_cur = 0
        self.n_cur = 0
    
    def schedule(self, n_for_prefetch = None) -> LLMKernel:
        if self.is_empty():
            return None
        if self.l_cur ==self.l:
            self.l_cur = 0

        if self.algo == 'mixed-sarathi':
            assert self.prefill_chunk > 0
            kernel_n = min(self.prefill_chunk, self.n - self.n_cur)
        elif self.algo == 'prefetch-mixed' or self.algo == 'prefetch-thread':
            # assert self.prefill_chunk > 0
            # kernel_n = min(self.prefill_chunk, self.n - self.n_cur)
            assert n_for_prefetch <= self.n - self.n_cur
            kernel_n = n_for_prefetch
        else:
            kernel_n = self.n
        kernel_l = self.l

        if self.l_cur + kernel_l > self.l:
            raise ValueError("prefill layers exceeds model layers")
        if self.n_cur + kernel_n > self.n:
            raise ValueError("prefill length exceeds input length")

        prefill_kernel = LLMKernel('prefill',
                                   self.tasks[0].req.model, 
                                   kernel_n,
                                   self.l_cur, 
                                   self.l_cur + kernel_l - 1)
        self.l_cur += kernel_l
        self.n_cur += kernel_n

        return prefill_kernel
    
    def is_done(self):
        return self.l_cur == self.l and self.n_cur == self.n and self.tasks is not []
    
    def reset(self):
        self.tasks = []
        self.l = 0
        self.n = 0
        self.l_cur = 0
        self.n_cur = 0
    
    def is_empty(self):
        return self.tasks == []

class DecodePool:
    def __init__(self, algo):
        self.algo = algo

        self.tasks = []
        self.l = 0
        self.n = 0
        self.l_cur = 0
    
    def new_tasks(self, tasks: List[Task]):
        self.tasks = []
        while len(tasks) > 0:
            self.tasks.append(tasks.pop(0))
        self.l = self.tasks[0].req.model.num_layers
        self.n = len(self.tasks)
        self.l_cur = 0
        
        # self.layer_weights = int(self.tasks[0].req.model.model_size / self.l)
        # self.layer_kv = 0
        # for task in self.tasks:
        #     ctx = task.req.input_len + task.n_cur
        #     self.layer_kv += (task.req.model.kv_cache_size_per_token * ctx // self.l)
    
    def schedule(self, prefetch_l = None) -> LLMKernel:
        if self.is_empty():
            return None
        
        kernel_n = self.n
        if self.algo == 'prefetch-mixed':
            raise NotImplementedError
        elif self.algo == 'prefetch-thread':
            raise NotImplementedError
        else:
            kernel_l = self.l

        decode_tasks_ctx = []
        for task in self.tasks:
            decode_tasks_ctx.append(task.req.input_len + task.n_cur)
        decode_kernel = LLMKernel('decode',
                                  self.tasks[0].req.model,
                                  kernel_n,
                                  self.l_cur,
                                  self.l_cur + kernel_l - 1,
                                  decode_tasks_ctx
        )
        self.l_cur += kernel_l
        assert self.l_cur <= self.l

        return decode_kernel
    
    # def get_prefetch_layers(self, mem_3d_size: int) -> int:
    #     prefetch_l = 0
    #     while prefetch_l * (self.layer_weights + self.layer_kv) * 2 <= mem_3d_size:
    #         if prefetch_l + self.l_cur >= self.l:
    #             return prefetch_l
    #         prefetch_l += 1
    #     prefetch_l -= 1
    #     if prefetch_l == 0:
    #         num_tokens = 0
    #         for task in self.tasks:
    #             num_tokens += (task.req.input_len + task.n_cur)
    #         print(f"{num_tokens} tokens in {len(self.tasks)} decode tasks")
    #         raise ValueError(f"3D memory {mem_3d_size/1e9} GB is too small to prefetch one layer with {self.layer_weights * 2 /1e9} GB weights and {self.layer_kv * 2 /1e9} GB kv cache")
    #     return prefetch_l
    
    def is_done(self):
        return self.l_cur == self.l and self.tasks != []
    
    def reset(self):
        self.tasks = []
        self.l = 0
        self.n = 0
        self.l_cur = 0
    
    def is_empty(self):
        return self.tasks == []
    

class Scheduler:
    def __init__(self, algo: str = 'baseline', # algo: baseline (request level), continuous, mixed-splitfuse, mixed-sarathi, or prefetch
                 mem_3d_size: int = 0, # 3D memory size
                 prefill_chunk: int = 0, # prefill chunk size, only used in mixed-sarathi and prefetch
    ):
        if algo == 'mixed-sarathi':
            assert prefill_chunk > 0
        if algo == 'prefetch-mixed':
            assert mem_3d_size > 0
        self.algo = algo
        self.prefill_pool = PrefillPool(algo, prefill_chunk)
        self.decode_pool = DecodePool(algo)
        self.mem_3d_size = mem_3d_size
        self.prefill_chunk = prefill_chunk

    def run(self, time, prefill_fifo, decode_fifo, accept_new_req):
        t_cur = time

        if self.algo == 'baseline':
            if not self.decode_pool.is_empty():
                sim_kernel = SimKernel(None, self.decode_pool.schedule())
            elif decode_fifo != []:
                for task in decode_fifo:
                    # all deocde tasks should arrive before current time
                    assert t_cur >= task.t_arrival
                logging.debug(f"Basline Scheduler (t={t_cur}): running {len(decode_fifo)} decode tasks")
                self.decode_pool.new_tasks(decode_fifo)
                # run batched decode only
                sim_kernel = SimKernel(None, self.decode_pool.schedule())
            else:
                new_prefill_tasks = []
                if t_cur < prefill_fifo[0].t_arrival:
                    # current time is before the next prefill task arrival, update current time to the prefill task arrival time
                    new_prefill_task = prefill_fifo.pop(0)
                    t_cur = new_prefill_task.t_arrival
                    logging.debug(f"Baseline Scheduler (t={t_cur}): update current time to {new_prefill_task.t_arrival}")
                    new_prefill_tasks.append(new_prefill_task)
                else:
                    while t_cur >= prefill_fifo[0].t_arrival:
                        # all prefill tasks should arrive before current time
                        new_prefill_task = prefill_fifo.pop(0)
                        new_prefill_tasks.append(new_prefill_task)
                logging.debug(f"Baseline Scheduler (t={t_cur}): run {len(new_prefill_tasks)} prefill tasks")
                self.prefill_pool.new_tasks(new_prefill_tasks)
                sim_kernel = SimKernel(self.prefill_pool.schedule(), None)
        elif self.algo == 'continuous':
            new_prefill_tasks = []
            if prefill_fifo != []:
                while t_cur >= prefill_fifo[0].t_arrival:
                    new_prefill_task = prefill_fifo.pop(0)
                    new_prefill_tasks.append(new_prefill_task)
                    if prefill_fifo == []:
                        break
            if new_prefill_tasks != []:
                self.prefill_pool.new_tasks(new_prefill_tasks)

            if not self.prefill_pool.is_empty():
                # prioritize prefill over decode
                sim_kernel = SimKernel(self.prefill_pool.schedule(), None)
                logging.debug(f"Scheduler (t={t_cur}): running {len(new_prefill_tasks)} prefill tasks")
            else:
                if not self.decode_pool.is_empty():
                    sim_kernel = SimKernel(None, self.decode_pool.schedule())
                elif decode_fifo != []:
                    for task in decode_fifo:
                        # all deocde tasks should arrive before current time
                        assert t_cur >= task.t_arrival
                    logging.debug(f"Scheduler (t={t_cur}): running {len(decode_fifo)} decode tasks")
                    self.decode_pool.new_tasks(decode_fifo)
                    sim_kernel = SimKernel(None, self.decode_pool.schedule())
                else:
                    t_cur = prefill_fifo[0].t_arrival
                    logging.debug(f"Scheduler (t={t_cur}): update current time to {prefill_fifo[0].t_arrival}")
                    self.prefill_pool.new_tasks([prefill_fifo.pop(0)])
                    sim_kernel = SimKernel(self.prefill_pool.schedule(), None)

        elif self.algo == 'mixed-splitfuse':
            raise NotImplementedError
        elif self.algo == 'mixed-sarathi':
            # the SARATHI version of mixed scheduling,
            if self.decode_pool.is_empty() and decode_fifo != []:
                for task in decode_fifo:
                    # all deocde tasks should arrive before current time
                    assert t_cur >= task.t_arrival
                logging.debug(f"Scheduler (t={t_cur}): add {len(decode_fifo)} decode tasks to decode pool")
                self.decode_pool.new_tasks(decode_fifo)
            if self.prefill_pool.is_empty() and prefill_fifo != [] and accept_new_req:
                if t_cur >= prefill_fifo[0].t_arrival:
                    # there is a prefill task arriving before current time
                    new_prefill_tasks = []
                    while t_cur >= prefill_fifo[0].t_arrival:
                        new_prefill_tasks.append(prefill_fifo.pop(0))
                        if prefill_fifo == []:
                            break
                    self.prefill_pool.new_tasks(new_prefill_tasks)
                    logging.debug(f"Scheduler (t={t_cur}): add {len(new_prefill_tasks)} prefill tasks to prefill pool")
                elif self.decode_pool.is_empty():
                    # no decode task in the pool, update current time to the prefill task arrival time
                    new_prefill_task = prefill_fifo.pop(0)
                    logging.debug(f"Scheduler (t={t_cur}): update current time to {new_prefill_task.t_arrival}")
                    t_cur = new_prefill_task.t_arrival
                    logging.debug(f"Scheduler (t={t_cur}): add prefill task to prefill pool")
                    self.prefill_pool.new_tasks([new_prefill_task])
            if accept_new_req:
                sim_kernel = SimKernel(self.prefill_pool.schedule(), self.decode_pool.schedule())
            else:
                sim_kernel = SimKernel(None, self.decode_pool.schedule())
        elif self.algo == 'prefetch-mixed':
            # like mixed continuous batching, prefill and decode are always in the same layer
            # but we prefetch KV cache (ONLY!!) for decode during FC layer computation
            # only 1 layer of KV in 3D memory, so capacity should not be a problem
            # adjust the prefill chunk size to make sure we have enough time for prefetching
            # the first layer needs longer time, since there is only QKF proj for prefetching

            # if the decode pool is empty, add decode tasks from the fifo to the pool
            if self.decode_pool.is_empty() and decode_fifo != []:
                for task in decode_fifo:
                    # all deocde tasks should arrive before current time
                    assert t_cur >= task.t_arrival
                logging.debug(f"Scheduler (t={t_cur}): add {len(decode_fifo)} decode tasks to decode pool")
                self.decode_pool.new_tasks(decode_fifo)
            
            if self.prefill_pool.is_empty() and prefill_fifo != [] and accept_new_req:
                if t_cur >= prefill_fifo[0].t_arrival:
                    # there are prefill tasks arriving before current time
                    new_prefill_tasks = []
                    while t_cur >= prefill_fifo[0].t_arrival:
                        new_prefill_tasks.append(prefill_fifo.pop(0))
                        if prefill_fifo == []:
                            break
                    self.prefill_pool.new_tasks(new_prefill_tasks)
                    logging.debug(f"Scheduler (t={t_cur}): add {len(new_prefill_tasks)} prefill tasks to prefill pool")
                elif self.decode_pool.is_empty():
                    # there is no decode task in the pool
                    # if decode_pool is not empty, we should just run the decode tasks without waiting for prefill
                    # if no decode task in the pool, update current time to the next prefill task arrival time
                    new_prefill_task = prefill_fifo.pop(0)
                    logging.debug(f"Scheduler (t={t_cur}): update current time to {new_prefill_task.t_arrival}")
                    t_cur = new_prefill_task.t_arrival
                    logging.debug(f"Scheduler (t={t_cur}): add prefill task to prefill pool")
                    self.prefill_pool.new_tasks([new_prefill_task])

            if not self.prefill_pool.is_empty():
                if not self.decode_pool.is_empty():
                    # both prefill and decode are running, decode pool l_cur should be 0
                    assert self.decode_pool.l_cur == 0
                    if accept_new_req:
                        # max_n = (self.prefill_pool.n - self.prefill_pool.n_cur) + self.decode_pool.n
                        max_n = (self.prefill_pool.n - self.prefill_pool.n_cur)
                        if max_n < self.prefill_chunk:
                            # not enough tokens to prefetch, run normal mixed continuous batching without prefetching
                            prefetch = None
                            prefill_kernel = self.prefill_pool.schedule(self.prefill_pool.n - self.prefill_pool.n_cur)
                        else:
                            prefetch = 'layer-kv'
                            # prefill_kernel = self.prefill_pool.schedule(self.prefill_chunk - self.decode_pool.n)
                            prefill_kernel = self.prefill_pool.schedule(self.prefill_chunk)
                    else:
                        prefetch = None
                        prefill_kernel = None
                    decode_kernel = self.decode_pool.schedule()
                else:
                    prefetch = None
                    prefill_kernel = self.prefill_pool.schedule(self.prefill_pool.n - self.prefill_pool.n_cur)
                    decode_kernel = None
            elif not self.decode_pool.is_empty():
                prefetch = None
                prefill_kernel = None
                decode_kernel = self.decode_pool.schedule()            
            else:
                raise ValueError("Both prefill and decode pools are empty")
            sim_kernel = SimKernel(prefill_kernel, decode_kernel, prefetch)
        elif self.algo == 'prefetch-thread':
            # thread switching between prefill and decode
            # prefill all layers, decode some layers, prefetch KV and weights for decode
            # num of decode layers depends on the 3D memory size
            # adjust the prefill chunk size to make sure we have enough time for prefetching

            # if the decode pool is empty, add decode tasks from the fifo to the pool
            if self.decode_pool.is_empty() and decode_fifo != []:
                for task in decode_fifo:
                    # all deocde tasks should arrive before current time
                    assert t_cur >= task.t_arrival
                # limit the decode pool size
                max_decode_tasks = 2048
                if len(decode_fifo) > max_decode_tasks:
                    logging.debug(f"Scheduler (t={t_cur}): add {max_decode_tasks} decode tasks to decode pool")
                    decode_tasks = decode_fifo[:max_decode_tasks]
                    self.decode_pool.new_tasks(decode_tasks)
                    # pop the first 2048 tasks from the decode fifo
                    # for some reason, decode_fifo = decode_fifo[2048:] does not work
                    for i in range(max_decode_tasks):
                        decode_fifo.pop(0)
                else:
                    logging.debug(f"Scheduler (t={t_cur}): add {len(decode_fifo)} decode tasks to decode pool")
                    self.decode_pool.new_tasks(decode_fifo)
            
            if self.prefill_pool.is_empty() and prefill_fifo != [] and accept_new_req:
                if t_cur >= prefill_fifo[0].t_arrival:
                    # there are prefill tasks arriving before current time
                    new_prefill_tasks = []
                    while t_cur >= prefill_fifo[0].t_arrival:
                        new_prefill_tasks.append(prefill_fifo.pop(0))
                        if prefill_fifo == []:
                            break
                    self.prefill_pool.new_tasks(new_prefill_tasks)
                    logging.debug(f"Scheduler (t={t_cur}): add {len(new_prefill_tasks)} prefill tasks to prefill pool")
                elif self.decode_pool.is_empty():
                    # there is no decode task in the pool
                    # if decode_pool is not empty, we should just run the decode tasks without waiting for prefill
                    # if no decode task in the pool, update current time to the next prefill task arrival time
                    new_prefill_task = prefill_fifo.pop(0)
                    logging.debug(f"Scheduler (t={t_cur}): update current time to {new_prefill_task.t_arrival}")
                    t_cur = new_prefill_task.t_arrival
                    logging.debug(f"Scheduler (t={t_cur}): add prefill task to prefill pool")
                    self.prefill_pool.new_tasks([new_prefill_task])

            if not self.prefill_pool.is_empty():
                if not self.decode_pool.is_empty():
                    if accept_new_req:
                        max_n = self.prefill_pool.n - self.prefill_pool.n_cur
                        if max_n < self.prefill_chunk:
                            # not enough tokens to prefetch, run normal mixed continuous batching without prefetching
                            prefetch = None
                            prefill_kernel = self.prefill_pool.schedule(self.prefill_pool.n - self.prefill_pool.n_cur)
                            decode_kernel = self.decode_pool.schedule(self.decode_pool.l - self.decode_pool.l_cur)
                        else:
                            prefetch = 'weights-kv'
                            prefill_kernel = self.prefill_pool.schedule(self.prefill_chunk)
                            prefetch_l = self.decode_pool.get_prefetch_layers(self.mem_3d_size)
                            decode_kernel = self.decode_pool.schedule(prefetch_l)
                    else:
                        prefetch = None
                        prefill_kernel = None
                        decode_kernel = self.decode_pool.schedule(self.decode_pool.l - self.decode_pool.l_cur)
                else:
                    prefetch = None
                    prefill_kernel = self.prefill_pool.schedule(self.prefill_pool.n - self.prefill_pool.n_cur)
                    decode_kernel = None
            elif not self.decode_pool.is_empty():
                prefetch = None
                prefill_kernel = None
                decode_kernel = self.decode_pool.schedule(self.decode_pool.l - self.decode_pool.l_cur)
            else:
                raise ValueError("Both prefill and decode pools are empty")
            sim_kernel = SimKernel(prefill_kernel, decode_kernel, prefetch)

        return sim_kernel, t_cur
    
    def is_done(self):
        return self.prefill_pool.is_empty() and self.decode_pool.is_empty()
    
    def update(self, time: int, requests: Dict[int, Request], decode_fifo: List[Task]):
        # find all finished tasks in prefill pool
        n_finished = self.prefill_pool.n_cur
        n_total = 0
        finished_tasks = 0
        for task in self.prefill_pool.tasks:
            n_total += task.req.input_len
            if n_total <= n_finished:
                # task is finished
                finished_tasks += 1
                req_id = task.req.id
                requests[req_id].output_one_token(time)
                decode_task = DecodeTask(task.req, 
                                         time, 0)
                decode_fifo.append(decode_task)
                logging.debug(f"Update (t={time}): request {req_id} prefill finished, add decode task to decode fifo (l={len(decode_fifo)})")
            else:
                break
        # remove finished tasks from prefill pool
        for i in range(finished_tasks):
            self.prefill_pool.n -= self.prefill_pool.tasks[i].req.input_len
            self.prefill_pool.n_cur -= self.prefill_pool.tasks[i].req.input_len
        self.prefill_pool.tasks = self.prefill_pool.tasks[finished_tasks:]

        if self.prefill_pool.is_done():
            self.prefill_pool.reset()

        if self.decode_pool.is_done():
            for task in self.decode_pool.tasks:
                req_id = task.req.id
                finished = requests[req_id].output_one_token(time)
                if not finished:
                    decode_task = DecodeTask(task.req, 
                                             time, 
                                             task.n_cur + 1)
                    decode_fifo.append(decode_task)
                    logging.debug(f"Update (t={time}): request {req_id} output one token, add decode task to decode fifo (l={len(decode_fifo)})")
                else:
                    logging.debug(f"Update (t={time}): request {req_id} finished")

            self.decode_pool.reset()
            # if decode_fifo != []:
            #     logging.debug(f"Update (t={time}): add {len(decode_fifo)} decode tasks from fifo to pool")
            #     self.decode_pool.new_tasks(decode_fifo)
            # else:
            #     self.decode_pool.reset()