
import math

class HardwareNode:
    def __init__(self, name, flops, mem_bw, mem_size, io_alpha, io_bw,
                 mem_3d_bw = 0, mem_3d_size = 0):
        self.name = name
        self.flops = flops
        self.mem_bw = mem_bw # bytes per second
        self.mem_size = mem_size # bytes
        self.io_alpha = io_alpha # seconds
        self.io_bw = io_bw # bytes per second
        self.mem_3d_bw = mem_3d_bw
        self.mem_3d_size = mem_3d_size

        self.tflops = flops / 1e12
        self.mem_bw_gb = mem_bw / 1e9
        self.mem_size_gb = mem_size / 1e9
        self.io_beta = 1 / io_bw # seconds to transfer 1 byte
        self.mem_3d_bw_gb = mem_3d_bw / 1e9
        self.mem_3d_size_gb = mem_3d_size / 1e9

class Hardware:
    def __init__(self, node: HardwareNode, num_nodes: int, 
                 parallelism, io_algo: str = ''):
        self.node = node
        self.num_nodes = num_nodes
        self.io_algo = io_algo
        self.parallelism = parallelism # 'ep#_tp#_pp#_cp#'

        for para_type in ['ep', 'tp', 'pp', 'cp', 'dp']:
            if para_type in parallelism:
                self.__setattr__(para_type, int(parallelism.split(para_type)[1].split('_')[0]))
            else:
                self.__setattr__(para_type, 1)

        if num_nodes % self.pp != 0:
            raise ValueError(f"num_nodes {num_nodes} must be divisible by pp {self.pp}")
        
        if self.ep * self.tp * self.pp * self.cp != num_nodes:
            raise ValueError(f"num_nodes {num_nodes} must be equal to ep {self.ep} * tp {self.tp} * pp {self.pp} * cp {self.cp}")

        self.flops = node.flops * num_nodes
        self.mem_bw = node.mem_bw * num_nodes
        self.mem_size = node.mem_size * num_nodes
        self.mem_3d_bw = node.mem_3d_bw * num_nodes
        self.mem_3d_size = node.mem_3d_size * num_nodes
        self.tflops = self.flops / 1e12
        self.mem_bw_gb = self.mem_bw / 1e9
        self.mem_size_gb = self.mem_size / 1e9
        self.mem_3d_bw_gb = self.mem_3d_bw / 1e9
        self.mem_3d_size_gb = self.mem_3d_size / 1e9
    
    def get_allreduce_latency(self, num_bytes: int, num_nodes = None) -> float:
        alpha = self.node.io_alpha
        beta = self.node.io_beta
        if num_nodes is None:
            p = self.tp
        else:
            p = num_nodes
        n = num_bytes
        if num_nodes == 1:
            return 0.0
        if self.io_algo == '':
            t = 0.0
        elif self.io_algo == 'ring':
            t = 2 * (p-1) * (alpha + beta * n / p)
        elif self.io_algo == '2d_ring':
            # 2D Ring Allreduce
            sqrt_p = math.ceil(math.sqrt(p))
            t = 2 * (sqrt_p - 1) * (2 * alpha + beta * n / sqrt_p)
        elif self.io_algo == 'multishot':
            t = 2 * (alpha + beta * n)
        else:
            raise ValueError(f"Unknown IO algorithm {self.io_algo}")
        return t

H100 = HardwareNode(
    name="H100",
    flops=1000.0e12, # 1000 TFLOPS
    mem_bw=3.0e12, # 3 TB/s
    mem_size=80.0e9, # 80 GB
    io_alpha=0.92e-6, # 8.92 us, from LLMCompass
    io_bw=450.0e9, # 450 GB/s per direction, 4th gen NVLink
)

A100 = HardwareNode(
    name="A100",
    flops=1000.0e12, # 1000 TFLOPS
    mem_bw=1.5e12, # 1.5 TB/s
    mem_size=80.0e9, # 80 GB
    io_alpha=5.92e-6, # 8.92 us, from LLMCompass
    io_bw=300.0e9, # 450 GB/s per direction, 4th gen NVLink
)

H100_3D = HardwareNode(
    name="h100",
    flops=1000.0e12, # 1000 TFLOPS
    mem_bw=3.0e12, # 3 TB/s
    mem_size=80.0e9, # 80 GB
    io_alpha=8.92e-6, # 8.92 us, from LLMCompass
    io_bw=450.0e9, # 450 GB/s per direction, 4th gen NVLink
    mem_3d_bw=18.0e12, # 18 TB/s
    mem_3d_size=9.0e9, # 9 GB
)

Our_3D = HardwareNode(
    name="our",
    flops=713.7e12, # 713.7
    mem_bw=3.0e12, # 3 TB/s
    mem_size=80.0e9, # 80 GB
    io_alpha=8.92e-6, # 8.92 us, from LLMCompass
    io_bw=450.0e9, # 450 GB/s per direction, 4th gen NVLink
    mem_3d_bw=18.0e12, # 18 TB/s
    mem_3d_size=9.0e9, # 9 GB
)

System_Num_Nodes = {'llama3-70B':  {'8k': 16, '32k': 32, '128k': 64},
                    'llama3-405B': {'8k': 32, '32k': 64, '128k': 128},
                    'opt-175B':    {'8k': 24, '32k': 48, '128k': 96}}