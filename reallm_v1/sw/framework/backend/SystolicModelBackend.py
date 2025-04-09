from math import ceil
from .expr import *
from bsg.utils.BSGBaseDataclass import BSGBaseDataclass


@dataclass
class LayerPerformance( BSGBaseDataclass ):
    compute_lat: float


class SystolicModelBackend:

    def __init__( self, network, arch ) -> None:
        self.network = network
        self.arch = arch


    def run( self ):
        result = {}
        for E in self.network.iter():
            kwargs = {}
            kwargs.update(E.out_args())
            kwargs.update(E.in_args())
            kwargs.update(E.attr_args())
            kwargs.update({P:self.network.tensors[T].shape for P,T in E.in_args().items() if T is not None})
            kwargs.update({P:self.network.tensors[T].shape for P,T in E.out_args().items() if T is not None})
            func = self.get_func(E)
            layer_res = func(**kwargs)
            result[E.id] = layer_res
        return result


    def get_func( self, expr ):
        return {
            Input:          self.input,
            Param:          self.param,
            Output:         self.output,
            Constant:       self.constant,
            Reshape:        self.reshape,
            Matmul:         self.matmul,
            Relu:           self.relu,
            Batchnorm:      self.batchnorm,
            Conv2D:         self.conv2d,
            Maxpool:        self.maxpool,
            Add:            self.add,
            GlobalAvgpool:  self.globalavgpool,
            Gemm:           self.gemm,
        } [type(expr)]


    def input( self, Y ):
        return None


    def param( self, Y, value ):
        return None


    def output( self, X ):
        return None


    def constant( self, output, value ):
        return None


    def reshape( self, reshaped, data, shape, allowzero ):
        return None


    def matmul( self, Y, A, B ):
        return None


    def relu( self, Y, X ):
        return None


    def batchnorm( self, Y, running_mean, running_var, X, scale, B, input_mean, input_var, epsilon, momentum, training_mode ):
        return None


    def conv2d( self, Y, X, W, B, kernel_shape, strides, pads, dilations, group, auto_pad ):

        SA_DWIDTH = self.arch.D_sa_act 
        SA_H      = self.arch.H_sa
        SA_W      = self.arch.W_sa
        ACC_DEPTH = self.arch.N_accum_ch_els

        n,c,ih,iw = X
        k,c,fh,fw = W
        n,k,oh,ow = Y

        fh, fw = kernel_shape
        dh, dw = dilations
        sh, sw = strides
        ph0, pw0, ph1, pw1 = pads


        num_macs     = n * k * oh * ow * fh * fw * c
        wt_intensity = n * oh * ow

        is_cb = (self.arch.N_wi_poi < wt_intensity)

        if c % SA_H != 0: c += SA_H - (c % SA_H)
        if k % SA_W != 0: k += SA_W - (k % SA_W)

        num_blocks  = ceil(wt_intensity / ACC_DEPTH) * ceil(c/SA_H) * ceil(k/SA_W) * fh * fw
        block_depth = min(wt_intensity, ACC_DEPTH)

        weight_block_load_lat = (SA_H * SA_W * SA_DWIDTH) / self.arch.WL_bw

        write_back_lat = block_depth / self.arch.f_sa

        if is_cb:
            compute_lat = weight_block_load_lat + ((num_blocks*block_depth+SA_H+SA_W+2)/self.arch.f_sa) + write_back_lat
        else:
            compute_lat = (weight_block_load_lat * num_blocks) + (block_depth+SA_H+SA_W+2)/self.arch.f_sa + write_back_lat

        compute_perf = num_macs / compute_lat
        ideal_lat    = num_macs / self.arch.GEMM_perf_peak
        mac_util     = compute_perf / self.arch.GEMM_perf_peak

        return LayerPerformance( compute_lat = compute_lat )


    def maxpool( self, Y, Indices, X, kernel_shape, pads, strides, dilations, auto_pad, ceil_mode, storage_order ):
        SA_H = self.arch.H_sa
        SA_W = self.arch.W_sa

        n,c,ih,iw = X
        n,k,oh,ow = Y

        fh, fw = kernel_shape
        dh, dw = dilations
        sh, sw = strides
        ph0, pw0, ph1, pw1 = pads

        if c % SA_H != 0: c += SA_H - (c % SA_H)
        if k % SA_W != 0: k += SA_W - (k % SA_W)

        num_vectors = ceil(c/SA_H) * ceil(k/SA_W) * fh * fw
        fwd_cycles = num_vectors + SA_H + SA_W + 2
        fwd_latency = fwd_cycles / self.arch.f_sa

        write_back_lat = oh*ow*n*ceil(k/SA_W) / self.arch.f_sa

        compute_lat = fwd_latency + write_back_lat
        compute_perf = (n*k*oh*ow*fh*fw) / compute_lat

        return LayerPerformance( compute_lat = compute_lat )


    def add( self, C, A, B ):
        SA_H = self.arch.H_sa
        SA_W = self.arch.W_sa

        n,c,ih,iw = A

        if c % SA_H != 0: c += SA_H - (c % SA_H)

        num_vectors = 2*n*ih*iw*max(ceil(c/SA_H), ceil(c/SA_W))
        fwd_cycles = num_vectors + SA_H + SA_W + 2
        fwd_latency = fwd_cycles / self.arch.f_sa
        write_back_lat = num_vectors / self.arch.f_sa

        compute_lat = fwd_latency + write_back_lat
        compute_perf = (n*c*ih*iw) / compute_lat

        return LayerPerformance( compute_lat = compute_lat )


    def globalavgpool( self, Y, X ):
        SA_H = self.arch.H_sa
        SA_W = self.arch.W_sa

        n,c,ih,iw = X
        n,k,oh,ow = n,c,1,1
        fh,fw = ih,iw

        if c % SA_H != 0: c += SA_H - (c % SA_H)
        if k % SA_W != 0: k += SA_W - (k % SA_W)

        num_vectors = ceil(c/SA_H) * ceil(k/SA_W) * fh * fw
        fwd_cycles = num_vectors + SA_H + SA_W + 2
        fwd_latency = fwd_cycles / self.arch.f_sa

        write_back_lat = oh*ow*n*ceil(k/SA_W) / self.arch.f_sa

        compute_lat = fwd_latency + write_back_lat
        compute_perf = (n*k*oh*ow*fh*fw) / compute_lat

        return LayerPerformance( compute_lat = compute_lat )


    def gemm( self, Y, A, B, C, alpha, beta, transA, transB ):

        SA_DWIDTH = self.arch.D_sa_act 
        SA_H      = self.arch.H_sa
        SA_W      = self.arch.W_sa
        ACC_DEPTH = self.arch.N_accum_ch_els

        i, j = A[0 if not transA else 1], A[1 if not transA else 0]
        j, k = B[0 if not transB else 1], B[1 if not transB else 0]

        num_macs     = i * k * j
        wt_intensity = i

        is_cb = (self.arch.N_wi_poi < wt_intensity)

        if j % SA_H != 0: j += SA_H - (j % SA_H)
        if k % SA_W != 0: k += SA_W - (k % SA_W)

        ### blocking ###

        i_bar = ceil( i / ceil(ACC_DEPTH / 2) )
        j_bar = ceil( j / SA_H )
        k_bar = ceil( k / SA_W )
        s     = ceil( i / i_bar )

        weight_block_load_lat = (SA_H * SA_W * SA_DWIDTH) / self.arch.WL_bw

        num_blocks  = i_bar * j_bar * k_bar
        write_back_lat = s / self.arch.f_sa

        if is_cb:
            compute_lat = weight_block_load_lat + ((num_blocks* s +SA_H+SA_W+2)/self.arch.f_sa) + write_back_lat
        else:
            compute_lat = (weight_block_load_lat * num_blocks) + (s +SA_H+SA_W+2)/self.arch.f_sa + write_back_lat

        compute_perf = num_macs / compute_lat
        ideal_lat    = num_macs / self.arch.GEMM_perf_peak
        mac_util     = compute_perf / self.arch.GEMM_perf_peak

        return LayerPerformance( compute_lat = compute_lat )

