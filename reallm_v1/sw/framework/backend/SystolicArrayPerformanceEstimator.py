from __future__ import annotations

from bsg.framework.Network import *
from bsg.framework.backend.ShapeInference import ShapeInference
from bsg.utils.BSGBaseDataclass import BSGBaseDataclass
from math import ceil
from typing import Any, Dict
import copy
import numpy as np
import sys
import os

################################################################################
#
@dataclass
class GemmLayerPerf( BSGBaseDataclass ):
    name                 :  str    =  None
    type                 :  str    =  None

    I                    :  int    =  None
    J                    :  int    =  None
    K                    :  int    =  None
    I_BAR                :  int    =  None
    J_BAR                :  int    =  None
    K_BAR                :  int    =  None
    K_BAR_WB             :  int    =  None
    S                    :  int    =  None

    num_macs             :  int    =  None
    wt_intensity         :  int    =  None
    is_compute_bound     :  bool   =  None

    wt_block_lat         :  float  =  None
    act_block_lat        :  float  =  None
    comp_push_block_lat  :  float  =  None
    comp_trav_block_lat  :  float  =  None

    num_blocks_A         :  int    =  None
    num_blocks_W         :  int    =  None
    num_blocks_O         :  int    =  None

    compute_lat          :  float  =  None

    gb_reads :  int    =  None
    wt_reads :  int    =  None

    compute_perf         :  float  =  None
    ideal_lat            :  float  =  None
    mac_util             :  float  =  None


################################################################################
#
@dataclass
class Conv2DLayerPerf( BSGBaseDataclass ):
    name                 :  str    =  None
    type                 :  str    =  None

    N                    :  int    =  None
    C                    :  int    =  None
    K                    :  int    =  None
    IH                   :  int    =  None
    IW                   :  int    =  None
    FH                   :  int    =  None
    FW                   :  int    =  None
    OH                   :  int    =  None
    OW                   :  int    =  None
    PH0                  :  int    =  None
    PW0                  :  int    =  None
    PH1                  :  int    =  None
    PW1                  :  int    =  None
    SH                   :  int    =  None
    SW                   :  int    =  None
    DH                   :  int    =  None
    DW                   :  int    =  None

    num_macs             :  int    =  None
    wt_intensity         :  int    =  None
    is_compute_bound     :  bool   =  None

    wt_block_lat         :  float  =  None
    act_block_lat        :  float  =  None
    comp_push_block_lat  :  float  =  None
    comp_trav_block_lat  :  float  =  None

    num_blocks_A         :  int    =  None
    num_blocks_W         :  int    =  None
    num_blocks_O         :  int    =  None

    gb_reads :  int    =  None
    wt_reads :  int    =  None

    compute_lat          :  float  =  None

    compute_perf         :  float  =  None
    ideal_lat            :  float  =  None
    mac_util             :  float  =  None


################################################################################
#
@dataclass
class ActivationLayerPerf( BSGBaseDataclass ):
    name          :  str    =  None
    type          :  str    =  None

    N             :  int    =  None
    C             :  int    =  None
    H             :  int    =  None
    W             :  int    =  None

    num_ops       :  int    =  None
    num_vectors   :  int    =  None

    fwd_cycles    :  int    =  None
    act_cycles    :  int    =  None
    rev_cycles    :  int    =  None
    total_cycles  :  int    =  None

    compute_lat   :  float  =  None

    compute_perf  :  float  =  None
    ideal_lat     :  float  =  None
    mac_util      :  float  =  None


################################################################################
#
class SystolicArrayPerformanceEstimator:

    ########################################################
    #
    def __init__( self, network, arch ) -> None:
        self.network     = network
        self.arch        = arch
        self.alloc       = SystolicArrayPerformanceEstimatorAllocator(self.network)
        self.shape_infer = ShapeInference(self.network)

    ########################################################
    #
    def run( self, in_dict ) -> None:
        self.shape_infer.run(in_dict)

        perfs = {}
        for E in self.network.iter():
            kwargs = { **E.get_output(symtable=self.alloc)
                     , **E.get_inputs(symtable=self.alloc)
                     , **E.get_attrs()
                     }
            rseult = None
            if E.type in self._cap:
                try:
                    kernel = self._cap[E.type]
                    result = kernel(self, E, **kwargs)
                except:
                    pass
            if result is None:
                if E.type not in ["Param", "Transpose", "Reshape", "ReshapeStatic"]:
                    print( f"Warning, skipping perf est for {E}", file=sys.stderr )
            else:
                perfs[E.id] = result

        return perfs

    ########################################################
    #
    def add( self, expr, Z, A, B ):

        # current limitation, shapes have to be equal (no broadcasting)
        assert A == B

        ### ARCH PARAMS ###

        SA_act_width = self.arch.SA_act_width
        SA_H         = self.arch.SA_H
        SA_W         = self.arch.SA_W
        SA_freq      = self.arch.SA_freq
        ACC_ch_els   = self.arch.ACC_ch_els
        WL_bw        = self.arch.WL_bw
        GB_bw        = self.arch.GB_bw
        N_wi_poi     = self.arch.N_wi_poi
        SA_HW_min    = min(SA_H, SA_W)

        ### EST. PERF ###

        perf = ActivationLayerPerf()
        perf.name = expr.id
        perf.type = expr.type

        perf.N, perf.C, perf.H, perf.W = A

        perf.num_ops = perf.N * perf.C * perf.H * perf.W

        if perf.C % SA_H != 0:
            perf.C += SA_H - (perf.C % SA_H)

        perf.num_vectors = 2 * perf.N * perf.H * perf.W * ceil( perf.C / SA_HW_min )

        perf.fwd_cycles   = perf.num_vectors + SA_H + SA_W + 2
        perf.act_cycles   = 1
        perf.rev_cycles   = perf.fwd_cycles
        perf.total_cycles = perf.fwd_cycles + perf.act_cycles + perf.rev_cycles

        perf.compute_lat = (perf.total_cycles / SA_freq) * 1e9

        perf.compute_perf = (perf.num_ops / perf.compute_lat) * 1e9
        perf.ideal_lat    = (perf.num_ops / self.arch.AU_perf_peak) * 1e9
        perf.mac_util     = perf.compute_perf / self.arch.AU_perf_peak

        return perf

    ########################################################
    #
    def batchnorm( self, expr, Z, A, gamma, beta, mean, var, epsilon ):

        ### ARCH PARAMS ###

        SA_act_width = self.arch.SA_act_width
        SA_H         = self.arch.SA_H
        SA_W         = self.arch.SA_W
        SA_freq      = self.arch.SA_freq
        ACC_ch_els   = self.arch.ACC_ch_els
        WL_bw        = self.arch.WL_bw
        GB_bw        = self.arch.GB_bw
        N_wi_poi     = self.arch.N_wi_poi
        SA_HW_min    = min(SA_H, SA_W)

        ### EST. PERF ###

        perf = ActivationLayerPerf()
        perf.name = expr.id
        perf.type = expr.type

        perf.N, perf.C, perf.H, perf.W = A

        perf.num_ops = perf.N * perf.C * perf.H * perf.W

        if perf.C % SA_H != 0:
            perf.C += SA_H - (perf.C % SA_H)

        #perf.I = perf.N * perf.H * perf.W
        #perf.J = perf.C

        #perf.I_BAR = ceil( perf.I / ACC_ch_els )
        #perf.J_BAR = ceil( perf.J / SA_HW_min )
        #perf.S     = ceil( perf.I / perf.I_BAR )

        perf.num_vectors = perf.N * perf.H * perf.W * ceil( perf.C / SA_HW_min )

        perf.fwd_cycles   = perf.num_vectors + SA_H + SA_W + 2
        perf.act_cycles   = 1
        perf.rev_cycles   = perf.fwd_cycles
        perf.total_cycles = perf.fwd_cycles + perf.act_cycles + perf.rev_cycles

        perf.compute_lat = (perf.total_cycles / SA_freq) * 1e9

        perf.compute_perf = (perf.num_ops / perf.compute_lat) * 1e9
        perf.ideal_lat    = (perf.num_ops / self.arch.AU_perf_peak) * 1e9
        perf.mac_util     = perf.compute_perf / self.arch.AU_perf_peak

        return perf

    ########################################################
    #
    def conv2d( self, expr, Z, A, W, B, pads, strides, dilations ):

        #fid = open(f"trace_{expr.id}.txt","w")

        ### ARCH PARAMS ###
        SA_act_width = self.arch.SA_act_width
        SA_H         = self.arch.SA_H
        SA_W         = self.arch.SA_W
        SA_freq      = self.arch.SA_freq
        ACC_ch_els   = self.arch.ACC_ch_els
        WL_bw        = self.arch.WL_bw
        GB_bw        = self.arch.GB_bw
        N_wi_poi     = self.arch.N_wi_poi
        SA_HW_min    = min(SA_H, SA_W)

        N,C,IH,IW = A
        K,C,FH,FW = W
        N,K,OH,OW = Z

        orig_C = C
        orig_K = K

        PH0, PW0, PH1, PW1 = pads
        SH, SW = strides
        DH, DW = dilations

        num_macs = N * K * OH * OW * FH * FW * C

        if not self.arch.GB_has_im2col:
            if C % SA_H != 0:
                C += SA_H - (C % SA_H)

            if K % SA_W != 0:
                K += SA_W - (K % SA_W)

        else:
            pass

        gemm = self.gemm( expr=expr
                        , Z=(OH*OW, K)
                        , A=(OH*OW, C*FH*FW)
                        , B=(C*FH*FW, K)
                        , C=None
                        , trace=False
                        )

        # Don't use these from gemm, but recompute with our num_macs which is
        # calculated is before the C/K padding.
        compute_perf = (num_macs / gemm.compute_lat) * 1e9
        ideal_lat    = (num_macs / self.arch.GEMM_perf_peak) * 1e9
        mac_util     = compute_perf / self.arch.GEMM_perf_peak

        #fid.close()

        return Conv2DLayerPerf ( N=N, C=orig_C, K=orig_K, IH=IH, IW=IW, FH=FH, FW=FW, OH=OH, OW=OW
                               , PH0=PH0, PW0=PW0, PH1=PH1, PW1=PW1
                               , SH=SH, SW=SW, DH=DH, DW=DW

                               , num_macs = num_macs
                               , wt_intensity = gemm.wt_intensity
                               , is_compute_bound = gemm.is_compute_bound

                               , wt_block_lat = gemm.wt_block_lat
                               , act_block_lat = gemm.act_block_lat
                               , comp_push_block_lat = gemm.comp_push_block_lat
                               , comp_trav_block_lat = gemm.comp_trav_block_lat

                               , num_blocks_A = gemm.num_blocks_A
                               , num_blocks_W = gemm.num_blocks_W
                               , num_blocks_O = gemm.num_blocks_O

                               , compute_lat = gemm.compute_lat
                               , compute_perf = compute_perf
                               , ideal_lat = ideal_lat
                               , mac_util = mac_util

                               , gb_reads=gemm.gb_reads
                               , wt_reads=gemm.wt_reads

                                  , name = expr.id
                                  , type = expr.type
                               )

    ########################################################
    #
    def gemm( self, expr, Z, A, B, C, trace=True ):
        
        if len(A) > 2:
            foo = self.gemm(expr, Z[-2:], A[-2:], B[-2:], C, trace)
            m = 1
            for a in A[:-2]:
                m = m * a
            return foo * m

        #if trace:
        #    #fid = open(f"trace_{expr.id}.txt","w")
        #else:
        #    #fid = open(os.devnull, "w")

        ### ARCH PARAMS ###
        SA_act_width = self.arch.SA_act_width
        SA_H         = self.arch.SA_H
        SA_W         = self.arch.SA_W
        SA_freq      = self.arch.SA_freq
        ACC_ch_els   = self.arch.ACC_ch_els
        WL_bw        = self.arch.WL_bw
        GB_bw        = self.arch.GB_bw
        N_wi_poi     = self.arch.N_wi_poi
        SA_HW_min    = min(SA_H, SA_W)

        I, J = A
        J, K = B
        assert Z[0] == I and Z[1] == K, f"{Z} != ({I},{K})"

        I_BAR    = ceil( I / ceil(ACC_ch_els / 2) )
        J_BAR    = ceil( J / SA_H )
        K_BAR    = ceil( K / SA_W )
        K_BAR_WB = ceil( K / SA_H )
        S        = ceil( I / I_BAR )

        num_macs         = I * J * K
        wt_intensity     = I
        is_compute_bound = (N_wi_poi < wt_intensity)

        wt_block_lat        = int(((SA_H * SA_W * SA_act_width / 8) / WL_bw) * 1e9)
        act_block_lat       = int(((SA_H * S / 8) / GB_bw) * 1e9)
        comp_push_block_lat = int(((S) / SA_freq) * 1e9)
        comp_trav_block_lat = int(((SA_H + SA_W + 2) / SA_freq) * 1e9)

        num_blocks_A = I_BAR * J_BAR
        num_blocks_W = J_BAR * K_BAR
        num_blocks_O = I_BAR * K_BAR

        block_schedule = []
        for i in range(I_BAR):
            for k in range(K_BAR):
                for j in range(J_BAR):
                    A_name = "I" + str((i,j))
                    W_name = "W" + str((j,k))
                    O_name = "O" + str((i,k))
                    block_schedule.append( (A_name, W_name, O_name) )

        wt_buf    = [ None, None ]
        wt_lock   = [ 0, 0 ]
        wt_ptr    = 0

        gb_reads = 0
        wt_reads = 0

        sa_edge_lock = 0

        current_time = 0

        for A_name,W_name,O_name in block_schedule:

            if W_name:
                # new weight block to load into the SA
                if W_name not in wt_buf:
                    # make sure the weight buffer is not being used (ie. the
                    # computation using the weight buffer is complete).
                    locked_until = wt_lock[wt_ptr]
                    if current_time < locked_until:
                        #fid.write(f"{current_time}:\t*STALLED WL -> WT[{wt_ptr}] (until {locked_until})\n")
                        current_time = locked_until

                    # load the weight block and swap wt pointer
                    #fid.write(f"{current_time}:\tLD {W_name} => WT[{wt_ptr}] \n")
                    wt_buf[wt_ptr] = W_name
                    wt_lock[wt_ptr] = current_time + wt_block_lat
                    wt_ptr = int(not(wt_ptr))

                    wt_reads += (SA_H * SA_W)

            if O_name:
                # get the weight ptr
                sa_ptr = wt_buf.index(W_name)

                # stall if the previous block is still being pushed into the
                # systolic array
                if current_time < sa_edge_lock:
                    #fid.write(f"{current_time}:\t*STALLED COMPUTE -> EDGE LOCK (until {sa_edge_lock})\n")
                    current_time = sa_edge_lock

                # stall if the weight block is not ready
                locked_until = wt_lock[sa_ptr]
                if current_time < locked_until:
                    #fid.write(f"{current_time}:\t*STALLED COMPUTE -> WT[{sa_ptr}] (until {locked_until})\n")
                    current_time = locked_until

                # start the computation
                #fid.write(f"{current_time}:\tCOMPUTE {A_name} @ {W_name} => {O_name} \n")
                sa_edge_lock    = current_time + comp_push_block_lat
                wt_lock[sa_ptr] = sa_edge_lock + comp_trav_block_lat

                gb_reads += (SA_H * S)

        #fid.close()

        #                                      This is the time it takes to move the last
        #                                      block from acc to gb (same as a forward
        #                                      pass compute, just in reverse)
        #                                      vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        simulated_compute_lat = max(wt_lock) + (comp_push_block_lat + comp_trav_block_lat)

        compute_perf = (num_macs / simulated_compute_lat) * 1e9
        ideal_lat    = (num_macs / self.arch.GEMM_perf_peak) * 1e9
        mac_util     = compute_perf / self.arch.GEMM_perf_peak

        return GemmLayerPerf ( I = I
                             , J = J
                             , K = K

                             , I_BAR = I_BAR
                             , J_BAR = J_BAR
                             , K_BAR = K_BAR
                             , K_BAR_WB = K_BAR_WB
                             , S = S

                             , num_macs = num_macs
                             , wt_intensity = wt_intensity
                             , is_compute_bound = is_compute_bound

                             , wt_block_lat = wt_block_lat
                             , act_block_lat = act_block_lat
                             , comp_push_block_lat = comp_push_block_lat
                             , comp_trav_block_lat = comp_trav_block_lat

                             , num_blocks_A = num_blocks_A
                             , num_blocks_W = num_blocks_W
                             , num_blocks_O = num_blocks_O

                             , compute_lat = simulated_compute_lat
                             , compute_perf = compute_perf
                             , ideal_lat = ideal_lat
                             , mac_util = mac_util
                             ,gb_reads = gb_reads
                             ,wt_reads = wt_reads

                                  , name = expr.id
                                  , type = expr.type
                             )

    ########################################################
    #
    def globalavgpool( self, expr, Z, A ):

        ### ARCH PARAMS ###
        SA_act_width = self.arch.SA_act_width
        SA_H         = self.arch.SA_H
        SA_W         = self.arch.SA_W
        SA_freq      = self.arch.SA_freq
        ACC_ch_els   = self.arch.ACC_ch_els
        WL_bw        = self.arch.WL_bw
        GB_bw        = self.arch.GB_bw
        N_wi_poi     = self.arch.N_wi_poi
        SA_HW_min    = min(SA_H, SA_W)

        N,C,H,W = A

        num_ops = N * C * ((H * W) - 1 + 1)

        if C % SA_H != 0:
            C += SA_H - (C % SA_H)

        I = N*H*W
        J = C

        I_BAR = ceil( I / ACC_ch_els )
        J_BAR = ceil( J / SA_HW_min )
        S     = ceil( I / I_BAR )

        num_vectors = N * H * W * ceil( C/SA_HW_min )
        rtn_vectors = N * ceil( C/SA_HW_min )

        fwd_cycles   = num_vectors + SA_H + SA_W + 2
        act_cycles   = 3 # div ~ 3 cycles
        rev_cycles   = rtn_vectors + SA_H + SA_W + 2
        total_cycles = fwd_cycles + act_cycles + rev_cycles

        compute_lat = (total_cycles / SA_freq) * 1e9

        compute_perf = (num_ops / compute_lat) * 1e9
        ideal_lat    = (num_ops / self.arch.AU_perf_peak) * 1e9
        mac_util     = compute_perf / self.arch.AU_perf_peak

        return ActivationLayerPerf( N=N, C=C, H=H, W=W
                                  , num_ops = num_ops
                                  , num_vectors = num_vectors
                                  , fwd_cycles = fwd_cycles
                                  , act_cycles = act_cycles
                                  , rev_cycles = rev_cycles
                                  , total_cycles = total_cycles

                                  , compute_lat = compute_lat
                                  , compute_perf = compute_perf
                                  , ideal_lat = ideal_lat
                                  , mac_util = mac_util

                                  , name = expr.id
                                  , type = expr.type
                                  )


    ########################################################
    #
    def maxpool( self, expr, Z, A, kernel, pads, strides, dilations ):

        ### ARCH PARAMS ###
        SA_act_width = self.arch.SA_act_width
        SA_H         = self.arch.SA_H
        SA_W         = self.arch.SA_W
        SA_freq      = self.arch.SA_freq
        ACC_ch_els   = self.arch.ACC_ch_els
        WL_bw        = self.arch.WL_bw
        GB_bw        = self.arch.GB_bw
        N_wi_poi     = self.arch.N_wi_poi
        SA_HW_min    = min(SA_H, SA_W)

        N,C,H,W = A
        _,_,OH,OW = Z

        FH,FW = kernel

        num_ops = N * C * FH * FW * OH * OW

        if C % SA_H != 0:
            C += SA_H - (C % SA_H)

        num_vectors = N *  H *  W * ceil( C/SA_HW_min )
        rtn_vectors = N * OH * OW * ceil( C/SA_HW_min )

        fwd_cycles   = num_vectors + SA_H + SA_W + 2
        act_cycles   = rtn_vectors * FH * FW
        rev_cycles   = rtn_vectors + SA_H + SA_W + 2
        total_cycles = fwd_cycles + act_cycles + rev_cycles

        compute_lat = (total_cycles / SA_freq) * 1e9

        compute_perf = (num_ops / compute_lat) * 1e9
        ideal_lat    = (num_ops / self.arch.AU_perf_peak) * 1e9
        mac_util     = compute_perf / self.arch.AU_perf_peak

        return ActivationLayerPerf( N=N, C=C, H=H, W=W
                                  , num_ops = num_ops
                                  , num_vectors = num_vectors
                                  , fwd_cycles = fwd_cycles
                                  , act_cycles = act_cycles
                                  , rev_cycles = rev_cycles
                                  , total_cycles = total_cycles

                                  , compute_lat = compute_lat
                                  , compute_perf = compute_perf
                                  , ideal_lat = ideal_lat
                                  , mac_util = mac_util

                                  , name = expr.id
                                  , type = expr.type
                                  )

    ########################################################
    #
    def param( self, expr, Z, value ):
        pass

    ########################################################
    #
    def relu( self, expr, Z, A ):
        pass

    ########################################################
    #
    def reshape( self, expr, Z, A, shape ):
        pass

    ########################################################
    #
    def transpose( self, expr, Z, A, axes ):
        pass

    def lrn( self, expr, Z, A, alpha, beta, bias, size ):
        pass

    def dropout( self, expr, Z, A, ratio, seed ):
        pass

    def softmax( self, expr, Z, A, axis ):
        pass

    ########################################################
    #
    _cap = {
        "Add"                 :  add,
        "Batchnorm"           :  batchnorm,
        "BatchnormFusedRelu"  :  batchnorm,
        "Conv2D"              :  conv2d,
        "Conv2DFusedRelu"     :  conv2d,
        "Gemm"                :  gemm,
        "GemmFusedRelu"       :  gemm,
        "GlobalAvgpool"       :  globalavgpool,
        "Maxpool"             :  maxpool,
        "Param"               :  param,
        "Relu"                :  relu,
        "ReshapeStatic"       :  reshape,
        "Transpose"           :  transpose,
        "LRN"                 :  lrn,
        "Dropout"             :  dropout,
        "Softmax"             :  softmax,
    }


################################################################################
#
class SystolicArrayPerformanceEstimatorAllocator:

    ########################################################
    #
    def __init__( self, network ) -> None:
        self.network = network

    ########################################################
    #
    def __getitem__( self, key ) -> Any:
        return self.network.lookup_tensor(key).shape

    ########################################################
    #
    def __setitem__( self, key, value ) -> None:
        self.network.lookup_tensor(key).shape = value

    ########################################################
    #
    def __contains__( self, key ) -> bool:
        return self.network.lookup_tensor(key) is not None

