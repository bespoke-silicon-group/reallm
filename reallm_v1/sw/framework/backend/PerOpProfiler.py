
from __future__ import annotations

from bsg.framework.Network import *
from bsg.utils.BSGBaseDataclass import BSGBaseDataclass
from typing import Any, List
from dataclasses import dataclass
import numpy as np
import copy
import sys
from math import ceil


@dataclass
class OpProfile( BSGBaseDataclass ):
    layer_name      : str = None
    layer_type      : str = None
    act_size_in     : int = None
    act_size_out    : int = None
    ops             : int = None



class PerOpProfiler:

    def __init__( self, network ) -> None:
        self.network = network
        self.alloc   = Allocator(self.network)

    def run( self, _unused ) -> None:
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

    def add( self, expr, Z, A, B ):
        assert A == B
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = np.prod(A) + np.prod(B),
            act_size_out = np.prod(Z),
            ops = np.prod(Z),
        )

    def batchnorm( self, expr, Z, A, gamma, beta, mean, var, epsilon ):
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = np.prod(A),
            act_size_out = np.prod(Z),
            ops = 2*np.prod(Z),
        )

    def conv2d( self, expr, Z, A, W, B, pads, strides, dilations ):
        K,C,FH,FW = W
        N,_,OH,OW = Z
        num_macs = N * K * OH * OW * FH * FW * C
        if B:
            num_macs += np.prod(B)

        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = np.prod(A),
            act_size_out = np.prod(Z),
            ops = num_macs,
        )

    def gemm( self, expr, Z, A, B, C, trace=True ):
        I, J = A
        J, K = B
        num_macs = I * J * K
        if C:
            num_macs += np.prod(C)
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = np.prod(A),
            act_size_out = np.prod(Z),
            ops = num_macs,
        )

    def globalavgpool( self, expr, Z, A ):
        N, C, H, W = A
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = np.prod(A),
            act_size_out = np.prod(Z),
            ops = N * C * (H * W - 1 + 1),
        )

    def maxpool( self, expr, Z, A, kernel, pads, strides, dilations ):
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = np.prod(A),
            act_size_out = np.prod(Z),
            ops = np.prod(kernel) * np.prod(Z),
        )

    def param( self, expr, Z, value ):
        pass

    def relu( self, expr, Z, A ):
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = 0,
            act_size_out = 0,
            ops = 0,
        )

    def reshape( self, expr, Z, A, shape ):
        pass
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = 0,
            act_size_out = 0,
            ops = 0,
        )

    def transpose( self, expr, Z, A, axes ):
        pass
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = 0,
            act_size_out = 0,
            ops = 0,
        )

    def lrn( self, expr, Z, A, alpha, beta, bias, size ):
        pass
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = 0,
            act_size_out = 0,
            ops = 0,
        )

    def dropout( self, expr, Z, A, ratio, seed ):
        pass
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = 0,
            act_size_out = 0,
            ops = 0,
        )

    def softmax( self, expr, Z, A, axis ):
        pass
        return OpProfile(
            layer_name = expr.id,
            layer_type = expr.type,
            act_size_in = 0,
            act_size_out = 0,
            ops = 0,
        )

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


class Allocator:

    def __init__( self, network ) -> None:
        self.network = network

    def __getitem__( self, key ) -> Any:
        return self.network.lookup_tensor(key).shape

    def __setitem__( self, key, value ) -> None:
        self.network.lookup_tensor(key).shape = value

    def __contains__( self, key ) -> bool:
        return self.network.lookup_tensor(key) is not None

