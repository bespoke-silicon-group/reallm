from __future__ import annotations
from bsg.framework.operators import *
from bsg.framework.Expr import *
from bsg.framework.Pattern import *

################################################################################
#
class SystolicArrayOptimizer:

    ########################################################
    #
    def __init__( self, network ) -> None:
        self.network = network

    ########################################################
    #
    def run( self ) -> None:
        for E in self.network.iter():
            self.visit(E)

    ########################################################
    #
    def visit( self, E ) -> None:
        if isinstance(E, Gemm):
            sinks_Y = self.network.lookup_sinks(E.Z)
            if len(sinks_Y) == 1 and isinstance(sinks_Y[0], Relu):
                self.fuse_gemm_relu(
                    E_gemm = E,
                    E_relu = sinks_Y[0],
                )

        if isinstance(E, Conv2D):
            sinks_Y = self.network.lookup_sinks(E.Z)
            if len(sinks_Y) == 1 and isinstance(sinks_Y[0], Relu):
                self.fuse_conv2d_relu(
                    E_conv2d = E,
                    E_relu = sinks_Y[0],
                )

        if isinstance(E, Batchnorm):
            sinks_Y = self.network.lookup_sinks(E.Z)
            if len(sinks_Y) == 1 and isinstance(sinks_Y[0], Relu):
                self.fuse_batchnorm_relu(
                    E_batchnorm = E,
                    E_relu = sinks_Y[0],
                )

    ########################################################
    #
    def fuse_gemm_relu( self, E_gemm, E_relu ) -> None:
        self.network.remove_tensors(self.network.lookup_tensor(E_relu.A))
        self.network.remove_exprs(E_relu)
        self.network.remove_exprs(E_gemm)
        self.network.add_exprs(
            GemmFusedRelu( id = E_gemm.id + "_relu_fused"
                         , A  = E_gemm.A
                         , B  = E_gemm.B
                         , C  = E_gemm.C
                         , Z  = E_relu.Z
                         )
        )

    ########################################################
    #
    def fuse_conv2d_relu( self, E_conv2d, E_relu ) -> None:
        self.network.remove_tensors(self.network.lookup_tensor(E_relu.A))
        self.network.remove_exprs(E_relu)
        self.network.remove_exprs(E_conv2d)
        self.network.add_exprs(
            Conv2DFusedRelu( id = E_conv2d.id + "_relu_fused"
                           , A         = E_conv2d.A
                           , W         = E_conv2d.W
                           , B         = E_conv2d.B
                           , Z         = E_relu.Z
                           , pads      = E_conv2d.pads
                           , strides   = E_conv2d.strides
                           , dilations = E_conv2d.dilations
                           )
        )

    ########################################################
    #
    def fuse_batchnorm_relu( self, E_batchnorm, E_relu ) -> None:
        self.network.remove_tensors(self.network.lookup_tensor(E_relu.A))
        self.network.remove_exprs(E_relu)
        self.network.remove_exprs(E_batchnorm)
        self.network.add_exprs(
            BatchnormFusedRelu( id      = E_batchnorm.id + "_relu_fused"
                              , A       = E_batchnorm.A
                              , gamma   = E_batchnorm.gamma
                              , beta    = E_batchnorm.beta
                              , mean    = E_batchnorm.mean
                              , var     = E_batchnorm.var
                              , Z       = E_relu.Z
                              , epsilon = E_batchnorm.epsilon
                              )
        )
