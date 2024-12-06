from bsg.framework.operators import *
from bsg.framework.Pattern import *
import numpy as np


class BaselineOptimizer:

    def __init__( self, network ):
        self.network = network


    def run( self ):
        for E in self.network.iter():
            self.visit(E)


    def visit( self, E ):

        ### Find Reshape operators that have constant shapes and convert them
        ### to ReshapeStatic operators.
        if Is(Reshape, shape=Is(Param)).match(E):
            E_reshape = E
            if len(self.network.lookup_sinks(E_reshape.shape)) == 1:
                T_param = self.network.lookup_tensor(E_reshape.shape)
                E_param = self.network.lookup_src(E_reshape.shape)

                self.network.remove_tensors(T_param)
                self.network.remove_exprs(E_param)
                self.network.remove_exprs(E_reshape)

                self.network.add_exprs(
                    ReshapeStatic( id    = E_reshape.id + "_static"
                                 , Z     = E_reshape.Z
                                 , A     = E_reshape.A
                                 , shape = E_param.value
                                 )
                    )

        ### Find Transpose operators on constant tensors and pre-process the
        ### transpose and remove the Transpose operator.
        if Is(Transpose, A=Is(Param)).match(E):
            E_transpose = E
            if len(self.network.lookup_sinks(E_transpose.A)) == 1:
                all_sinks = [(sink.id, arg) for sink in self.network.lookup_sinks(E_transpose.Z) for arg in sink.connected_to(E_transpose.Z)]

                E_param = self.network.lookup_src(E_transpose.A)
                E_param.value = np.transpose(E_param.value, E_transpose.axes)

                self.network.remove_tensors(self.network.lookup_tensor(E_transpose.Z))
                self.network.remove_exprs(E_transpose)

                for E_id, E_arg in all_sinks:
                    self.network.connect_expr(E_param.Z, E_id, E_arg)

