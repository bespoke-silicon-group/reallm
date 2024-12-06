from bsg.framework.Expr import *
from bsg.framework.operators.Transpose import Transpose
from dataclasses import dataclass


@dataclass
class Gemm( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None
    C  : ArgIn  = None


@register_onnx( "MatMul" )
def from_onnx( node, kwargs ):
    return Gemm( id = node.name
               , Z  = kwargs["Y"]
               , A  = kwargs["A"]
               , B  = kwargs["B"]
               )


@register_onnx( "Gemm" )
def from_onnx( node, kwargs ):
    assert kwargs["alpha"] in { None, 1.0 }
    assert kwargs["beta"]  in { None, 1.0 }

    exprs = []

    A_name = kwargs["A"]
    B_name = kwargs["B"]

    if kwargs["transA"] == 1:
        A_name += "_transposed"

    if kwargs["transB"] == 1:
        B_name += "_transposed"

    gemm_expr = Gemm( id = node.name
                    , Z  = kwargs["Y"]
                    , A  = A_name
                    , B  = B_name
                    , C  = kwargs["C"]
                    )

    if kwargs["transA"] == 1:
        exprs.append(
            Transpose( id = gemm_expr.id + "_transA"
                     , Z  = A_name
                     , A  = kwargs["A"]
                     )
        )
    if kwargs["transB"] == 1:
        exprs.append(
            Transpose( id = gemm_expr.id + "_transB"
                     , Z  = B_name
                     , A  = kwargs["B"]
                     )
        )

    exprs.append( gemm_expr )

    return exprs

