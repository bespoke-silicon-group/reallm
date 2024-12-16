from framework.Expr import *
from dataclasses import dataclass


@dataclass
class GlobalAvgpool( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None


@register_onnx( "GlobalAveragePool" )
def from_onnx( node, kwargs ):
    return GlobalAvgpool( id = node.name
                        , Z  = kwargs["Y"]
                        , A  = kwargs["X"]
                        )


