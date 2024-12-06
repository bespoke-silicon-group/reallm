from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Transpose( Expr ):
    id   : str     = None
    Z    : ArgOut  = None
    A    : ArgIn   = None
    axes : ArgAttr = None


@register_onnx( "Transpose" )
def from_onnx( node, kwargs ):
    return Transpose( id   = node.name
                    , Z    = kwargs["transposed"]
                    , A    = kwargs["data"]
                    , axes = kwargs["perm"]
                    )

