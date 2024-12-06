from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Gather( Expr ):
    id      : str     = None
    Z       : ArgOut  = None
    A       : ArgIn   = None
    indices : ArgIn   = None
    axis    : ArgAttr = None


@register_onnx( "Gather" )
def from_onnx( node, kwargs ):
    return Gather( id      = node.name
                 , Z       = kwargs["output"]
                 , A       = kwargs["data"]
                 , indices = kwargs["indices"]
                 , axis    = kwargs["axis"]
                 )

