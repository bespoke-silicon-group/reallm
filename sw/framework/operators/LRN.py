from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class LRN( Expr ):
    id    : str     = None
    Z     : ArgOut  = None
    A     : ArgIn   = None
    alpha : ArgAttr = 0.0001
    beta  : ArgAttr = 0.75
    bias  : ArgAttr = 1.0
    size  : ArgAttr = None


@register_onnx( "LRN" )
def from_onnx( node, kwargs ):
    return LRN( id    = node.name
              , Z     = kwargs["Y"]
              , A     = kwargs["X"]
              , alpha = kwargs["alpha"]
              , beta  = kwargs["beta"]
              , bias  = kwargs["bias"]
              , size  = kwargs["size"]
              )

