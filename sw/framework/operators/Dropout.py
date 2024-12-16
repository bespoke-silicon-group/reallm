from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Dropout( Expr ):
    id    : str     = None
    Z     : ArgOut  = None
    A     : ArgIn   = None
    ratio : ArgIn   = None
    seed  : ArgAttr = 1e-5


@register_onnx( "Dropout" )
def from_onnx( node, kwargs ):
    return Dropout( id      = node.name
                  , Z       = kwargs["output"]
                  , A       = kwargs["data"]
                  , ratio   = kwargs["ratio"]
                  , seed    = kwargs["seed"]
                  )

