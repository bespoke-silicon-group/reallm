from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Reshape( Expr ):
    id    : str     = None
    Z     : ArgOut  = None
    A     : ArgIn   = None
    shape : ArgIn   = None


@register_onnx( "Reshape" )
def from_onnx( node, kwargs ):
    assert kwargs["allowzero"] in {None, 0}

    return Reshape( id    = node.name
                  , Z     = kwargs["reshaped"]
                  , A     = kwargs["data"]
                  , shape = kwargs["shape"]
                  )

