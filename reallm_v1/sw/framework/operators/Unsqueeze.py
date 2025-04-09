from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Unsqueeze( Expr ):
    id   : str     = None
    Z    : ArgOut  = None
    A    : ArgIn   = None
    axes : ArgIn   = None


@register_onnx( "Unsqueeze" )
def from_onnx( node, kwargs ):
    return Unsqueeze( id   = node.name
                    , Z    = kwargs["expanded"]
                    , A    = kwargs["data"]
                    , axes = kwargs["axes"]
                    )

