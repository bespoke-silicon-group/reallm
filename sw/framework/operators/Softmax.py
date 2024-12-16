from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Softmax( Expr ):
    id   : str     = None
    Z    : ArgOut  = None
    A    : ArgIn   = None
    axis : ArgAttr = None


@register_onnx( "Softmax" )
def from_onnx( node, kwargs ):
    return Softmax( id    = node.name
                  , Z     = kwargs["output"]
                  , A     = kwargs["input"]
                  , axis  = kwargs["axis"]
                  )

