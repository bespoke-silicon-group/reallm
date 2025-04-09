from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Cast( Expr ):
    id   : str     = None
    Z    : ArgOut  = None
    A    : ArgIn   = None
    type : ArgAttr = None

@register_onnx( "Cast" )
def from_onnx( node, kwargs ):
    return Cast( id   = node.name
               , Z    = kwargs["output"]
               , A    = kwargs["input"]
               , type = kwargs["to"]
               )
