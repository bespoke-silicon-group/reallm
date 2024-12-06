from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Concat( Expr ):
    id   : str    = None
    Z    : ArgOut = None
    A    : ArgIn  = None
    axis : ArgAttr  = None


@register_onnx( "Concat" )
def from_onnx( node, kwargs ):
    return Concat( id = node.name
                 , Z  = kwargs["concat_result"]
                 , A  = kwargs["inputs"]
                 , axis = kwargs["axis"]
                 )

