from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Split( Expr ):
    id   : str    = None
    Z    : ArgOut = None
    A    : ArgIn  = None
    split: ArgIn  = None
    axis : ArgAttr  = None


@register_onnx( "Split" )
def from_onnx( node, kwargs ):
    return Split( id = node.name
                 , Z  = kwargs["outputs"]
                 , A  = kwargs["input"]
                 , split = kwargs["split"]
                 , axis = kwargs["axis"]
                 )

