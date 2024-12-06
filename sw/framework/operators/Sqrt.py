from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Sqrt( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None


@register_onnx( "Sqrt" )
def from_onnx( node, kwargs ):
    return Sqrt( id = node.name
               , Z  = kwargs["Y"]
               , A  = kwargs["X"]
               )

