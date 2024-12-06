from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Relu( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None


@register_onnx( "Relu" )
def from_onnx( node, kwargs ):
    return Relu( id = node.name
               , Z  = kwargs["Y"]
               , A  = kwargs["X"]
               )

