from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Pow( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None


@register_onnx( "Pow" )
def from_onnx( node, kwargs ):
    return Pow( id = node.name
              , Z  = kwargs["Z"]
              , A  = kwargs["X"]
              , B  = kwargs["Y"]
              )

