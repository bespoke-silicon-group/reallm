from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Mul( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None


@register_onnx( "Mul" )
def from_onnx( node, kwargs ):
    return Mul( id = node.name
              , Z  = kwargs["C"]
              , A  = kwargs["A"]
              , B  = kwargs["B"]
              )

