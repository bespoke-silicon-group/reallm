from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Div( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None


@register_onnx( "Div" )
def from_onnx( node, kwargs ):
    return Div( id = node.name
              , Z  = kwargs["C"]
              , A  = kwargs["A"]
              , B  = kwargs["B"]
              )

