from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Sub( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None


@register_onnx( "Sub" )
def from_onnx( node, kwargs ):
    return Sub( id = node.name
              , Z  = kwargs["C"]
              , A  = kwargs["A"]
              , B  = kwargs["B"]
              )

