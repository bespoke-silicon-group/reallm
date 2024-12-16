from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Add( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None

@register_onnx( "Add" )
def from_onnx( node, kwargs ):
    return Add( id = node.name
              , Z  = kwargs["C"]
              , A  = kwargs["A"]
              , B  = kwargs["B"]
              )
