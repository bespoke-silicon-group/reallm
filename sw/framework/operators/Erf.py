from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Erf( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None


@register_onnx( "Erf" )
def from_onnx( node, kwargs ):
    return Erf( id = node.name
              , Z  = kwargs["output"]
              , A  = kwargs["input"]
              )

