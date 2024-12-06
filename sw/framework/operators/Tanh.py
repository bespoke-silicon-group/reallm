from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Tanh( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None


@register_onnx("Tanh")
def from_onnx( node, kwargs ):
    return Tanh( id = node.name
               , Z  = kwargs["output"]
               , A  = kwargs["input"]
               )


