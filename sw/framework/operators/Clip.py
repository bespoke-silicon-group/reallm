from bsg.framework.Expr import *
from dataclasses import dataclass
from bsg.framework.operators.Relu import Relu

@dataclass
class Clip( Expr ):
    id  : str    = None
    Z   : ArgOut = None
    A   : ArgIn  = None
    min : ArgIn  = None
    max : ArgIn  = None

@register_onnx( "Clip" )
def from_onnx( node, kwargs ):
    return Relu( id = node.name
               , Z  = kwargs["output"]
               , A  = kwargs["input"]
               )
    #return Clip( id  = node.name
    #           , Z   = kwargs["output"]
    #           , A   = kwargs["input"]
    #           , min = kwargs["min"]
    #           , max = kwargs["max"]
    #           )
