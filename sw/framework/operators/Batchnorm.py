from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Batchnorm( Expr ):
    id      : str     = None
    Z       : ArgOut  = None
    A       : ArgIn   = None
    gamma   : ArgIn   = None
    beta    : ArgIn   = None
    mean    : ArgIn   = None
    var     : ArgIn   = None
    epsilon : ArgAttr = 1e-5

@register_onnx( "BatchNormalization" )
def from_onnx( node, kwargs ):
    assert kwargs["running_mean"]  in { None }
    assert kwargs["running_var"]   in { None }
    assert kwargs["training_mode"] in { None, 0 }

    return Batchnorm( id      = node.name
                    , Z       = kwargs["Y"]
                    , A       = kwargs["X"]
                    , gamma   = kwargs["scale"]
                    , beta    = kwargs["B"]
                    , mean    = kwargs["input_mean"]
                    , var     = kwargs["input_var"]
                    , epsilon = default_attr( kwargs["epsilon"], 1e-5 )
                    )
