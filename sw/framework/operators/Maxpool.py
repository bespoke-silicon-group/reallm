from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Maxpool( Expr ):
    id        : str     = None
    Z         : ArgOut  = None
    A         : ArgIn   = None
    kernel    : ArgAttr = None
    pads      : ArgAttr = (0,0,0,0)
    strides   : ArgAttr = (1,1)
    dilations : ArgAttr = (1,1)


@register_onnx( "MaxPool" )
def from_onnx( node, kwargs ):
    assert kwargs["auto_pad"]      in { None, "NOTSET" }
    assert kwargs["ceil_mode"]     in { None, 0 }
    assert kwargs["storage_order"] in { None, 0 }
    assert kwargs["Indices"]       in { None }

    return Maxpool( id        = node.name
                  , Z         = kwargs["Y"]
                  , A         = kwargs["X"]
                  , kernel    = kwargs["kernel_shape"]
                  , pads      = default_attr( kwargs["pads"], (0,0,0,0) )
                  , strides   = default_attr( kwargs["strides"], (1,1) )
                  , dilations = default_attr( kwargs["dilations"], (1,1) )
                  )

