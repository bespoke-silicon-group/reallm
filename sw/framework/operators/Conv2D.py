from framework.Expr import *
from dataclasses import dataclass


@dataclass
class Conv2D( Expr ):
    id        : str     = None
    Z         : ArgOut  = None
    A         : ArgIn   = None
    W         : ArgIn   = None
    B         : ArgIn   = None
    pads      : ArgAttr = (0,0,0,0)
    strides   : ArgAttr = (1,1)
    dilations : ArgAttr = (1,1)


@register_onnx( "Conv" )
def from_onnx( node, kwargs ):
    assert kwargs["auto_pad"] in { None, "NOTSET" }
    # TODO # assert kwargs["group"]    in { None, 1 }, kwargs["group"]

    # note: there is a kernel_shape attribute but we assume that the lowest two
    # dims of W will equal kernel_shape so we ignore that attribute.

    return Conv2D( id=node.name
                 , Z         = kwargs["Y"]
                 , A         = kwargs["X"]
                 , W         = kwargs["W"]
                 , B         = kwargs["B"]
                 , pads      = default_attr( kwargs["pads"], (0,0,0,0) )
                 , strides   = default_attr( kwargs["strides"], (1,1) )
                 , dilations = default_attr( kwargs["dilations"], (1,1) )
                 )

