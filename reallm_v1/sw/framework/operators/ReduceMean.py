from framework.Expr import *
from dataclasses import dataclass


@dataclass
class ReduceMean( Expr ):
    id       : str     = None
    Z        : ArgOut  = None
    A        : ArgIn   = None
    axes     : ArgAttr = None
    keepdims : ArgAttr = None


@register_onnx( "ReduceMean" )
def from_onnx( node, kwargs ):
    return ReduceMean( id       = node.name
                     , Z        = kwargs["reduced"]
                     , A        = kwargs["data"]
                     , axes     = kwargs["axes"]
                     , keepdims = kwargs["keepdims"]
                     )

