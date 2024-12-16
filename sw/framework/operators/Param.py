from framework.Expr import *
from dataclasses import dataclass, field


@dataclass
class Param( Expr ):
    id    : str     = None
    Z     : ArgOut  = None
    value : ArgAttr = field(repr=False, default=None)


@register_onnx( "Constant" )
def from_onnx( node, kwargs ):
    assert kwargs["value"] is not None

    return Param( id    = node.name
                , Z     = kwargs["output"]
                , value = kwargs["value"]
                )

