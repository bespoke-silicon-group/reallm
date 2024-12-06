from bsg.framework.Expr import *
from dataclasses import dataclass


@dataclass
class Conv2DFusedRelu( Expr ):
    id        : str     = None
    Z         : ArgOut  = None
    A         : ArgIn   = None
    W         : ArgIn   = None
    B         : ArgIn   = None
    pads      : ArgAttr = (0,0,0,0)
    strides   : ArgAttr = (1,1)
    dilations : ArgAttr = (1,1)

