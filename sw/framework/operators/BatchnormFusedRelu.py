from bsg.framework.Expr import *
from dataclasses import dataclass

@dataclass
class BatchnormFusedRelu( Expr ):
    id      : str     = None
    Z       : ArgOut  = None
    A       : ArgIn   = None
    gamma   : ArgIn   = None
    beta    : ArgIn   = None
    mean    : ArgIn   = None
    var     : ArgIn   = None
    epsilon : ArgAttr = 1e-5
