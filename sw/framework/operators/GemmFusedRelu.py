from framework.Expr import *
from dataclasses import dataclass


@dataclass
class GemmFusedRelu( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None
    C  : ArgIn  = None

