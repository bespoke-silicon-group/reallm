from framework.Expr import *
from dataclasses import dataclass


@dataclass
class ReshapeStatic( Expr ):
    id    : str     = None
    Z     : ArgOut  = None
    A     : ArgIn   = None
    shape : ArgAttr = None

