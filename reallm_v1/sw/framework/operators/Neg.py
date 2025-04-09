from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Neg( Expr ):
	id : str = None
	Y : ArgOut = None
	X : ArgIn = None

@register_onnx( "Neg" )
def from_onnx( node, kwargs ):
	return Neg( id = node.name
		, Y = kwargs["Y"]
		, X = kwargs["X"]
	)
