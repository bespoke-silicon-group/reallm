from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Sigmoid( Expr ):
	id : str = None
	Y : ArgOut = None
	X : ArgIn = None

@register_onnx( "Sigmoid" )
def from_onnx( node, kwargs ):
	return Sigmoid( id = node.name
		, Y = kwargs["Y"]
		, X = kwargs["X"]
	)
