from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Where( Expr ):
	id : str = None
	output : ArgOut = None
	condition : ArgIn = None
	X : ArgIn = None
	Y : ArgIn = None

@register_onnx( "Where" )
def from_onnx( node, kwargs ):
	return Where( id = node.name
		, output = kwargs["output"]
		, condition = kwargs["condition"]
		, X = kwargs["X"]
		, Y = kwargs["Y"]
	)
