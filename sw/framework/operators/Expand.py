from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Expand( Expr ):
	id : str = None
	output : ArgOut = None
	input : ArgIn = None
	shape : ArgIn = None

@register_onnx( "Expand" )
def from_onnx( node, kwargs ):
	return Expand( id = node.name
		, output = kwargs["output"]
		, input = kwargs["input"]
		, shape = kwargs["shape"]
	)
