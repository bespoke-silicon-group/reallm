from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Sin( Expr ):
	id : str = None
	output : ArgOut = None
	input : ArgIn = None

@register_onnx( "Sin" )
def from_onnx( node, kwargs ):
	return Sin( id = node.name
		, output = kwargs["output"]
		, input = kwargs["input"]
	)
