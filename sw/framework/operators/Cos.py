from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Cos( Expr ):
	id : str = None
	output : ArgOut = None
	input : ArgIn = None

@register_onnx( "Cos" )
def from_onnx( node, kwargs ):
	return Cos( id = node.name
		, output = kwargs["output"]
		, input = kwargs["input"]
	)
