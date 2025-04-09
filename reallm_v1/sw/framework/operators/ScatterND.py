from framework.Expr import *
from dataclasses import dataclass

@dataclass
class ScatterND( Expr ):
	id : str = None
	output : ArgOut = None
	data : ArgIn = None
	indices : ArgIn = None
	updates : ArgIn = None

@register_onnx( "ScatterND" )
def from_onnx( node, kwargs ):
	return ScatterND( id = node.name
		, output = kwargs["output"]
		, data = kwargs["data"]
		, indices = kwargs["indices"]
		, updates = kwargs["updates"]
	)
