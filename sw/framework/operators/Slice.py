from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Slice( Expr ):
	id : str = None
	output : ArgOut = None
	data : ArgIn = None
	starts : ArgIn = None
	ends : ArgIn = None
	axes : ArgIn = None
	steps : ArgIn = None

@register_onnx( "Slice" )
def from_onnx( node, kwargs ):
	return Slice( id = node.name
		, output = kwargs["output"]
		, data = kwargs["data"]
		, starts = kwargs["starts"]
		, ends = kwargs["ends"]
		, axes = kwargs["axes"]
		, steps = kwargs["steps"]
	)
