from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Range( Expr ):
	id : str = None
	output : ArgOut = None
	start : ArgIn = None
	limit : ArgIn = None
	delta : ArgIn = None

@register_onnx( "Range" )
def from_onnx( node, kwargs ):
	return Range( id = node.name
		, output = kwargs["output"]
		, start = kwargs["start"]
		, limit = kwargs["limit"]
		, delta = kwargs["delta"]
	)
