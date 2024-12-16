from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Shape( Expr ):
	id : str = None
	shape : ArgOut = None
	data : ArgIn = None

@register_onnx( "Shape" )
def from_onnx( node, kwargs ):
	return Shape( id = node.name
		, shape = kwargs["shape"]
		, data = kwargs["data"]
	)
