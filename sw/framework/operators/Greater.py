from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Greater( Expr ):
	id : str = None
	C : ArgOut = None
	A : ArgIn = None
	B : ArgIn = None

@register_onnx( "Greater" )
def from_onnx( node, kwargs ):
	return Greater( id = node.name
		, C = kwargs["C"]
		, A = kwargs["A"]
		, B = kwargs["B"]
	)
