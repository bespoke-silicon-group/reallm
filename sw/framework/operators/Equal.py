from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Equal( Expr ):
	id : str = None
	C : ArgOut = None
	A : ArgIn = None
	B : ArgIn = None

@register_onnx( "Equal" )
def from_onnx( node, kwargs ):
	return Equal( id = node.name
		, C = kwargs["C"]
		, A = kwargs["A"]
		, B = kwargs["B"]
	)
