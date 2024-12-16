from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Trilu( Expr ):
	id : str = None
	output : ArgOut = None
	input : ArgIn = None
	k : ArgIn = None
	upper : ArgAttr = None

@register_onnx( "Trilu" )
def from_onnx( node, kwargs ):
	return Trilu( id = node.name
		, output = kwargs["output"]
		, input = kwargs["input"]
		, k = kwargs["k"]
		, upper = kwargs["upper"]
	)
