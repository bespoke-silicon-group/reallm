from framework.Expr import *
from dataclasses import dataclass

@dataclass
class ConstantOfShape( Expr ):
	# Generate a tensor with given value and shape.
	id : str = None
	output : ArgOut = None
	input : ArgIn = None # 1D tensor. The shape of the expected output tensor. If empty tensor is given, the output would be a scalar. All values must be >= 0.
	value : ArgAttr = None

@register_onnx( "ConstantOfShape" )
def from_onnx( node, kwargs ):
	return ConstantOfShape( id = node.name
		, output = kwargs["output"]
		, input = kwargs["input"]
		, value = kwargs["value"]
	)
