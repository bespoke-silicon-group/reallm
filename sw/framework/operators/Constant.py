from framework.Expr import *
from dataclasses import dataclass

@dataclass
class Constant( Expr ):
	id : str = None
	output : ArgOut = None
	sparse_value : ArgAttr = None
	value : ArgAttr = None
	value_float : ArgAttr = None
	value_floats : ArgAttr = None
	value_int : ArgAttr = None
	value_ints : ArgAttr = None
	value_string : ArgAttr = None
	value_strings : ArgAttr = None

@register_onnx( "Constant" )
def from_onnx( node, kwargs ):
	return Constant( id = node.name
		, output = kwargs["output"]
		, sparse_value = kwargs["sparse_value"]
		, value = kwargs["value"]
		, value_float = kwargs["value_float"]
		, value_floats = kwargs["value_floats"]
		, value_int = kwargs["value_int"]
		, value_ints = kwargs["value_ints"]
		, value_string = kwargs["value_string"]
		, value_strings = kwargs["value_strings"]
	)
