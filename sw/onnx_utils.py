"""
Collection of utility functions to work with onnx models.

Public Types:
    + TensorShape

    + OpInputPrototype
    + OpOutputPrototype
    + OpAttributePrototype
    + OpIOAPrototype

    + OpInput
    + OpOutput
    + OpAttribute
    + OpIOA

Public Enums:
    + OPT_REQUIRED, OPT_OPTIONAL, OPT_VARIADIC

Public Functions:
    ## operator based functions
    + get_operator_prototype

    ## model based functions
    + get_model_opset
    + get_model_input_shapes
    + get_model_output_shapes
    + get_model_initializer_np_data
    + get_model_node_optypes

    ## value info based functions
    + get_valueinfo_shape
    + get_valueinfo_np_dtype

    ## tensor based functions
    + get_tensor_np_data

    ## node based functions
    + get_node_input_args
    + get_node_output_args
    + get_node_attribute_args
    + get_node_args
    + get_node_kwargs

    ## helper functions
    + dtype_onnx_to_np
    + dtype_np_to_onnx
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import onnx.checker
import onnx.numpy_helper
import onnx.version_converter
from onnx import ModelProto, NodeProto, TensorProto, ValueInfoProto


@dataclass
class OpInputPrototype:
    name: str
    type: str
    option: int  # Use OPT_* enum


@dataclass
class OpOutputPrototype:
    name: str
    type: str
    option: int  # Use OPT_* enum


@dataclass
class OpAttributePrototype:
    name: str
    type: str
    option: int  # Use OPT_* enum (note: variadic attributes are not supported)
    default: Any


@dataclass
class OpInput:
    name: str
    value: str | list[str]  # List of values if variadic


@dataclass
class OpOutput:
    name: str
    value: str | list[str]  # List of values if variadic


@dataclass
class OpAttribute:
    name: str
    value: Any


OpIOAPrototype = tuple[
    list[OpInputPrototype],
    list[OpOutputPrototype],
    list[OpAttributePrototype],
]

OpIOA = tuple[
    list[OpInput],
    list[OpOutput],
    list[OpAttribute],
]

# A tensor shape is a list of ints. The length of the list is the
# dimensionallity of the tensor. Onnx models support dynamic dimensions where
# the size of the dimension is determined at runtime. In this case, rather than
# being set to an int, the dimension is given a name (string).
TensorShape = list[int | str]


# Argument option enum
OPT_REQUIRED = 0  # must specify
OPT_OPTIONAL = 1  # optional to specify
OPT_VARIADIC = 2  # variadic, can specify 1+ items


def get_operator_prototype(
    operator: str,
    opset: int | None = None,
) -> OpIOAPrototype:
    """
    Get the operator prototype for the given onnx opset version. The prototype
    is a collection of 3 lists for the input, output, and attribute argument
    prototypes.

    Args:
        operator (str): Name of the onnx operators.
        opset (int | None, optional): Opset version. If None, uses the latest
            version. Defaults to None.

    Raises:
        NotImplementedError: Internal error for unhandled attribute type.
        RuntimeError: Failure to find prototype for the given operator and opset
            version.

    Returns:
        OpIOAPrototype: The prototype as a collection of lists of input, output,
            and attribute prototypes.
    """

    global _prototypes_all_versions

    if "_prototypes_all_versions" not in globals():
        _prototypes_all_versions = {}
        for schema in onnx.defs.get_all_schemas_with_history():
            if not schema.deprecated and (
                schema.domain == "" or schema.domain == "ai.onnx"
            ):
                prototype = {"inputs": [], "outputs": [], "attrs": []}
                if schema.inputs:
                    for n in schema.inputs:
                        prototype["inputs"].append(
                            OpInputPrototype(
                                n.name,
                                n.type_str,
                                int(n.option),
                            )
                        )

                if schema.outputs:
                    for n in schema.outputs:
                        prototype["outputs"].append(
                            OpOutputPrototype(
                                n.name,
                                n.type_str,
                                int(n.option),
                            )
                        )

                if schema.attributes:
                    for n, attr in schema.attributes.items():
                        if attr.default_value.type == attr.default_value.UNDEFINED:
                            default_value = None
                        elif attr.default_value.type == attr.default_value.INT:
                            default_value = attr.default_value.i
                        elif attr.default_value.type == attr.default_value.FLOAT:
                            default_value = attr.default_value.f
                        elif attr.default_value.type == attr.default_value.STRING:
                            default_value = attr.default_value.s
                        elif attr.default_value.type == attr.default_value.INTS:
                            default_value = list(attr.default_value.ints)
                        elif attr.default_value.type == attr.default_value.FLOATS:
                            default_value = list(attr.default_value.floats)
                        elif attr.default_value.type == attr.default_value.STRINGS:
                            default_value = list(attr.default_value.strings)
                        else:
                            raise NotImplementedError(
                                f"attribute type not implemented: {attr.default_value.type}"
                            )

                        prototype["attrs"].append(
                            OpAttributePrototype(
                                n,
                                attr.type.name,
                                OPT_REQUIRED if attr.required else OPT_OPTIONAL,
                                default_value,
                            )
                        )

                if schema.name not in _prototypes_all_versions:
                    _prototypes_all_versions[schema.name] = {}
                version = int(schema.since_version)
                _prototypes_all_versions[schema.name][version] = prototype

    versions = _prototypes_all_versions[operator]
    for v, prototype in sorted(versions.items(), reverse=True):
        if not opset or v <= opset:
            return (prototype["inputs"], prototype["outputs"], prototype["attrs"])

    raise RuntimeError(
        f"could not find operator prototypes for '{operator}' opset version '{opset}'"
    )


def get_model_opset(model: ModelProto) -> int:
    """
    Get the opset version of the onnx model.

    Args:
        model (ModelProto): Onnx model.

    Returns:
        int: Opset version of the onnx model.
    """
    return int(model.opset_import[0].version)


def get_model_input_shapes(
    model: ModelProto,
    keep_init: bool = False,
    dyn_map: dict[str, int] | None = None,
    dyn_default: int | None = None,
) -> dict[str, TensorShape]:
    """
    Get all of the input tensors and their shapes from the onnx model.

    Args:
        model (ModelProto): Onnx model.
        keep_init (bool, optional): Keep any initializer tensors found in the
            input graph in the result. Defaults to False.
        dyn_map (dict[str, int] | None, optional): Map for dynamic tensor
            dimensions to specific values. Defaults to None.
        dyn_default (int | None, optional): Default tensor dimension for any
            dynamic tensor dimension. Defaults to None.

    Returns:
        dict[str, TensorShape]: Dictionary of input tensor names and their
            shapes.
    """
    result = {}
    for value_info in model.graph.input:
        result[value_info.name] = get_valueinfo_shape(value_info, dyn_map, dyn_default)
    if not keep_init:
        for tensor in model.graph.initializer:
            result.pop(tensor.name, None)
    return result


def get_model_output_shapes(
    model: ModelProto,
    keep_init: bool = False,
    dyn_map: dict[str, int] | None = None,
    dyn_default: int | None = None,
) -> dict[str, TensorShape]:
    """
    Get all of the output tensors and their shapes from the onnx model.

    Args:
        model (ModelProto): Onnx model.
        keep_init (bool, optional): Keep any initializer tensors found in the
            output graph in the result. Defaults to False.
        dyn_map (dict[str, int] | None, optional): Map for dynamic tensor
            dimensions to specific values. Defaults to None.
        dyn_default (int | None, optional): Default tensor dimension for any
            dynamic tensor dimension. Defaults to None.

    Returns:
        dict[str, TensorShape]: Dictionary of output tensor names and their
            shapes.
    """
    result = {}
    for value_info in model.graph.output:
        result[value_info.name] = get_valueinfo_shape(value_info, dyn_map, dyn_default)
    if not keep_init:
        for tensor in model.graph.initializer:
            result.pop(tensor.name, None)
    return result


def get_model_initializer_np_data(model: ModelProto) -> dict[str, np.ndarray]:
    """
    Get all of the initializer tensors as numpy arrays from the onnx model.

    Args:
        model (ModelProto): Onnx model.

    Returns:
        dict[str, np.ndarray]: Dictionary of initializer tensor names and their
            numpy arrays.
    """
    return {n.name: get_tensor_np_data(n) for n in model.graph.initializer}


def get_model_node_optypes(model: ModelProto) -> list[tuple[str, str]]:
    """
    Get all of the node names and their operator type names from the onnx model.

    Args:
        model (ModelProto): Onnx model.

    Returns:
        list[tuple[str, str]]: List of node names and their operator type names.
    """
    # NOTE: we use a list rather than a dictionary because nodes are not
    # required to have a name, unlike tensors/value_infos which technically
    # don't need a name but are pretty useless without them.
    return [(node.name, node.op_type) for node in model.graph.node]


def get_valueinfo_shape(
    value_info: ValueInfoProto,
    dyn_map: dict[str, int] | None = None,
    dyn_default: int | None = None,
) -> TensorShape:
    """
    Get the shape of a value info object.

    Args:
        value_info (ValueInfoProto): Value info object.
        dyn_map (dict[str, int] | None, optional): Map for dynamic tensor
            dimensions to specific values. Defaults to None.
        dyn_default (int | None, optional): Default tensor dimension for any
            dynamic tensor dimension. Defaults to None.

    Returns:
        TensorShape: Shape of the value info object.
    """
    shape = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_param == "":
            shape.append(d.dim_value)
        else:
            if dyn_map is not None and d.dim_param in dyn_map:
                shape.append(dyn_map[d.dim_param])
            elif dyn_default is not None:
                shape.append(dyn_default)
            else:
                shape.append(d.dim_param)
    return shape


def get_valueinfo_np_dtype(value_info: ValueInfoProto) -> np.dtype:
    """
    Get the numpy datatype from a value info object.

    Args:
        value_info (ValueInfoProto): Value info object.

    Returns:
        np.dtype: Numpy datatype of the value info object.
    """
    return dtype_onnx_to_np(value_info.type.tensor_type.elem_type)


def get_tensor_np_data(tensor: TensorProto) -> np.ndarray:
    """
    Get the numpy data from a tensor object.

    Args:
        tensor (TensorProto): Tensor object.

    Returns:
        np.ndarray: Numpy array of the tensor object.
    """
    return onnx.numpy_helper.to_array(tensor).astype(dtype_onnx_to_np(tensor.data_type))


def get_node_input_args(
    node: NodeProto,
    opset: int | None = None,
) -> list[OpInput]:
    """
    Get the input arguments for the given node.

    Args:
        node (NodeProto): Node object.
        opset (int | None, optional): Opset version to match the operator to.
            If None, uses the latest version. Defaults to None.

    Returns:
        list[OpInput]: List of input arguments for the node.
    """
    result: list[OpInput] = []
    input_args, _, _ = get_operator_prototype(node.op_type, opset)
    node_input_list = [i for i in node.input]
    for idx, input_name in enumerate(node_input_list):
        arg = input_args[idx]
        is_variadic = arg.option == OPT_VARIADIC
        if not is_variadic:
            op = OpInput(arg.name, input_name)
            result.append(op)
        else:
            # the rest of the inputs are variadic to this argument
            op = OpInput(arg.name, node_input_list[idx:])
            result.append(op)
            break
    return result


def get_node_output_args(
    node: NodeProto,
    opset: int = None,
) -> list[OpOutput]:
    """
    Get the output arguments for the given node.

    Args:
        node (NodeProto): Node object.
        opset (int | None, optional): Opset version to match the operator to.
            If None, uses the latest version. Defaults to None.

    Returns:
        list[OpOutput]: List of output arguments for the node.
    """
    result: list[OpOutput] = []
    _, output_args, _ = get_operator_prototype(node.op_type, opset)
    node_output_list = [i for i in node.output]
    for idx, output_name in enumerate(node_output_list):
        arg = output_args[idx]
        is_variadic = arg.option == OPT_VARIADIC
        if not is_variadic:
            op = OpOutput(arg.name, output_name)
            result.append(op)
        else:
            # the rest of the outputs are variadic to this argument
            op = OpOutput(arg.name, node_output_list[idx:])
            result.append(op)
            break
    return result


def get_node_attribute_args(
    node: NodeProto,
    opset: int = None,
) -> list[OpAttribute]:
    """
    Get the attribute arguments for the given node.

    Args:
        node (NodeProto): Node object.
        opset (int | None, optional): Opset version to match the operator to.
            If None, uses the latest version. Defaults to None.

    Returns:
        list[OpAttribute]: List of attribute arguments for the node.
    """
    result = []
    _, _, attr_args = get_operator_prototype(node.op_type, opset)
    for attr in node.attribute:
        arg = list(filter(lambda x: x.name == attr.name, attr_args))[0]
        if arg.type == "TENSOR":
            attr = OpAttribute(attr.name, onnx.numpy_helper.to_array(attr.t))
        elif arg.type == "FLOAT":
            attr = OpAttribute(attr.name, attr.f)
        elif arg.type == "INTS":
            attr = OpAttribute(attr.name, list(attr.ints))
        elif arg.type == "INT":
            attr = OpAttribute(attr.name, attr.i)
        elif arg.type == "STRING":
            attr = OpAttribute(attr.name, attr.s)
        else:
            raise NotImplementedError(f"attribute type not implemented: {arg.type}")
        result.append(attr)
    return result


def get_node_args(
    node: NodeProto,
    opset: int = None,
) -> OpIOA:
    """
    Get the input, output, and attribute arguments for the given node.

    Args:
        node (NodeProto): Node object.
        opset (int | None, optional): Opset version to match the operator to.
            If None, uses the latest version. Defaults to None.

    Returns:
        OpIOA: Tuple of input, output, and attribute arguments for the node.
    """
    return (
        get_node_input_args(node, opset),
        get_node_output_args(node, opset),
        get_node_attribute_args(node, opset),
    )


def get_node_kwargs(
    node: NodeProto,
    opset: int = None,
) -> dict[str, Any]:
    """
    Generates a kwargs dictionary from the node object. This dictionary contains
    all input, output and attribute arguements. If the node does not define an
    arguement the value is set to None. This is useful for passing the node's
    args to a function.

    Args:
        node (NodeProto): Node object.
        opset (int | None, optional): Opset version to match the operator to.
            If None, uses the latest version. Defaults to None.

    Returns:
        dict[str, Any]: Kwargs dictionary of the node object.
    """
    in_args, out_args, attr_args = get_operator_prototype(node.op_type, opset)
    kwargs = {}
    kwargs.update({n.name: None for n in in_args})
    kwargs.update({n.name: None for n in out_args})
    kwargs.update({n.name: None for n in attr_args})
    kwargs.update({n.name: n.value for n in get_node_input_args(node, opset)})
    kwargs.update({n.name: n.value for n in get_node_output_args(node, opset)})
    kwargs.update({n.name: n.value for n in get_node_attribute_args(node, opset)})
    return kwargs


def dtype_onnx_to_np(onnx_dtype: int) -> np.dtype:
    """
    Convert an onnx tensor datatype to the equivilant numpy datatype.

    Args:
        onnx_dtype (int): Onnx tensor datatype.

    Returns:
        np.dtype: Corresponding numpy datatype.
    """
    return {
        TensorProto.DOUBLE: np.dtype(np.float64),
        TensorProto.FLOAT: np.dtype(np.float32),
        TensorProto.INT64: np.dtype(np.int64),
        TensorProto.INT32: np.dtype(np.int32),
        TensorProto.INT16: np.dtype(np.int16),
        TensorProto.INT8: np.dtype(np.int8),
        TensorProto.UINT64: np.dtype(np.uint64),
        TensorProto.UINT32: np.dtype(np.uint32),
        TensorProto.UINT16: np.dtype(np.uint16),
        TensorProto.UINT8: np.dtype(np.uint8),
        TensorProto.BOOL: np.dtype(bool),
    }[onnx_dtype]


def dtype_np_to_onnx(np_dtype: np.dtype) -> int:
    """
    Convert a numpy datatype to the equivilant onnx tensor datatype.

    Args:
        np_dtype (np.dtype): Numpy datatype.

    Returns:
        int: Corresponding onnx tensor datatype.
    """
    return {
        np.dtype(np.float64): TensorProto.DOUBLE,
        np.dtype(np.float32): TensorProto.FLOAT,
        np.dtype(np.int64): TensorProto.INT64,
        np.dtype(np.int32): TensorProto.INT32,
        np.dtype(np.int16): TensorProto.INT16,
        np.dtype(np.int8): TensorProto.INT8,
        np.dtype(np.uint64): TensorProto.UINT64,
        np.dtype(np.uint32): TensorProto.UINT32,
        np.dtype(np.uint16): TensorProto.UINT16,
        np.dtype(np.uint8): TensorProto.UINT8,
        np.dtype(bool): TensorProto.BOOL,
    }[np_dtype]


__all__ = [
    "OPT_REQUIRED",
    "OPT_OPTIONAL",
    "OPT_VARIADIC",
    "TensorShape",
    "OpInputPrototype",
    "OpInputPrototype",
    "OpInputPrototype",
    "OpIOAPrototype",
    "OpInput",
    "OpOutput",
    "OpAttribute",
    "OpIOA",
    "get_operator_prototype",
    "get_model_opset",
    "get_model_input_shapes",
    "get_model_output_shapes",
    "get_model_initializer_np_data",
    "get_model_node_optypes",
    "get_valueinfo_shape",
    "get_valueinfo_np_dtype",
    "get_tensor_np_data",
    "get_node_input_args",
    "get_node_output_args",
    "get_node_attribute_args",
    "get_node_args",
    "get_node_kwargs",
    "dtype_onnx_to_np",
    "dtype_np_to_onnx",
]
