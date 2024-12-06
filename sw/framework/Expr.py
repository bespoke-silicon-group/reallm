from dataclasses import dataclass, fields
from typing import List, Union
import uuid

class ArgIn: pass
    # Expression input argument. These inputs are string symbols that can be
    # used as a key to a symbol table to get the real input argument object.

class ArgOut: pass
    # Expression output argument. These outputs are string symbols that can be
    # used as a key to a symbol table to get the real output argument object.

class ArgAttr: pass
    # Expression attribute argument. Attributes are instance constants that
    # change the behaviour of the expression.

################################################################################
#
_expr_onnx_node_proto_converters = {}

def register_onnx( op_type ):
    def wrapper( func ):
        _expr_onnx_node_proto_converters[op_type] = func
        return func
    return wrapper

def get_onnx_node_expr( node ):
    return _expr_onnx_node_proto_converters[node.op_type]


################################################################################
#
def default_attr( attr, default ):
    return attr if attr is not None else default


################################################################################
#
@dataclass
class Tensor:
    id    : str       = None
    type  : str       = None
    shape : List[int] = None


################################################################################
#
@dataclass
class Expr:

    def __post_init__( self ):
        """
            After an Expr has been created, we add a type field (which is the
            name of the class) and we make sure that an id has been set. If an
            id has not been set, generate a random unique id.
        """
        self.type = self.__class__.__name__
        if not hasattr(self,"id") or self.id is None or self.id == "":
            self.id = str(uuid.uuid4())
        self.network = None

    def get_inputs( self, symtable=None ):
        """
           Return the expr's inputs as a dict that maps the arg name to the
           attached tensor symbol. If a symtable is provided, the arg name is
           instead mapped to the value in the symtable that is keyed by the
           tensor symbol.
        """
        result = {}
        for f in fields(self):
            if "ArgIn" in str(f.type):
                tensor_sym = getattr(self, f.name)
                if symtable is None:
                    result[f.name] = tensor_sym
                elif tensor_sym in symtable:
                    result[f.name] = symtable[tensor_sym]
                else:
                    result[f.name] = None
        return result

    ########################################################
    #
    def get_attrs( self ):
        """
           Return the expr attributes as a dict that maps the arg name to the
           attribute value.
        """
        result = {}
        for f in fields(self):
            if "ArgAttr" in str(f.type):
                result[f.name] = getattr(self, f.name)
        return result

    ########################################################
    #
    def get_output( self, symtable=None ):
        """
           Return the expr's output as a tuple of arg name, attached tensor
           symbol. If a symtable is provided, the arg name is instead mapped to
           the value in the symtable that is keyed by the tensor symbol.
        """
        for f in fields(self):
            if "ArgOut" in str(f.type):
                tensor_sym = getattr(self, f.name)
                if symtable is None:
                    return {f.name: tensor_sym}
                elif tensor_sym in symtable:
                    return {f.name: symtable[tensor_sym]}
                else:
                    return {f.name: None}
        return None

    ########################################################
    #
    def connected_to( self, T_id ):
        """
            Return a list of all arguments (inputs and outputs) connected to
            the tensor with the given id.
        """
        io_args = { **self.get_inputs(), **self.get_output() }
        return [arg for arg,sym in io_args.items() if sym == T_id]

    ########################################################
    #
    def disconnect( self, T_id ):
        """
            Remove any arguments that are connected to the tensor with the
            given id. Returns a list of all the args that were connected to
            the tensor.
        """
        result = self.connected_to(T_id)
        for arg in result:
            setattr(self,arg,None)
        return result

    ########################################################
    #
    def set_network( self, network ):
        """
            Associate the expression with a network.
        """
        assert self.network is None or network is None, "Expr {self.id} already associated with network {self.network.name}"
        self.network = network

