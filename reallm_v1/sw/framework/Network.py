from framework.Expr import *
from framework.operators import *
from onnx_utils import *
import onnx
import os
import pickle
import numpy as np

class Network:

    def __init__( self, name="" ):
        """
            A network object is a graph that is represented with 4 simple
            dictionary objects: tensors, exprs, srcnodes, and sinknodes.

            The first two are the tensors and exprs dicts which store the edges
            and nodes of the graph respectively. The keys of these dicts are
            the id's of the Tensor and Expr objects for easy lookups.

            The other two dicts define the source and sinks of the edges in the
            graph (ie. tensors). The srcnodes dict maps tensor ids to the
            expression id that generates the tensor object (ie. expression
            outputs). The sinknodes dict maps tensor ids to a set of expression
            ids that use that tensor (ie. expression inputs).

            The network also maintains a set for the input edges and a set for
            the output edges of the graph. An optional name can also be given
            to the network object.
        """
        self.name      = name

        self.inputs    = set()
        self.outputs   = set()

        self.tensors   = {}
        self.exprs     = {}
        self.srcnodes  = {}
        self.sinknodes = {}


    def __str__( self ):
        """
            Create a string representation of the network.
        """
        lines = []
        lines.append(f"=========================")
        lines.append(f"Network:\n\t{self.name}")
        lines.append(f"=========================")
        lines.append(f"Inputs:\n\t{self.inputs}")
        lines.append(f"=========================")
        lines.append(f"Tensors:")
        for T in self.tensors.values(): lines.append(f"\t{str(T)}")
        lines.append(f"=========================")
        lines.append(f"Operations:")
        for E in self.iter(): lines.append(f"\t{str(E)}")
        lines.append(f"=========================")
        lines.append(f"Outputs:\n\t{self.outputs}")
        lines.append(f"=========================")
        return "\n".join(lines)


    def lookup_tensor( self, T_id ):
        """
            Return the network's Tensor object with the given id.
        """
        if T_id is None:
            return None
        return self.tensors[T_id]


    def lookup_expr( self, E_id ):
        """
            Return the network's Expr object with the given id.
        """
        if E_id is None:
            return None
        return self.exprs[E_id]


    def lookup_src( self, T_id ):
        """
            Return the Expr object that is the source for the tensor with the
            given id.
        """
        if T_id is None:
            return None
        return self.lookup_expr(self.srcnodes[T_id])


    def lookup_sinks( self, T_id ):
        """
            Return the Expr objects that are sinks for the tensor with the
            given id.
        """
        if T_id is None:
            return None

        sinks = []
        for E_id in self.sinknodes[T_id]:
            S = self.lookup_expr(E_id)
            if S is not None:
                sinks.append(S)
        return sinks


    def lookup_connections( self, T_id ):
        """
            Return a list of (Expr, str) pairs representing all of the expr
            arguments connected to the tensor with the given id.
        """
        if T_id is None:
            return None

        conns = []
        for S in self.lookup_sinks(T_id):
            for arg in S.connected_to(T_id):
                conns.append((S,arg))
        return conns


    def add_inputs( self, T_to_add ):
        """
            Add the given tensors to the network input set. Can accept a single
            Tensor or a collection of Tensors. Returns the number of new
            Tensors added to the input set.
        """
        if isinstance(T_to_add, Tensor):
            T_to_add = [T_to_add]

        skip_count = 0
        for T in T_to_add:
            assert isinstance(T,Tensor)
            if T.id in self.inputs:
                skip_count += 1
                continue
            self.inputs.add(T.id)

        return len(T_to_add) - skip_count


    def add_outputs( self, T_to_add ):
        """
            Add the given tensors to the network output set. Can accept a
            single Tensor or a collection of Tensors. Returns the number of new
            Tensors added to the output set.
        """
        if isinstance(T_to_add, Tensor):
            T_to_add = [T_to_add]

        skip_count = 0
        for T in T_to_add:
            assert isinstance(T,Tensor)
            if T.id in self.outputs:
                skip_count += 1
                continue
            self.outputs.add(T.id)

        return len(T_to_add) - skip_count


    def add_exprs( self, E_to_add, overwrite=False ):
        """
            Add the given expressions to the network. Can accept a single Epxr
            or a collection of Exprs. When an expression is added, any tensors
            connected to the inputs and outputs of the expression are also
            added to the network (if they are not already a part of the
            network) and the connection between the expression and tensor is
            established. Returns the number of expressions added to the
            network.
        """
        if isinstance(E_to_add, Expr):
            E_to_add = [E_to_add]

        skip_count = 0
        for E in E_to_add:
            assert isinstance(E,Expr)

            if E.id in self.exprs and not overwrite:
                skip_count += 1
                continue

            self.exprs[E.id] = E
            E.set_network(self)
            
            io_args = { **E.get_inputs(), **E.get_output() }
            for arg,sym in io_args.items():
                if sym is not None:
                    if isinstance(sym, List):
                        # print(f"Warning: {E.id} has multiple {arg} tensors:")
                        # print(f"\t{sym}")
                        for s in sym:
                            self.add_tensors(Tensor(id=s))
                            self.connect_expr(T_id=s, E_id=E.id, E_arg=arg)
                            # print(f"added tensor {s} to expr {E.id}")
                    else:
                        self.add_tensors(Tensor(id=sym))
                        self.connect_expr(T_id=sym, E_id=E.id, E_arg=arg)

        return len(E_to_add) - skip_count


    def add_tensors( self, T_to_add, overwrite=False ):
        """
            Add the given Tensors to the network. Can accept a single Tensor or
            a collection of Tensors. Returns the number of tensors added to the
            network.
        """
        if isinstance(T_to_add, Tensor):
            T_to_add = [T_to_add]

        skip_count = 0
        for T in T_to_add:
            assert isinstance(T,Tensor)
            if T.id in self.tensors and not overwrite:
                skip_count += 1
                continue
            self.tensors  [T.id] = T
            self.srcnodes [T.id] = None
            self.sinknodes[T.id] = set()

        return len(T_to_add) - skip_count


    def connect_expr( self, T_id, E_id, E_arg ):
        """
            Connect the tensor to the expression + argument with the given id
            or name.
        """
        E = self.lookup_expr(E_id)
        T = self.lookup_tensor(T_id)

        # concat input A has multiple tensors
        if E.type == "Concat" and E_arg == "A":
            if hasattr(E, "A"):
                if T.id not in E.A:
                    E.A.append(T.id)
            else:
                E.A = [T.id]
            self.sinknodes[T.id].add(E.id)
        # split output Z has multiple tensors
        elif E.type == "Split" and E_arg == "Z":
            if hasattr(E, "Z"):
                if T.id not in E.Z:
                    E.Z.append(T.id)
            else:
                E.Z = [T.id]
            self.srcnodes[T.id] = E.id
        else:

            setattr(E, E_arg, T.id)

            if E_arg in E.get_output():
                self.srcnodes[T.id] = E.id

            if E_arg in E.get_inputs():
                self.sinknodes[T.id].add(E.id)


    def remove_inputs( self, T_to_remove ):
        """
            Remove the given Tensors from the network input set. Can accept a
            single Tensor or a collection of Tensors.
        """
        if isinstance(T_to_remove, Tensor):
            T_to_remove = [T_to_remove]

        for T in T_to_remove:
            assert isinstance(T,Tensor)
            if T.id in self.inputs:
                self.inputs.remove(T.id)


    def remove_outputs( self, T_to_remove ):
        """
            Remove the given Tensors from the network output set. Can accept a
            single Tensor or a collection of Tensors.
        """
        if isinstance(T_to_remove, Tensor):
            T_to_remove = [T_to_remove]

        for T in T_to_remove:
            assert isinstance(T,Tensor)
            if T.id in self.outputs:
                self.outputs.remove(T.id)


    def remove_exprs( self, E_to_remove ):
        """
            Remove the given Exprs from the network. Can accept a single Expr
            or a collection of Exprs. Tensors whose source was the removed Expr
            will have their srcnode set to None.
        """
        if isinstance(E_to_remove, Expr):
            E_to_remove = [E_to_remove]

        for E in E_to_remove:
            assert isinstance(E,Expr)

            for T_id in E.get_output().values():
                if T_id and T_id in self.srcnodes:
                    self.srcnodes[T_id] = None

            for T_id in E.get_inputs().values():
                if T_id and T_id in self.sinknodes:
                    self.sinknodes[T_id].remove(E.id)

            self.exprs.pop(E.id)
            E.set_network(None)


    def remove_tensors( self, T_to_remove ):
        """
            Remove the given Tensors from the network. Can accept a single
            Tensor or a collection of Tensors. Exprs connected to tensors that
            are removed will have the tensors disconnected first.
        """
        if isinstance(T_to_remove, Tensor):
            T_to_remove = [T_to_remove]

        for T in T_to_remove:
            assert isinstance(T,Tensor)

            src_E = self.lookup_src(T.id)
            if src_E is not None:
                src_E.disconnect(T.id)

            sink_Es = self.lookup_sinks(T.id)
            for sink_E in sink_Es:
                if sink_E is not None:
                    sink_E.disconnect(T.id)

            self.remove_inputs(T)
            self.remove_outputs(T)

            self.tensors.pop(T.id)
            self.srcnodes.pop(T.id)
            self.sinknodes.pop(T.id)


    def iter( self ):
        """
            Iterable generator to traverse the network graph nodes (ie. Expr)
            in a topological ordering.
        """
        tracker = set()
        def _recursive( T_id ):
            E = self.lookup_src(T_id)
            if E and E.id not in tracker:
                tracker.add(E.id)
                for expr_in_T_id in E.get_inputs().values():
                    if isinstance(expr_in_T_id, list):
                        for e in expr_in_T_id:
                            yield from _recursive(e)
                    else:
                        if expr_in_T_id is not None:
                            yield from _recursive(expr_in_T_id)
                yield E
        for T_id in self.outputs:
            yield from _recursive(T_id)


    def save( self, filename=None ):
        """
            Save the Network object as a pickled file. If a filename is not
            given, an auto generated filename is used instead.
        """
        filename = f"{self.name}_network.pkl" if filename is None else filename
        with open(filename, "wb") as f:
            pickle.dump(self, f)
        return filename


    @classmethod
    def load( cls, filename ):
        """
            Load a pickled Network object.
        """
        with open(filename, "rb") as f:
            result = pickle.load(f)
        return result


    @classmethod
    def from_onnx( cls, filename, load_external_data = True ):
        """
            Create a Network object from the given onnx file.
        """
        network = cls(os.path.splitext(os.path.basename(filename))[0])
        onnx_model = onnx.load(filename, load_external_data=load_external_data)
        opset = get_model_opset(onnx_model)

        ### Add Param Exprs ###
        for tensor in onnx_model.graph.initializer:
            if load_external_data:
                value = get_tensor_np_data(tensor)
            else:
                value = np.ones(tensor.dims, dtype=dtype_onnx_to_np(tensor.data_type))
            network.add_exprs(
                Param( id=f"PARAM-{tensor.name}"
                     , Z=tensor.name
                     , value=value
                     )
            )

        ### Add All Other Exprs ###
        for node in onnx_model.graph.node:
            func = get_onnx_node_expr(node)
            expr = func(node, get_node_kwargs(node, opset))
            network.add_exprs(expr)

        ### Set Inputs ###
        initializer_names = [T.name for T in onnx_model.graph.initializer]
        for T in onnx_model.graph.input:
            if T.name not in initializer_names:
                network.add_inputs(network.lookup_tensor(T.name))

        ### Set Outputs ###
        for T in onnx_model.graph.output:
            network.add_outputs(network.lookup_tensor(T.name))

        return network

    @classmethod
    def from_onnx_partition( cls, filename, T, P, C,
                            load_external_data = True ):
        """
            Create a Network object from the given onnx file,
            each operator is partitioned based on parallelisms.
            T: tensor parallelism
            P: pipeline parallelism
            C: context parallelism
        """
        network = cls(os.path.splitext(os.path.basename(filename))[0])
        onnx_model = onnx.load(filename, load_external_data=load_external_data)
        opset = get_model_opset(onnx_model)

        ### Add Param Exprs ###
        for tensor in onnx_model.graph.initializer:
            if load_external_data:
                value = get_tensor_np_data(tensor)
            else:
                value = np.ones(tensor.dims, dtype=dtype_onnx_to_np(tensor.data_type))
            
            # partition the weight tensor based on tensor parallelism
            if 'weight' in tensor.name:
                if 'qkv' in tensor.name:
                    # qkv projection weight, split on the last dimension
                    value = np.split(value, T, axis=-1)[0]
                elif 'attn' in tensor.name:
                    # attention output projection weight, split on the first dimension
                    value = np.split(value, T, axis=0)[0]
                elif 'ffn1' in tensor.name:
                    # FFN layer 1 weight, split on the last dimension
                    value = np.split(value, T, axis=-1)[0]
                elif 'ffn2' in tensor.name:
                    # FFN layer 2 weight, split on the first dimension
                    value = np.split(value, T, axis=0)[0]
            elif 'split' in tensor.name:
                value = value / T
            network.add_exprs(
                Param( id=f"PARAM-{tensor.name}"
                     , Z=tensor.name
                     , value=value
                     )
            )

        ### Add All Other Exprs ###
        for node in onnx_model.graph.node:
            func = get_onnx_node_expr(node)
            expr = func(node, get_node_kwargs(node, opset))
            network.add_exprs(expr)

        ### Set Inputs ###
        initializer_names = [T.name for T in onnx_model.graph.initializer]
        for T in onnx_model.graph.input:
            if T.name not in initializer_names:
                network.add_inputs(network.lookup_tensor(T.name))

        ### Set Outputs ###
        for T in onnx_model.graph.output:
            network.add_outputs(network.lookup_tensor(T.name))

        return network