import numpy as np
from framework.Pattern import *
from framework.operators import *
from dataclasses import fields

class ShapeInference:

    def __init__( self, network ):
        self.network = network
        self.alloc   = ShapeInferenceAllocator(network)
        self.unk_idx = 0

    def run( self, in_dict ):
        for T_id in self.network.inputs:
            self.alloc[T_id] = in_dict[T_id]

        for E in self.network.iter():
            kwargs = { **E.get_inputs(symtable=self.alloc), **E.get_attrs() }
            result = self._cap[E.type](self, **kwargs)
            if result is None:
                tensor_sym = getattr(E, "A")
                srcnode = self.network.lookup_src(tensor_sym)
                print(f"ShapeInference: {E.type} failed to infer shape. Inputs: {kwargs}")
                print(f"A Source node: {srcnode}")
                raise Exception(f"ShapeInference: {E.type} failed to infer shape. Inputs: {kwargs}")
            for out_arg, out_T_id in E.get_output().items():
                self.alloc[out_T_id] = result
            
    def run_symbolic( self, preset_in_dict = {} ):

        # for some input/output tensors of those operators, we add a data attribute to them if they are determined
        # since many of them are determined or used by the tensor shape

        # Set input values if they are determined
        for T_id, value in preset_in_dict.items():
            T = self.network.lookup_tensor(T_id)
            if T is not None:
                T.data = value

        # Add data attribute to output of all Param operators
        for E in self.network.iter():
            if E.type == "Param":
                output_tensor = self.network.lookup_tensor(E.Z)
                output_tensor.data = E.value

        # Some operators are often used to determine the shape
        for E in self.network.iter():
            if E.type == "Add" or E.type == "Sub" or E.type == "Mul" or E.type == "Div":
                A = self.network.lookup_tensor(E.A)
                B = self.network.lookup_tensor(E.B)
                if hasattr(A, 'data') and hasattr(B, 'data'):
                    if E.type == "Add":
                        result = np.add(A.data, B.data)
                    elif E.type == "Sub":
                        result = np.subtract(A.data, B.data)
                    elif E.type == "Mul":
                        result = np.multiply(A.data, B.data)
                    elif E.type == "Div":
                        result = np.divide(A.data, B.data)
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.data = result
                    output_tensor.shape = result.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.shape = self._cap[E.type](self, A.shape, B.shape)
            elif E.type == "Cast":
                A = self.network.lookup_tensor(E.A)
                if hasattr(A, 'data'):
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.data = A.data
                    output_tensor.shape = A.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.shape = self.cast(A.shape, None)
            elif E.type == "Concat":
                input_tensors = [self.network.lookup_tensor(a) for a in E.A]
                if all([hasattr(tensor, 'data') for tensor in input_tensors]):
                    axis = E.axis
                    result = np.concatenate([tensor.data for tensor in input_tensors], axis=axis)
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.data = result
                    output_tensor.shape = result.shape
                else:
                    axis = E.axis
                    output_axis_dim = 0
                    update_shape = True
                    for tensor in input_tensors:
                        if isinstance(tensor.shape[axis], str):
                            update_shape = False
                            break
                        else:
                            output_axis_dim += tensor.shape[axis]
                    if update_shape:
                        output_tensor = self.network.lookup_tensor(E.Z)
                        output_tensor.shape = list(input_tensors[0].shape)
                        output_tensor.shape[axis] = output_axis_dim
                        output_tensor.shape = tuple(output_tensor.shape)
            elif E.type == "Gather":
                indices_tensor = self.network.lookup_tensor(E.indices)
                data_tensor = self.network.lookup_tensor(E.A)
                if hasattr(indices_tensor, 'data') and hasattr(data_tensor, 'data'):
                    data = np.array(data_tensor.data)
                    indices = np.array(indices_tensor.data, dtype=int)
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.data = np.take(data, indices, axis=0)
                    output_tensor.shape = output_tensor.data.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.shape = self.gather(data_tensor.shape, indices_tensor.shape, E.axis)
            elif E.type == "Reshape" or E.type == "ReshapeStatic":
                if E.type == "Reshape":
                    shape = self.network.lookup_tensor(E.shape)
                else:
                    shape = E.shape
                A = self.network.lookup_tensor(E.A)
                if hasattr(shape, 'data'):
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_tensor.shape = self.reshapestatic(A.shape, shape.data)
                if hasattr(A, 'data'):
                    output_tensor.data = np.reshape(A.data, output_tensor.shape)
            elif E.type == "Shape":
                data_shape = self.network.lookup_tensor(E.data).shape
                if all([isinstance(dim, int) or isinstance(dim, np.int64) for dim in data_shape]):
                    output_tensor = self.network.lookup_tensor(E.shape)
                    output_tensor.data = np.array(data_shape, dtype=int)
                    assert output_tensor.shape == (len(data_shape),)
            elif E.type == "Slice":
                # if axes are omitted (==NONE), they are set to [0, 1, ..., len(data.shape) - 1]
                # if steps are omitted (==NONE), they are set to [1, 1, ..., 1]
                data_tensor = self.network.lookup_tensor(E.data)
                starts_tensor = self.network.lookup_tensor(E.starts)
                ends_tensor = self.network.lookup_tensor(E.ends)
                axes_tensor = self.network.lookup_tensor(E.axes)
                steps_tensor = self.network.lookup_tensor(E.steps)
                # if hasattr(data_tensor, 'data') and hasattr(starts_tensor, 'data') and hasattr(ends_tensor, 'data'):
                if hasattr(starts_tensor, 'data') and hasattr(ends_tensor, 'data'):
                    if hasattr(data_tensor, 'data'):
                        data = np.array(data_tensor.data)
                    else:
                        if all([isinstance(dim, int) or isinstance(dim, np.int64) for dim in data_tensor.shape]):
                            data = np.zeros(data_tensor.shape)
                        else:
                            continue
                    starts = np.array(starts_tensor.data, dtype=int)
                    ends = np.array(ends_tensor.data, dtype=int)
                    if axes_tensor is None:
                        axes = list(range(len(data.shape)))
                    else:
                        if hasattr(axes_tensor, 'data'):
                            axes = np.array(axes_tensor.data, dtype=int)
                        else:
                            continue
                    if steps_tensor is None:
                        steps = np.ones(len(axes), dtype=int)
                    else:
                        if hasattr(steps_tensor, 'data'):
                            steps = np.array(steps_tensor.data, dtype=int)
                        else:
                            continue
                    output_tensor = self.network.lookup_tensor(E.output)
                    output = np.copy(data)
                    for i, axis in enumerate(axes):
                        start = starts[i]
                        end = ends[i]
                        if start < 0:
                            start += data.shape[axis]
                        if end < 0:
                            end += data.shape[axis]
                        step = steps[i]
                        if step > 0:
                            start = np.clip(start, 0, data.shape[axis])
                            end = np.clip(end, 0, data.shape[axis])
                        else:
                            start = np.clip(start, 0, data.shape[axis] - 1)
                            end = np.clip(end, -1, data.shape[axis] - 1)
                        output = np.take(output, range(start, end, step), axis=axis) 
                    if hasattr(data_tensor, 'data'):
                        output_tensor.data = output
                    output_tensor.shape = output.shape
                else:
                    # output shape depends on the values of starts, ends, axes, and steps
                    pass
            elif E.type == "Equal":
                A = self.network.lookup_tensor(E.A)
                B = self.network.lookup_tensor(E.B)
                if hasattr(A, 'data') and hasattr(B, 'data'):
                    result = np.equal(A.data, B.data)
                    output_tensor = self.network.lookup_tensor(E.C)
                    output_tensor.data = result
                    output_tensor.shape = result.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.C)
                    output_tensor.shape = self.pow(A.shape, B.shape)
            elif E.type == "ConstantOfShape":
                input_tensor = self.network.lookup_tensor(E.input)
                if hasattr(input_tensor, 'data'):
                    output_tensor = self.network.lookup_tensor(E.output)
                    # val = self.network.lookup_tensor(E.value).data
                    val = E.value
                    output_tensor.data = np.full(input_tensor.data, val)
                    output_tensor.shape = output_tensor.data.shape
                else:
                    # output shape depends on the values of input
                    pass
            elif E.type == "Unsqueeze":
                data_tensor = self.network.lookup_tensor(E.A)
                axes_tensor = self.network.lookup_tensor(E.axes)
                if hasattr(data_tensor, 'data') and hasattr(axes_tensor, 'data'):
                    axes = tuple(np.array(axes_tensor.data, dtype=int))
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_shape = list(data_tensor.shape)
                    data = np.array(data_tensor.data)
                    output_tensor.data = np.expand_dims(data, axes)
                    output_tensor.shape = output_tensor.data.shape
                elif hasattr(axes_tensor, 'data'):
                    axes = tuple(np.array(axes_tensor.data, dtype=int))
                    output_tensor = self.network.lookup_tensor(E.Z)
                    output_shape = list(data_tensor.shape)
                    for axis in axes:
                        output_shape.insert(axis, 1)
                    output_tensor.shape = tuple(output_shape)
                else:
                    # output shape depends on the values of axes
                    pass
            elif E.type == "Range":
                start_tensor = self.network.lookup_tensor(E.start)
                limit_tensor = self.network.lookup_tensor(E.limit)
                delta_tensor = self.network.lookup_tensor(E.delta)
                if hasattr(start_tensor, 'data') and hasattr(limit_tensor, 'data') and hasattr(delta_tensor, 'data'):
                    start = start_tensor.data
                    limit = limit_tensor.data
                    delta = delta_tensor.data
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.data = np.arange(start, limit, delta)
                    output_tensor.shape = output_tensor.data.shape
                else:
                    # output shape depends on the values of start, limit, and delta
                    pass
            elif E.type == "Expand":
                input_tensor = self.network.lookup_tensor(E.input)
                shape_tensor = self.network.lookup_tensor(E.shape)
                if hasattr(input_tensor, 'data') and hasattr(shape_tensor, 'data'):
                    input_data = input_tensor.data
                    shape_data = shape_tensor.data
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.data = input_data * np.ones(shape_data, dtype=input_data.dtype)
                    output_tensor.shape = output_tensor.data.shape
                elif hasattr(shape_tensor, 'data'):
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.shape = shape_tensor.data
                else:
                    # output shape depends on the values of shape
                    pass
            elif E.type == "Where":
                condition_tensor = self.network.lookup_tensor(E.condition)
                X_tensor = self.network.lookup_tensor(E.X)
                Y_tensor = self.network.lookup_tensor(E.Y)
                if hasattr(condition_tensor, 'data') and hasattr(X_tensor, 'data') and hasattr(Y_tensor, 'data'):
                    condition = condition_tensor.data
                    X = X_tensor.data
                    Y = Y_tensor.data
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.data = np.where(condition, X, Y)
                    output_tensor.shape = output_tensor.data.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.shape = self.where(condition_tensor.shape, X_tensor.shape, Y_tensor.shape)
            elif E.type == "Trilu":
                # Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s). 
                # The attribute "upper" determines whether the upper or lower part is retained. 
                # If k = 0, the triangular part on and above/below the main diagonal is retained. 
                input_tensor = self.network.lookup_tensor(E.input)
                k_tensor = self.network.lookup_tensor(E.k)
                upper = E.upper
                if hasattr(input_tensor, 'data') and hasattr(k_tensor, 'data'):
                    input_data = input_tensor.data
                    k = k_tensor.data
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.data = np.triu(input_data, k=k) if upper else np.tril(input_data, k=k)
                    output_tensor.shape = output_tensor.data.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.shape = self.trilu(input_tensor.shape, k_tensor.data, upper)
            elif E.type == "Greater":
                # with broadcasting
                A_tensor = self.network.lookup_tensor(E.A)
                B_tensor = self.network.lookup_tensor(E.B)
                if hasattr(A_tensor, 'data') and hasattr(B_tensor, 'data'):
                    A = A_tensor.data
                    B = B_tensor.data
                    output_tensor = self.network.lookup_tensor(E.C)
                    output_tensor.data = np.greater(A, B)
                    output_tensor.shape = output_tensor.data.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.C)
                    output_tensor.shape = self.pow(A_tensor.shape, B_tensor.shape)
            elif E.type == "ScatterND":
                data_tensor = self.network.lookup_tensor(E.data)
                indices_tensor = self.network.lookup_tensor(E.indices)
                updates_tensor = self.network.lookup_tensor(E.updates)
                if hasattr(data_tensor, 'data') and hasattr(indices_tensor, 'data') and hasattr(updates_tensor, 'data'):
                    data = data_tensor.data
                    indices = indices_tensor.data
                    updates = updates_tensor.data
                    output = np.copy(data)
                    update_indices = indices.shape[:-1]
                    for idx in np.ndindex(update_indices):
                        output[tuple(indices[idx])] = updates[idx]
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.data = output
                    output_tensor.shape = output.shape
                else:
                    output_tensor = self.network.lookup_tensor(E.output)
                    output_tensor.shape = self.scatternd(data_tensor.shape, indices_tensor.shape, updates_tensor.shape)
            else:
                inputs = {}
                attrs = {}
                output = {}
                for f in fields(E):
                    if "ArgIn" in str(f.type):
                        tensor_sym = getattr(E, f.name)
                        if isinstance(tensor_sym, list):
                            # the input A of concat is a list of tensors
                            inputs[f.name] = [self.network.lookup_tensor(t).shape for t in tensor_sym]
                        else:
                            tensor = self.network.lookup_tensor(tensor_sym)
                            if tensor is None:
                                inputs[f.name] = None
                            else:
                                tensor_shape = tensor.shape
                                inputs[f.name] = tensor_shape
                    if "ArgAttr" in str(f.type):
                        attrs[f.name] = getattr(E, f.name)
                    if "ArgOut" in str(f.type):
                        output[f.name] = getattr(E, f.name)
                output_shape = self._cap[E.type](self, **inputs, **attrs)
                for output in output.values():
                    output_tensor = self.network.lookup_tensor(output)
                    output_tensor.shape = output_shape



    def add( self, A, B ):
        # if len(A) > len(B):
        #     return A
        # elif len(B) > len(A):
        #     return B
        # else:
        #     result = []
        #     for a,b in zip(A,B):
        #         result.append(max(a,b))
        #     return tuple(result)

        # multidirectional broadcasting
        return self.multidirectional_broadcast_with_unknown(A, B)


    def batchnorm( self, A, gamma, beta, mean, var, epsilon ):
        return A


    def conv2d( self, A, W, B, pads, strides, dilations ):
        n, ic, ih, iw      = A
        k, wc, fh, fw      = W
        ph0, pw0, ph1, pw1 = pads
        sh, sw             = strides
        dh, dw             = dilations

        oh = ((ih + ph0 + ph1 - dh * (fh - 1) -1) // sh) + 1
        ow = ((iw + pw0 + pw1 - dw * (fw - 1) -1) // sw) + 1

        return (n,k,oh,ow)


    def gemm( self, A, B, C ):
        if len(A) == 2:
            return (A[0], B[1])
        elif len(A) > 2:
            result = list(A[:-1])
            result.append(B[-1])
            return tuple(result)


    def globalavgpool( self, A ):
        n, c = A[0], A[1]
        return (n,c,1,1)


    def maxpool( self, A, kernel, pads, strides, dilations ):
        n, c, ih, iw       = A
        fh, fw             = kernel
        ph0, pw0, ph1, pw1 = pads
        sh, sw             = strides
        dh, dw             = dilations

        oh = ((ih + ph0 + ph1 - dh * (fh - 1) -1) // sh) + 1
        ow = ((iw + pw0 + pw1 - dw * (fw - 1) -1) // sw) + 1

        return (n,c,oh,ow)


    def param( self, value ):
        return value.shape


    def relu( self, A ):
        return A

    def sqrt( self, A ):
        return A

    def pow( self, A, B ):
        # multidirectional broadcasting
        return self.multidirectional_broadcast_with_unknown(A, B)

    def lrn( self, A, alpha, beta, bias, size ):
        return A

    def dropout( self, A, ratio, seed ):
        return A
    
    def softmax( self, A, axis ):
        return A

    def clip( self, A, min, max ):
        return A

    def reshapestatic( self, A, shape ):
        # I'm too lazy to implement... so just use a numpy reshape call on an
        # empty array. One feature that numpy doesn't have is that a reshape of
        # a dim to 0 implies that you want to keep that dim the same.
        # new_shape = list(map(lambda x: A[x[0]] if x[1] == 0 else x[1], enumerate(shape)))
        # return np.empty(A).reshape(new_shape).shape
        """
        Infers the output shape for a Reshape operation with support for unknown dimensions.
        
        :param A: Tuple of integers or strings (e.g., 'unk__<id>') representing the input tensor A shape.
        :param shape: Tuple of integers or strings specifying the target shape, including -1 or unknowns.
        :return: Tuple representing the inferred output shape, including any unknowns.
        :raises ValueError: If the shape cannot be inferred due to invalid inputs.
        """
        # Compute total number of known elements in data_shape
        def element_count(dims):
            count = 1
            for dim in dims:
                if isinstance(dim, int) or isinstance(dim, np.int64):
                    count *= dim
                elif isinstance(dim, str) and dim.startswith("unk__"):
                    return -1  # Unknown size
                else:
                    raise ValueError(f"Invalid dimension: {dim}")
            return count

        data_shape = A
        num_elements = element_count(data_shape)

        # Validate and process the target shape
        inferred_index = -1  # Index of the -1 dimension
        inferred_elements = num_elements
        output_shape = []

        for i, dim in enumerate(shape):
            if dim == -1:
                if inferred_index != -1:
                    raise ValueError("Only one dimension can be set to -1.")
                inferred_index = i
                output_shape.append(-1)  # Placeholder for now
            elif dim == 0:
                # Without allowzero, 0 means taking the corresponding dimension from data_shape
                if i >= len(data_shape):
                    raise ValueError("Shape contains 0 but input data_shape has insufficient dimensions.")
                output_shape.append(data_shape[i])
                if isinstance(data_shape[i], int) and inferred_elements > 0:
                    inferred_elements //= data_shape[i]
            elif isinstance(dim, str) and dim.startswith("unk__"):
                output_shape.append(dim)
                inferred_elements = -1  # inferred_elements becomes unknown
            elif dim > 0:
                output_shape.append(dim)
                if inferred_elements > 0:
                    inferred_elements //= dim
            else:
                raise ValueError(f"Invalid dimension: {dim}")

        # Handle the -1 dimension
        if inferred_index != -1:
            if inferred_elements <= 0:
                output_shape[inferred_index] = f'unk__{self.unk_idx}'
                self.unk_idx += 1
            else:
                output_shape[inferred_index] = inferred_elements

        # Assign new unknown indices if any -1 remains or unknowns are inferred
        for i, dim in enumerate(output_shape):
            if dim == -1:
                output_shape[i] = f"unk__{self.unk_idx}"
                self.unk_idx += 1

        return tuple(output_shape)


    def reshape( self, A, shape ):
        results = []
        if isinstance(shape[0], str):
            # This shouldn't happen, return unknown shape for now
            print(f'Warning: reshape shape {shape} is unknown')
            for _ in range(len(A)):
                results.append('unk__' + str(self.unk_idx))
                self.unk_idx += 1
            return tuple(results)
        for _ in range(shape[0]):
            results.append('unk__' + str(self.unk_idx))
            self.unk_idx += 1
        return results


    def transpose( self, A, axes ):
        if axes:
            return tuple([A[a] for a in axes])
        else:
            return A[::-1]

    def gather( self, A, indices, axis ):
        # if len(indices) == 0:
        #     return (1, A[1])
        # else:
        #     result = list(indices)
        #     result.append(A[1])
        #     return tuple(result)
        # seems like the above code is wrong
        if axis == None:
            axis = 0
        results = [None] * (len(indices) + len(A) - 1)
        for i in range(0, axis):
            results[i] = A[i]
        for i in range(axis, axis + len(indices)):
            results[i] = indices[i - axis]
        for i in range(axis + len(indices), len(results)):
            results[i] = A[i - len(indices) + 1]
        return tuple(results)
    

    def reducemean( self, A, axes, keepdims ):
        x = [a if a >= 0 else len(A) + a for a in axes]
        result = []
        if keepdims == 1 or keepdims == True:
            for i in range(len(A)):
                if i in x:
                    result.append(1)
                else:
                    result.append(A[i])
        else:
            for i in range(len(A)):
                if i not in x:
                    result.append(A[i])
        return tuple(result)

    def unsqueeze( self, A, axes ):
        # result = list(A)
        # for a in axes:
        #     result.insert(a, 1)
        # return tuple(result)
        # the above code only works when A is empty tuple (i.e., scalar)
        # we need the values of the axes to be inserted
        # axes is a 1D tensor, axes[0] is the number of dimensions to be inserted
        if A == (): # scalar
            return tuple([1] * axes[0])

        result = [None] * (len(A) + axes[0])
        for i in range(len(result)):
            result[i] = f'unk__{self.unk_idx}'
            self.unk_idx += 1
        return tuple(result)
           
    def cast( self, A, type ):
        return A
    
    def concat( self, A, axis ):
        # all tensors in A have the same shape except for the axis dimension
        result = list(A[0])
        for shape in A[1:]:
            if isinstance(shape[axis], str):
                result[axis] = f'unk__{self.unk_idx}'
                self.unk_idx += 1
                break
            result[axis] += shape[axis]
        return tuple(result)
    
    def cos ( self, input ):
        return input
    
    def constantofshape( self, input , value ):
        # input (1D tensor) is the shape of the output tensor
        # return unknown shape for now
        results = []
        for _ in range(input[0]):
            results.append('unk__' + str(self.unk_idx))
            self.unk_idx += 1
        return tuple(results)
    
    def shape( self, data ):
        results = [len(data)]
        return tuple(results)
    
    def expand( self, input, shape ):
        # shape (1D tensor) is the shape of the output tensor
        results = []
        if isinstance(shape[0], str):
            # This shouldn't happen
            print(f'Warning: expand shape {shape} is unknown')
            results.append('unk__' + str(self.unk_idx))
            self.unk_idx += 1
            return tuple(results)
        for _ in range(shape[0]):
            results.append('unk__' + str(self.unk_idx))
            self.unk_idx += 1
        return tuple(results)
    
    def neg( self, X ):
        return X
    
    def range_( self, start, limit, delta ):
        # shape depends on the values of start, limit, and delta
        # return unknown shape for now
        results = ['unk__' + str(self.unk_idx)]
        self.unk_idx += 1
        return tuple(results)
    
    def scatternd( self, data, indices, updates ):
        return data
    
    def trilu( self, input, k, upper ):
        return input
    
    def where( self, condition, X, Y ):
        # output shape equal to the broadcasted shape of condition, X, and Y.
        return self.multidirectional_broadcast_with_unknown(condition, X, Y)
    
    def slice( self, data, starts, ends, axes, steps ):
        # Slice uses the starts, ends, axes and steps inputs 
        # to select a sub-tensor of its input data tensor
        # return the shape as data with unknown dimensions
        results = []
        for _ in range(len(data)):
            results.append('unk__' + str(self.unk_idx))
            self.unk_idx += 1
        return tuple(results)
    
    def multidirectional_broadcast_with_unknown(self, *shapes):
        """
        Computes the resulting shape from multidirectional broadcasting of input shapes,
        supporting unknown dimensions with prefix 'unk__'.
        ONNX multidirectional broadcasting rule: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md

        :param shapes: Tuple of shapes (each shape is a tuple of integers or strings like 'unk__<id>').
        :return: The resulting broadcasted shape as a tuple, or raises ValueError if not broadcastable.
        """
        def is_compatible(dim, max_dim):
            """Check if a dimension is compatible with max_dim."""
            if isinstance(max_dim, str) and max_dim.startswith("unk__"):
                return True # Unknown max_dim is compatible with any dim
            if isinstance(dim, str) and dim.startswith("unk__"):
                return True  # Unknown dims are compatible with any max_dim
            return dim == 1 or dim == max_dim

        # Find the maximum number of dimensions among all shapes
        max_dims = max(len(shape) for shape in shapes)

        # Pad shapes with 1s to match the maximum dimensions
        padded_shapes = [
            (1,) * (max_dims - len(shape)) + shape for shape in shapes
        ]

        # Compute the result shape
        result_shape = []
        for dims in zip(*padded_shapes):
            # Dimension rules: all dims must be 1, max_dim, or 'unk__<id>'
            known_dims = [dim for dim in dims if not (isinstance(dim, str) and dim.startswith("unk__"))]
            unknown_dims = [dim for dim in dims if isinstance(dim, str) and dim.startswith("unk__")]
            
            max_known_dim = 1
            max_unknown_dim = 1
            if known_dims:
                max_known_dim = max(known_dims)  # Use the largest known dimension
            if unknown_dims:
                max_unknown_dim = unknown_dims[0]  # Pick one of the unknown dims arbitrarily as max_dim

            if max_unknown_dim == 1:
                max_dim = max_known_dim
            else:
                if max_known_dim == 1:
                    max_dim = max_unknown_dim
                else:
                    # Not sure if we should use the known or unknown dim as max_dim
                    # max_dim = f'unk__{self.unk_idx}'
                    # self.unk_idx += 1
                    max_dim = max_known_dim

            if all(is_compatible(dim, max_dim) for dim in dims):
                result_shape.append(max_dim)
            else:
                raise ValueError(f"Shapes {shapes} are not broadcastable.")

        return tuple(result_shape)


    _cap = {
        "Add"                 :  add,
        "Unsqueeze"           :  unsqueeze,
        "Batchnorm"           :  batchnorm,
        "BatchnormFusedRelu"  :  batchnorm,
        "Conv2D"              :  conv2d,
        "Conv2DFusedRelu"     :  conv2d,
        "Gemm"                :  gemm,
        "GemmFusedRelu"       :  gemm,
        "GlobalAvgpool"       :  globalavgpool,
        "Maxpool"             :  maxpool,
        "Param"               :  param,
        "Relu"                :  relu,
        "Sqrt"                :  sqrt,
        "ReshapeStatic"       :  reshapestatic,
        "Reshape"             :  reshape,
        "Transpose"           :  transpose,
        "LRN"                 :  lrn,
        "Dropout"             :  dropout,
        "Softmax"             :  softmax,
        "Clip"                :  clip,
        "Gather"              :  gather,
        "ReduceMean"          :  reducemean,
        "Sub"                 :  add,
        "Div"                 :  add,
        "Mul"                 :  add,
        "Pow"                 :  pow,
        "Cast"                :  cast,
        "Erf"                 :  relu,
        "Tanh"                :  relu,
        "Concat"              :  concat,
        "ConstantOfShape"     :  constantofshape,
        'Cos'                 :  cos,     
        "Equal"               :  pow,
        "Expand"              :  expand,
        "Greater"             :  pow,        
        "Neg"                 :  neg,
        "Range"               :  range_,
        "ScatterND"           :  scatternd,
        "Shape"               :  shape,
        "Sigmoid"             :  neg,
        "Sin"                 :  cos, 
        "Trilu"               :  trilu,
        "Where"               :  where,
        "Slice"               :  slice,
    }




class ShapeInferenceAllocator:

    def __init__( self, network ):
        self.network = network


    def __getitem__( self, key ):
        return self.network.lookup_tensor(key).shape


    def __setitem__( self, key, value ):
        self.network.lookup_tensor(key).shape = value


    def __contains__( self, key ):
        return self.network.lookup_tensor(key) is not None

