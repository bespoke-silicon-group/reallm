import numpy as np


class ShapeInference:

    def __init__( self, network ):
        self.network = network
        self.alloc   = ShapeInferenceAllocator(network)


    def run( self, in_dict ):
        for T_id in self.network.inputs:
            self.alloc[T_id] = in_dict[T_id]

        for E in self.network.iter():
            kwargs = { **E.get_inputs(symtable=self.alloc), **E.get_attrs() }
            result = self._cap[E.type](self, **kwargs)
            for out_arg, out_T_id in E.get_output().items():
                self.alloc[out_T_id] = result



    def add( self, A, B ):
        if len(A) > len(B):
            return A
        elif len(B) > len(A):
            return B
        else:
            result = []
            for a,b in zip(A,B):
                result.append(max(a,b))
            return tuple(result)


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
        return A

    def lrn( self, A, alpha, beta, bias, size ):
        return A

    def dropout( self, A, ratio, seed ):
        return A
    
    def softmax( self, A, axis ):
        return A

    def clip( self, A, min, max ):
        return A

    def reshape( self, A, shape ):
        # I'm too lazy to implement... so just use a numpy reshape call on an
        # empty array. One feature that numpy doesn't have is that a reshape of
        # a dim to 0 implies that you want to keep that dim the same.
        new_shape = list(map(lambda x: A[x[0]] if x[1] == 0 else x[1], enumerate(shape)))
        return np.empty(A).reshape(new_shape).shape


    def transpose( self, A, axes ):
        if axes:
            return tuple([A[a] for a in axes])
        else:
            return A[::-1]

    def gather( self, A, indices, axis ):
        if len(indices) == 0:
            return (1, A[1])
        else:
            result = list(indices)
            result.append(A[1])
            return tuple(result)

    def reducemean( self, A, axes, keepdims ):
        x = [a if a >= 0 else len(A) + a for a in axes]
        result = []
        for i in range(len(A)):
            if i not in x:
                result.append(A[i])
        return tuple(result)

    def unsqueeze( self, A, axes ):
        result = list(A)
        for a in axes:
            result.insert(a, 1)
        return result
           
    def cast( self, A, type ):
        return A


    _cap = {
        "Add"                 :  add,
        "Unsqueeze"                 :  unsqueeze,
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
        "ReshapeStatic"       :  reshape,
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
        "Cast"                 :  cast,
        "Erf"                 :  relu,
        "Tanh"                 :  relu,
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

