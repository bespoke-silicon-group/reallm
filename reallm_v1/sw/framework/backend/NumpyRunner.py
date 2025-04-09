import numpy as np


################################################################################
#
class NumpyRunner:

    ########################################################
    #
    def __init__( self, network ):
        self.network = network
        self.alloc   = NumpyRunnerAllocator()

    ########################################################
    #
    def run( self, in_dict ):
        for T_id in self.network.inputs:
            self.alloc[T_id] = in_dict[T_id]
        for E in self.network.iter():
            kwargs = { **E.get_inputs(symtable=self.alloc), **E.get_attrs() }
            result = self._cap[E.type](self, **kwargs)
            for _, T_id in E.get_output().items():
                self.alloc[T_id] = result
        return { T_id: self.alloc[T_id] for T_id in self.network.outputs }

    ########################################################
    #
    def add( self, A, B ):
        return A + B

    ########################################################
    #
    def batchnorm( self, A, gamma, beta, mean, var, epsilon ):
        A_norm = ((A - mean[None,:,None,None]) / np.sqrt(var + epsilon)[None,:,None,None])
        return A_norm * gamma[None,:,None,None] + beta[None,:,None,None]

    ########################################################
    #
    def conv2d( self, A, W, B, pads, strides, dilations ):
        n, c, ih, iw       = A.shape
        k, c, fh, fw       = W.shape
        ph0, pw0, ph1, pw1 = pads
        sh, sw             = strides
        dh, dw             = dilations

        oh = ((ih + ph0 + ph1 - dh * (fh - 1) -1) // sh) + 1
        ow = ((iw + pw0 + pw1 - dw * (fw - 1) -1) // sw) + 1

        row_idx = np.tile(np.repeat(dh*np.arange(fh), fw), c).reshape(-1, 1) + sh*np.repeat(np.arange(oh), ow).reshape(1, -1)
        col_idx = np.tile(dw*np.arange(fw), fh * c).reshape(-1, 1) + sw*np.tile(np.arange(ow), oh)
        ch_idx  = np.repeat(np.arange(c), fh*fw).reshape(-1, 1)

        A_pad = np.pad(A, ((0,0), (0,0), (ph0,ph1), (pw0,pw1)))
        A_col = np.concatenate(A_pad[:,ch_idx,row_idx,col_idx], axis=-1)
        W_col = W.reshape((k,-1))

        result = np.array(np.hsplit((W_col @ A_col), n)).reshape((n,k,oh,ow))
        if B is not None:
            result += B.reshape(1,k,1,1)
        return result

    ########################################################
    #
    def gemm( self, A, B, C ):
        result = A @ B
        if C is not None:
            result += C
        return result

    ########################################################
    #
    def globalavgpool( self, A ):
        n,c,_,_ = A.shape
        return np.average(A, axis=(2,3)).reshape((n,c,1,1))

    ########################################################
    #
    def maxpool( self, A, kernel, pads, strides, dilations ):
        n, c, ih, iw       = A.shape
        fh, fw             = kernel
        ph0, pw0, ph1, pw1 = pads
        sh, sw             = strides
        dh, dw             = dilations

        oh = ((ih + ph0 + ph1 - dh * (fh - 1) -1) // sh) + 1
        ow = ((iw + pw0 + pw1 - dw * (fw - 1) -1) // sw) + 1

        assert dh == 1 and dw == 1, "max pool dilations not currently supported" 

        row_idx = np.repeat(np.arange(fh), fw).reshape(-1, 1) + sh*np.repeat(np.arange(oh), ow).reshape(1, -1)
        col_idx = np.tile(np.arange(fw), fh).reshape(-1, 1) + sw*np.tile(np.arange(ow), oh)

        A_pad = np.pad(A, ((0,0), (0,0), (ph0,ph1), (pw0,pw1)), 'constant', constant_values=np.NINF)
        A_col = A_pad[:,:,row_idx,col_idx]
        return np.max(A_col, axis=2).reshape((n, c, oh, ow))

    ########################################################
    #
    def param( self, value ):
        return value

    ########################################################
    #
    def relu( self, A ):
        return np.clip(A, 0.0, None)

    ########################################################
    #
    def reshape( self, A, shape ):
        new_shape = list(map(lambda x: A.shape[x[0]] if x[1] == 0 else x[1], enumerate(shape)))
        return A.reshape(new_shape)

    ########################################################
    #
    def transpose( self, A, axes ):
        return A.transpose(axes)

    ########################################################
    #
    _cap = {
        "Add"            :  add,
        "Batchnorm"      :  batchnorm,
        "Conv2D"         :  conv2d,
        "Gemm"           :  gemm,
        "GlobalAvgpool"  :  globalavgpool,
        "Maxpool"        :  maxpool,
        "Param"          :  param,
        "Relu"           :  relu,
        "ReshapeStatic"  :  reshape,
        "Transpose"      :  transpose,
    }


################################################################################
#
class NumpyRunnerAllocator:

    ########################################################
    #
    def __init__( self ):
        self.mem = {}

    ########################################################
    #
    def __getitem__( self, key ):
        return self.mem[key]

    ########################################################
    #
    def __setitem__( self, key, value ):
        self.mem[key] = value

    ########################################################
    #
    def __contains__( self, key ):
        return key in self.mem

