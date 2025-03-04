from framework.Expr import *
from framework.backend.PerformanceSim import register_roofline_simulator, register_uarch_simulator
from framework.operators.Transpose import Transpose
from LLMCompass.software_model.matmul import *
from LLMCompass.software_model.utils import Tensor, data_type_dict
from dataclasses import dataclass


@dataclass
class Gemm( Expr ):
    id : str    = None
    Z  : ArgOut = None
    A  : ArgIn  = None
    B  : ArgIn  = None
    C  : ArgIn  = None


@register_onnx( "MatMul" )
def from_onnx( node, kwargs ):
    return Gemm( id = node.name
               , Z  = kwargs["Y"]
               , A  = kwargs["A"]
               , B  = kwargs["B"]
               )


@register_onnx( "Gemm" )
def from_onnx( node, kwargs ):
    assert kwargs["alpha"] in { None, 1.0 }
    assert kwargs["beta"]  in { None, 1.0 }

    exprs = []

    A_name = kwargs["A"]
    B_name = kwargs["B"]

    if kwargs["transA"] == 1:
        A_name += "_transposed"

    if kwargs["transB"] == 1:
        B_name += "_transposed"

    gemm_expr = Gemm( id = node.name
                    , Z  = kwargs["Y"]
                    , A  = A_name
                    , B  = B_name
                    , C  = kwargs["C"]
                    )

    if kwargs["transA"] == 1:
        exprs.append(
            Transpose( id = gemm_expr.id + "_transA"
                     , Z  = A_name
                     , A  = kwargs["A"]
                     )
        )
    if kwargs["transB"] == 1:
        exprs.append(
            Transpose( id = gemm_expr.id + "_transB"
                     , Z  = B_name
                     , A  = kwargs["B"]
                     )
        )

    exprs.append( gemm_expr )

    return exprs

@register_roofline_simulator( "Gemm" )
def roofline_sim(node_perf, node_bw, input_shapes: dict, data_bytes=2, saved_results_file_path=None, debug=False):

    flops = node_perf
    mem_bw = node_bw
    A_shape = input_shapes["A"]
    B_shape = input_shapes["B"]
    assert len(A_shape) > 1
    assert len(B_shape) > 1

    A_pre_shape = A_shape[:-2]
    B_pre_shape = B_shape[:-2]
    A_shape = A_shape[-2:]
    B_shape = B_shape[-2:]
    if A_shape[1] != B_shape[0]:
        print(f"A_shape: {A_shape}, B_shape: {B_shape}")
    assert A_shape[1] == B_shape[0]

    # Batched Matmul: A: [batch_size, M, K], B: [batch_size, K, N], C: [batch_size, M, N]
    #             or  A: [batch_size, M, K], B: [K, N], C: [batch_size, M, N]
    #             or  A: [M, K], B: [batch_size, K, N], C: [batch_size, M, N]
    num_matmul = 1
    if len(A_pre_shape) > 0 and len(B_pre_shape) > 0:
        assert A_pre_shape == B_pre_shape
        for dim in A_pre_shape:
            num_matmul *= dim
    elif len(A_pre_shape) > 0:
        assert len(B_pre_shape) == 0
        for dim in A_pre_shape:
            num_matmul *= dim
    elif len(B_pre_shape) > 0:
        assert len(A_pre_shape) == 0
        for dim in B_pre_shape:
            num_matmul *= dim
    
    # num_A = 1
    # num_B = 1
    # for dim in A_pre_shape:
    #     num_A *= dim
    # for dim in B_pre_shape:
    #     num_B *= dim
    
    def sim(num_matmul, A_shape, B_shape, flops, mem_bw, data_bytes, debug):
        # Memory access
        A_load_time = A_shape[0] * A_shape[1] * data_bytes / mem_bw
        B_load_time = B_shape[0] * B_shape[1] * data_bytes / mem_bw

        # Computation
        compute_time = A_shape[0] * A_shape[1] * B_shape[1] * 2 / flops

        matmul_time = max(A_load_time + B_load_time, compute_time)
        total_time = num_matmul * matmul_time

        if debug:
            print("=========================================")
            print(f"Running roofline simulator for Gemm")
            print(f"flops: {flops:.2E}, Mem_bw: {mem_bw:.2E} byte/s")
            print(f"A: {input_shapes['A']}, B: {input_shapes['B']}")
            print(f"num_matmul: {num_matmul}, A_shape: {A_shape}, B_shape: {B_shape}")
            print(f"A_load_time: {A_load_time:.3E}s, B_load_time: {B_load_time:.3E}s, compute_time: {compute_time:.3E}, matmul_time: {matmul_time:.3E}s")
            print(f"total_time: {total_time:.3E}s")
        
        return A_load_time, B_load_time, compute_time, matmul_time, total_time

    A_load_time, B_load_time, compute_time, matmul_time, total_time = sim(num_matmul, A_shape, B_shape, flops, mem_bw, data_bytes, debug)

    # overwrite = False
    # if not os.path.exists(saved_results_file_path) or overwrite:
    #     with open(saved_results_file_path, "w") as f:
    #         f.write("flops, mem_bw, num_A, num_B, A_shape, B_shape, num_matmul, A_load_time, B_load_time, matmul_time, total_time\n")


    # check if the result is already saved
    # A_shape_str = 'x'.join([str(dim) for dim in A_shape])
    # B_shape_str = 'x'.join([str(dim) for dim in B_shape])
    # hash_str = f"{flops}, {mem_bw}, {num_A}, {num_B}, {A_shape_str}, {B_shape_str}"
    # found = False
    # with open(saved_results_file_path, "r") as f:
    #     lines = f.readlines()
    #     for line in lines[1:]:
    #         if line.startswith(str(hash_str)):
    #             found = True
    #             _, _, _, _, _, _, _, A_load_time, B_load_time, matmul_time, total_time = line.split(", ")
    #             A_load_time = int(float(A_load_time))
    #             B_load_time = int(float(B_load_time))
    #             matmul_time = int(float(matmul_time))
    #             total_time = int(float(total_time))
    #             if debug:
    #                 print(f"Found saved result for {hash_str}, flops: {flops}, mem_bw: {mem_bw}")
    #             break
    # if not found:
    #     A_load_time, B_load_time, num_matmul, matmul_time, total_time = sim(A_shape, B_shape, num_A, num_B, flops, mem_bw, debug)
    #     with open(saved_results_file_path, "a") as f:
    #         f.write(f"{flops}, {mem_bw}, {num_A}, {num_B}, {A_shape_str}, {B_shape_str}, {num_matmul}, {A_load_time}, {B_load_time}, {matmul_time}, {total_time}\n")


    return total_time

@register_uarch_simulator( "Gemm" )
def uarch_sim(node, input_shapes: dict, compile_mode="heuristic-GPU", saved_results_file_path=None, debug=False):
    A_shape = input_shapes["A"]
    B_shape = input_shapes["B"]

    assert len(A_shape) > 1
    assert len(B_shape) > 1

    A_pre_shape = A_shape[:-2]
    B_pre_shape = B_shape[:-2]
    A_shape = A_shape[-2:]
    B_shape = B_shape[-2:]
    assert A_shape[1] == B_shape[0]

    # Batched Matmul: A: [batch_size, M, K], B: [batch_size, K, N], C: [batch_size, M, N]
    #             or  A: [batch_size, M, K], B: [K, N], C: [batch_size, M, N]
    #             or  A: [M, K], B: [batch_size, K, N], C: [batch_size, M, N]
    num_matmul = 1
    if len(A_pre_shape) > 0 and len(B_pre_shape) > 0:
        assert A_pre_shape == B_pre_shape
        for dim in A_pre_shape:
            num_matmul *= dim
    elif len(A_pre_shape) > 0:
        assert len(B_pre_shape) == 0
        for dim in A_pre_shape:
            num_matmul *= dim
    elif len(B_pre_shape) > 0:
        assert len(A_pre_shape) == 0
        for dim in B_pre_shape:
            num_matmul *= dim
    
    A_shape = [num_matmul] + list(A_shape)
    B_shape = [num_matmul] + list(B_shape)

    data_type = data_type_dict["fp16"]
    input1 = Tensor(A_shape, data_type)
    input2 = Tensor(B_shape, data_type)
    if len(A_shape) > 2:
        mm = BatchedMatmul(data_type)
    else:
        mm = Matmul(data_type)
    _ = mm(input1, input2)
    total_time = mm.compile_and_simulate(node, compile_mode)
    return total_time
