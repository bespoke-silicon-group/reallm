# %%
import numpy as np
import os, sys
sys.path.append('../')
sys.path.append('../LLMCompass')
from LLMCompass.design_space_exploration.dse import template_to_system, read_architecture_template
from LLMCompass.software_model.utils import Tensor, data_type_dict
from LLMCompass.software_model.matmul import *
import matplotlib.pyplot as plt
# %%
out_file = 'plot_matmul_lat.csv'
hw_name = 'GH100'
hw_specs = read_architecture_template(f'../LLMCompass/configs/{hw_name}.json')
lc_system = template_to_system(hw_specs)
device = lc_system.device
compile_mode = 'heuristic-GPU'
data_type = data_type_dict["fp16"]

exist_matmul_sizes = set()

if os.path.exists(out_file):
    # read all existing matmul sizes
    f = open(out_file, 'r')
    lines = f.readlines()
    for line in lines[1:]:
        B, M, K, N, _ = line.strip().split(',')
        if B == '1':
            exist_matmul_sizes.add((int(M), int(K), int(N)))
        else:
            exist_matmul_sizes.add((int(B), int(M), int(K), int(N)))
    f.close()
else:
    f = open(out_file, 'w')
    f.write('B, M, K, N, latency\n')
    f.close()

M = 8
K = 128
Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
Ns_lat = []
for N in Ns:
    matmul_size = (M, K, N)
    if matmul_size in exist_matmul_sizes:
        print(f'Skipping Matmul Size {matmul_size}')
        f = open(out_file, 'r')
        lines = f.readlines()
        for line in lines[1:]:
            B_str, M_str, K_str, N_str, lat = line.strip().split(',')
            if B_str == '1' and float(M_str) == matmul_size[0] and float(K_str) == matmul_size[1] and float(N_str) == matmul_size[2]:
                Ns_lat.append(float(lat))
                f.close()
                break
    else:
        print(f'Evaluate Matmul Size {matmul_size}')
        f = open(out_file, 'a')
        if len(matmul_size) == 3:
            M = matmul_size[0]
            K = matmul_size[1]
            N = matmul_size[2]
            mm = Matmul(data_type)
            input1 = Tensor([M, K], data_type)
            input2 = Tensor([K, N], data_type)
            _ = mm(input1, input2)
            lat = mm.compile_and_simulate(device, compile_mode)
            f.write(f'1, {M}, {K}, {N}, {lat}\n')
        elif len(matmul_size) == 4:
            B = matmul_size[0]
            M = matmul_size[1]
            K = matmul_size[2]
            N = matmul_size[3]
            mm = BatchedMatmul(data_type)
            input1 = Tensor([B, M, K], data_type)
            input2 = Tensor([B, K, N], data_type)
            _ = mm(input1, input2)
            lat = mm.compile_and_simulate(device, compile_mode)
            f.write(f'{B}, {M}, {K}, {N}, {lat}\n')
        Ns_lat.append(lat)
        print(f'{lat * 1e6} us')
        f.close()

M = 128
N = 512
Ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
Ks_lat = []
for K in Ks:
    matmul_size = (M, K, N)
    if matmul_size in exist_matmul_sizes:
        print(f'Skipping Matmul Size {matmul_size}')
        f = open(out_file, 'r')
        lines = f.readlines()
        for line in lines[1:]:
            B_str, M_str, K_str, N_str, lat = line.strip().split(',')
            if B_str == '1' and float(M_str) == matmul_size[0] and float(K_str) == matmul_size[1] and float(N_str) == matmul_size[2]:
                Ks_lat.append(float(lat))
                f.close()
                break
    else:
        print(f'Evaluate Matmul Size {matmul_size}')
        f = open(out_file, 'a')
        if len(matmul_size) == 3:
            M = matmul_size[0]
            K = matmul_size[1]
            N = matmul_size[2]
            mm = Matmul(data_type)
            input1 = Tensor([M, K], data_type)
            input2 = Tensor([K, N], data_type)
            _ = mm(input1, input2)
            lat = mm.compile_and_simulate(device, compile_mode)
            f.write(f'1, {M}, {K}, {N}, {lat}\n')
        elif len(matmul_size) == 4:
            B = matmul_size[0]
            M = matmul_size[1]
            K = matmul_size[2]
            N = matmul_size[3]
            mm = BatchedMatmul(data_type)
            input1 = Tensor([B, M, K], data_type)
            input2 = Tensor([B, K, N], data_type)
            _ = mm(input1, input2)
            lat = mm.compile_and_simulate(device, compile_mode)
            f.write(f'{B}, {M}, {K}, {N}, {lat}\n')
        Ks_lat.append(lat)
        print(f'{lat * 1e6} us')
        f.close()

# %%
# Interpolation
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# 2x2 plot: top: Ns, bottom: Ks
# left: interpolation, right: error comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for i, (Xs, all_lat, title) in enumerate([(Ns, Ns_lat, "N"), (Ks, Ks_lat, "K")]):
    # Example simulation data (Replace with your actual measured latencies)
    values = np.array(Xs)  # Matrix dimension N
    latencies = np.array(all_lat)  # Latency in ms

    # Example simulation data (Replace with actual data)
    # 1. Linear Interpolation
    linear_interp = interp1d(values, latencies, kind='linear')

    # 2. Polynomial Interpolation (Choose degree carefully)
    degree = min(len(values) - 1, 3)  # Limiting to cubic to avoid overfitting
    poly_coeffs = np.polyfit(values, latencies, degree)  # Fit polynomial
    poly_func = np.poly1d(poly_coeffs)  # Create polynomial function

    # Define new N values for estimation
    N_interp = np.linspace(values[0], values[-1], 2000)
    latency_linear = linear_interp(N_interp)
    latency_poly = poly_func(N_interp)

    # Compare predictions at intermediate values
    # generate 10 random values for N
    np.random.seed(0)
    test_points = []
    for j in range(1, len(values) - 1):
        new_test_points = (np.random.choice(range(values[j], values[j + 1]), 1, replace=False))
        for test_point in new_test_points:
            test_points.append(test_point)
    test_points = sorted(test_points)
    simulated_latencies = []
    simulated_matmul_sizes = []
    if i == 0:
        M = 8
        K = 128
        for N in test_points:
            simulated_matmul_sizes.append((M, K, N))
    else:
        M = 128
        N = 512
        for K in test_points:
            simulated_matmul_sizes.append((M, K, N))
    for matmul_size in simulated_matmul_sizes:
        if matmul_size in exist_matmul_sizes:
            f = open(out_file, 'r')
            lines = f.readlines()
            for line in lines[1:]:
                B_str, M_str, K_str, N_str, lat = line.strip().split(',')
                if B_str == '1' and float(M_str) == matmul_size[0] and float(K_str) == matmul_size[1] and float(N_str) == matmul_size[2]:
                    simulated_latencies.append(float(lat))
                    f.close()
                    print(f'Skipping Matmul Size {matmul_size}')
                    break
        else:
            M, K, N = matmul_size
            mm = Matmul(data_type)
            input1 = Tensor([M, K], data_type)
            input2 = Tensor([K, N], data_type)
            _ = mm(input1, input2)
            lat = mm.compile_and_simulate(device, compile_mode)
            simulated_latencies.append(lat)
            with open(out_file, 'a') as f:
                f.write(f'1, {M}, {K}, {N}, {lat}\n')
        print(f'M={M}, K={K}, N={N}, Latency={lat}')

    predicted_linear = linear_interp(test_points)
    predicted_poly = poly_func(test_points)
    # Compute absolute errors
    error_linear = np.abs(predicted_linear - simulated_latencies) / simulated_latencies
    error_poly = np.abs(predicted_poly - simulated_latencies) / simulated_latencies

    # Two Plots: left: interpolations, right: error comparison
    ax1 = axes[i][0]
    ax1.scatter(values, latencies, color='red', label="Simulated Data")
    ax1.plot(N_interp, latency_linear, label="Linear Interpolation", linestyle="dashed")
    ax1.plot(N_interp, latency_poly, label=f"Polynomial Interpolation (Degree {degree})", linestyle="dashdot")
    ax1.set_xlabel("N (Matrix Dimension)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_title("Linear vs. Polynomial Interpolation for MatMul Latency")
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[i][1]
    ax2.bar(test_points, error_linear, width=100, color='blue', alpha=0.7, label="Linear Interpolation")
    ax2.bar(test_points, error_poly, width=100, color='green', alpha=0.7, label=f"Polynomial Interpolation (Degree {degree})")
    ax2.set_xlabel("N (Matrix Dimension)")
    ax2.set_ylabel("Relative Error")
    ax2.set_title("Prediction Error Comparison")
    ax2.legend()
    ax2.grid(True)


# %%

