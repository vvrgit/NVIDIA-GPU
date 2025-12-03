import torch
import time

# Check GPU
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Matrix size
N = 3000

# Generate random matrices on CPU
A_cpu = torch.randn(N, N)
B_cpu = torch.randn(N, N)

# -----------------------------
# 1️⃣ CPU Computation Time
# -----------------------------
start = time.time()
C_cpu = torch.matmul(A_cpu, B_cpu)
cpu_time = time.time() - start
print(f"CPU Time: {cpu_time:.4f} seconds")

# -----------------------------
# 2️⃣ GPU Computation Time
# -----------------------------
device = torch.device("cuda")

# Move data to GPU
A_gpu = A_cpu.to(device)
B_gpu = B_cpu.to(device)

# GPU warm-up (important!)
for _ in range(5):
    torch.matmul(A_gpu, B_gpu)

torch.cuda.synchronize()     # wait for GPU to finish

start = time.time()
C_gpu = torch.matmul(A_gpu, B_gpu)
torch.cuda.synchronize()     # ensure timing is correct
gpu_time = time.time() - start
print(f"GPU Time: {gpu_time:.4f} seconds")

# Speedup
print(f"Speedup (CPU/GPU): {cpu_time / gpu_time:.2f}x")
