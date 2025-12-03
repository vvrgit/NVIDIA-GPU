from numba import cuda
import time
import numpy as np

print(cuda.gpus)
print(cuda.get_current_device())

# Data
N = 10000000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)


# CUDA kernel
@cuda.jit
def vector_add(a, b, c):
    pos = cuda.grid(1)          # Get global thread ID
    if pos < a.size:
        c[pos] = a[pos] + b[pos]



# Allocate GPU memory
a_gpu = cuda.to_device(a)
b_gpu = cuda.to_device(b)
c_gpu = cuda.device_array_like(a)

# Threads per block and blocks per grid
threads_per_block = 256
blocks_per_grid = (a.size + (threads_per_block - 1)) // threads_per_block

# Run GPU kernel
start = time.time()
vector_add[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)
cuda.synchronize()  # Important!
end = time.time()

print("GPU Time:", end - start)

# Copy result back
c_gpu_result = c_gpu.copy_to_host()