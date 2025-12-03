import numpy as np
import time

# Data
N = 10000000
a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

# CPU computation
start = time.time()
c_cpu = a + b
end = time.time()

print("CPU Time:", end - start)