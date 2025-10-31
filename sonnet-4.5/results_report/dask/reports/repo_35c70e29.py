import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import numpy as np
import dask.array as da

shape = [1]
axis = 1

x_np = np.random.rand(*shape)
x_da = da.from_array(x_np, chunks='auto')

print("Testing squeeze with shape=[1], axis=1 (out of bounds)")
print("=" * 60)

try:
    result = np.squeeze(x_np, axis=axis)
    print("NumPy: No error raised (unexpected)")
except Exception as e:
    print(f"NumPy error: {type(e).__name__}: {e}")

try:
    result = da.squeeze(x_da, axis=axis).compute()
    print("Dask: No error raised (unexpected)")
except Exception as e:
    print(f"Dask error: {type(e).__name__}: {e}")