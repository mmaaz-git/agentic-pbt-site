import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask
import dask.array as da

print(f"Dask version: {dask.__version__}")

# Test dask's reshape_blockwise directly
from dask.array import reshape_blockwise

arr = da.arange(1).reshape(1, 1)
print(f"Original array shape: {arr.shape}")

try:
    result = reshape_blockwise(arr, shape=(1,))
    print(f"Dask reshape_blockwise result shape: {result.shape}")
except Exception as e:
    print(f"Dask reshape_blockwise failed with error: {e}")

# Test regular reshape
result_normal = arr.reshape((1,))
print(f"Normal reshape result shape: {result_normal.shape}")