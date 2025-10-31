import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask.array as da
from xarray.compat.dask_array_compat import reshape_blockwise

# Create a (1, 1) dask array
arr = da.arange(1).reshape(1, 1)
print(f"Input array shape: {arr.shape}")
print(f"Input array: {arr.compute()}")

# Try to reshape to (1,)
result = reshape_blockwise(arr, (1,))
print(f"Expected shape: (1,)")
print(f"Actual shape: {result.shape}")

# Verify the failure
try:
    assert result.shape == (1,), f"Expected shape (1,), but got {result.shape}"
    print("Test PASSED")
except AssertionError as e:
    print(f"Test FAILED: {e}")