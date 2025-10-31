import numpy as np
import dask.array as da

# Create a simple array
arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)

# Create dask array with multiple chunks
dask_arr = da.from_array(arr, chunks=2)

print(f"Array: {arr}")
print(f"Dask array chunks: {dask_arr.chunks}")
print(f"Requesting k=5 (equal to array size)")

# This should return indices of all 5 elements sorted by their values
# But it crashes when k equals array size with multiple chunks
try:
    result = da.argtopk(dask_arr, 5).compute()
    print(f"Result: {result}")
except Exception as e:
    import traceback
    print(f"\nError occurred: {e}")
    print("\nFull traceback:")
    traceback.print_exc()