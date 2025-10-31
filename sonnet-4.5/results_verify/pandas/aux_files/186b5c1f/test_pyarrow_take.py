import pyarrow as pa
import numpy as np

# Test pyarrow take with empty indices
print("Testing pyarrow array.take with empty indices:")
pa_arr = pa.array([1, 2, 3], type=pa.int64())
print(f"Original array: {pa_arr}")

# Test with empty list
print("\n1. Empty Python list:")
try:
    result = pa_arr.take([])
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with empty numpy array (float64 - default)
print("\n2. Empty numpy array (float64 default):")
empty_float = np.array([])
print(f"   Empty array dtype: {empty_float.dtype}")
try:
    result = pa_arr.take(empty_float)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with empty numpy array (int64)
print("\n3. Empty numpy array (int64):")
empty_int = np.array([], dtype=np.int64)
print(f"   Empty array dtype: {empty_int.dtype}")
try:
    result = pa_arr.take(empty_int)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test pyarrow compute.take
print("\n4. Using pa.compute.take:")
try:
    result = pa.compute.take(pa_arr, pa.array([], type=pa.int64()))
    print(f"Success with pa.compute.take! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")