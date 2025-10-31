import numpy as np

# Test numpy.take with empty indices
print("Testing numpy.take with empty indices:")
arr = np.array([1, 2, 3])
try:
    result = np.take(arr, [])
    print(f"Success! np.take([1,2,3], []) = {result}")
    print(f"Result dtype: {result.dtype}")
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test with different index dtypes
print("\nTesting numpy.take with empty float64 indices:")
empty_float = np.array([], dtype=np.float64)
try:
    result = np.take(arr, empty_float)
    print(f"Success with float64 indices! Result: {result}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting numpy.take with empty int64 indices:")
empty_int = np.array([], dtype=np.int64)
try:
    result = np.take(arr, empty_int)
    print(f"Success with int64 indices! Result: {result}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")