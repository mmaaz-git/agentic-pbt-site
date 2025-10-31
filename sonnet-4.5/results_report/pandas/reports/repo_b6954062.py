from pandas.arrays import SparseArray
import numpy as np

# Create a SparseArray with integer dtype
arr = SparseArray([1, 2, 3], dtype=np.int64)
print(f"Original array: {arr}")
print(f"Original type: {type(arr)}")
print()

# Call astype with np.float64 (plain numpy dtype, not SparseDtype)
result = arr.astype(np.float64)

print(f"Result after astype(np.float64): {result}")
print(f"Result type: {type(result)}")
print()

print(f"Expected type: <class 'pandas.core.arrays.sparse.array.SparseArray'>")
print(f"Actual type: {type(result)}")
print()

# Try to use SparseArray methods on the result
try:
    print(f"Attempting to call to_dense() on result...")
    dense = result.to_dense()
    print(f"to_dense() succeeded: {dense}")
except AttributeError as e:
    print(f"Error: {e}")
    print("Result is not a SparseArray, it's a numpy.ndarray!")