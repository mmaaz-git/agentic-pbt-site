import numpy as np
from pandas.arrays import SparseArray

# Test the basic reproduction case
arr = SparseArray([0], fill_value=0, dtype=np.int64)
result = arr.astype(np.float64)

print(f"Result type: {type(result)}")
print(f"Is SparseArray: {isinstance(result, SparseArray)}")
print(f"Result value: {result}")

# Test with a slightly different case
arr2 = SparseArray([0, 0, 0], fill_value=0, dtype=np.int64)
result2 = arr2.astype(np.float64)
print(f"\nTest 2 - Result type: {type(result2)}")
print(f"Test 2 - Is SparseArray: {isinstance(result2, SparseArray)}")

# Test with non-fill values
arr3 = SparseArray([0, 1, 0], fill_value=0, dtype=np.int64)
result3 = arr3.astype(np.float64)
print(f"\nTest 3 - Result type: {type(result3)}")
print(f"Test 3 - Is SparseArray: {isinstance(result3, SparseArray)}")