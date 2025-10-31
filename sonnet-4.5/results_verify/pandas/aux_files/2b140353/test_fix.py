#!/usr/bin/env python3
"""Test the proposed fix for SparseArray.cumsum bug"""

from pandas.arrays import SparseArray
import numpy as np

print("Testing the proposed fix logic:")

# Original buggy code would do:
# return SparseArray(self.to_dense()).cumsum()
# Which creates a new SparseArray with same fill_value, causing recursion

# Proposed fix would do:
# return SparseArray(self.to_dense().cumsum())
# Which creates SparseArray from the already cumsum'd dense array

sparse = SparseArray([1, 2, 3], fill_value=0)
print(f"Original SparseArray: {sparse}")
print(f"Fill value: {sparse.fill_value}")

# Convert to dense
dense = sparse.to_dense()
print(f"\nDense array: {dense}")

# Apply cumsum on dense array
dense_cumsum = dense.cumsum()
print(f"Dense cumsum: {dense_cumsum}")

# Create SparseArray from cumsum result (this is what the fix does)
result = SparseArray(dense_cumsum)
print(f"\nResult SparseArray: {result}")
print(f"Result values: {result.to_numpy()}")

# Verify that this gives the correct result
expected = np.array([1, 3, 6])
print(f"\nExpected values: {expected}")
print(f"Are they equal? {np.array_equal(result.to_numpy(), expected)}")

# Test with more complex case
print("\n" + "="*50)
print("Testing with sparse array that has actual sparse values:")

# Create a truly sparse array with zeros
sparse2 = SparseArray([1, 0, 2, 0, 3], fill_value=0)
print(f"SparseArray: {sparse2}")

dense2 = sparse2.to_dense()
print(f"Dense: {dense2}")

dense_cumsum2 = dense2.cumsum()
print(f"Dense cumsum: {dense_cumsum2}")

result2 = SparseArray(dense_cumsum2)
print(f"Result: {result2}")

expected2 = np.array([1, 1, 3, 3, 6])
print(f"Expected: {expected2}")
print(f"Are they equal? {np.array_equal(result2.to_numpy(), expected2)}")