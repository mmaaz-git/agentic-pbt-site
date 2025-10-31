#!/usr/bin/env python3
"""Test to understand expected cumsum behavior"""

import numpy as np
from pandas.arrays import SparseArray

# Test numpy cumsum behavior
print("NumPy cumsum behavior:")
print("-" * 40)
arr1 = np.array([1, 0, 0, 2])
print(f"Input: {arr1}")
print(f"cumsum: {np.cumsum(arr1)}")  # Expected: [1, 1, 1, 3]

arr2 = np.array([1])
print(f"Input: {arr2}")
print(f"cumsum: {np.cumsum(arr2)}")  # Expected: [1]

arr3 = np.array([0, 0, 1, 0, 2])
print(f"Input: {arr3}")
print(f"cumsum: {np.cumsum(arr3)}")  # Expected: [0, 0, 1, 1, 3]

# Test with NaN
arr4 = np.array([1, np.nan, 2, np.nan])
print(f"Input: {arr4}")
print(f"cumsum: {np.cumsum(arr4)}")  # NaN propagates

print("\n" + "="*50 + "\n")

# Test SparseArray with NaN fill value (should work according to code)
print("SparseArray with NaN fill value:")
print("-" * 40)
try:
    sparse_nan = SparseArray([1, np.nan, 2, np.nan])
    print(f"Input: {sparse_nan}")
    print(f"Fill value: {sparse_nan.fill_value}")
    print(f"_null_fill_value: {sparse_nan._null_fill_value}")
    result = sparse_nan.cumsum()
    print(f"cumsum result: {result}")
    print(f"As dense: {result.to_dense()}")
except Exception as e:
    print(f"Error: {e}")

print("\n")

# Check what to_dense() returns
print("Understanding to_dense():")
print("-" * 40)
sparse1 = SparseArray([1, 0, 0, 2])
print(f"SparseArray: {sparse1}")
dense1 = sparse1.to_dense()
print(f"to_dense() result: {dense1}")
print(f"Type: {type(dense1)}")
print(f"What cumsum should return: {np.cumsum(dense1)}")