#!/usr/bin/env python3
"""Debug test to understand the internals"""

from pandas.arrays import SparseArray
import numpy as np

# Create a sparse array with only fill values
sparse = SparseArray([0, 0, 0])
print(f"SparseArray: {sparse}")
print(f"Fill value: {sparse.fill_value}")
print(f"Sparse values: {sparse._sparse_values}")
print(f"Sparse index: {sparse._sparse_index}")
print(f"Sparse index indices: {sparse._sparse_index.indices}")
print(f"Length of sparse values: {len(sparse._sparse_values)}")

# What happens when we try to use np.argmin on empty array?
print("\nTesting np.argmin on empty array:")
try:
    result = np.argmin([])
    print(f"np.argmin([]): {result}")
except Exception as e:
    print(f"np.argmin([]) raises: {type(e).__name__}: {e}")

print("\nTesting np.argmax on empty array:")
try:
    result = np.argmax([])
    print(f"np.argmax([]): {result}")
except Exception as e:
    print(f"np.argmax([]) raises: {type(e).__name__}: {e}")