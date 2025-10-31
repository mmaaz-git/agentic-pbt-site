#!/usr/bin/env python3
"""Verify the proposed fix works"""

import pandas.arrays
import numpy as np

# Test with the proposed fix logic
sparse = pandas.arrays.SparseArray([1, 0, 2, 0, 3], fill_value=0)
print("Original sparse array:", sparse)

# What the current code does (infinite recursion):
# return SparseArray(self.to_dense()).cumsum()

# What the fix would do:
dense = sparse.to_dense()
print("Dense array:", dense)

cumsum_dense = dense.cumsum()
print("Dense cumsum:", cumsum_dense)

fixed_result = pandas.arrays.SparseArray(cumsum_dense)
print("Fixed result (wrap after cumsum):", fixed_result)

# Verify this matches expected behavior
expected = [1, 1, 3, 3, 6]
print(f"\nExpected: {expected}")
print(f"Actual: {list(fixed_result)}")
print(f"Match: {list(fixed_result) == expected}")

# Test with other fill values
print("\n" + "="*50 + "\n")
print("Testing with fill_value=1:")
sparse2 = pandas.arrays.SparseArray([1, 1, 2, 1, 3], fill_value=1)
print(f"Original: {sparse2}")
print(f"_null_fill_value: {sparse2._null_fill_value}")

# This should work correctly since _null_fill_value is False
# and go through the problematic line 1550