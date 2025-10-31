#!/usr/bin/env python3
"""Test script to reproduce the SparseArray argmin/argmax bug"""

from pandas.arrays import SparseArray
import numpy as np

print("=" * 60)
print("Testing SparseArray argmin/argmax bug")
print("=" * 60)

# Test case 1: Array with only fill value (0)
print("\nTest 1: SparseArray([0])")
try:
    sparse = SparseArray([0])
    print(f"sparse array: {sparse}")
    print(f"fill_value: {sparse.fill_value}")
    result = sparse.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"argmin() raised: {type(e).__name__}: {e}")

try:
    sparse = SparseArray([0])
    result = sparse.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"argmax() raised: {type(e).__name__}: {e}")

# Compare with numpy behavior
print(f"\nnumpy.argmin([0]): {np.argmin([0])}")
print(f"numpy.argmax([0]): {np.argmax([0])}")

# Test case 2: Array with multiple fill values
print("\n" + "=" * 60)
print("Test 2: SparseArray([0, 0, 0])")
try:
    sparse = SparseArray([0, 0, 0])
    print(f"sparse array: {sparse}")
    result = sparse.argmin()
    print(f"argmin() result: {result}")
except Exception as e:
    print(f"argmin() raised: {type(e).__name__}: {e}")

try:
    sparse = SparseArray([0, 0, 0])
    result = sparse.argmax()
    print(f"argmax() result: {result}")
except Exception as e:
    print(f"argmax() raised: {type(e).__name__}: {e}")

print(f"\nnumpy.argmin([0, 0, 0]): {np.argmin([0, 0, 0])}")
print(f"numpy.argmax([0, 0, 0]): {np.argmax([0, 0, 0])}")

# Test case 3: Array with non-fill values (should work)
print("\n" + "=" * 60)
print("Test 3: SparseArray([0, 1, 2])")
try:
    sparse = SparseArray([0, 1, 2])
    print(f"sparse array: {sparse}")
    print(f"argmin() result: {sparse.argmin()}")
    print(f"argmax() result: {sparse.argmax()}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test case 4: Check to_dense consistency
print("\n" + "=" * 60)
print("Test 4: Consistency check with to_dense()")
test_data = [0, 0, 0]
sparse = SparseArray(test_data)
dense = sparse.to_dense()
print(f"sparse array: {sparse}")
print(f"dense array: {dense}")
print(f"Type of dense: {type(dense)}")
try:
    sparse_min = sparse.argmin()
    print(f"sparse.argmin(): {sparse_min}")
except Exception as e:
    print(f"sparse.argmin() raised: {type(e).__name__}: {e}")

print(f"np.argmin(dense): {np.argmin(dense)}")