#!/usr/bin/env python3
"""Test script to reproduce the pandas SparseArray.cumsum bug"""

import sys
import traceback
from pandas.arrays import SparseArray
import numpy as np

# Test 1: Basic reproduction test
print("Test 1: Basic reproduction test")
print("-" * 40)
try:
    sparse = SparseArray([1, 0, 0, 2])
    print(f"Created SparseArray: {sparse}")
    result = sparse.cumsum()
    print(f"Result of cumsum: {result}")
except RecursionError as e:
    print(f"RecursionError caught: {e}")
    print("Traceback (last 5 lines):")
    tb_lines = traceback.format_exc().split('\n')
    print('\n'.join(tb_lines[-10:]))
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\n")

# Test 2: Test with zero fill value (should work according to bug report)
print("Test 2: Test with zero/NaN fill value")
print("-" * 40)
try:
    sparse = SparseArray([0, 0, 1, 0, 2])
    print(f"Created SparseArray with zeros: {sparse}")
    print(f"Fill value: {sparse.fill_value}")
    result = sparse.cumsum()
    print(f"Result of cumsum: {result}")
    print(f"As dense: {result.to_dense()}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")

print("\n")

# Test 3: Test with simple case [1]
print("Test 3: Test with simple case [1]")
print("-" * 40)
try:
    sparse = SparseArray([1])
    print(f"Created SparseArray: {sparse}")
    result = sparse.cumsum()
    print(f"Result of cumsum: {result}")
except RecursionError as e:
    print(f"RecursionError caught: {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\n")

# Test 4: Check fill_value behavior
print("Test 4: Check fill_value behavior")
print("-" * 40)
sparse1 = SparseArray([1, 0, 0, 2])
print(f"SparseArray([1, 0, 0, 2]) fill_value: {sparse1.fill_value}")
print(f"_null_fill_value: {sparse1._null_fill_value}")

sparse2 = SparseArray([0, 0, 1, 0])
print(f"SparseArray([0, 0, 1, 0]) fill_value: {sparse2.fill_value}")
print(f"_null_fill_value: {sparse2._null_fill_value}")

sparse3 = SparseArray([1, np.nan, 2, np.nan])
print(f"SparseArray([1, nan, 2, nan]) fill_value: {sparse3.fill_value}")
print(f"_null_fill_value: {sparse3._null_fill_value}")