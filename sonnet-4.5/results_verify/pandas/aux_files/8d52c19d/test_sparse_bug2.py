#!/usr/bin/env python3
"""Test the SparseArray astype() bug - corrected version"""

from pandas.arrays import SparseArray
import numpy as np

# Simple test of the roundtrip issue
print("=== Testing Roundtrip Issue ===")
values = [1, 2, 3]
arr = SparseArray(values, dtype=np.int64)
print(f"Original arr type: {type(arr)}")
print(f"Original arr: {arr}")

arr_float = arr.astype(np.float64)
print(f"\nAfter astype(np.float64):")
print(f"  Type: {type(arr_float)}")
print(f"  Value: {arr_float}")
print(f"  Has to_dense() method: {hasattr(arr_float, 'to_dense')}")

# Try to continue the roundtrip
if isinstance(arr_float, np.ndarray):
    print("\nCannot continue roundtrip - arr_float is ndarray, not SparseArray")
    print("Attempting arr_float.astype(np.int64) will give ndarray, not SparseArray")
    arr_int = arr_float.astype(np.int64)
    print(f"  Type of arr_int: {type(arr_int)}")
    print(f"  Value of arr_int: {arr_int}")
else:
    arr_int = arr_float.astype(np.int64)
    print(f"\nAfter roundtrip back to int64:")
    print(f"  Type: {type(arr_int)}")
    print(f"  Value: {arr_int}")
    # Check if values are preserved
    print(f"  Values equal: {np.array_equal(arr.to_dense(), arr_int.to_dense())}")