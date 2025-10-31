#!/usr/bin/env python3
"""Test expected behavior when k equals array size"""

import numpy as np
import dask.array as da

# Test with NumPy for comparison
print("NumPy behavior:")
arr = np.array([5, 1, 3, 6, 2])
print(f"Original array: {arr}")

# argsort returns all indices
sorted_indices = np.argsort(arr)[::-1]  # Reverse to get largest to smallest
print(f"np.argsort (reversed): {sorted_indices}")

# argpartition with k=5 (all elements)
partitioned = np.argpartition(arr, -5)[-5:]
print(f"np.argpartition(-5)[-5:]: {partitioned}")

# What we'd expect from argtopk(5) - top 5 elements sorted
expected_top5 = np.argsort(arr)[::-1]
print(f"Expected argtopk(5): {expected_top5}")

print("\n" + "="*50)
print("Testing dask behavior with single chunk (works):")
darr = da.from_array(arr, chunks=5)
try:
    result = da.argtopk(darr, 5).compute()
    print(f"da.argtopk(5) with single chunk: {result}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*50)
print("Testing what happens with k > array size:")
try:
    result = da.argtopk(darr, 10).compute()
    print(f"da.argtopk(10) on size-5 array: {result}")
except Exception as e:
    print(f"Expected error with k > size: {e}")

print("\n" + "="*50)
print("Testing with negative k:")
darr = da.from_array(arr, chunks=2)
try:
    result = da.argtopk(darr, -2).compute()
    print(f"da.argtopk(-2): {result}")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "="*50)
print("Testing k=-5 (all elements, smallest to largest):")
darr_single = da.from_array(arr, chunks=5)
try:
    result = da.argtopk(darr_single, -5).compute()
    print(f"da.argtopk(-5) with single chunk: {result}")
except Exception as e:
    print(f"ERROR: {e}")

darr_multi = da.from_array(arr, chunks=2)
try:
    result = da.argtopk(darr_multi, -5).compute()
    print(f"da.argtopk(-5) with multiple chunks: {result}")
except Exception as e:
    print(f"ERROR with multiple chunks: {e}")