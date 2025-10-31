#!/usr/bin/env python3
"""Test NumPy behavior for edge cases"""

import numpy as np

arr = np.array([5, 1, 3, 6, 2])
print("Array:", arr)
print("Array size:", len(arr))

# Test argsort - always returns all indices
print("\nnp.argsort():")
print("  Result:", np.argsort(arr))
print("  Result reversed:", np.argsort(arr)[::-1])

# Test argpartition with k equal to size
print("\nnp.argpartition with k=5 (equal to size):")
try:
    result = np.argpartition(arr, 4)  # kth element (0-indexed)
    print("  np.argpartition(arr, 4):", result)
    print("  Last 5 elements:", result[-5:])
except Exception as e:
    print(f"  Error: {e}")

print("\nnp.argpartition with k=-5:")
try:
    result = np.argpartition(arr, -5)
    print("  np.argpartition(arr, -5):", result)
    print("  Last 5 elements:", result[-5:])
except Exception as e:
    print(f"  Error: {e}")

# Test with k > size
print("\nnp.argpartition with k > size:")
try:
    result = np.argpartition(arr, 10)
    print("  np.argpartition(arr, 10):", result)
except Exception as e:
    print(f"  Error: {e}")

# What about negative k > size?
print("\nnp.argpartition with k=-10 (abs > size):")
try:
    result = np.argpartition(arr, -10)
    print("  np.argpartition(arr, -10):", result)
except Exception as e:
    print(f"  Error: {e}")