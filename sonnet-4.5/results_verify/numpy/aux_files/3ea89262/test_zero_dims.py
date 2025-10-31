#!/usr/bin/env python3
"""Test if numpy generally supports zero-dimension arrays"""

import numpy as np

print("Testing numpy array support for zero dimensions...")

# Test regular numpy arrays with zero dimensions
arr1 = np.zeros((0, 5))
print(f"np.zeros((0, 5)) - Shape: {arr1.shape}, Size: {arr1.size}")

arr2 = np.zeros((5, 0))
print(f"np.zeros((5, 0)) - Shape: {arr2.shape}, Size: {arr2.size}")

arr3 = np.array([[], [], []])
print(f"np.array([[], [], []]) - Shape: {arr3.shape}, Size: {arr3.size}")

# Test operations on zero-dimension arrays
arr4 = np.zeros((3, 0))
arr5 = np.zeros((0, 4))
print(f"\nZero-dimension arrays are valid in numpy: {isinstance(arr4, np.ndarray)}")

# Test matrix multiplication with zero dimensions
try:
    result = np.dot(np.zeros((2, 0)), np.zeros((0, 3)))
    print(f"Matrix multiplication (2,0) @ (0,3) works: Shape = {result.shape}")
except Exception as e:
    print(f"Matrix multiplication with zero dims failed: {e}")

# Test that matrices are just special arrays
m = np.matrix("1 2; 3 4")
print(f"\nnp.matrix is subclass of ndarray: {isinstance(m, np.ndarray)}")