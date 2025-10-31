#!/usr/bin/env python3
"""Test how NumPy handles ravel on various empty arrays."""

import numpy as np

print("Testing NumPy's behavior with empty arrays:")
print("=" * 60)

test_cases = [
    # shape, description
    ((0,), "1D empty array"),
    ((0, 0), "2D fully empty array"),
    ((3, 0), "2D array with zero columns"),
    ((0, 3), "2D array with zero rows"),
    ((5, 0, 2), "3D with zero in middle dimension"),
    ((2, 3, 0), "3D with zero in last dimension"),
    ((0, 0, 0), "3D fully empty"),
]

for shape, desc in test_cases:
    arr = np.empty(shape, dtype=np.float64)
    raveled = np.ravel(arr)

    print(f"\n{desc}:")
    print(f"  Input shape: {shape}")
    print(f"  Input size: {arr.size}")
    print(f"  Raveled shape: {raveled.shape}")
    print(f"  Raveled size: {raveled.size}")

    # Verify it's indeed a 1D array
    assert raveled.ndim == 1, f"Expected 1D array but got {raveled.ndim}D"
    # Verify size preservation
    assert raveled.size == arr.size, f"Size mismatch: {raveled.size} != {arr.size}"

print("\n" + "=" * 60)
print("Conclusion: NumPy successfully ravels ALL empty arrays to 1D shape (0,)")
print("regardless of how the zero dimensions are distributed.")