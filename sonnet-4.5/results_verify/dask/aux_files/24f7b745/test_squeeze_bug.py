#!/usr/bin/env python3
"""Test script to reproduce the squeeze bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env')

import numpy as np
import dask.array as da

print("Testing NumPy and Dask squeeze with out-of-bounds axis...")
print("=" * 60)

# Test case from bug report
shape = [1]
axis = 1

print(f"\nTest 1: shape={shape}, axis={axis}")

x_np = np.random.rand(*shape)
x_da = da.from_array(x_np, chunks='auto')

print("\nNumPy behavior:")
try:
    result = np.squeeze(x_np, axis=axis)
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"Module: {type(e).__module__}")

print("\nDask behavior:")
try:
    result = da.squeeze(x_da, axis=axis).compute()
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"Module: {type(e).__module__}")

# Test additional cases
print("\n" + "=" * 60)
print("\nTest 2: shape=[2, 3], axis=2")
shape = [2, 3]
axis = 2

x_np = np.random.rand(*shape)
x_da = da.from_array(x_np, chunks='auto')

print("\nNumPy behavior:")
try:
    result = np.squeeze(x_np, axis=axis)
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

print("\nDask behavior:")
try:
    result = da.squeeze(x_da, axis=axis).compute()
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# Test with negative out of bounds axis
print("\n" + "=" * 60)
print("\nTest 3: shape=[1], axis=-2")
shape = [1]
axis = -2

x_np = np.random.rand(*shape)
x_da = da.from_array(x_np, chunks='auto')

print("\nNumPy behavior:")
try:
    result = np.squeeze(x_np, axis=axis)
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

print("\nDask behavior:")
try:
    result = da.squeeze(x_da, axis=axis).compute()
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

# Test with valid axis that has size > 1 (should raise ValueError)
print("\n" + "=" * 60)
print("\nTest 4: shape=[2, 1], axis=0 (size > 1, should raise ValueError)")
shape = [2, 1]
axis = 0

x_np = np.random.rand(*shape)
x_da = da.from_array(x_np, chunks='auto')

print("\nNumPy behavior:")
try:
    result = np.squeeze(x_np, axis=axis)
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")

print("\nDask behavior:")
try:
    result = da.squeeze(x_da, axis=axis).compute()
    print(f"No error, result shape: {result.shape}")
except Exception as e:
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")