#!/usr/bin/env python3
"""Test script to reproduce the dask.array.eye bug"""

import sys
import traceback
import numpy as np
import dask.array as da

print("=" * 60)
print("Testing dask.array.eye bug report")
print("=" * 60)

# First, let's verify NumPy behavior
print("\n1. NumPy eye function with N=2, M=3:")
np_result = np.eye(2, M=3, k=0)
print(f"   Shape: {np_result.shape}")
print(f"   Result:\n{np_result}")

# Now test the reported bug case
print("\n2. Testing dask.array.eye with N=2, M=3, chunks=3:")
try:
    arr = da.eye(2, chunks=3, M=3, k=0)
    print(f"   Created array with shape {arr.shape}")
    print(f"   Chunks: {arr.chunks}")
    result = arr.compute()
    print(f"   Computed successfully!")
    print(f"   Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test case that should work according to bug report
print("\n3. Testing dask.array.eye with N=2, M=3, chunks=2 (chunks < M):")
try:
    arr = da.eye(2, chunks=2, M=3, k=0)
    print(f"   Created array with shape {arr.shape}")
    print(f"   Chunks: {arr.chunks}")
    result = arr.compute()
    print(f"   Computed successfully!")
    print(f"   Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Test square matrix case
print("\n4. Testing dask.array.eye with N=3, M=3, chunks=3 (square matrix):")
try:
    arr = da.eye(3, chunks=3, M=3, k=0)
    print(f"   Created array with shape {arr.shape}")
    print(f"   Chunks: {arr.chunks}")
    result = arr.compute()
    print(f"   Computed successfully!")
    print(f"   Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Additional test cases
print("\n5. Testing dask.array.eye with N=4, M=2, chunks=3 (N > M, chunks > M):")
try:
    arr = da.eye(4, chunks=3, M=2, k=0)
    print(f"   Created array with shape {arr.shape}")
    print(f"   Chunks: {arr.chunks}")
    result = arr.compute()
    print(f"   Computed successfully!")
    print(f"   Result:\n{result}")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n6. Checking the actual chunks assignment issue:")
print("   Looking at the code at line 624...")
print("   The issue is chunks=(chunks, chunks) where chunks=vchunks[0]")
print("   This means both row and column chunks are set to the same value")
print("   But vchunks and hchunks may be different for non-square matrices!")