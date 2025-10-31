#!/usr/bin/env python3
"""Test to check the internal implementation details of dask.diff"""
import numpy as np
import dask.array as da
import dask.array.routines as dar
import inspect

print("="*60)
print("EXAMINING INTERNAL IMPLEMENTATION OF dask.array.diff")
print("="*60)

# First, let's check what imports are available in the routines module
print("\n1. Checking imports in dask.array.routines module:")
print("-"*40)

# Check if broadcast_to is imported from dask.array.core
source = inspect.getsource(dar.diff)
print("Source code of diff function (showing lines with broadcast_to):")
for i, line in enumerate(source.split('\n'), 1):
    if 'broadcast_to' in line:
        print(f"  Line {i}: {line.strip()}")

# Now let's trace through what actually happens
print("\n2. Testing the actual execution:")
print("-"*40)

# Create test array
arr = da.from_array(np.array([1, 2, 3, 4, 5]), chunks=3)

# Test with prepend
print("\nTesting with prepend=10:")
result_prepend = da.diff(arr, prepend=10)
print(f"  Result type: {type(result_prepend)}")
print(f"  Result: {result_prepend.compute()}")

# Test with append
print("\nTesting with append=10:")
result_append = da.diff(arr, append=10)
print(f"  Result type: {type(result_append)}")
print(f"  Result: {result_append.compute()}")

# Let's check what broadcast_to functions are available
print("\n3. Checking broadcast_to functions available:")
print("-"*40)

# Check dask's broadcast_to
from dask.array.core import broadcast_to as dask_broadcast_to
print(f"dask.array.core.broadcast_to: {dask_broadcast_to}")
print(f"  Returns: {type(dask_broadcast_to(10, (1,)))}")

# Check numpy's broadcast_to
print(f"\nnp.broadcast_to: {np.broadcast_to}")
print(f"  Returns: {type(np.broadcast_to(10, (1,)))}")

# Test the difference
print("\n4. Testing the difference between implementations:")
print("-"*40)

scalar_val = 42
shape = (1, 3)

# Using dask's broadcast_to
dask_result = dask_broadcast_to(scalar_val, shape)
print(f"dask.array.core.broadcast_to({scalar_val}, {shape}):")
print(f"  Type: {type(dask_result)}")
print(f"  Is dask array: {isinstance(dask_result, da.Array)}")
if isinstance(dask_result, da.Array):
    print(f"  Computed value: {dask_result.compute()}")

# Using numpy's broadcast_to
numpy_result = np.broadcast_to(scalar_val, shape)
print(f"\nnp.broadcast_to({scalar_val}, {shape}):")
print(f"  Type: {type(numpy_result)}")
print(f"  Is dask array: {isinstance(numpy_result, da.Array)}")
print(f"  Value: {numpy_result}")

# Check lazy evaluation principle
print("\n5. Checking lazy evaluation principle:")
print("-"*40)

# Create a dask array and append to it
arr = da.from_array(np.array([1, 2, 3]), chunks=2)

# With the current implementation (using np.broadcast_to for append)
result = da.diff(arr, append=10)
print(f"Result with append is lazy (dask array): {isinstance(result, da.Array)}")
print(f"Result graph size: {len(result.__dask_graph__())}")

# Check consistency
print("\n6. Verifying the inconsistency claim:")
print("-"*40)
print("According to the bug report:")
print("  - Line 593 uses: broadcast_to (from dask.array.core)")
print("  - Line 603 uses: np.broadcast_to (from numpy)")
print("\nThis creates an inconsistency where:")
print("  - prepend parameter uses dask's lazy broadcast_to")
print("  - append parameter uses numpy's eager broadcast_to")
print("\nWhile both produce correct results, the implementation is inconsistent.")