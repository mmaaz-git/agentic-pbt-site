#!/usr/bin/env python3
"""
Minimal reproduction of the RangeIndex negative size bug.
"""

import sys
import os
import numpy as np

# Add the xarray environment to path
sys.path.insert(0, '/home/npc/miniconda/lib/python3.13/site-packages')

from xarray.indexes.range_index import RangeIndex
import xarray as xr

print("=== Testing RangeIndex.arange with mismatched step direction ===")
print()

# The problematic case: positive step going from 1.0 to 0.0
print("Test case: RangeIndex.arange(1.0, 0.0, 1.0, dim='x')")
print("Expected: size should be 0 (empty range)")
print()

# Create the index
index = RangeIndex.arange(1.0, 0.0, 1.0, dim="x")

# Show the problematic negative size
print(f"Actual result:")
print(f"  index.size = {index.size}")
print(f"  index.start = {index.start}")
print(f"  index.stop = {index.stop}")
print(f"  index.step = {index.step}")
print()

# Compare with NumPy's behavior
print("NumPy comparison:")
np_result = np.arange(1.0, 0.0, 1.0)
print(f"  np.arange(1.0, 0.0, 1.0) = {np_result}")
print(f"  np.arange(1.0, 0.0, 1.0).size = {np_result.size}")
print()

# Try to use this index in an xarray Dataset
print("=== Creating xarray Dataset with the negative-sized index ===")
try:
    coords = xr.Coordinates.from_xindex(index)
    ds = xr.Dataset(coords=coords)
    print(f"Dataset created: {ds}")
    print()

    # Try to add a data variable - this will fail due to negative dimension
    print("Attempting to add a data variable...")
    ds["temperature"] = xr.DataArray(np.zeros(abs(index.size)), dims=["x"])
    print("Success (should not happen)")

except Exception as e:
    print(f"Error when using negative-sized index: {e}")
    print(f"Error type: {type(e).__name__}")