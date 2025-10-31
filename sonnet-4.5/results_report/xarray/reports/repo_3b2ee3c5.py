#!/usr/bin/env python3
"""Minimal reproduction of the RangeIndex.linspace bug with num=1"""

import sys
import traceback
import numpy as np

# First show that NumPy handles this case correctly
print("NumPy's behavior with num=1 and endpoint=True:")
print("np.linspace(0.0, 1.0, num=1, endpoint=True):", np.linspace(0.0, 1.0, num=1, endpoint=True))
print("np.linspace(5.0, 10.0, num=1, endpoint=True):", np.linspace(5.0, 10.0, num=1, endpoint=True))
print()

# Now try xarray's RangeIndex.linspace
print("Attempting xarray.indexes.RangeIndex.linspace with num=1 and endpoint=True:")
print("Code: RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim='x')")
print()

try:
    from xarray.indexes import RangeIndex
    idx = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim='x')
    print("Success! Result:", idx)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()