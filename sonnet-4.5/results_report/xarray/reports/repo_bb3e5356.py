#!/usr/bin/env python3
"""Minimal reproduction of the ZeroDivisionError bug in xarray.indexes.RangeIndex.linspace"""

from xarray.indexes import RangeIndex

# This should create a single-point index at position 0.0
# but instead crashes with ZeroDivisionError
try:
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
    print(f"Success! Created index with size {index.size}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError occurred: {e}")
    import traceback
    traceback.print_exc()