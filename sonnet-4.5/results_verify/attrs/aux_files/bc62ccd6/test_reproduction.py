#!/usr/bin/env python3
"""Reproduce the bug from the report"""

import sys
import traceback

# First, test the simple reproduction case from the bug report
print("=" * 60)
print("Testing simple reproduction from bug report")
print("=" * 60)

try:
    from dask.dataframe.io.io import sorted_division_locations

    L = ['A', 'B', 'C', 'D', 'E', 'F']
    print(f"Input list: {L}")
    print(f"Calling sorted_division_locations(L, chunksize=2)")

    divisions, locations = sorted_division_locations(L, chunksize=2)
    print(f"Success! Result: divisions={divisions}, locations={locations}")

except Exception as e:
    print(f"ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Now test the hypothesis test case
print("\n" + "=" * 60)
print("Testing with hypothesis test inputs")
print("=" * 60)

try:
    from dask.dataframe.io.io import sorted_division_locations

    seq = [0]
    chunksize = 1
    print(f"Input: seq={seq}, chunksize={chunksize}")

    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    print(f"Success! Result: divisions={divisions}, locations={locations}")

except Exception as e:
    print(f"ERROR: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test with different types of inputs
print("\n" + "=" * 60)
print("Testing with numpy array and pandas Series")
print("=" * 60)

try:
    import numpy as np
    import pandas as pd

    # Test with numpy array
    arr = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
    print(f"Testing with numpy array: {arr}")
    divisions, locations = sorted_division_locations(arr, chunksize=2)
    print(f"numpy array Success! Result: divisions={divisions}, locations={locations}")

    # Test with pandas Series
    series = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])
    print(f"\nTesting with pandas Series: {list(series)}")
    divisions, locations = sorted_division_locations(series, chunksize=2)
    print(f"pandas Series Success! Result: divisions={divisions}, locations={locations}")

except Exception as e:
    print(f"ERROR with numpy/pandas: {e}")
    traceback.print_exc()