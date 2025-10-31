#!/usr/bin/env python3
"""Test if the function works with numpy arrays and pandas Series."""

import numpy as np
import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

# Test with numpy array
print("Testing with numpy array...")
arr = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
try:
    divisions, locations = sorted_division_locations(arr, chunksize=2)
    print(f"SUCCESS with numpy array: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"FAILED with numpy array: {type(e).__name__}: {e}")

# Test with pandas Series
print("\nTesting with pandas Series...")
series = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])
try:
    divisions, locations = sorted_division_locations(series, chunksize=2)
    print(f"SUCCESS with pandas Series: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"FAILED with pandas Series: {type(e).__name__}: {e}")

# Test with pandas Index
print("\nTesting with pandas Index...")
index = pd.Index(['A', 'B', 'C', 'D', 'E', 'F'])
try:
    divisions, locations = sorted_division_locations(index, chunksize=2)
    print(f"SUCCESS with pandas Index: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"FAILED with pandas Index: {type(e).__name__}: {e}")