#!/usr/bin/env python3
"""Debug the bug to understand what's happening"""

from dask.dataframe.io.io import sorted_division_locations
import traceback

# Let's trace exactly where the error occurs
L = ['A', 'B', 'C', 'D', 'E', 'F']

try:
    result = sorted_division_locations(L, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Let's also check what tolist does
from dask.dataframe.dispatch import tolist

print("\n\nTesting tolist directly:")
print(f"tolist on list: ")
try:
    result = tolist(['A', 'B', 'C'])
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")

import numpy as np
print(f"\ntolist on numpy array: ")
try:
    result = tolist(np.array(['A', 'B', 'C']))
    print(f"Success: {result}")
except Exception as e:
    print(f"Error: {e}")