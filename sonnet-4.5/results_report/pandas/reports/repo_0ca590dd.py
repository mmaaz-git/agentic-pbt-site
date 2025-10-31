#!/usr/bin/env python3
"""
Minimal reproduction of SparseArray.density ZeroDivisionError on empty array.
"""

import pandas.core.arrays as arr

# Create an empty SparseArray with fill_value=0
sparse = arr.SparseArray([], fill_value=0)

# This should work but causes a ZeroDivisionError
density = sparse.density
print(f"Density of empty SparseArray: {density}")