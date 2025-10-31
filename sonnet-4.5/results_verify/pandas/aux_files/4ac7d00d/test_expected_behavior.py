#!/usr/bin/env python3
"""Test what would be reasonable behavior for empty array density"""

import numpy as np

print("What would be reasonable for density of empty array?")
print("\nDensity is defined as: 'The percent of non-fill_value points, as decimal'")
print("\nFor an empty array:")
print("- Number of non-fill_value points = 0")
print("- Total number of points = 0")
print("- Density = 0/0 which is undefined mathematically")

print("\nCommon approaches for 0/0 in similar contexts:")
print("\n1. Return 0.0 (treat as 'no density'):")
print("   - Pros: Doesn't crash, consistent with 'no non-fill values'")
print("   - Cons: 0/0 ≠ 0 mathematically")

print("\n2. Return NaN (not a number):")
print("   - Pros: Mathematically correct for undefined")
print("   - Cons: May break downstream code expecting float")

print("\n3. Raise exception:")
print("   - Pros: Forces explicit handling")
print("   - Cons: Breaks for valid empty arrays")

print("\n4. Return 1.0 (100% dense since no sparse values):")
print("   - Pros: Conceptually, empty has no sparsity")
print("   - Cons: Counter-intuitive")

print("\nLet's see how similar libraries handle this:")

# NumPy behavior with similar operations
print("\n" + "="*50)
print("NumPy handling of 0/0:")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = np.divide(0, 0)
    print(f"np.divide(0, 0) = {result} (returns nan)")

# Check pandas DataFrame density behavior
print("\n" + "="*50)
print("Pandas DataFrame.sparse.density behavior:")
import pandas as pd

# Empty DataFrame
df_empty = pd.DataFrame({"A": pd.arrays.SparseArray([])})
try:
    density = df_empty.sparse.density
    print(f"Empty DataFrame sparse density: {density}")
except Exception as e:
    print(f"Empty DataFrame raises: {e}")

# DataFrame with single sparse column
df_single = pd.DataFrame({"A": pd.arrays.SparseArray([1])})
print(f"Single element DataFrame sparse density: {df_single.sparse.density}")

print("\n" + "="*50)
print("What about Series.sparse.density?")
series_empty = pd.Series(pd.arrays.SparseArray([]), dtype="Sparse[float]")
try:
    density = series_empty.sparse.density
    print(f"Empty Series sparse density: {density}")
except Exception as e:
    print(f"Empty Series raises: {e}")

print("\n" + "="*50)
print("\nConclusion:")
print("The bug report is about SparseArray.density crashing with ZeroDivisionError.")
print("This happens because the implementation divides by sp_index.length without checking for 0.")
print("Whether this should return 0.0, NaN, or raise an exception depends on the design intent.")