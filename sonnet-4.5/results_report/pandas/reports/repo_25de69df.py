import numpy as np
import pandas as pd
from pandas.core.arrays.sparse import SparseArray

# Test case showing the bug
arr = SparseArray([np.nan], fill_value=0.0)

print("Expected behavior (pandas Series):")
s = pd.Series([np.nan])
print(f"  pd.Series([np.nan]).sum(skipna=False) = {s.sum(skipna=False)}")
print(f"  pd.Series([np.nan]).sum(skipna=True) = {s.sum(skipna=True)}")

print("\nActual behavior (SparseArray):")
print(f"  SparseArray([np.nan]).sum(skipna=False) = {arr.sum(skipna=False)}")
print(f"  SparseArray([np.nan]).sum(skipna=True) = {arr.sum(skipna=True)}")

print("\n--- More examples ---")
print("\nExample with mix of NaN and regular values:")
arr2 = SparseArray([1.0, np.nan, 2.0], fill_value=0.0)
s2 = pd.Series([1.0, np.nan, 2.0])

print(f"Data: [1.0, np.nan, 2.0]")
print(f"  pd.Series.sum(skipna=False) = {s2.sum(skipna=False)}")
print(f"  SparseArray.sum(skipna=False) = {arr2.sum(skipna=False)}")
print(f"  pd.Series.sum(skipna=True) = {s2.sum(skipna=True)}")
print(f"  SparseArray.sum(skipna=True) = {arr2.sum(skipna=True)}")

print("\n--- Internal details ---")
print(f"\nFor SparseArray([np.nan]):")
print(f"  sp_values: {arr.sp_values}")
print(f"  fill_value: {arr.fill_value}")
print(f"  _valid_sp_values: {arr._valid_sp_values}")
print(f"  Length of sp_values: {len(arr.sp_values)}")
print(f"  Length of _valid_sp_values: {len(arr._valid_sp_values)}")