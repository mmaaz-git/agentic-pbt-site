import numpy as np
from pandas.arrays import SparseArray
import pandas as pd

print("=" * 60)
print("Testing SparseArray max/min behavior with skipna=False")
print("=" * 60)

# Test case from the bug report
arr = SparseArray([1.0, np.nan, 0.0], fill_value=0.0)

print(f"\nArray values: {[1.0, np.nan, 0.0]}")
print(f"Fill value: 0.0")
print(f"SparseArray: {arr}")

# Test max with skipna=False
result_max_false = arr.max(skipna=False)
print(f"\nSparseArray.max(skipna=False): {result_max_false}")
print(f"Type: {type(result_max_false)}, Is NaN: {pd.isna(result_max_false)}")

# Test max with skipna=True
result_max_true = arr.max(skipna=True)
print(f"\nSparseArray.max(skipna=True): {result_max_true}")
print(f"Type: {type(result_max_true)}, Is NaN: {pd.isna(result_max_true)}")

# Test min with skipna=False
result_min_false = arr.min(skipna=False)
print(f"\nSparseArray.min(skipna=False): {result_min_false}")
print(f"Type: {type(result_min_false)}, Is NaN: {pd.isna(result_min_false)}")

# Test min with skipna=True
result_min_true = arr.min(skipna=True)
print(f"\nSparseArray.min(skipna=True): {result_min_true}")
print(f"Type: {type(result_min_true)}, Is NaN: {pd.isna(result_min_true)}")

print("\n" + "=" * 60)
print("Comparing with NumPy behavior")
print("=" * 60)

numpy_arr = np.array([1.0, np.nan, 0.0])
print(f"\nNumPy array: {numpy_arr}")
print(f"np.max(): {np.max(numpy_arr)}")
print(f"np.nanmax(): {np.nanmax(numpy_arr)}")

print("\n" + "=" * 60)
print("Comparing with Dense pandas Series behavior")
print("=" * 60)

series = pd.Series([1.0, np.nan, 0.0])
print(f"\nPandas Series: {series.values}")
print(f"Series.max(skipna=False): {series.max(skipna=False)}")
print(f"Series.max(skipna=True): {series.max(skipna=True)}")
print(f"Series.min(skipna=False): {series.min(skipna=False)}")
print(f"Series.min(skipna=True): {series.min(skipna=True)}")

print("\n" + "=" * 60)
print("Testing different scenarios")
print("=" * 60)

# Test with NaN in fill_value
arr_nan_fill = SparseArray([1.0, np.nan, 0.0], fill_value=np.nan)
print(f"\nArray with NaN fill_value:")
print(f"Values: {[1.0, np.nan, 0.0]}, fill_value: np.nan")
print(f"max(skipna=False): {arr_nan_fill.max(skipna=False)}")
print(f"max(skipna=True): {arr_nan_fill.max(skipna=True)}")

# Test without NaN values
arr_no_nan = SparseArray([1.0, 2.0, 0.0], fill_value=0.0)
print(f"\nArray without NaN:")
print(f"Values: {[1.0, 2.0, 0.0]}, fill_value: 0.0")
print(f"max(skipna=False): {arr_no_nan.max(skipna=False)}")
print(f"max(skipna=True): {arr_no_nan.max(skipna=True)}")

# Test with all NaN values
arr_all_nan = SparseArray([np.nan, np.nan, np.nan], fill_value=0.0)
print(f"\nArray with all NaN:")
print(f"Values: {[np.nan, np.nan, np.nan]}, fill_value: 0.0")
print(f"max(skipna=False): {arr_all_nan.max(skipna=False)}")
print(f"max(skipna=True): {arr_all_nan.max(skipna=True)}")