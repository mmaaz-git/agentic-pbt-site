import pandas as pd
from pandas.core.arrays._utils import to_numpy_dtype_inference
from pandas._libs import lib
import numpy as np

# Test case from bug report
arr = pd.array(['a', 'b', 'c'], dtype="string")

print(f"Array: {arr}")
print(f"Array dtype: {arr.dtype}")
print(f"Is numeric dtype: {pd.api.types.is_numeric_dtype(arr.dtype)}")

# Call the function with dtype=None
dtype, na_value = to_numpy_dtype_inference(arr, None, lib.no_default, False)

print(f"\nResult when calling with dtype=None:")
print(f"dtype: {dtype}")
print(f"na_value: {na_value}")

# Check the assertion from bug report
print(f"\nIs dtype None? {dtype is None}")

# Let's trace through the logic step by step
print("\n--- Tracing through function logic ---")
print(f"dtype is None: {None is None}")
print(f"is_numeric_dtype(arr.dtype): {pd.api.types.is_numeric_dtype(arr.dtype)}")
print(f"So we go to else branch at line 43-44")
print(f"Which sets dtype_given = True")

# Test with a numeric array for comparison
print("\n--- Testing with numeric array for comparison ---")
numeric_arr = pd.array([1, 2, 3], dtype="Int64")
print(f"Numeric Array: {numeric_arr}")
print(f"Numeric Array dtype: {numeric_arr.dtype}")
print(f"Is numeric dtype: {pd.api.types.is_numeric_dtype(numeric_arr.dtype)}")

dtype2, na_value2 = to_numpy_dtype_inference(numeric_arr, None, lib.no_default, False)
print(f"\nResult for numeric array with dtype=None:")
print(f"dtype: {dtype2}")
print(f"na_value: {na_value2}")

# Test with explicit dtype
print("\n--- Testing string array with explicit dtype ---")
dtype3, na_value3 = to_numpy_dtype_inference(arr, np.dtype('O'), lib.no_default, False)
print(f"Result with explicit object dtype:")
print(f"dtype: {dtype3}")
print(f"na_value: {na_value3}")