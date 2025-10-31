import pandas as pd
import numpy as np

print("Testing pandas behavior with all-NA arrays...\n")

# Test with regular pandas nullable arrays
print("1. Testing Int64 nullable array with all NAs:")
int_array_na = pd.array([pd.NA, pd.NA, pd.NA], dtype='Int64')
print(f"Array: {int_array_na}")
print(f"all(skipna=True): {int_array_na.all(skipna=True)}")
print(f"any(skipna=True): {int_array_na.any(skipna=True)}")
print(f"all(skipna=False): {int_array_na.all(skipna=False)}")
print(f"any(skipna=False): {int_array_na.any(skipna=False)}")

print("\n2. Testing boolean nullable array with all NAs:")
bool_array_na = pd.array([pd.NA, pd.NA], dtype='boolean')
print(f"Array: {bool_array_na}")
print(f"all(skipna=True): {bool_array_na.all(skipna=True)}")
print(f"any(skipna=True): {bool_array_na.any(skipna=True)}")
print(f"all(skipna=False): {bool_array_na.all(skipna=False)}")
print(f"any(skipna=False): {bool_array_na.any(skipna=False)}")

print("\n3. Testing empty boolean array:")
empty_array = pd.array([], dtype='boolean')
print(f"Array: {empty_array}")
print(f"all(skipna=True): {empty_array.all(skipna=True)}")
print(f"any(skipna=True): {empty_array.any(skipna=True)}")

print("\n4. Testing ArrowExtensionArray boolean with NAs:")
import pyarrow as pa
from pandas.core.arrays.arrow import ArrowExtensionArray

arrow_bool_na = ArrowExtensionArray(pa.array([None, None], type=pa.bool_()))
print(f"Array: {arrow_bool_na}")
print(f"all(skipna=True): {arrow_bool_na.all(skipna=True)}")
print(f"any(skipna=True): {arrow_bool_na.any(skipna=True)}")

print("\n5. Testing ArrowExtensionArray int with NAs:")
arrow_int_na = ArrowExtensionArray(pa.array([None, None], type=pa.int64()))
print(f"Array: {arrow_int_na}")
print(f"all(skipna=True): {arrow_int_na.all(skipna=True)}")
print(f"any(skipna=True): {arrow_int_na.any(skipna=True)}")