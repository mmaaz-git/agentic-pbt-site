import pandas.arrays as pa
import numpy as np
import pandas as pd

print("=" * 60)
print("Testing BooleanArray any() and all() behavior with NA values")
print("=" * 60)

# Test 1: All NA array
arr_all_na = pa.BooleanArray(np.array([False], dtype='bool'),
                              np.array([True], dtype='bool'))

print(f"\nTest 1: all-NA array")
print(f"Array: {arr_all_na}")
print(f"any(): {arr_all_na.any()}")
print(f"all(): {arr_all_na.all()}")
print(f"Expected: both should be pd.NA (according to bug report)")
print(f"Actual any() is NA: {pd.isna(arr_all_na.any())}")
print(f"Actual all() is NA: {pd.isna(arr_all_na.all())}")

# Test with skipna=False
print(f"\nWith skipna=False:")
print(f"any(skipna=False): {arr_all_na.any(skipna=False)}")
print(f"all(skipna=False): {arr_all_na.all(skipna=False)}")
print(f"any(skipna=False) is NA: {pd.isna(arr_all_na.any(skipna=False))}")
print(f"all(skipna=False) is NA: {pd.isna(arr_all_na.all(skipna=False))}")

# Test 2: [False, NA]
arr_false_na = pa.BooleanArray(np.array([False, False], dtype='bool'),
                                np.array([False, True], dtype='bool'))
print(f"\n\nTest 2: [False, NA]")
print(f"Array: {arr_false_na}")
print(f"any(): {arr_false_na.any()} (bug report says should be NA)")
print(f"all(): {arr_false_na.all()} (bug report says should be False)")
print(f"any(skipna=False): {arr_false_na.any(skipna=False)}")
print(f"all(skipna=False): {arr_false_na.all(skipna=False)}")

# Test 3: [True, NA]
arr_true_na = pa.BooleanArray(np.array([True, False], dtype='bool'),
                               np.array([False, True], dtype='bool'))
print(f"\n\nTest 3: [True, NA]")
print(f"Array: {arr_true_na}")
print(f"any(): {arr_true_na.any()} (bug report says should be True)")
print(f"all(): {arr_true_na.all()} (bug report says should be NA)")
print(f"any(skipna=False): {arr_true_na.any(skipna=False)}")
print(f"all(skipna=False): {arr_true_na.all(skipna=False)}")

# Test what documentation says
print("\n" + "=" * 60)
print("Documentation examples from help(pa.BooleanArray):")
print("=" * 60)

# Example from docs
print("\n>>> pd.array([pd.NA], dtype=\"boolean\").any()")
arr_doc = pd.array([pd.NA], dtype="boolean")
print(f"Result: {arr_doc.any()}")

print("\n>>> pd.array([pd.NA], dtype=\"boolean\").all()")
print(f"Result: {arr_doc.all()}")

print("\n>>> pd.array([pd.NA], dtype=\"boolean\").any(skipna=False)")
print(f"Result: {arr_doc.any(skipna=False)}")

print("\n>>> pd.array([pd.NA], dtype=\"boolean\").all(skipna=False)")
print(f"Result: {arr_doc.all(skipna=False)}")

# Test empty array case (from docs)
print("\n>>> pd.array([], dtype=\"boolean\").any()")
empty = pd.array([], dtype="boolean")
print(f"Result: {empty.any()}")

print("\n>>> pd.array([], dtype=\"boolean\").all()")
print(f"Result: {empty.all()}")